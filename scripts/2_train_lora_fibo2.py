# -*- coding: utf-8 -*-
"""
2_train_lora_fibo2.py — Entrenamiento LoRA para Fibo 2.0 (BRIA, Transformer NF4)
LoRA training for Fibo 2.0 (BRIA, NF4 Transformer)

Dos fases: el pre-cache ya dejó los latentes y los embeddings en la caché, así que aquí
no se usa el text encoder (Qwen3-VL-4B) ni se codifica ninguna imagen: solo el
transformer NF4, el LoRA y el optimizador.

Lee configuración desde train_settings_fibo2.json si existe.
Reads configuration from train_settings_fibo2.json if present.
"""
import os
import gc
import math
import time
import random
import json
import signal
import sys
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load, load_file
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from i18n import t
import lora_options
import training_metrics
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import fibo2_caption
import fibo2_lora

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# El modelo NF4 publicado en Hugging Face (lo mismo que el conversor genera en local).
HF_REPO_ID = "AcademiaSD/Fibo-2.0-NF4-for-LoRA-Training"

# La receta de BRIA (fine_tuning/train_lora.py del modelo):
#   - sigma: 90% logit-normal con la media que da el tamaño de la imagen, 10% uniforme.
#   - x_t = sigma * ruido + (1 - sigma) * x0;  objetivo = ruido - x0;  MSE en fp32.
#   - text dropout 10% (caption vacío), context dropout 20% en modo edición.
#   - AdamW (wd 0.01), grad clip 1.0. El LR lo manda el esquema de la GUI (coseno por defecto).
TEXT_DROPOUT = 0.1
CONTEXT_DROPOUT = 0.2
# Los textos del caption solo se ven a partir de este nivel de ruido (como en el pipeline).
TEXT_IMAGES_MAX_SIGMA = 0.9

DEFAULTS = {
    "model_id": "Models/Fibo2_Dev-NF4",
    "cache_dir": "./Cache_Data/cached_data_fibo2",
    "output_dir": "./Outputs/fibo2_lora_output",
    "total_steps": 1200,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 3e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 8,
    "lora_alpha": 8,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    "use_lokr": False,
    "lokr_factor": 16,
    "lora_targets": "all",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "seed": -1,
    "timestep_sampling": "fibo",
    "preview_every": 50,
    "preview_steps": 30,
    "preview_cfg": 5.0,
    "preview_size": 0,
    "preview_width": 0,
    "preview_height": 0,
    "use_turbo": False,
    "turbo_lora_strength": 15.0,
    "preview_caption_mode": "custom",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_fibo2.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID          = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = cfg.get("grad_accum_steps",  DEFAULTS["grad_accum_steps"])
LR                = cfg.get("lr",                DEFAULTS["lr"])
MIN_LR_RATIO      = cfg.get("min_lr_ratio",      DEFAULTS["min_lr_ratio"])
WARMUP_STEPS      = cfg.get("warmup_steps",      DEFAULTS["warmup_steps"])
LR_SCHEDULER      = cfg.get("lr_scheduler",      DEFAULTS["lr_scheduler"])
LR_CYCLES         = cfg.get("lr_cycles",         DEFAULTS["lr_cycles"])
LORA_RANK         = cfg.get("lora_rank",         DEFAULTS["lora_rank"])
LORA_ALPHA        = cfg.get("lora_alpha",        DEFAULTS["lora_alpha"])
LORA_TARGETS      = "all" if cfg.get("lora_targets", DEFAULTS["lora_targets"]) == "all" else "blocks"
# rsLoRA escala el LoRA por alpha/√rank en vez de alpha/rank; LoRA+ da a lora_B un LR ratio veces mayor.
# Las dos cambian cómo aprende, no el formato: el LoRA exportado es estándar (ver export_alpha).
# LoKr (LyCORIS) sustituye a LoRA: cada capa aprende w1 ⊗ (w2_a·w2_b); con él no se usan las otras dos.
USE_RSLORA        = bool(cfg.get("use_rslora",   DEFAULTS["use_rslora"]))
USE_LORAPLUS      = bool(cfg.get("use_loraplus", DEFAULTS["use_loraplus"]))
LORAPLUS_RATIO    = float(cfg.get("loraplus_ratio", DEFAULTS["loraplus_ratio"]))
USE_LOKR          = bool(cfg.get("use_lokr",     DEFAULTS["use_lokr"]))
LOKR_FACTOR       = max(1, int(cfg.get("lokr_factor", DEFAULTS["lokr_factor"])))
if USE_LOKR:
    USE_RSLORA = USE_LORAPLUS = False
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_WIDTH     = int(cfg.get("preview_width",  DEFAULTS["preview_width"]))
PREVIEW_HEIGHT    = int(cfg.get("preview_height", DEFAULTS["preview_height"]))
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

# Previews rápidas: menos pasos y sin guía. En Fibo 2.0 Dev no hay un LoRA destilado
# (el modelo "Edit Turbo" es otro checkpoint, no un LoRA), así que el interruptor solo
# recorta los pasos y quita la guía: la imagen sale más floja, pero la preview tarda la mitad.
USE_TURBO           = cfg.get("use_turbo",           DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = float(cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"]))

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./Cache_Data/cached_data_fibo2_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./Outputs/fibo2_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Output Dir'):<22}: {OUTPUT_DIR}")
print(f"  {t('Total Steps'):<22}: {TOTAL_STEPS}")
print(f"  {t('Learning Rate'):<22}: {LR}")
print(f"  {t('LR Scheduler'):<22}: {lora_options.lr_scheduler_label(LR_SCHEDULER, WARMUP_STEPS, LR_CYCLES)}")
print(f"  {'LoRA Rank/Alpha':<22}: {LORA_RANK}/{LORA_ALPHA}")
print(f"  {'rsLoRA / LoRA+':<22}: {'on' if USE_RSLORA else 'off'} / {f'x{LORAPLUS_RATIO:g}' if USE_LORAPLUS else 'off'}")
print(f"  {'LoKr':<22}: {f'factor {LOKR_FACTOR}' if USE_LOKR else 'off'}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG} | {t('Preview Size')} {PREVIEW_SIZE or t('Training')}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")
print(f"  {'Fast Previews':<22}: {t('ON ({n} steps)', n=int(TURBO_LORA_STRENGTH)) if USE_TURBO else t('OFF')}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")

if SEED > 0:
    torch.manual_seed(SEED)
    random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_hf_token():
    if os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token_data = json.load(f)
                token = token_data.get("token", "").strip()
                if token:
                    print("✓ " + t("Using HF Token"))
                    return token
        except Exception:
            pass
    return None


def enable_hf_file_progress():
    try:
        import tqdm
        import tqdm.auto
        import tqdm.std

        def patch_tqdm(cls):
            orig_init = cls.__init__
            def new_init(self, *args, **kwargs):
                kwargs['disable'] = False
                kwargs['mininterval'] = 0.5
                orig_init(self, *args, **kwargs)
            cls.__init__ = new_init

        patch_tqdm(tqdm.std.tqdm)
        patch_tqdm(tqdm.auto.tqdm)
        if hasattr(tqdm, 'tqdm'):
            patch_tqdm(tqdm.tqdm)

        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    except Exception:
        pass

    os.environ["TQDM_DISABLE"] = "0"
    os.environ["TQDM_MININTERVAL"] = "0.5"


def ensure_model_downloaded(local_path):
    if (os.path.exists(os.path.join(local_path, "transformer", "index.json"))
            and os.path.exists(os.path.join(local_path, "vae", "config.json"))):
        print("[OK] " + t("Local model found at: {path}", path=local_path))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=HF_REPO_ID))

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(t("huggingface_hub is required. Install it with: pip install huggingface_hub"))

    # El entrenamiento no usa el text encoder: ya esta todo en el pre-cache.
    downloaded_path = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["text_encoder_*/*"],
    )

    print("[OK] " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    """
    El nivel de ruido de cada muestra.
      fibo:          la receta de BRIA: 90% logit-normal con la media del tamaño, 10% uniforme.
      logit_normal: logit-normal con esa media, siempre.
      shift:         uniforme desplazado, como el resto de entrenadores del estudio.
    """
    mu = calculate_shift(image_seq_len, *shift_cfg)
    if TIMESTEP_SAMPLING == "shift":
        u = torch.rand(batch_size, device=device)
        e_mu = math.exp(mu)
        sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
        return sigma.clamp(1e-4, 1.0 - 1e-4)
    normal = torch.sigmoid(torch.randn(batch_size, device=device) + mu)
    if TIMESTEP_SAMPLING == "logit_normal":
        return normal
    # "fibo": el 10% de las veces un sigma uniforme sin desplazar.
    pick_uniform = torch.rand(batch_size, device=device) < 0.1
    uniform = torch.rand(batch_size, device=device)
    return torch.where(pick_uniform, uniform, normal)


def load_nf4_transformer(model_dir):
    """
    Construye el Transformer vacío desde transformer/config.json y lo rellena con la
    caché NF4 (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    """
    from diffusers import BriaFibo2Transformer2DModel

    cache_dir = os.path.join(model_dir, "transformer")
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(t("index.json not found in the NF4 cache: {path}", path=cache_dir))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = BriaFibo2Transformer2DModel.from_config(BriaFibo2Transformer2DModel.load_config(cache_dir))

    weights_dir = os.path.join(cache_dir, "weights")

    def set_module(name, module):
        parent_name, _, child_name = name.rpartition(".")
        setattr(transformer.get_submodule(parent_name) if parent_name else transformer, child_name, module)

    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}
            bias_data = f.get_tensor("bias") if "bias" in f.keys() else None

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device="cpu")
        weight.bnb_quantized = True

        # En meta: nn.Linear.__init__ reserva y rellena con kaiming una matriz FP32 que se tira
        # en la línea siguiente.
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info.get("bias", False),
                                quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        if bias_data is not None:
            layer.bias = torch.nn.Parameter(bias_data.to(torch.bfloat16), requires_grad=False)
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=info.get("bias", False),
                                    dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    transformer.load_state_dict(load_file(os.path.join(cache_dir, index["others"])), strict=False, assign=True)

    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=missing[:5]))

    print("[OK] " + t("NF4 layers: {a} | BF16 layers: {b}", a=len(index['quantized']), b=len(index['unquantized'])))
    return transformer


def is_block_target(name):
    """Las lineales de la atención y del MLP de los 44 bloques (y de los 2 refiner)."""
    if not (name.startswith("layers.") or name.startswith("noise_refiner.")):
        return False
    return any(part in name for part in (".attention.", ".feed_forward.", ".cross_attention."))


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab Fibo 2.0",
        "ss_sd_model_name": "Fibo-2.0-dev",
        "ss_base_model_version": "Fibo-2.0-dev",
        "ss_network_module": "lycoris.kohya" if USE_LOKR else "peft.LoraModel",
        "ss_network_dim": LORA_RANK,
        "ss_network_alpha": export_alpha(),
        "lora_targets": LORA_TARGETS,
        "ss_learning_rate": LR,
        "ss_lr_scheduler": LR_SCHEDULER,
        "ss_lr_warmup_steps": WARMUP_STEPS,
        "ss_max_train_steps": TOTAL_STEPS,
        "ss_steps": step,
        "ss_batch_size_per_device": BATCH_SIZE,
        "ss_gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "ss_seed": SEED,
        "ss_mixed_precision": "bf16",
    }

    # La GUI lee estas claves al exportar o descargar para poner el sufijo _rs / _plus / _lokr al nombre.
    if USE_RSLORA:
        meta["rslora"] = "true"
        meta["rslora_train_alpha"] = LORA_ALPHA
    if USE_LORAPLUS:
        meta["loraplus_ratio"] = LORAPLUS_RATIO
    if USE_RSLORA or USE_LORAPLUS:
        meta["ss_network_args"] = json.dumps({"rslora": USE_RSLORA,
                                              "loraplus_lr_ratio": LORAPLUS_RATIO if USE_LORAPLUS else None})
    if USE_LOKR:
        meta["lokr_factor"] = LOKR_FACTOR
        meta["ss_network_args"] = json.dumps({"algo": "lokr", "factor": LOKR_FACTOR})

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "fibo2_lora"

    # La resolución la fija el pre-caché del proyecto.
    pc_json = os.path.join(CACHE_DIR, f"pre_cache_settings_{PROJECT_NAME}.json")
    if os.path.exists(pc_json):
        with open(pc_json, "r", encoding="utf-8") as f:
            area = json.load(f).get("target_area")
        if area:
            side = int(round(float(area) ** 0.5))
            meta["ss_resolution"] = f"({side},{side})"

    # safetensors exige que todos los valores sean str.
    return {k: str(v) for k, v in meta.items()}


def export_alpha():
    return lora_options.export_alpha(LORA_ALPHA, LORA_RANK, USE_RSLORA)


def _comfy_key(module):
    """La clave de ComfyUI de una capa de diffusers que se mapea 1:1."""
    old, new, group = fibo2_lora.classify(module)
    if old is None:
        return module
    return module[: -len(old)] + new


def _export_lora(model, path, step):
    # Claves diffusion_model.<capa de ComfyUI>.lora_A/lora_B + alpha por capa: es el
    # formato que ComfyUI carga directamente. Su Fibo 2.0 es el modelo repack, con la
    # atencion (qkv), el MLP (w13) y la atencion cruzada (to_kv) fundidas: las capas que
    # se funden se rearman en un LoRA de rango multiple (ver fibo2_lora).
    if USE_LOKR:
        clean, skipped = fibo2_lora.build_lokr(model.lokr_layers(), _comfy_key, export_alpha())
        if skipped:
            print("  [!] " + t("The LoKr cannot be written in ComfyUI's fused layers: {n} layers are left out.",
                              n=len(skipped)))
        save_file(clean, path, metadata=build_lora_metadata(step))
        return
    pairs = {}
    for k, v in get_peft_model_state_dict(model).items():
        module, _, rest = k.replace("base_model.model.", "").partition(".lora_")
        pairs.setdefault(module, {})["A" if rest.startswith("A") else "B"] = v
    clean, skipped = fibo2_lora.build_lora(pairs, export_alpha())
    if skipped:
        print("  [!] " + t("{n} layers could not be mapped to ComfyUI.", n=len(skipped)))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLFlux2.from_pretrained(
                MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
        return cls.vae


def decode_latents(vae, latents):
    """
    Del latente a la imagen: se deshace la normalización del BatchNorm del VAE (en el
    formato parcheado, como lo hace el pipeline) y se decodifica.
    """
    lat = BriaFibo2Pipeline._patchify_latents(latents.float())
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    scale = (1 / torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps)).view(1, -1, 1, 1)
    scale = scale.to(lat.device, lat.dtype)
    lat = lat / scale + mean
    lat = BriaFibo2Pipeline._unpatchify_latents(lat)
    return vae.decode(lat.to(vae.dtype), return_dict=False)[0]


def denoise(model, scheduler, sample, neg, size, generator):
    """
    La pasada de denoise de la preview: los mismos pasos que el pipeline (sigmas del 1 al
    1/pasos con el desplazamiento por el tamaño de la imagen) y la guía con el caption vacío.
    """
    device = "cuda"
    H, W = size
    patch = model.config.patch_size
    n_tokens = (H // patch) * (W // patch)

    latents = torch.randn((1, model.config.in_channels, H, W), generator=generator, device=device, dtype=torch.float32)
    embed = sample["emb"].unsqueeze(0).to(device)
    mask = sample.get("mask")
    mask = None if mask is None else mask.unsqueeze(0).to(device)
    base_embed = sample.get("base_emb")
    base_mask = sample.get("base_msk")
    if base_embed is not None:
        base_embed = base_embed.unsqueeze(0).to(device)
        base_mask = None if base_mask is None else base_mask.unsqueeze(0).to(device)

    refs = sample.get("ref")
    # El latente de la imagen de antes ya viene con su lote (1, canales, alto, ancho).
    # El transformer lo espera en bf16, como los tokens de la imagen.
    context = None if refs is None else [refs.to(device=device, dtype=torch.bfloat16)]

    steps = int(TURBO_LORA_STRENGTH) if USE_TURBO else int(PREVIEW_STEPS)
    steps = max(1, steps)
    use_cfg = (not USE_TURBO) and PREVIEW_CFG > 1.0 and neg is not None

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(n_tokens, scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    scheduler.set_begin_index(0)

    for i, t in enumerate(scheduler.timesteps):
        timestep = t.expand(1).to(torch.float32) / 1000
        use_text_images = base_embed is None or float(t) / 1000 <= TEXT_IMAGES_MAX_SIGMA
        pred = model(
            hidden_states=latents.to(torch.bfloat16),
            timestep=timestep,
            encoder_hidden_states=embed if use_text_images else base_embed,
            encoder_attention_mask=mask if use_text_images else base_mask,
            context_latents=context,
            return_dict=False,
        )[0].float()
        if use_cfg:
            neg_pred = model(
                hidden_states=latents.to(torch.bfloat16),
                timestep=timestep,
                encoder_hidden_states=neg[0].unsqueeze(0).to(device),
                encoder_attention_mask=neg[1].unsqueeze(0).to(device) if neg[1] is not None else None,
                context_latents=context,
                return_dict=False,
            )[0].float()
            pred = neg_pred + PREVIEW_CFG * (pred - neg_pred)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    return latents


def run_preview(model, scheduler, sample, neg, size, step):
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(model, scheduler, sample, neg, size, g)

            vae = VaeHolder.get().to("cuda")
            try:
                image = decode_latents(vae, latents)
            except torch.OutOfMemoryError:
                # Con 8 GB y el entrenamiento cargado (1024² o más) no cabe entera: por mosaicos
                # usa ~1 GB en vez de ~4 GB, a cambio de alguna marca leve en las uniones.
                print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                image = decode_latents(vae, latents)
            proc = BriaFibo2ImageProcessor(vae_scale_factor=16)
            pil = proc.postprocess(image, output_type="pil")[0]
            vae.to("cpu")

        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        pil.save(out)
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
        if was_training:
            model.train()
        free_vram()


def check_custom_prompt(has_custom, edit_mode=False):
    """
    El trainer no tiene text encoder (es la razón de ser del pre-cache): el prompt manual
    lo codifica el Pre-Cache, que al relanzarlo salta las imágenes ya cacheadas.
    Aquí solo se avisa si falta o si el de los ajustes ya no es el codificado.
    """
    if PREVIEW_CAPTION_MODE != "custom":
        return
    if not has_custom:
        print("\n[PREVIEW] " + t("Custom mode but the cache has no encoded prompt: set it, Save JSON and run Pre-Cache "
                                "(it only re-encodes the texts). Using the first caption meanwhile."))
        return

    wanted = fibo2_caption.with_trigger(PREVIEW_CUSTOM_PROMPT, TRIGGER_WORD, is_edit=edit_mode)
    prompt_file = os.path.join(CACHE_DIR, "_custom_prompt.txt")
    encoded = open(prompt_file, "r", encoding="utf-8").read() if os.path.exists(prompt_file) else ""
    print(f"\n[PREVIEW] {t('Custom prompt')}: '{encoded}'")
    if wanted and wanted != encoded:
        print("[PREVIEW] " + t("The new prompt is being encoded on CPU (Save JSON); previews switch to it when ready."))


_live_mtime = None


def reload_live_settings():
    """
    Ajustes en caliente: si train_settings_fibo2.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT
    global USE_TURBO, TURBO_LORA_STRENGTH, SEED

    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return []
    if _live_mtime is None or mtime == _live_mtime:
        # La primera llamada solo memoriza la fecha: el guardado previo al lanzamiento no es un cambio.
        _live_mtime = _live_mtime or mtime
        return []

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []  # guardado a medias: se vuelve a intentar en el paso siguiente
    _live_mtime = mtime

    changes = []

    def fresh(key, cast, current):
        if key not in raw:
            return current
        try:
            value = cast(raw[key])
        except (TypeError, ValueError):
            return current
        if value != current:
            changes.append(f"{key}: {current} -> {value}")
        return value

    TOTAL_STEPS           = fresh("total_steps",           int,   TOTAL_STEPS)
    SAVE_EVERY            = fresh("save_every",            int,   SAVE_EVERY)
    LR                    = fresh("lr",                    float, LR)
    MAX_GRAD_NORM         = fresh("max_grad_norm",         float, MAX_GRAD_NORM)
    PREVIEW_EVERY         = fresh("preview_every",         int,   PREVIEW_EVERY)
    PREVIEW_STEPS         = fresh("preview_steps",         int,   PREVIEW_STEPS)
    PREVIEW_CFG           = fresh("preview_cfg",           float, PREVIEW_CFG)
    PREVIEW_SIZE          = fresh("preview_size",          int,   PREVIEW_SIZE)
    PREVIEW_WIDTH         = fresh("preview_width",         int,   PREVIEW_WIDTH)
    PREVIEW_HEIGHT        = fresh("preview_height",        int,   PREVIEW_HEIGHT)
    PREVIEW_CAPTION_MODE  = fresh("preview_caption_mode",  str,   PREVIEW_CAPTION_MODE)
    PREVIEW_CUSTOM_PROMPT = fresh("preview_custom_prompt", lambda v: str(v).strip(), PREVIEW_CUSTOM_PROMPT)
    USE_TURBO             = fresh("use_turbo",             bool,  USE_TURBO)
    TURBO_LORA_STRENGTH   = fresh("turbo_lora_strength",   float, TURBO_LORA_STRENGTH)
    SEED                  = fresh("seed",                  int,   SEED)  # solo previews
    return changes


def collate_base(embeds, masks, base_embeds, base_masks, use_base):
    """
    Para cada muestra se elige el embedding con o sin los textos dibujados del caption
    (por encima del nivel de ruido 0.9 el modelo los deja fuera, como en la inferencia)
    y todos se rellenan al mismo tamaño.
    """
    chosen_e, chosen_m = [], []
    for i, e in enumerate(embeds):
        b = base_embeds[i] if use_base[i] else None
        if b is not None:
            chosen_e.append(b)
            chosen_m.append(base_masks[i] if base_masks[i] is not None
                            else torch.ones(b.shape[1], dtype=torch.bool))
        else:
            chosen_e.append(e)
            chosen_m.append(masks[i] if masks[i] is not None
                            else torch.ones(e.shape[1], dtype=torch.bool))
    max_len = max(x.shape[1] for x in chosen_e)
    emb = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in chosen_e])
    msk = torch.stack([F.pad(m, (0, max_len - m.shape[0]), value=False) for m in chosen_m])
    return emb, msk


def collate_text(embeds, masks):
    # Con batch > 1 los prompts tienen longitudes distintas: se rellenan a la derecha y se enmascaran.
    max_len = max(e.shape[1] for e in embeds)
    if all(e.shape[1] == max_len for e in embeds):
        return torch.stack(embeds), None
    emb = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in embeds])
    # Una muestra sin máscara (caché de una versión anterior) cuenta como sin relleno.
    msk = torch.stack([F.pad(m if m is not None else torch.ones(e.shape[1], dtype=torch.bool),
                            (0, max_len - e.shape[1]), value=False)
                       for e, m in zip(embeds, masks)])
    return emb, msk


def train_fibo2():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_model_downloaded(local_path=MODEL_ID)

    print(t("Loading {name}...", name="Fibo 2.0 Transformer (NF4)"))
    t0 = time.time()
    transformer = load_nf4_transformer(MODEL_ID)
    transformer.to("cuda")
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    patch = transformer.config.patch_size

    transformer.enable_gradient_checkpointing()

    # "blocks": atención + MLP de los 44 bloques. "all": además el perceptor, la modulación
    # adaLN y las capas de entrada y salida (415 lineales, la receta de BRIA).
    all_linears = [name for name, m in transformer.named_modules()
                   if isinstance(m, (torch.nn.Linear, Linear4bit))]
    target_modules = [name for name in all_linears
                      if LORA_TARGETS == "all" or is_block_target(name)]
    print(t("Target LoRA layers: {n} ({kind})", n=len(target_modules), kind=LORA_TARGETS))

    # Reanudar es continuar EL MISMO LoRA: capas, rank y alpha salen del checkpoint, no de la GUI.
    # Si se construyera con otros, el optimizador no encajaría y quedaría una mezcla de pesos.
    # Las capas se leen de los PESOS guardados: adapter_config.json las guarda abreviadas.
    # El checkpoint dice además si era LoRA o LoKr: se sigue con el mismo método.
    lora_weights = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    lokr_weights = os.path.join(RESUME_DIR, lora_options.LOKR_WEIGHTS)
    if os.path.exists(STEP_FILE) and (os.path.exists(lora_weights) or os.path.exists(lokr_weights)):
        saved_lokr = not os.path.exists(lora_weights)
        if saved_lokr != USE_LOKR:
            print("\n[!] " + t("Resuming with the checkpoint's method: {m} (settings ask for {m2}; it applies to new trainings).",
                              m="LoKr" if saved_lokr else "LoRA", m2="LoKr" if USE_LOKR else "LoRA"))
        USE_LOKR = saved_lokr
    resume_weights = lokr_weights if USE_LOKR else lora_weights
    if os.path.exists(STEP_FILE) and os.path.exists(resume_weights) and USE_LOKR:
        with safe_open(resume_weights, framework="pt", device="cpu") as f:
            saved_targets = sorted({k.rsplit(".", 1)[0] for k in f.keys()})
        with open(os.path.join(RESUME_DIR, lora_options.LOKR_CONFIG), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
        if saved != (len(target_modules), LORA_RANK, LORA_ALPHA) or saved_cfg["lokr_factor"] != LOKR_FACTOR:
            print("\n[!] " + t("Resuming with the checkpoint's LoKr: {n} layers, rank {r}, alpha {a}, factor {f} "
                              "(settings ask for {n2} layers, rank {r2}, alpha {a2}, factor {f2}; they apply to new trainings).",
                              n=saved[0], r=saved[1], a=saved[2], f=saved_cfg["lokr_factor"],
                              n2=len(target_modules), r2=LORA_RANK, a2=LORA_ALPHA, f2=LOKR_FACTOR))
        target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]
        LOKR_FACTOR = saved_cfg["lokr_factor"]
    elif os.path.exists(STEP_FILE) and os.path.exists(resume_weights):
        with safe_open(resume_weights, framework="pt", device="cpu") as f:
            saved_targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1) for k in f.keys()})
        with open(os.path.join(RESUME_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
        if saved != (len(target_modules), LORA_RANK, LORA_ALPHA):
            print("\n[!] " + t("Resuming with the checkpoint's LoRA: {n} layers, rank {r}, alpha {a} "
                              "(settings ask for {n2} layers, rank {r2}, alpha {a2}; they apply to new trainings).",
                              n=saved[0], r=saved[1], a=saved[2], n2=len(target_modules), r2=LORA_RANK, a2=LORA_ALPHA))
        target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]
        # rsLoRA lo guarda PEFT en adapter_config.json; LoRA+ va en lora_options.json. Un checkpoint
        # anterior a estas opciones no tiene ninguno de los dos: se entrenó sin ellas.
        opts_file = os.path.join(RESUME_DIR, "lora_options.json")
        opts = {}
        if os.path.exists(opts_file):
            with open(opts_file, "r", encoding="utf-8") as f:
                opts = json.load(f)
        saved_opts = (bool(saved_cfg.get("use_rslora", False)), bool(opts.get("use_loraplus", False)),
                      float(opts.get("loraplus_ratio", LORAPLUS_RATIO)))
        if saved_opts[:2] != (USE_RSLORA, USE_LORAPLUS):
            print("\n[!] " + t("Resuming with the checkpoint's options: rsLoRA {a}, LoRA+ {b} "
                              "(settings ask for rsLoRA {a2}, LoRA+ {b2}; they apply to new trainings).",
                              a=saved_opts[0], b=saved_opts[1], a2=USE_RSLORA, b2=USE_LORAPLUS))
        USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO = saved_opts

    if USE_LOKR:
        USE_RSLORA = USE_LORAPLUS = False
        model = lora_options.LoKrModel(transformer, target_modules, LORA_RANK, LORA_ALPHA, LOKR_FACTOR)
    else:
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
            target_modules=target_modules, use_dora=False, init_lora_weights=True, use_rslora=USE_RSLORA
        )

        model = get_peft_model(transformer, lora_config)
    # LoRA y estados del optimizador en FP32: en 8/16 bits se pierde el detalle fino (caras).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # LoRA+: lora_B en su propio grupo, con el LR multiplicado (lr_mult se aplica en cada paso).
    if USE_LORAPLUS:
        lora_b = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
        lora_a = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" not in n]
        optimizer = torch.optim.AdamW([{"params": lora_a, "lr_mult": 1.0},
                                       {"params": lora_b, "lr_mult": LORAPLUS_RATIO}],
                                      lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        return LR * lora_options.lr_factor(LR_SCHEDULER, step, TOTAL_STEPS, WARMUP_STEPS, MIN_LR_RATIO, LR_CYCLES)

    start_step = 0
    lora_weights_path = resume_weights
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(lora_weights_path):
        print("=" * 65)
        print(t("Checkpoint detected! Restoring state..."))
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(lora_weights_path, "rb") as f:
                saved_sd = load(f.read())
            if USE_LOKR:
                model.load_lokr_state_dict(saved_sd)
            else:
                set_peft_model_state_dict(model, {k: v.float() for k, v in saved_sd.items()})
            optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
            print(t("Resuming training from step {n}...", n=start_step))
        except Exception as e:
            print("[!] " + t("Warning reading checkpoint: {error}", error=e))
            start_step = 0
        print("=" * 65)

    last_step_executed = start_step

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return
        print("\n" + t("Saving checkpoint state at step {n}...", n=current_s))
        os.makedirs(RESUME_DIR, exist_ok=True)
        model.save_pretrained(RESUME_DIR)
        # Solo puede quedar uno: los pesos del otro método serían de un entrenamiento anterior.
        stale = os.path.join(RESUME_DIR, "adapter_model.safetensors" if USE_LOKR else lora_options.LOKR_WEIGHTS)
        if os.path.exists(stale):
            os.remove(stale)
        with open(os.path.join(RESUME_DIR, "lora_options.json"), "w", encoding="utf-8") as f:
            json.dump({"use_loraplus": USE_LORAPLUS, "loraplus_ratio": LORAPLUS_RATIO}, f)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        ckpt = os.path.join(OUTPUT_DIR, f"Fibo2_LoRA_step_{current_s}.safetensors")
        _export_lora(model, ckpt, current_s)
        print("✓ " + t("Checkpoint saved at step {n}: {path}", n=current_s, path=ckpt))

    def handle_signal(sig, frame):
        nonlocal last_step_executed
        print("\n[!] " + t("Stop signal received ({sig}).", sig=sig))
        save_checkpoint_now(last_step_executed)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
    except Exception:
        pass

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pin = torch.cuda.is_available()
    cache_data, buckets = {}, defaultdict(list)

    def load_sample(nombre):
        # lat: objetivo (en _custom no existe: solo aporta el tamaño). En edición además
        # ref (latente de la imagen de antes) y, si el caption lleva textos, sus embeddings sin ellos.
        sample = {}
        for key, suffix, cast in (("lat", "latent", torch.float32), ("emb", "embed", torch.bfloat16),
                                  ("msk", "mask", torch.bool), ("ref", "ref", torch.float32),
                                  ("base_emb", "base_embed", torch.bfloat16), ("base_msk", "base_mask", torch.bool)):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                t = torch.load(path, weights_only=True).to(cast)
                sample[key] = t.pin_memory() if pin else t
        return sample

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = load_sample(nombre)
        buckets[tuple(cache_data[nombre]["lat"].shape[-2:])].append(nombre)

    edit_mode = any("ref" in v for v in cache_data.values())

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        cache_data["_custom"] = load_sample("_custom")
    check_custom_prompt("_custom" in cache_data, edit_mode)
    custom_mtime = [os.path.getmtime(f"{CACHE_DIR}/_custom_embed.pt") if "_custom" in cache_data else None]

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_negcfg_embed.pt"):
        neg = (torch.load(f"{CACHE_DIR}/_negcfg_embed.pt", weights_only=True).to(torch.bfloat16),
               torch.load(f"{CACHE_DIR}/_negcfg_mask.pt", weights_only=True).bool())
    null_embed = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        null_embed = (torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True).to(torch.bfloat16),
                      torch.load(f"{CACHE_DIR}/_neg_mask.pt", weights_only=True).bool())

    all_preview_names = sorted(k for k in cache_data.keys() if not k.startswith("_"))

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "custom" and "_custom" in cache_data:
            return "_custom"
        elif PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    running_loss, t_step_avg, grad_norm = 0.0, 0.0, 0.0
    kind = t("before/after pairs") if edit_mode else t("images")
    print("\n" + t("STARTING TRAINING! {n} {kind} in {b} buckets.", n=len(all_preview_names), kind=kind, b=len(buckets)))

    reload_live_settings()
    metrics = training_metrics.MetricsLog(OUTPUT_DIR, start_step)
    step = start_step
    try:
        # while y no range(): TOTAL_STEPS puede cambiar en caliente, en ambos sentidos.
        while step < TOTAL_STEPS:
            step += 1
            last_step_executed = step

            changes = reload_live_settings()
            if changes:
                print("\n[LIVE] " + t("Settings reloaded without stopping:"))
                for c in changes:
                    print(f"[LIVE]   {c}")
                if any(c.startswith(("preview_custom_prompt", "preview_caption_mode")) for c in changes):
                    check_custom_prompt("_custom" in cache_data, edit_mode)
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True)

            H, W = size
            B = latents.shape[0]

            sigma  = sample_sigma(B, (H // patch) * (W // patch), "cuda", shift_cfg)

            # Por encima de este nivel de ruido el caption va sin los textos dibujados,
            # como en la inferencia y en la receta de BRIA (solo si el caption los tiene).
            use_base = [bool(s) > TEXT_IMAGES_MAX_SIGMA and cache_data[n].get("base_emb") is not None
                        for n, s in zip(names, sigma)]
            if any(use_base):
                embeds, masks = collate_base([cache_data[n]["emb"] for n in names],
                                              [cache_data[n].get("msk") for n in names],
                                              [cache_data[n].get("base_emb") for n in names],
                                              [cache_data[n].get("base_msk") for n in names],
                                              use_base)
            else:
                embeds, masks = collate_text([cache_data[n]["emb"] for n in names],
                                             [cache_data[n].get("msk") for n in names])
            embeds = embeds.to("cuda", non_blocking=True)
            masks = None if masks is None else masks.to("cuda", non_blocking=True)

            noise  = torch.randn_like(latents)
            t_exp  = sigma.view(-1, 1, 1, 1)

            noisy = (t_exp * noise + (1 - t_exp) * latents).to(torch.bfloat16)
            target = noise - latents

            # El caption vacío sustituye al caption en el 10% de las muestras (text dropout).
            # El caption vacío tiene otra longitud que los del lote: con batch > 1 el cambio
            # es para todo el lote (la referencia de BRIA entrena con batch 1 y acumulación).
            if null_embed is not None and random.random() < TEXT_DROPOUT:
                embeds = null_embed[0].unsqueeze(0).to("cuda").expand(B, -1, -1, -1).contiguous()
                masks = None

            # En edición, el 20% de las veces se quitan los latentes de la imagen de antes
            # (context dropout): el texto que la describía se queda.
            context = None
            if edit_mode:
                refs = torch.cat([cache_data[n]["ref"] for n in names]).to("cuda", non_blocking=True)
                if random.random() >= CONTEXT_DROPOUT:
                    context = [refs.to(torch.bfloat16)]

            pred = model(
                hidden_states=noisy,
                encoder_hidden_states=embeds,
                encoder_attention_mask=masks,
                timestep=sigma,
                context_latents=context,
                return_dict=False,
            )[0]

            loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
            loss.backward()
            step_loss = loss.item() * GRAD_ACCUM_STEPS
            running_loss += step_loss

            # La norma solo existe al aplicar el optimizador: entre medias se muestra la última.
            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for gparam in optimizer.param_groups:
                    gparam["lr"] = lr_at(step) * gparam.get("lr_mult", 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            t_step     = time.time() - t0
            t_step_avg = t_step if t_step_avg == 0 else 0.1 * t_step + 0.9 * t_step_avg
            eta_s      = (TOTAL_STEPS - step) * t_step_avg
            eta        = f"{int(eta_s//3600):02d}:{int((eta_s%3600)//60):02d}:{int(eta_s%60):02d}"
            pct        = step / TOTAL_STEPS
            barra      = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))

            avg_loss = running_loss / max(1, step - start_step)
            progress_line = (
                f"{STEP_WORD} {step:4d}/{TOTAL_STEPS} [{barra}] {pct*100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {lr_at(step):.2e} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)
            metrics.log(step, step_loss, lr_at(step), grad_norm, t_step, TOTAL_STEPS)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                # El servidor codifica en CPU los prompts nuevos: aquí solo se relee el embedding si cambió.
                custom_path = f"{CACHE_DIR}/_custom_embed.pt"
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = load_sample("_custom")
                    check_custom_prompt(True, edit_mode)
                p_name = get_preview_sample(step)
                sample = cache_data[p_name]
                # Tamaño de la preview: el del objetivo; en edición, el de la imagen de antes
                # (como en ComfyUI, la salida hereda el tamaño de la referencia).
                ref = sample.get("ref", sample.get("lat", next(iter(cache_data.values()))["lat"]))
                print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
                H, W = ref.shape[-2], ref.shape[-1]
                if PREVIEW_WIDTH > 0 and PREVIEW_HEIGHT > 0 and not edit_mode:
                    # Lados explícitos en píxeles: VAE 8x, lados pares.
                    H, W = max(2, round(PREVIEW_HEIGHT / 8)), max(2, round(PREVIEW_WIDTH / 8))
                    H, W = H - H % 2, W - W % 2
                elif PREVIEW_SIZE > 0:
                    # El VAE de Fibo 2.0 baja 8: el area de la preview es la del latent por 64.
                    s = PREVIEW_SIZE / 8 / math.sqrt(H * W)
                    H, W = max(2, round(H * s)), max(2, round(W * s))
                    H, W = H - H % 2, W - W % 2
                run_preview(model, scheduler, sample, neg, (H, W), step)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    final = os.path.join(OUTPUT_DIR, "Fibo2_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print("✓ " + t("Final LoRA saved to: {path}", path=final))


if __name__ == "__main__":
    train_fibo2()
