# -*- coding: utf-8 -*-
"""
2_train_lora_mossimage.py — Entrenamiento LoRA para MOSS-Image (Transformer NF4)
LoRA training for MOSS-Image (NF4 Transformer)

Lee configuración desde train_settings_mossimage.json si existe.
Reads configuration from train_settings_mossimage.json if present.

Entrena sobre MOSS-Image Base o SFT (ajuste base_model): los dos comparten text encoder y VAE, así
que la caché sirve para ambos. El muestreo del ruido es el de la última etapa del preentrenamiento
de OpenMOSS (informe técnico de MOSS-Image, tabla 6) y las previews siguen MOSSImagePipeline:
CFG con el bloque 22 saltado en la rama negativa. El LoRA se exporta con los nombres originales
(diffusion_model.layers.N.proj.*), los del nodo de MOSS-Image para ComfyUI.

LoRA de edición (caché con <nombre>_ctrl.pt): el antes entra como referencia (ver
moss_image_model.py) y se entrena encima del LoRA chinoll/Moss-Image-Edit, que se suma congelado a
la salida de cada capa. El LoRA exportado es solo lo aprendido encima: en ComfyUI se usa con el
parche Moss-Image-Edit (nodo Apply MOSS Edit), o fusionado con él al exportar.
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
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load, load_file
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import training_metrics
from moss_image_model import MOSSImageTransformer

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/MOSS-Image-NF4-for-LoRA-Training"

# Carpeta del transformer de cada variante dentro del modelo NF4.
TRANSFORMER_FOLDERS = {"base": "transformer", "sft": "transformer_sft"}
SOURCE_MODELS = {"base": "OpenMOSS-Team/MOSS-Image-Base", "sft": "OpenMOSS-Team/MOSS-Image-SFT"}

# Ruido del entrenamiento original: sigma con logit-normal (mu, s) recortada a SIGMA_CUTOFF y
# estirada a [0, 1], más un 1 % uniforme. Con mu negativa en la fórmula del informe, la mayoría de
# los pasos caen en ruido alto.
SIGMA_MU, SIGMA_S, SIGMA_CUTOFF, SIGMA_UNIFORM = -1.09, 0.95, 0.95, 0.01
# Inferencia de MOSSImagePipeline: desplazamiento 5 y bloque 22 saltado en la rama negativa.
PREVIEW_SHIFT = 5.0
UNCOND_SKIP_LAYERS = (22,)

# Adaptador turbo de 4 pasos para las previews: darask0/MOSS-Image_4-step_adapter_lora (Apache-2.0,
# destilado de MOSS-Image-SFT con DMAD a partir de lodestones/MOSS-Image-SFT-8-steps-dmd). No es un
# LoRA normal: LoRA FP32 sin fusionar + time_embedding y bias de modulación propios en FP32 + su
# sampler "data-space" sin CFG. Se fija la revisión que se ha probado.
TURBO_REPO = "darask0/MOSS-Image_4-step_adapter_lora"
TURBO_REVISION = "1ec327b9fac5a0aa47426aaba6565892616ce30e"
TURBO_FILES = ("lora.safetensors", "time_embedding.safetensors", "dataspace.safetensors")

# LoRA de edición sobre el que se entrenan los LoRAs de edición: chinoll/Moss-Image-Edit (rank 64,
# formato "moss-edit-lora" del nodo ComfyUI-Moss-Image-Edit). Se fija la revisión que se ha probado.
EDIT_REPO = "chinoll/Moss-Image-Edit"
EDIT_REVISION = "3be9c40037e1b89b17aa85ff6f8ee805872aa558"
EDIT_FILES = ("adapter_config.json", "adapter_model.safetensors")
EDIT_FOLDER = os.path.join("LoRAs", "Moss-Image-Edit")

DEFAULTS = {
    "model_id": "Models/MOSS-Image-NF4",
    "base_model": "base",
    "cache_dir": "./Cache_Data/cached_data_mossimage",
    "output_dir": "./Outputs/mossimage_lora_output",
    "total_steps": 500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 4e-4,
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
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "seed": 42,
    "timestep_sampling": "moss",
    "use_turbo": False,
    "turbo_lora_strength": 1.0,
    "edit_lora_strength": 1.0,
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 3.0,
    "preview_size": 0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_mossimage.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID          = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
BASE_MODEL        = cfg.get("base_model",        DEFAULTS["base_model"])
if BASE_MODEL not in TRANSFORMER_FOLDERS:
    BASE_MODEL = DEFAULTS["base_model"]
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
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()
USE_TURBO         = cfg.get("use_turbo",         DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"])
# Fuerza del LoRA Moss-Image-Edit base. Se fija al arrancar: lo aprendido encima depende de ella.
EDIT_LORA_STRENGTH = float(cfg.get("edit_lora_strength", DEFAULTS["edit_lora_strength"]))

TRIGGER_WORD     = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_mossimage_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./mossimage_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

print(f"  {t('Model ID'):<22}: {MODEL_ID} ({SOURCE_MODELS[BASE_MODEL]})")
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
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG} | {t('Preview Size')} {PREVIEW_SIZE or t('Training')}")
print(f"  {'Turbo LoRA':<22}: {t('ON (strength {s})', s=TURBO_LORA_STRENGTH) if USE_TURBO else t('OFF')}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")

if SEED > 0:
    torch.manual_seed(SEED)
    random.seed(SEED)


VRAM_TOTAL = 0
# True cuando la caché es de pares antes/después (lo decide train_mossimage al leerla).
EDIT_MODE = False


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


def ensure_model_downloaded(local_path, repo_id):
    folder = TRANSFORMER_FOLDERS[BASE_MODEL]
    if (os.path.exists(os.path.join(local_path, folder, "index.json"))
            and os.path.exists(os.path.join(local_path, "vae", "config.json"))):
        print("[OK] " + t("Local model found at: {path}", path=local_path))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=repo_id))

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(t("huggingface_hub is required. Install it with: pip install huggingface_hub"))

    # Solo el transformer elegido y el VAE: el text encoder ya está en el pre-cache.
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        # Rutas exactas: "*.json" traería también los config.json de los otros text encoders.
        allow_patterns=[f"{folder}/*", "vae/*", "model_index.json", "LICENSE", "NOTICE.md", "README.md"],
    )

    print("[OK] " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def sample_sigma(batch_size, device):
    if TIMESTEP_SAMPLING == "uniform":
        return torch.rand(batch_size, device=device).clamp(1e-3, 1.0)
    # sigma = sigmoid(s·z - mu) con la cola por encima del corte descartada (inversa de la CDF
    # truncada), estirada a [0, 1]; un 1 % de las muestras, uniformes.
    z_max = (math.log(SIGMA_CUTOFF / (1 - SIGMA_CUTOFF)) + SIGMA_MU) / SIGMA_S
    p_max = 0.5 * (1 + math.erf(z_max / math.sqrt(2)))
    z = torch.special.ndtri(torch.rand(batch_size, device=device) * p_max)
    sigma = torch.sigmoid(SIGMA_S * z - SIGMA_MU) / SIGMA_CUTOFF
    uniform = torch.rand(batch_size, device=device) < SIGMA_UNIFORM
    sigma = torch.where(uniform, torch.rand(batch_size, device=device), sigma)
    return sigma.clamp(1e-3, 1.0)


def stack_embeds(embeds, device):
    """Embeddings de distinta longitud -> [B, S, 4096] rellenados con ceros + máscara [B, S]."""
    n = max(e.shape[0] for e in embeds)
    out = torch.zeros(len(embeds), n, embeds[0].shape[-1], dtype=torch.bfloat16, device=device)
    mask = torch.zeros(len(embeds), n, dtype=torch.long, device=device)
    for i, e in enumerate(embeds):
        out[i, :e.shape[0]] = e.to(device, non_blocking=True)
        mask[i, :e.shape[0]] = 1
    return out, mask


def load_nf4_transformer(model_dir):
    """
    Construye el Transformer vacío desde config.json y lo rellena con la caché NF4
    (weights/ + others.safetensors) de la variante elegida. No hace falta el Transformer en BF16.
    """
    cache_dir = os.path.join(model_dir, TRANSFORMER_FOLDERS[BASE_MODEL])
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(t("index.json not found in the NF4 cache: {path}", path=cache_dir))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = MOSSImageTransformer.from_config(MOSSImageTransformer.load_config(cache_dir))

    weights_dir = os.path.join(cache_dir, "weights")

    def set_module(name, module):
        parent_name, _, child_name = name.rpartition(".")
        setattr(transformer.get_submodule(parent_name) if parent_name else transformer, child_name, module)

    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device="cpu")
        weight.bnb_quantized = True

        # En meta: nn.Linear.__init__ reserva y rellena con kaiming una matriz FP32 que se tira
        # en la línea siguiente.
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=False, quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        set_module(name, layer)

    transformer.load_state_dict(load_file(os.path.join(cache_dir, index["others"])), strict=False, assign=True)

    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=missing[:5]))

    print("[OK] " + t("NF4 layers: {a} | BF16 layers: {b}", a=len(index['quantized']), b=len(index['unquantized'])))
    return transformer


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab MOSS-Image",
        "ss_sd_model_name": SOURCE_MODELS[BASE_MODEL].split("/")[-1],
        "ss_base_model_version": "MOSS-Image",
        "ss_network_module": "lycoris.kohya" if USE_LOKR else "peft.LoraModel",
        "ss_network_dim": LORA_RANK,
        "ss_network_alpha": export_alpha(),
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
    meta["ss_output_name"] = PROJECT_NAME or trigger or "mossimage_lora"
    # La GUI lee edit_base_lora para ofrecer la exportación fusionada con el LoRA de edición.
    if EDIT_MODE:
        meta["edit_base_lora"] = EDIT_REPO
        meta["edit_base_lora_revision"] = EDIT_REVISION
        meta["edit_base_lora_strength"] = EDIT_LORA_STRENGTH
        meta["usage"] = "Edit LoRA: load it together with the Moss-Image-Edit patch (Apply MOSS Edit node)."

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


def comfy_name(name):
    return "diffusion_model." + name


def _export_lora(model, path, step):
    # Claves diffusion_model.<capa>.lora_A/lora_B + alpha por capa: sin él, ComfyUI aplicaría
    # el LoRA con escala 1 en lugar de alpha/rank.
    if USE_LOKR:
        save_file(lora_options.lokr_export(model, comfy_name, export_alpha()), path, metadata=build_lora_metadata(step))
        return
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        layer, _, suffix = k.replace("base_model.model.", "").partition(".lora_")
        key = comfy_name(layer)
        clean[f"{key}.lora_{suffix}"] = v.to(torch.bfloat16).cpu().contiguous()
        if suffix == "A.weight":
            clean[key + ".alpha"] = torch.tensor(float(export_alpha()))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def load_edit_lora(transformer):
    """
    Suma el LoRA chinoll/Moss-Image-Edit, congelado y en BF16, a la salida de cada capa que toca
    (q, k, v, o y MLP de los 42 bloques). Se engancha a la capa NF4 antes de envolverla con el LoRA
    que se entrena: PEFT y LoKr la conservan como capa base, así que los dos se suman.
    """
    folder = os.path.join(MODEL_ID, EDIT_FOLDER)
    if not all(os.path.exists(os.path.join(folder, f)) for f in EDIT_FILES):
        from huggingface_hub import hf_hub_download
        print("  " + t("Downloading the Moss-Image-Edit LoRA (~{mb} MB): {repo}", mb=300, repo=EDIT_REPO))
        for f in EDIT_FILES:
            hf_hub_download(EDIT_REPO, f, revision=EDIT_REVISION, local_dir=folder, token=get_hf_token())
    path = os.path.join(folder, "adapter_model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        config = json.loads(f.metadata()["adapter_config"])
    weights = load_file(path, device="cuda")

    def make_hook(down, up, scale):
        def hook(module, args, output):
            return output + (args[0].to(down.dtype) @ down.t()) @ up.t() * scale
        return hook

    for name, spec in config["modules"].items():
        down, up = weights[name + ".lora_A"], weights[name + ".lora_B"]
        scale = spec["alpha"] / spec["rank"] * EDIT_LORA_STRENGTH
        transformer.get_submodule(name).register_forward_hook(make_hook(down, up, scale))
    print("[OK] " + t("Moss-Image-Edit LoRA loaded: {n} layers, rank {r}, strength {s}",
                      n=len(config["modules"]), r=next(iter(config["modules"].values()))["rank"], s=EDIT_LORA_STRENGTH))


def denoise(model, scheduler, emb, H, W, generator, neg=None, refs=None):
    device = "cuda"
    latents = torch.randn((1, 128, H, W), generator=generator, device=device, dtype=torch.float32)
    # Como Apply MOSS Edit: las referencias van en las dos ramas del CFG.
    references = [refs] if refs else None

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    scheduler.set_begin_index(0)

    conds = [(stack_embeds([emb], device), ())]
    use_cfg = neg is not None and PREVIEW_CFG > 1.0
    if use_cfg:
        conds.append((stack_embeds([neg], device), UNCOND_SKIP_LAYERS))

    for t in scheduler.timesteps:
        timestep = t.expand(1)
        preds = []
        for (text, mask), skip in conds:
            preds.append(model(latents.to(torch.bfloat16), timestep, text, mask, skip_layers=skip,
                               references=references).float())
        pred = preds[0] if not use_cfg else preds[1] + PREVIEW_CFG * (preds[0] - preds[1])
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    return latents


class TurboHolder:
    # Se lee del disco una sola vez; queda en RAM fijada (no en VRAM) entre previews.
    layers = None
    time_path = None
    dataspace = None

    @classmethod
    def load(cls):
        folder = os.path.join(MODEL_ID, "LoRAs", "MOSS-Image_4-step_adapter")
        if not all(os.path.exists(os.path.join(folder, f)) for f in TURBO_FILES):
            from huggingface_hub import hf_hub_download
            print("  " + t("Downloading Turbo LoRA (~{mb} MB): {repo}", mb=1230, repo=TURBO_REPO))
            for f in TURBO_FILES:
                hf_hub_download(TURBO_REPO, f, revision=TURBO_REVISION, local_dir=folder)
        lora = load_file(os.path.join(folder, "lora.safetensors"))
        cls.layers = {}
        for k in lora:
            if k.endswith(".lora_down.weight"):
                name = k[len("diffusion_model."):-len(".lora_down.weight")]
                down, up = lora[k].float(), lora[k.replace(".lora_down.", ".lora_up.")].float()
                alpha = float(lora[f"diffusion_model.{name}.alpha"])
                cls.layers[name] = (down.pin_memory(), up.pin_memory(), alpha / down.shape[0])
        cls.time_path = load_file(os.path.join(folder, "time_embedding.safetensors"))
        cls.dataspace = load_file(os.path.join(folder, "dataspace.safetensors"))


def apply_turbo_lora(model):
    """
    Suma el adaptador turbo en FP32 a la salida de cada capa durante la preview, sin tocar el LoRA
    que se entrena, y cambia el camino del tiempo (time_embedding + bias de las dos modulaciones)
    por el suyo, también en FP32. Devuelve la función que lo deshace todo.
    """
    if TurboHolder.layers is None:
        TurboHolder.load()
    inner = model.base_model.model

    def make_hook(down, up, scale):
        def hook(module, args, output):
            return output + ((args[0].float() @ down.t()) @ up.t() * scale).to(output.dtype)
        return hook

    hooks = []
    for name, (down, up, scale) in TurboHolder.layers.items():
        down, up = down.to("cuda", non_blocking=True), up.to("cuda", non_blocking=True)
        hooks.append(inner.get_submodule(name).register_forward_hook(make_hook(down, up, scale * TURBO_LORA_STRENGTH)))

    tp = {k: v.to("cuda", torch.float32) for k, v in TurboHolder.time_path.items()}
    saved = {name: getattr(inner, name) for name in ("time_embedding", "shared_block_modulation", "video_head_modulation")}
    with torch.device("meta"):
        time_embedding = torch.nn.Sequential(torch.nn.Linear(256, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072))
    time_embedding.load_state_dict({k[len("time_embedding."):]: v for k, v in tp.items() if k.startswith("time_embedding.")}, assign=True)
    inner.time_embedding = time_embedding
    for name in ("shared_block_modulation", "video_head_modulation"):
        old = saved[name]
        with torch.device("meta"):
            layer = torch.nn.Linear(old.in_features, old.out_features)
        layer.load_state_dict({"weight": old.weight.float(), "bias": tp[name + ".bias"]}, assign=True)
        setattr(inner, name, layer)

    def undo():
        for h in hooks:
            h.remove()
        hooks.clear()
        for name, module in saved.items():
            setattr(inner, name, module)
    return undo


def denoise_turbo(model, emb, H, W, generator, refs=None):
    """Sampler data-space del adaptador: una sola z; x = z + Q·x0, x0 = z - v, sin CFG ni schedule."""
    Q = TurboHolder.dataspace["latent_mix.Q"].to("cuda", torch.float32)
    codes = TurboHolder.dataspace["step_codes"].tolist()
    z = torch.randn((1, 128, H, W), generator=generator, device="cuda", dtype=torch.float32)
    x0 = torch.zeros_like(z)
    text, mask = stack_embeds([emb], "cuda")
    references = [refs] if refs else None
    for code in codes[:max(1, PREVIEW_STEPS)]:
        x = z + torch.einsum("oc,bchw->bohw", Q, x0)
        x0 = z - model(x.to(torch.bfloat16), torch.tensor([code], device="cuda"), text, mask,
                       references=references).float()
    return x0


def run_preview(model, scheduler, emb, neg, size, step, refs=None):
    # refs: en edición, los latentes de referencia de la preview (el antes, o las imágenes elegidas).
    refs = [r.to("cuda") for r in refs] if refs else None
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    undo_turbo = apply_turbo_lora(model) if USE_TURBO else None
    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            if undo_turbo:
                latents = denoise_turbo(model, emb, H, W, g, refs)
                # Los pesos del adaptador en la GPU ya no hacen falta: se liberan antes del VAE.
                undo_turbo()
            else:
                latents = denoise(model, scheduler, emb, H, W, g, neg, refs)

            vae = VaeHolder.get().to("cuda")
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float()
            bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).float() + vae.config.batch_norm_eps)
            lat = (latents * bn_std + bn_mean).view(1, 32, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, H * 2, W * 2)
            lat = lat.to(vae.dtype)
            try:
                img = vae.decode(lat, return_dict=False)[0]
            except torch.OutOfMemoryError:
                # Con 8 GB y el entrenamiento cargado no siempre cabe entera: por mosaicos usa mucha
                # menos memoria, a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(lat, return_dict=False)[0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out)
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
        if undo_turbo:
            undo_turbo()
        if was_training:
            model.train()
        free_vram()


def check_custom_prompt(has_custom):
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

    wanted = PREVIEW_CUSTOM_PROMPT
    if TRIGGER_WORD and wanted and TRIGGER_WORD.lower() not in wanted.lower():
        wanted = f"{TRIGGER_WORD}, {wanted}"
    prompt_file = os.path.join(CACHE_DIR, "_custom_prompt.txt")
    encoded = open(prompt_file, "r", encoding="utf-8").read() if os.path.exists(prompt_file) else ""
    print(f"\n[PREVIEW] {t('Custom prompt')}: '{encoded}'")
    if wanted and wanted != encoded:
        print("[PREVIEW] " + t("The new prompt is being encoded on CPU (Save JSON); previews switch to it when ready."))


_live_mtime = None


def reload_live_settings():
    """
    Ajustes en caliente: si train_settings_mossimage.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución, modelo
    y carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED
    global USE_TURBO, TURBO_LORA_STRENGTH

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
    PREVIEW_CAPTION_MODE  = fresh("preview_caption_mode",  str,   PREVIEW_CAPTION_MODE)
    PREVIEW_CUSTOM_PROMPT = fresh("preview_custom_prompt", lambda v: str(v).strip(), PREVIEW_CUSTOM_PROMPT)
    SEED                 = fresh("seed",                  int,   SEED)  # solo previews
    USE_TURBO             = fresh("use_turbo",             bool,  USE_TURBO)
    TURBO_LORA_STRENGTH   = fresh("turbo_lora_strength",   float, TURBO_LORA_STRENGTH)
    return changes


def train_mossimage():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR, VRAM_TOTAL  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    global EDIT_MODE
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"
    # El pre-caché solo deja referencias (<nombre>_ctrl.pt) en un LoRA de edición.
    EDIT_MODE = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(CACHE_DIR))
    if EDIT_MODE:
        print(f"  {t('LoRA Type'):<22}: {t('edit (before/after pairs)')} | {t('on top of')} {EDIT_REPO}")
        if BASE_MODEL != "sft":
            print("  " + t("Note: Moss-Image-Edit was trained on SFT; it also works on Base."))

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    print(t("Loading {name}...", name=f"MOSS-Image {BASE_MODEL.upper()} Transformer (NF4)"))
    t0 = time.time()
    transformer = load_nf4_transformer(MODEL_ID)
    transformer.to("cuda")
    VRAM_TOTAL = torch.cuda.get_device_properties(0).total_memory
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    if EDIT_MODE:
        load_edit_lora(transformer)

    # Las previews integran con Euler sobre sigmas desplazadas por 5, como MOSSImageScheduler
    # (sin su ajuste a la rejilla de entrenamiento, que apenas cambia nada).
    scheduler = FlowMatchEulerDiscreteScheduler(shift=PREVIEW_SHIFT, use_dynamic_shifting=False)

    transformer.enable_gradient_checkpointing()

    # Atención (q, k, v, o) y MLP de los 42 bloques. El refinador de texto, los embedders, las
    # modulaciones y la salida no se tocan.
    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, Linear4bit)) and name.startswith("layers.")]
    print(t("Target LoRA layers: {n} ({kind})", n=len(target_modules), kind="blocks"))

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
        # rsLoRA lo guarda PEFT en adapter_config.json; LoRA+ va en lora_options.json.
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
        ckpt = os.path.join(OUTPUT_DIR, f"MossImage_LoRA_step_{current_s}.safetensors")
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
        # lat: objetivo (en _custom no existe); emb: [longitud, 4096]. En edición, ctrls: los
        # latentes de referencia (el antes; en _custom, las imágenes de referencia de la preview).
        sample = {}
        for key, suffix in (("lat", "latent"), ("emb", "embed")):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                t = torch.load(path, weights_only=True).to(torch.bfloat16)
                sample[key] = t.pin_memory() if pin else t
        ctrls, index = [], 1
        path = f"{CACHE_DIR}/{nombre}_ctrl.pt"
        while os.path.exists(path):
            t = torch.load(path, weights_only=True).to(torch.bfloat16)
            ctrls.append(t.pin_memory() if pin else t)
            index += 1
            path = f"{CACHE_DIR}/{nombre}_ctrl_{index}.pt"
        if ctrls:
            sample["ctrls"] = ctrls
        return sample

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = load_sample(nombre)
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        cache_data["_custom"] = load_sample("_custom")
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(f"{CACHE_DIR}/_custom_embed.pt") if "_custom" in cache_data else None]

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg = torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True).to(torch.bfloat16)

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
    kind = t("before/after pairs") if EDIT_MODE else t("images")
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
                    check_custom_prompt("_custom" in cache_data)
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True).float()
            text, mask = stack_embeds([cache_data[n]["emb"] for n in names], "cuda")

            sigma  = sample_sigma(latents.shape[0], "cuda")
            noise  = torch.randn_like(latents)
            t_exp  = sigma.view(-1, 1, 1, 1)

            noisy = ((1 - t_exp) * latents + t_exp * noise).to(torch.bfloat16)
            # El transformer predice la velocidad (ruido - x0); el timestep va en escala 0-1000.
            target = noise - latents

            # En edición, cada muestra lleva su antes como referencia (limpia, con timestep 0).
            references = ([[c.to("cuda", non_blocking=True) for c in cache_data[n]["ctrls"]] for n in names]
                          if EDIT_MODE else None)
            pred = model(noisy, sigma * 1000, text, mask, references=references)

            loss = F.mse_loss(pred.float(), target) / GRAD_ACCUM_STEPS
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

            # Con 8 GB, los distintos tamaños de imagen fragmentan la caché de CUDA hasta llenarla; en
            # Windows el driver pasa entonces a la RAM del sistema y cada paso tarda varias veces más.
            del pred, loss, noisy, latents, noise, target, text, mask, references
            if torch.cuda.memory_reserved() > 0.85 * VRAM_TOTAL:
                torch.cuda.empty_cache()

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
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                sample = cache_data[p_name]
                # Tamaño de la preview: el de la imagen; el prompt manual usa el de la primera del dataset.
                # En edición con el prompt manual, la proporción de la primera referencia (como en
                # ComfyUI, donde la salida sigue a la imagen que se edita) con el área de entrenamiento.
                first = cache_data[all_preview_names[0]]["lat"]
                ref = sample.get("lat", first)
                print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
                H, W = ref.shape[2], ref.shape[3]
                refs = sample.get("ctrls") if EDIT_MODE else None
                if refs and "lat" not in sample:
                    h, w = refs[0].shape[2], refs[0].shape[3]
                    s = math.sqrt(first.shape[2] * first.shape[3] / (h * w))
                    H, W = max(1, round(h * s)), max(1, round(w * s))
                if PREVIEW_SIZE > 0:
                    # Cada token es un parche de 16x16 px.
                    s = PREVIEW_SIZE / 16 / math.sqrt(H * W)
                    H, W = max(1, round(H * s)), max(1, round(W * s))
                run_preview(model, scheduler, sample["emb"], neg, (H, W), step, refs)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    final = os.path.join(OUTPUT_DIR, "MossImage_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print("✓ " + t("Final LoRA saved to: {path}", path=final))


if __name__ == "__main__":
    train_mossimage()
