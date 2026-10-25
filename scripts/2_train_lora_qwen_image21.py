# -*- coding: utf-8 -*-
"""
2_train_lora_qwen_image21.py — Entrenamiento LoRA para Qwen-Image 2.1 (Transformer NF4)
LoRA training for Qwen-Image 2.1 (NF4 Transformer)

Lee configuración desde train_settings_qwenimage21.json si existe.
Reads configuration from train_settings_qwenimage21.json if present.
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
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load, load_file
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import training_metrics

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

# slider_core vive junto a este script (también cuando otro script lo importa desde otra carpeta).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slider_core

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "Models/Qwen-Image21-NF4",
    "cache_dir": "./Cache_Data/cached_data_qwen_image21",
    "output_dir": "./Outputs/qwen_image21_lora_output",
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
    "lora_targets": "blocks",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "timestep_sampling": "shift",
    "preview_every": 0,
    "preview_steps": 30,
    "preview_cfg": 3.0,
    "preview_size": 0,
    "use_turbo": False,
    "turbo_lora_strength": 1.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_qwenimage21.json"

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
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

# Slider (docs/SLIDERS.md): fuerza al 100 % y push interno. El modo (edición o texto a imagen) lo fija
# el pre-caché en su manifiesto. Se activa si la caché es de un slider. Qwen-Image 2.1 aún no tiene
# Ultra: falta una prueba en GPU que elija sus bloques.
SLIDER = slider_core.read_settings(cfg)
SLIDER_RUN = None  # {"factor", "mode"} durante un entrenamiento slider: lo usa la exportación

# Previews rápidas: 4 pasos, sin CFG.
USE_TURBO           = cfg.get("use_turbo",           DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"])
TURBO_LORA_PATH     = os.path.join(MODEL_ID, "LoRAs", "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors")

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_qwen_image21_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./qwen_image21_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

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
print(f"  {'Turbo LoRA':<22}: {t('ON (strength {s})', s=TURBO_LORA_STRENGTH) if USE_TURBO else t('OFF')}")
if cfg.get("lora_type") == "slider":
    print(f"  {'Slider':<22}: strength at 100% / fuerza al 100 % {SLIDER['slider_strength_100']:g} | "
          f"push {SLIDER['slider_push']:g}")

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


def ensure_model_downloaded(local_path, repo_id):
    if (os.path.exists(os.path.join(local_path, "transformer", "index.json"))
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

    # El entrenamiento no usa el text encoder: ya está todo en el pre-cache.
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["text_encoder_*/*"],
    )

    print("[OK] " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192,
                    base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    mu = calculate_shift(image_seq_len, *shift_cfg)
    e_mu = math.exp(mu)
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    sigma = sigma.clamp(1e-4, 1.0 - 1e-4)
    # Mismo redondeo que en inferencia: el modelo recibe bf16(t * 1000) / 1000.
    return (sigma * 1000).to(torch.bfloat16).float() / 1000


def pack_latents(x):
    # Qwen-Image 2.1 no usa parches: cada latente 16x16 px es un token.
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2)


def unpack_latents(x, H, W):
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)


def make_img_mask(sample, H, W, device):
    # Cada ranura de imagen del text encoder representa un grupo 2x2 de latentes. En edición el
    # texto ya trae marcadas las ranuras del antes (imgmask); el objetivo va siempre al final.
    # Con batch > 1 el texto viene rellenado a la derecha: el relleno no son ranuras de imagen.
    text = torch.zeros(sample["emb"].shape[1], dtype=torch.bool, device=device)
    if "imgmask" in sample:
        text[:sample["imgmask"].shape[1]] = sample["imgmask"][0].to(device)
    return torch.cat([text, torch.ones(H * W // 4, dtype=torch.bool, device=device)]).unsqueeze(0)


def model_inputs(sample, latents, H, W, device):
    """
    Secuencia de imagen del transformer: en edición las referencias van delante, LIMPIAS (sin
    ruido), y el objetivo detrás; img_shapes y la máscara siguen ese orden. La predicción es
    siempre la del objetivo: los últimos H*W tokens. La preview puede llevar varias referencias
    (<image1>, <image2>...); el entrenamiento normal solo una.
    """
    ctrls = sample.get("ctrls") or ([sample["ctrl"]] if "ctrl" in sample else [])
    if ctrls:
        ctrls = [c.to(device, non_blocking=True) for c in ctrls]
        hidden = torch.cat([pack_latents(c).expand(latents.shape[0], -1, -1).to(latents.dtype) for c in ctrls]
                           + [latents], dim=1)
        shapes = [(1, c.shape[2], c.shape[3]) for c in ctrls] + [(1, H, W)]
    else:
        hidden, shapes = latents, [(1, H, W)]
    return hidden, [shapes] * latents.shape[0], make_img_mask(sample, H, W, device).expand(latents.shape[0], -1)


def load_nf4_transformer(model_dir):
    """
    Construye el Transformer vacío desde transformer/config.json y lo rellena con la
    caché NF4 (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    """
    cache_dir = os.path.join(model_dir, "transformer")
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(t("index.json not found in the NF4 cache: {path}", path=cache_dir))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = QwenImage21Transformer2DModel.from_config(QwenImage21Transformer2DModel.load_config(cache_dir))

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
        # en la línea siguiente. Era casi todo el tiempo de carga (~25 s de ~28 s).
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=False, quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=False, dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
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
        "trained_with": "AcademiaSD LoRAlab Qwen-Image 2.1",
        "ss_sd_model_name": "Qwen-Image-2.1",
        "ss_base_model_version": "Qwen-Image-2.1",
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

    if SLIDER_RUN:
        meta["slider"] = "true"
        meta["slider_mode"] = SLIDER_RUN["mode"]
        meta["slider_strength_100"] = SLIDER["slider_strength_100"]
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
    meta["ss_output_name"] = PROJECT_NAME or trigger or "qwen_image21_lora"

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


def _export_lora(model, path, step):
    # Formato PEFT/diffusers que carga ComfyUI. Se incluye alpha por capa: sin él,
    # ComfyUI aplicaría el LoRA con escala 1 en lugar de alpha/rank.
    if USE_LOKR:
        save_file(lora_options.lokr_export(model, lambda name: "transformer." + name, export_alpha()),
                  path, metadata=build_lora_metadata(step))
        return
    # En un slider, lora_B se escala para que la fuerza al 100 % (5) dé la posición 100 del dataset.
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        k = "transformer." + k.replace("base_model.model.", "")
        if SLIDER_RUN and k.endswith(".lora_B.weight"):
            v = v * SLIDER_RUN["factor"]
        clean[k] = v.to(torch.bfloat16).cpu().contiguous()
        if k.endswith(".lora_A.weight"):
            clean[k[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(export_alpha()))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLQwenImage21.from_pretrained(
                MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
        return cls.vae


def denoise(model, scheduler, sample, H, W, generator, neg=None):
    device = "cuda"
    latents = pack_latents(torch.randn((1, 64, H, W), generator=generator, device=device, dtype=torch.bfloat16))
    embed, mask = sample["emb"].to(device), sample["msk"].to(device)

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=calculate_shift(H * W))
    scheduler.set_begin_index(0)

    # El prefijo (texto) no depende del paso: se calcula en el primer paso y se reutiliza, como el pipeline oficial.
    # En edición el antes también es prefijo: se cachea con el texto en el primer paso.
    conds = [(sample, QwenImage21KVCache(len(model.transformer_blocks)))]
    # Sin CFG en edición: el negativo tendría que codificarse con la misma imagen de antes.
    use_cfg = neg is not None and PREVIEW_CFG > 1.0 and "ctrl" not in sample
    if use_cfg:
        conds.append(({"emb": neg[0], "msk": neg[1]}, QwenImage21KVCache(len(model.transformer_blocks))))

    for i, t in enumerate(scheduler.timesteps):
        timestep = t.expand(1).to(torch.bfloat16) / 1000
        preds = []
        for cond, cache in conds:
            msk = cond["msk"].to(device)
            hidden, shapes, img_mask = model_inputs(cond, latents, H, W, device)
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=cond["emb"].to(device),
                encoder_hidden_states_mask=None if msk.all() else msk,
                timestep=timestep,
                img_shapes=shapes,
                img_mask=img_mask,
                kv_cache=cache,
                kv_cache_mode="extract" if i == 0 else "cached",
                return_dict=False,
            )[0]
            preds.append(out[:, -H * W:])
        pred = preds[0] if not use_cfg else preds[1] + PREVIEW_CFG * (preds[0] - preds[1])
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    return latents


class TurboHolder:
    # Se lee del disco una sola vez; queda en RAM (no en VRAM) entre previews.
    layers = None
    alpha_over_rank = 1.0

    @classmethod
    def load(cls):
        with safe_open(TURBO_LORA_PATH, framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("lora_adapter_metadata", "{}"))
            cls.layers = {
                k[len("transformer."):-len(".lora_A.weight")]: (
                    f.get_tensor(k).to(torch.bfloat16).pin_memory(),
                    f.get_tensor(k.replace(".lora_A.", ".lora_B.")).to(torch.bfloat16).pin_memory(),
                )
                for k in f.keys() if k.endswith(".lora_A.weight")
            }
        cls.alpha_over_rank = meta.get("transformer.lora_alpha", 1) / meta.get("transformer.r", 1)


def apply_turbo_lora(model):
    """
    Suma el LoRA turbo a la salida de cada capa durante la preview, sin tocar el LoRA
    que se entrena. Devuelve los hooks para quitarlos al terminar.
    """
    if TurboHolder.layers is None:
        if not os.path.exists(TURBO_LORA_PATH):
            print("  [!] " + t("Turbo LoRA not found: {path}", path=TURBO_LORA_PATH))
            return []
        TurboHolder.load()

    scale = TURBO_LORA_STRENGTH * TurboHolder.alpha_over_rank

    def make_hook(a, b):
        def hook(module, args, output):
            return output + F.linear(F.linear(args[0].to(a.dtype), a), b).to(output.dtype) * scale
        return hook

    # Los pesos viven en RAM fijada: en cada preview solo se copian a la GPU, no se leen del disco.
    hooks = []
    for name, (a, b) in TurboHolder.layers.items():
        a, b = a.to("cuda", non_blocking=True), b.to("cuda", non_blocking=True)
        hooks.append(model.get_submodule("base_model.model." + name).register_forward_hook(make_hook(a, b)))

    print("  🚀 Turbo LoRA: " + t("{n} layers (strength {s})", n=len(hooks), s=TURBO_LORA_STRENGTH))
    return hooks


def run_preview(model, scheduler, sample, neg, size, step, slider=None):
    """
    slider: (capas LoRA, [(etiqueta, multiplicador)]). Genera la misma imagen con el LoRA a cada
    multiplicador y guarda una tira con el dial (misma semilla en todas).
    """
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    hooks = []
    if USE_TURBO:
        hooks = apply_turbo_lora(model)
        # El Turbo LoRA se entrenó sin shift_terminal: el 0.02 de la configuración base estropea el último paso.
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config, shift_terminal=None)

    try:
        dial = slider[1] if slider else [(None, None)]
        with torch.no_grad():
            all_latents = []
            for _, mult in dial:
                if mult is not None:
                    slider_core.set_multiplier(slider[0], mult)
                g = torch.Generator(device="cuda").manual_seed(actual_seed)
                all_latents.append(denoise(model, scheduler, sample, H, W, g, neg))
            # Los pesos del Turbo LoRA en la GPU ya no hacen falta: se liberan antes del VAE.
            for h in hooks:
                h.remove()
            hooks = []

            vae = VaeHolder.get().to("cuda")
            images = []
            for latents in all_latents:
                lat = unpack_latents(latents, H, W).to(vae.dtype).unsqueeze(2)
                mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
                std  = torch.tensor(vae.config.latents_std,  device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
                # El VAE devuelve RGBA; en las previews solo interesa el color.
                try:
                    img = vae.decode(lat * std + mean, return_dict=False)[0][:, :3, 0]
                except torch.OutOfMemoryError:
                    # Con 8 GB y el entrenamiento cargado (768² o más) no cabe entera: por mosaicos usa ~1 GB
                    # en vez de ~4.4 GB, a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                    print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                    torch.cuda.empty_cache()
                    vae.enable_tiling()
                    img = vae.decode(lat * std + mean, return_dict=False)[0][:, :3, 0]
                images.append(((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8"))
            vae.to("cpu")

        from PIL import Image, ImageDraw, ImageFont
        pil = [Image.fromarray(img) for img in images]
        result = pil[0]
        if slider:
            # Etiqueta grande y con fondo: la fuerza se tiene que leer aunque la tira se vea reducida.
            size = max(16, pil[0].width // 10)
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1: solo la fuente pequeña
                font = ImageFont.load_default()
            pad = size // 4
            result = Image.new("RGB", (sum(im.width for im in pil), pil[0].height))
            for i, (im, (label, _)) in enumerate(zip(pil, dial)):
                draw = ImageDraw.Draw(im)
                box = draw.textbbox((pad, pad), label, font=font)
                draw.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
                draw.text((pad, pad), label, fill=(255, 255, 0), font=font)
                result.paste(im, (i * pil[0].width, 0))
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        result.save(out)
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
        if slider:
            slider_core.set_multiplier(slider[0], 1.0)
        for h in hooks:
            h.remove()
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
    Ajustes en caliente: si train_settings_qwenimage21.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT
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
    PREVIEW_CAPTION_MODE  = fresh("preview_caption_mode",  str,   PREVIEW_CAPTION_MODE)
    PREVIEW_CUSTOM_PROMPT = fresh("preview_custom_prompt", lambda v: str(v).strip(), PREVIEW_CUSTOM_PROMPT)
    USE_TURBO             = fresh("use_turbo",             bool,  USE_TURBO)
    TURBO_LORA_STRENGTH   = fresh("turbo_lora_strength",   float, TURBO_LORA_STRENGTH)
    SEED                  = fresh("seed",                  int,   SEED)  # solo previews
    return changes


def collate_text(embeds, masks):
    # Con batch > 1 los prompts tienen longitudes distintas: se rellenan a la derecha y se enmascaran.
    max_len = max(e.shape[1] for e in embeds)
    if all(e.shape[1] == max_len for e in embeds):
        return torch.cat(embeds), None
    emb = torch.cat([F.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in embeds])
    msk = torch.cat([F.pad(m, (0, max_len - m.shape[1]), value=False) for m in masks])
    return emb, msk


def train_qwen_image21():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    global BATCH_SIZE, SLIDER_RUN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    # Slider: lo decide la caché (su manifiesto), que es lo que de verdad se va a entrenar.
    manifest = slider_core.read_manifest(CACHE_DIR)
    if cfg.get("lora_type") == "slider" and not manifest:
        print("\n[!] Slider LoRA but the cache is not a slider one: run Pre-Cache again.")
        print("[!] LoRA Slider pero la caché no es de un slider: vuelve a lanzar el Pre-Caché.")
        sys.exit(2)
    if manifest:
        groups, anchors, slider_mode = manifest
        SLIDER_RUN = {"factor": slider_core.export_factor(SLIDER), "mode": slider_mode}
        print(f"\nSlider ({'edit' if slider_mode == 'edit' else 'text-to-image'}): {slider_core.describe(groups, anchors)}")
        if SLIDER["slider_mode"] != slider_mode:
            print("[!] The slider mode changed after the Pre-Cache: training with the cached one. Run Pre-Cache to switch.")
            print("[!] El modo del slider cambió después del Pre-Caché: se entrena con el de la caché. Lanza el Pre-Caché para cambiarlo.")
        if BATCH_SIZE != 1:
            # El multiplicador del LoRA es uno por capa: un batch no puede mezclar pares de signos distintos.
            print(f"[i] Slider: batch {BATCH_SIZE} -> 1 (use Grad Accum to average more steps / usa Grad Accum para promediar más pasos)")
            BATCH_SIZE = 1
        if USE_LOKR:
            # El dial cambia la escala de las capas LoRA de PEFT; LoKr no las usa.
            print("[i] Slider: LoKr -> LoRA (the dial needs LoRA layers / el dial necesita capas LoRA)")
            USE_LOKR = False

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    print(t("Loading {name}...", name="Qwen-Image 2.1 Transformer (NF4)"))
    t0 = time.time()
    transformer = load_nf4_transformer(MODEL_ID)
    transformer.to("cuda")
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 8192),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 0.9),
    )

    transformer.enable_gradient_checkpointing()

    # "blocks": atención + MLP de los 32 bloques. La modulación, el timestep y las capas de
    # entrada/salida se dejan intactas: no hacen falta para una cara o un estilo, y tocarlas
    # rompe la combinación con LoRAs destilados como el turbo (que ajustan justo esas capas).
    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, Linear4bit))
                      and (LORA_TARGETS == "all" or name.startswith("transformer_blocks."))]
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
    slider_layers = slider_core.lora_layers(model) if manifest else None

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
        ckpt = os.path.join(OUTPUT_DIR, f"QwenImage21_LoRA_step_{current_s}.safetensors")
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
        # ctrl (latente de <image1>) e imgmask (huecos de las referencias en el texto).
        # ctrls = varias referencias: <image2> se guarda como <nombre>_ctrl_2.pt, etc.
        sample = {}
        for key, suffix, cast in (("lat", "latent", torch.bfloat16), ("emb", "embed", torch.bfloat16),
                                  ("msk", "mask", torch.bool), ("ctrl", "ctrl", torch.bfloat16),
                                  ("imgmask", "imgmask", torch.bool)):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                t = torch.load(path, weights_only=True).to(cast)
                sample[key] = t.pin_memory() if pin else t
        if "ctrl" in sample:
            ctrls = [sample["ctrl"]]
            index = 2
            while os.path.exists(f"{CACHE_DIR}/{nombre}_ctrl_{index}.pt"):
                extra = torch.load(f"{CACHE_DIR}/{nombre}_ctrl_{index}.pt", weights_only=True).to(torch.bfloat16)
                ctrls.append(extra.pin_memory() if pin else extra)
                index += 1
            if len(ctrls) > 1:
                sample["ctrls"] = ctrls
        return sample

    if manifest:
        # Slider: un latente por imagen; los pares los forma el Sampler. El texto es uno por imagen de
        # entrada en edición (Qwen3-VL la ve con el caption) y uno por grupo en texto a imagen.
        stems = [s for pos in groups.values() for s in pos.values()] + anchors
        slider_lat = {s: load_sample(s)["lat"] for s in stems}
        texts = stems if slider_mode == "edit" else {slider_core.caption_stem(s) for s in stems}
        slider_text = {}
        for name in texts:
            sample = load_sample(name)
            slider_text[name] = {k: sample[k] for k in ("emb", "msk", "imgmask") if k in sample}
        sampler = slider_core.Sampler(groups, anchors, slider_mode, SLIDER["slider_push"])
    else:
        for f in os.listdir(CACHE_DIR):
            if not f.endswith("_latent.pt"):
                continue
            nombre = f.replace("_latent.pt", "")
            cache_data[nombre] = load_sample(nombre)
            buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)

    edit_mode = any("ctrl" in v for v in cache_data.values())

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        cache_data["_custom"] = load_sample("_custom")
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(f"{CACHE_DIR}/_custom_embed.pt") if "_custom" in cache_data else None]

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg = (torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True).to(torch.bfloat16),
               torch.load(f"{CACHE_DIR}/_neg_mask.pt",  weights_only=True).bool())

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

    def slider_step():
        """
        Un paso del slider: el par (o el anchor) que elige el Sampler, cada imagen con su multiplicador
        del LoRA. En texto a imagen las dos imágenes del par comparten ruido y timestep: el paso solo
        ve su diferencia. Devuelve la loss ya dividida entre Grad Accum.
        """
        items = sampler.next()
        first = slider_lat[items[0]["lat"]]
        H, W = first.shape[2], first.shape[3]
        sigma = sample_sigma(1, H * W, "cuda", shift_cfg)
        t_exp = sigma.view(-1, 1, 1)
        noise = torch.randn_like(pack_latents(first.to("cuda", non_blocking=True)))
        total = 0.0
        for item in items:
            slider_core.set_multiplier(slider_layers, item["mult"])
            latent_packed = pack_latents(slider_lat[item["lat"]].to("cuda", non_blocking=True))
            noisy = ((1 - t_exp) * latent_packed + t_exp * noise).to(torch.bfloat16)
            target = noise - latent_packed
            batch = dict(slider_text[item["ctrl"] or item["emb"]])
            if item["ctrl"]:
                batch["ctrl"] = slider_lat[item["ctrl"]]
            msk = batch["msk"].to("cuda", non_blocking=True)
            hidden, shapes, img_mask = model_inputs(batch, noisy, H, W, "cuda")
            pred = model(
                hidden_states=hidden,
                encoder_hidden_states=batch["emb"].to("cuda", non_blocking=True),
                encoder_hidden_states_mask=None if msk.all() else msk,
                timestep=sigma,
                img_shapes=shapes,
                img_mask=img_mask,
                return_dict=False,
            )[0][:, -H * W:]
            # El backward antes de cambiar de multiplicador: el gradient checkpointing recalcula con el actual.
            loss = F.mse_loss(pred.float(), target.float()) / (GRAD_ACCUM_STEPS * len(items))
            loss.backward()
            total += loss.item()
        slider_core.set_multiplier(slider_layers, 1.0)
        return total

    running_loss, t_step_avg, grad_norm = 0.0, 0.0, 0.0
    if manifest:
        print(f"\nSTARTING SLIDER TRAINING / ¡ARRANCANDO ENTRENAMIENTO SLIDER! {slider_core.describe(groups, anchors)} "
              f"(anchors in {sampler.p_anchor():.0%} of the steps / en el {sampler.p_anchor():.0%} de los pasos).")
    else:
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
                    check_custom_prompt("_custom" in cache_data)
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()

            if manifest:
                step_loss = slider_step() * GRAD_ACCUM_STEPS
                running_loss += step_loss
            else:
                size = random.choice(list(buckets))
                names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
                latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True)
                embeds, masks = collate_text([cache_data[n]["emb"] for n in names], [cache_data[n]["msk"] for n in names])
                embeds = embeds.to("cuda", non_blocking=True)
                masks = None if masks is None else masks.to("cuda", non_blocking=True)

                H, W = size
                latent_packed = pack_latents(latents)
                B = latent_packed.shape[0]

                sigma  = sample_sigma(B, H * W, "cuda", shift_cfg)
                noise  = torch.randn_like(latent_packed)
                t_exp  = sigma.view(-1, 1, 1)

                noisy = ((1 - t_exp) * latent_packed + t_exp * noise).to(torch.bfloat16)
                target = noise - latent_packed

                # En edición el antes de cada muestra va delante, limpio. El modelo usa una sola fila de
                # img_mask para todo el batch: vale porque la imagen va antes de la instrucción, así que en
                # un mismo bucket las ranuras del antes caen en las mismas posiciones en todos los pares.
                batch = {**cache_data[names[0]], "emb": embeds}
                batch.pop("ctrls", None)
                if edit_mode:
                    batch["ctrl"] = torch.cat([cache_data[n]["ctrl"] for n in names])
                hidden, shapes, img_mask = model_inputs(batch, noisy, H, W, "cuda")
                pred = model(
                    hidden_states=hidden,
                    encoder_hidden_states=embeds,
                    encoder_hidden_states_mask=masks,
                    timestep=sigma,
                    img_shapes=shapes,
                    img_mask=img_mask,
                    return_dict=False,
                )[0][:, -H * W:]

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
                    check_custom_prompt(True)
                dial = None
                if manifest:
                    # Slider: el prompt de la preview (y en edición su imagen) con el dial a -100 %, -50 %, 0, 50 % y 100 %.
                    p_name, sample = "_custom", cache_data["_custom"]
                    ref = sample.get("ctrl", next(iter(slider_lat.values())))
                    dial = (slider_layers, [(f"{s:+g}" if s else "0", slider_core.strength_to_multiplier(s, SLIDER))
                                            for s in slider_core.preview_strengths(SLIDER)])
                else:
                    p_name = get_preview_sample(step)
                    sample = cache_data[p_name]
                    # Tamaño de la preview: el del objetivo; en edición, el de la imagen de antes
                    # (como en ComfyUI, la salida hereda el tamaño de la referencia).
                    ref = sample.get("ctrl", sample.get("lat", next(iter(cache_data.values()))["lat"]))
                print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
                H, W = ref.shape[2], ref.shape[3]
                if PREVIEW_SIZE > 0:
                    # Cada token es un parche de 16x16 px; lados pares (32 px) porque cada ranura de
                    # imagen de img_mask agrupa 2x2 tokens, como los buckets del pre-caché.
                    s = PREVIEW_SIZE / 32 / math.sqrt(H * W)
                    H, W = max(2, round(H * s) * 2), max(2, round(W * s) * 2)
                run_preview(model, scheduler, sample, neg, (H, W), step, dial)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    final = os.path.join(OUTPUT_DIR, "QwenImage21_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print("✓ " + t("Final LoRA saved to: {path}", path=final))


if __name__ == "__main__":
    train_qwen_image21()
