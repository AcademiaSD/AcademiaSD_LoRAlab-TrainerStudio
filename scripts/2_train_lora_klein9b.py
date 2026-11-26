# -*- coding: utf-8 -*-
"""
2_train_lora_klein9b.py — Entrenamiento LoRA para FLUX.2 [klein] 9B (Transformer NF4)
LoRA training for FLUX.2 [klein] 9B (NF4 Transformer)

Lee configuración desde train_settings_klein9b.json si existe.
Reads configuration from train_settings_klein9b.json if present.
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
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load, load_file
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
import cache_names
from i18n import t
import settings_limits
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import block_swap
import training_metrics
import preview_prompts
import xy_grid
import output_preservation

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "Models/FLUX.2-Klein-9B_NF4",
    "cache_dir": "./Cache_Data/cached_data_klein9b",
    "output_dir": "./Outputs/klein9b_lora_output",
    "total_steps": 1000,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 3e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 16,
    "lora_alpha": 16,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    "use_lokr": False,
    "lokr_factor": 16,
    "use_dora": False,
    "dora_normfix": True,
    "precision": "auto",
    "lora_targets": "blocks",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "seed": 42,
    "timestep_sampling": "shift",
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 4.0,
    "preview_size": 0,
    "preview_width": 0,
    "preview_height": 0,
    "use_turbo": False,
    "turbo_lora_strength": 1.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
    # Memoria (como LTX 2.5 / MiniMax-H3). VRAM Profile simula una tarjeta menor con un tope duro; los bloques
    # que no caben se aparcan y suben justo a tiempo. resident_blocks -1 = lo decide el plan.
    "vram_profile": "auto",
    "resident_blocks": -1,
    # Dónde esperan los bloques aparcados: "ram" (RAM fijada, lo más rápido), "disk" (mapeados al .safetensors
    # del modelo, sin RAM propia) o "auto" (RAM + disco: RAM mientras la RAM total no pase de ram_limit_gb,
    # 0 = sin límite, y queden ram_reserve_gb libres).
    "park_mode": "auto",
    "ram_limit_gb": 0.0,
    "ram_reserve_gb": 8.0,
    # Activaciones guardadas a RAM fijada mientras hay block swap (las entradas de cada bloque).
    "activation_offload": True,
}

CONFIG_PATH = os.environ.get("LORALAB_TRAIN_CONFIG") or "settings/train_settings_klein9b.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        cfg = settings_limits.check(cfg)
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
# DoRA (Liu et al., 2024): cada fila de salida aprende su magnitud aparte de la dirección (la de B·A).
# Es una variante de LoRA (PEFT, use_dora): convive con rsLoRA y LoRA+, no con LoKr. Cada paso descomprime
# el peso NF4 de cada capa para su norma, así que es más lento y pide algo más de VRAM.
USE_DORA          = bool(cfg.get("use_dora",     DEFAULTS["use_dora"]))
# DoRA NormFix: la exportación usa la norma por fila del peso original (dora_norms.safetensors junto al
# transformer NF4) en vez de la del NF4, que es la que divide ComfyUI. Sin esto el LoRA pierde el parecido
# en ComfyUI aunque las previews salgan bien. Solo cambia la exportación.
DORA_NORMFIX      = bool(cfg.get("dora_normfix", DEFAULTS["dora_normfix"]))
# Precisión de cálculo (Auto / BF16 / FP16): ver lora_options.resolve_precision. FP16_OK: las activaciones
# del modelo caben en FP16 (medido), si no Auto se queda en BF16.
FP16_OK           = True
PRECISION_CFG     = str(cfg.get("precision", DEFAULTS["precision"]) or "auto").lower()
PRECISION         = lora_options.resolve_precision(PRECISION_CFG, FP16_OK)
COMPUTE_DTYPE, ADAPTER_DTYPE = lora_options.precision_dtypes(PRECISION)
if USE_LOKR:
    USE_RSLORA = USE_LORAPLUS = USE_DORA = False
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
# Optimizador (AdamW, AdamW 8-bit, Prodigy, Adafactor). Con Prodigy el LR base es 1.0.
OPTIMIZER         = lora_options.optimizer_kind(cfg.get("optimizer"), "adamw")
LR                = lora_options.optimizer_lr(OPTIMIZER, LR)
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
PREVIEW_EVERY     = int(cfg.get("preview_every", DEFAULTS["preview_every"]) or 0)   # null (campo vacío) = 0, sin previews
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_WIDTH     = int(cfg.get("preview_width",  DEFAULTS["preview_width"]))
PREVIEW_HEIGHT    = int(cfg.get("preview_height", DEFAULTS["preview_height"]))
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()
PREVIEW_PROMPTS = preview_prompts.settings(cfg)   # Custom Prompts (hasta 4), cambian en caliente
PREVIEW_NOW = {"index": 0, "prompt": ""}          # preview en curso: nº dentro de la ronda y su prompt

# Previews rápidas: LoRA extraído de la resta FLUX.2 [klein] 9B (destilado) - Base 9B, de kalle07
# (FLUX Non-Commercial License). Destila pasos y CFG: 4 pasos sin CFG.
USE_TURBO           = cfg.get("use_turbo",           DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"])
TURBO_LORA_REPO     = "kalle07/FLUX.2-klein-9B-turbo-lora-set"
TURBO_LORA_FILE     = "Flux_Klein_9b_Turbo_lora_rank_64_bf16_standard.safetensors"

VRAM_PROFILE      = str(cfg.get("vram_profile", DEFAULTS["vram_profile"]) or "auto").strip().lower()
RESIDENT_BLOCKS   = int(cfg.get("resident_blocks", DEFAULTS["resident_blocks"]) if cfg.get("resident_blocks") is not None else -1)
PARK_MODE         = str(cfg.get("park_mode", DEFAULTS["park_mode"]) or "auto").strip().lower()
if PARK_MODE not in ("auto", "ram", "disk"):
    PARK_MODE = "auto"
try:
    RAM_LIMIT_GB  = max(0.0, float(cfg.get("ram_limit_gb", DEFAULTS["ram_limit_gb"]) or 0))
except (TypeError, ValueError):
    RAM_LIMIT_GB  = 0.0
RAM_RESERVE_GB    = float(cfg.get("ram_reserve_gb", DEFAULTS["ram_reserve_gb"]))
ACTIVATION_OFFLOAD = bool(cfg.get("activation_offload", DEFAULTS["activation_offload"]))

TRIGGER_WORD     = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_klein9b_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./klein9b_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

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
print(f"  {'DoRA':<22}: {('on, NormFix' if DORA_NORMFIX else 'on') if USE_DORA else 'off'}")
print(f"  {'Precision':<22}: {PRECISION.upper()}{' (auto)' if PRECISION_CFG not in ('bf16', 'fp16') else ''}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG} | {t('Preview Size')} {PREVIEW_SIZE or t('Training')}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")
print(f"  {'Turbo LoRA':<22}: {t('ON (strength {s})', s=TURBO_LORA_STRENGTH) if USE_TURBO else t('OFF')}")
print(f"  {'VRAM Profile':<22}: {VRAM_PROFILE} | {t('Resident Blocks')} {RESIDENT_BLOCKS if RESIDENT_BLOCKS >= 0 else 'auto'} | "
      f"{t('Block Swap Storage')} {PARK_MODE}{f' (RAM Limit {RAM_LIMIT_GB:g} GB)' if PARK_MODE == 'auto' and RAM_LIMIT_GB > 0 else ''}")

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
        ignore_patterns=["text_encoder/*"],
    )

    print("[OK] " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def sample_sigma(batch_size, image_seq_len, device):
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    # El mismo desplazamiento que la inferencia a 50 pasos (el pipeline lo calcula por resolución).
    e_mu = math.exp(compute_empirical_mu(image_seq_len, 50))
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def pack_latents(x):
    # Latente ya en parches 2x2 (128 canales): cada posición es un token.
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2)


def unpack_latents(x, H, W):
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)


def model_inputs(sample, latents, H, W, device):
    """
    Secuencia de imagen del transformer: el objetivo primero y, en edición, el antes detrás,
    LIMPIO (sin ruido) y con su propia coordenada temporal (t = 10), como en el pipeline.
    La predicción es siempre la del objetivo: los primeros H*W tokens.
    """
    B = latents.shape[0]
    img_ids = Flux2KleinPipeline._prepare_latent_ids(latents.new_empty(B, 1, H, W)).to(device)
    txt_ids = Flux2KleinPipeline._prepare_text_ids(sample["emb"]).to(device)
    if "ctrl" not in sample:
        return latents, img_ids, txt_ids
    ctrls = sample.get("ctrls") or [sample["ctrl"]]
    ctrls = [c.to(device, non_blocking=True) for c in ctrls]
    ctrl_ids = Flux2KleinPipeline._prepare_image_ids([c[:1] for c in ctrls]).to(device)
    hidden = torch.cat([latents] + [pack_latents(c).expand(B, -1, -1).to(latents.dtype) for c in ctrls], dim=1)
    return hidden, torch.cat([img_ids, ctrl_ids.expand(B, -1, -1)], dim=1), txt_ids


BLOCK_PREFIXES = ("transformer_blocks.", "single_transformer_blocks.")


def load_nf4_transformer(model_dir, device="cuda"):
    """
    Construye el Transformer vacío desde transformer/config.json y lo rellena con la
    caché NF4 (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    Todo va a `device` salvo el dato NF4 empaquetado de los bloques, que se queda mapeado a su fichero
    (sin leerlo a RAM): BlockSwap decide después qué bloques suben a la VRAM y dónde esperan los demás.
    """
    cache_dir = os.path.join(model_dir, "transformer")
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(t("index.json not found in the NF4 cache: {path}", path=cache_dir))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config(cache_dir))

    weights_dir = os.path.join(cache_dir, "weights")

    def set_module(name, module):
        parent_name, _, child_name = name.rpartition(".")
        setattr(transformer.get_submodule(parent_name) if parent_name else transformer, child_name, module)

    for name, info in index["quantized"].items():
        path = os.path.join(weights_dir, info["file"])
        in_block = name.startswith(BLOCK_PREFIXES)
        with safe_open(path, framework="pt", device="cpu") as f:
            weight_data = None if in_block else f.get_tensor("weight").to(device)
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}
        if in_block:
            weight_data = block_swap.mmap_tensor(path, "weight")

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device=device)
        weight.bnb_quantized = True

        # En meta: nn.Linear.__init__ reserva y rellena con kaiming una matriz FP32 que se tira
        # en la línea siguiente.
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=False, quant_type="nf4", compute_dtype=COMPUTE_DTYPE)
        layer.weight = weight
        layer._nf4_file = path
        if in_block:
            layer._mmap_home = weight.data
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=False, dtype=COMPUTE_DTYPE)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"]), device=device), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    transformer.load_state_dict(load_file(os.path.join(cache_dir, index["others"]), device=device), strict=False, assign=True)

    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=missing[:5]))

    block_swap.to_device_except_nf4(transformer, device)   # buffers recreados en CPU (rope...)
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
        "trained_with": "AcademiaSD LoRAlab FLUX.2 Klein 9B",
        "ss_sd_model_name": "FLUX.2-klein-base-9B",
        "ss_base_model_version": "Flux.2 Klein 9B",
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
        "ss_mixed_precision": PRECISION,
    }

    # La GUI lee estas claves al exportar o descargar para poner el sufijo _rs / _plus / _lokr al nombre.
    if USE_RSLORA:
        meta["rslora"] = "true"
        meta["rslora_train_alpha"] = LORA_ALPHA
    if USE_LORAPLUS:
        meta["loraplus_ratio"] = LORAPLUS_RATIO
    if USE_DORA:
        meta["dora"] = "true"
    if USE_RSLORA or USE_LORAPLUS or USE_DORA:
        args = {"rslora": USE_RSLORA, "loraplus_lr_ratio": LORAPLUS_RATIO if USE_LORAPLUS else None}
        if USE_DORA:
            args["dora_wd"] = True          # convención de kohya / LyCORIS
        meta["ss_network_args"] = json.dumps(args)
    if USE_LOKR:
        meta["lokr_factor"] = LOKR_FACTOR
        meta["ss_network_args"] = json.dumps({"algo": "lokr", "factor": LOKR_FACTOR})

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "klein9b_lora"

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
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        if "lora_magnitude_vector" in k:
            continue                        # DoRA: va como dora_scale, abajo
        k = "transformer." + k.replace("base_model.model.", "")
        clean[k] = v.to(torch.bfloat16).cpu().contiguous()
        if k.endswith(".lora_A.weight"):
            clean[k[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(export_alpha()))
    # DoRA: <capa>.dora_scale, la clave que leen ComfyUI y Forge (ver lora_options.dora_scales).
    # norm_out.linear: ComfyUI le aplica swap_scale_shift (BFL guarda shift, scale), así que sus filas van cambiadas.
    clean.update(lora_options.dora_export(model, lambda name: "transformer." + name, swapped=("norm_out.linear",)))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=(torch.bfloat16 if PRECISION == "bf16" else torch.float32))
        return cls.vae


def denoise(model, scheduler, sample, H, W, generator, neg=None):
    device = "cuda"
    latents = pack_latents(torch.randn((1, 128, H, W), generator=generator, device=device, dtype=COMPUTE_DTYPE))

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=compute_empirical_mu(H * W, PREVIEW_STEPS))
    scheduler.set_begin_index(0)

    # En edición el negativo lleva la misma imagen de antes: solo cambia el texto, como en el pipeline.
    conds = [sample]
    use_cfg = neg is not None and PREVIEW_CFG > 1.0
    if use_cfg:
        conds.append({**sample, "emb": neg})

    for t in scheduler.timesteps:
        timestep = (t / 1000).expand(1).to(COMPUTE_DTYPE)
        preds = []
        for cond in conds:
            hidden, img_ids, txt_ids = model_inputs(cond, latents, H, W, device)
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=cond["emb"].to(device),
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]
            preds.append(out[:, :H * W])
        pred = preds[0] if not use_cfg else preds[1] + PREVIEW_CFG * (preds[0] - preds[1])
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    return latents


# Nombres del formato original de BFL (el de ComfyUI) -> módulos de diffusers. Las qkv fusionadas
# se reparten en q/k/v: misma matriz down, filas consecutivas de la up.
TURBO_BLOCK_MAP = {
    "double_blocks": ("transformer_blocks", {
        "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
        "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
        "img_attn.proj": ("attn.to_out.0",),
        "txt_attn.proj": ("attn.to_add_out",),
        "img_mlp.0": ("ff.linear_in",),
        "img_mlp.2": ("ff.linear_out",),
        "txt_mlp.0": ("ff_context.linear_in",),
        "txt_mlp.2": ("ff_context.linear_out",),
        "img_attn.norm.query_norm": ("attn.norm_q",),
        "img_attn.norm.key_norm": ("attn.norm_k",),
        "txt_attn.norm.query_norm": ("attn.norm_added_q",),
        "txt_attn.norm.key_norm": ("attn.norm_added_k",),
    }),
    "single_blocks": ("single_transformer_blocks", {
        "linear1": ("attn.to_qkv_mlp_proj",),
        "linear2": ("attn.to_out",),
        "norm.query_norm": ("attn.norm_q",),
        "norm.key_norm": ("attn.norm_k",),
    }),
}
TURBO_TOP_MAP = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}


def turbo_targets(name):
    """Módulos de diffusers para una capa del LoRA turbo, en el orden de sus filas."""
    parts = name.split(".")
    if parts[0] in TURBO_BLOCK_MAP:
        prefix, layers = TURBO_BLOCK_MAP[parts[0]]
        return [f"{prefix}.{parts[1]}.{t}" for t in layers[".".join(parts[2:])]]
    return [TURBO_TOP_MAP[name]]


class TurboHolder:
    # Se lee del disco una sola vez; queda en RAM (no en VRAM) entre previews.
    layers = None
    norms = None

    @classmethod
    def load(cls):
        path = os.path.join(MODEL_ID, "LoRAs", TURBO_LORA_FILE)
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            print("  " + t("Downloading Turbo LoRA (~350 MB): {repo}", repo=TURBO_LORA_REPO))
            path = hf_hub_download(TURBO_LORA_REPO, TURBO_LORA_FILE, local_dir=os.path.join(MODEL_ID, "LoRAs"))

        # Claves diffusion_model.<capa>.lora_down/lora_up (sin alpha: escala 1) y .diff de las normas q/k.
        cls.layers, cls.norms = {}, {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                name = k[len("diffusion_model."):]
                if name.endswith(".diff"):
                    cls.norms[turbo_targets(name[:-len(".diff")])[0]] = f.get_tensor(k).to(COMPUTE_DTYPE)
                    continue
                if not name.endswith(".lora_down.weight"):
                    continue
                name = name[:-len(".lora_down.weight")]
                down = f.get_tensor(k).to(COMPUTE_DTYPE).pin_memory()
                up = f.get_tensor(k.replace(".lora_down.", ".lora_up.")).to(COMPUTE_DTYPE)
                targets = turbo_targets(name)
                if targets == ["norm_out.linear"]:
                    # BFL guarda (shift, scale); la AdaLayerNormContinuous de diffusers espera (scale, shift).
                    shift, scale = up.chunk(2)
                    up = torch.cat([scale, shift])
                for target, rows in zip(targets, up.chunk(len(targets))):
                    cls.layers[target] = (down, rows.contiguous().pin_memory())


def apply_turbo_lora(model):
    """
    Suma el LoRA turbo a la salida de cada capa durante la preview, sin tocar el LoRA que se
    entrena, y sus diferencias a las normas q/k. Devuelve la función que lo deshace todo.
    """
    if TurboHolder.layers is None:
        TurboHolder.load()

    scale = TURBO_LORA_STRENGTH

    def make_hook(a, b):
        def hook(module, args, output):
            return output + F.linear(F.linear(args[0].to(a.dtype), a), b).to(output.dtype) * scale
        return hook

    # Los pesos viven en RAM fijada: en cada preview solo se copian a la GPU, no se leen del disco.
    hooks = []
    for name, (a, b) in TurboHolder.layers.items():
        a, b = a.to("cuda", non_blocking=True), b.to("cuda", non_blocking=True)
        hooks.append(model.get_submodule("base_model.model." + name).register_forward_hook(make_hook(a, b)))

    # Las normas se restauran con una copia exacta, no restando: en BF16 la resta no devuelve el valor original.
    saved = {}
    for name, diff in TurboHolder.norms.items():
        w = model.get_submodule("base_model.model." + name).weight
        saved[name] = w.data.clone()
        w.data += diff.to(w.device, w.dtype) * scale

    def undo():
        for h in hooks:
            h.remove()
        hooks.clear()
        for name, w in saved.items():
            model.get_submodule("base_model.model." + name).weight.data.copy_(w)
        saved.clear()

    print("  🚀 Turbo LoRA: " + t("{a} layers + {b} norms (strength {s})", a=len(hooks), b=len(TurboHolder.norms), s=TURBO_LORA_STRENGTH))
    return undo


def run_preview(model, scheduler, sample, neg, size, step):
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    undo_turbo = apply_turbo_lora(model) if USE_TURBO else None
    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(model, scheduler, sample, H, W, g, neg)
            # Los pesos del Turbo LoRA en la GPU ya no hacen falta: se liberan antes del VAE.
            if undo_turbo:
                undo_turbo()

            vae = VaeHolder.get().to("cuda")
            lat = unpack_latents(latents, H, W).float()
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float()
            bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).float() + vae.config.batch_norm_eps)
            lat = Flux2KleinPipeline._unpatchify_latents(lat * bn_std + bn_mean).to(vae.dtype)
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
        out = os.path.join(OUTPUT_DIR, preview_prompts.preview_name(step, PREVIEW_NOW["index"]))
        preview_prompts.save_png(img, out, PREVIEW_NOW["prompt"])
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
        if undo_turbo:
            undo_turbo()
        if was_training:
            model.train()
        free_vram()


def check_custom_prompt(custom):
    """
    El trainer no tiene text encoder (es la razón de ser del pre-cache): los prompts propios los
    codifica el Pre-Cache, o el servidor en CPU al guardarlos con el entrenamiento en marcha.
    Aquí solo se dice cuáles hay y si alguno aún no está codificado.
    """
    if not preview_prompts.prompts(PREVIEW_PROMPTS):
        return
    print("\n[PREVIEW] " + t("Custom prompts:"))
    for line in preview_prompts.describe(PREVIEW_PROMPTS, CACHE_DIR, TRIGGER_WORD):
        print(line)
    if custom.missing(PREVIEW_PROMPTS):
        print("[PREVIEW] " + t("Prompts marked ... are not encoded yet: they are encoded on CPU when you Save JSON "
                              "(or by Pre-Cache). Until then previews use the ones that are ready, or the dataset captions."))


_live_mtime = None


def reload_live_settings():
    """
    Ajustes en caliente: si train_settings_klein9b.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED
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
    LR                    = fresh("lr",                    lambda v: lora_options.optimizer_lr(OPTIMIZER, float(v)), LR)
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
    SEED                 = fresh("seed",                  int,   SEED)  # solo previews
    preview_prompts.live_update(PREVIEW_PROMPTS, raw, changes)
    return changes


def swap_blocks(transformer):
    """Bloques en el orden en que corren: los 8 dobles y después los 24 simples."""
    return list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)


# Activaciones del paso (GiB) en función de los tokens, con checkpoint reentrante y LoRA rank 16. Medido en una
# RTX 5080 con 3.520 tokens (imágenes de ~1 MP): el pico sube 2,7 GB sobre lo ya colocado, optimizador incluido.
ACT_GB_BASE = 0.6
ACT_GB_PER_TOKEN = 0.00045
DORA_EXTRA_GB = 2.4


def setup_block_swap(transformer, model, card_gb, usable_gb, max_tokens):
    """
    Cuántos bloques quedan residentes y dónde esperan los demás. Lo que ya hay en VRAM (resto del transformer,
    LoRA, contexto) se mide; el optimizador (AdamW: 2 estados FP32 + el gradiente por parámetro) y las
    activaciones del paso se estiman.
    """
    blocks = swap_blocks(transformer)
    # Todo en GiB, como el tope del perfil (usable_gb).
    sizes = [block_swap.block_nf4_bytes(b) / (1024 ** 3) for b in blocks]
    # Lo asignado de verdad: la caché que deja la magnitud inicial de DoRA (cada peso descomprimido) no cuenta.
    free_vram()
    torch.cuda.synchronize()
    in_use = torch.cuda.memory_allocated() / (1024 ** 3)
    trainable_gb = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad) / (1024 ** 3)
    optim_gb = trainable_gb * 3
    act_gb = ACT_GB_BASE + max_tokens * ACT_GB_PER_TOKEN
    if USE_DORA:
        # DoRA descomprime cada peso para su norma, y en el backward de un bloque viven a la vez todos sus pesos
        # descomprimidos. Medido: el pico sube 2,5 GB más que sin DoRA (10,57 frente a 8,07 GB con 3.520 tokens).
        act_gb += DORA_EXTRA_GB
    # Margen para la fragmentación: con muchos tamaños de bucket la VRAM queda en trozos (medido: 1,1 GiB
    # reservados sin usar a los 34 pasos con 10 buckets). Si aun así falta, el paso aparca más bloques y se repite.
    headroom = 0.8
    free = usable_gb - headroom - in_use - optim_gb - act_gb
    if RESIDENT_BLOCKS >= 0:
        resident = min(RESIDENT_BLOCKS, len(blocks))
    else:
        resident = block_swap.plan_resident_sized(free, sizes)
    # Ni con todos los bloques aparcados queda sitio para el que corre y el siguiente. Con DoRA hace falta medio GB
    # más: sus pesos descomprimidos, de tamaños distintos, fragmentan la VRAM (medido: OOM con 2 GB libres en trozos).
    need = 2 * max(sizes) + (0.5 if USE_DORA else 0.0)
    if free < need:
        print("[VRAM] " + t("Warning: this run does not fit the {c:.0f} GB profile even with every block in RAM "
                            "(~{need:.1f} GB more). Pre-cache at a lower resolution, turn DoRA off or choose a larger "
                            "VRAM Profile.", c=card_gb, need=need - free))
    print("[VRAM] " + t("Plan: card {c:.1f} GB | in use {f:.2f} GB | optimizer ~{o:.2f} GB | activations ~{a:.2f} GB "
                        "({tok} tokens) | blocks {b:.2f} GB -> {r}/{n} resident",
                        c=card_gb, f=in_use, o=optim_gb, a=act_gb, tok=max_tokens, b=sum(sizes), r=resident, n=len(blocks)))
    swap = block_swap.BlockSwap(blocks, resident, pin=True, log=print, park=PARK_MODE,
                                ram_reserve_gb=RAM_RESERVE_GB, ram_limit_gb=RAM_LIMIT_GB)
    torch.cuda.synchronize()
    print("[VRAM] " + t("After placing the blocks: {gb:.2f} GB allocated", gb=torch.cuda.memory_allocated() / 1e9))
    print(block_swap.ram_report())
    return swap


def train_klein9b():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR, USE_DORA  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    card_gb, usable_gb = block_swap.apply_vram_profile(VRAM_PROFILE, log=print, t=t)

    # Output preservation: los prompts de clase se codifican antes de cargar el transformer (VRAM libre). En
    # LoRAs de edición no se usa.
    pres = output_preservation.Preservation(cfg, CACHE_DIR, OUTPUT_DIR, TRIGGER_WORD, os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_pre_cache_klein9b.py"),
                                            edit=any(n.endswith("_ctrl.pt") and not cache_names.is_reserved(n) for n in os.listdir(CACHE_DIR)))

    print(t("Loading {name}...", name="FLUX.2 Klein 9B Transformer (NF4)"))
    t0 = time.time()
    # Los bloques quedan mapeados a sus ficheros: setup_block_swap sube después los que caben.
    transformer = load_nf4_transformer(MODEL_ID)
    lora_options.to_compute_dtype(transformer, COMPUTE_DTYPE)
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # "blocks": atención + MLP de los 8 bloques dobles y los 24 simples. La modulación compartida,
    # el timestep y las capas de entrada/salida se dejan intactas: no hacen falta para una cara
    # o un estilo, y tocarlas rompe la combinación con LoRAs destilados.
    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, Linear4bit))
                      and (LORA_TARGETS == "all" or name.startswith(("transformer_blocks.", "single_transformer_blocks.")))]
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
        # DoRA también lo guarda PEFT en adapter_config.json; el checkpoint manda, como con rsLoRA.
        saved_dora = bool(saved_cfg.get("use_dora", False))
        if saved_dora != USE_DORA:
            print("\n[!] " + t("Resuming with the checkpoint's DoRA: {a} (settings ask for {b}; it applies to new trainings).",
                              a=saved_dora, b=USE_DORA))
        USE_DORA = saved_dora

    if USE_LOKR:
        USE_RSLORA = USE_LORAPLUS = USE_DORA = False
        model = lora_options.LoKrModel(transformer, target_modules, LORA_RANK, LORA_ALPHA, LOKR_FACTOR)
    else:
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
            target_modules=target_modules, use_dora=USE_DORA, init_lora_weights=True, use_rslora=USE_RSLORA
        )

        if USE_DORA:
            # La magnitud inicial descomprime cada peso NF4: con los bloques aún en el fichero, se sube a la GPU
            # solo para eso (y sin la copia profunda de PEFT).
            block_swap.patch_peft_dora_init(len(target_modules))
        model = get_peft_model(transformer, lora_config)
        if USE_DORA:
            lora_options.attach_dora_norms(model, os.path.join(MODEL_ID, "transformer"), DORA_NORMFIX, repo_id=HF_REPO_ID, root=MODEL_ID)
    # LoRA y estados del optimizador en FP32: en 8/16 bits se pierde el detalle fino (caras).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    # El LoRA de las capas con el peso aún en el fichero se creó en la CPU: a la GPU, sin subir los bloques.
    block_swap.to_device_except_nf4(model, "cuda")
    # Checkpoint REENTRANTE: el no reentrante guarda el peso NF4 de cada capa y lo fijaría en VRAM (no habría
    # block swap). El reentrante solo da gradiente si alguna entrada del bloque lo pide: la salida de
    # x_embedder lo pide (sin LoRA en esa capa, no lo haría).
    block_swap.enable_reentrant_checkpointing(transformer)

    def _inputs_require_grad(module, inputs, output):
        if torch.is_grad_enabled() and torch.is_tensor(output):
            output.requires_grad_(True)

    transformer.x_embedder.register_forward_hook(_inputs_require_grad)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # LoRA+: lora_B en su propio grupo, con el LR multiplicado (lr_mult se aplica en cada paso).
    if USE_LORAPLUS:
        lora_b = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
        lora_a = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" not in n]
        optimizer = lora_options.make_optimizer([{"params": lora_a, "lr_mult": 1.0},
                                       {"params": lora_b, "lr_mult": LORAPLUS_RATIO}],
                                      lr=LR, weight_decay=WEIGHT_DECAY, kind=OPTIMIZER, warmup=WARMUP_STEPS, default="adamw")
    else:
        optimizer = lora_options.make_optimizer(trainable, lr=LR, weight_decay=WEIGHT_DECAY, kind=OPTIMIZER, warmup=WARMUP_STEPS, default="adamw")
    scaler = lora_options.grad_scaler(PRECISION)   # FP16: loss escalada; BF16: no hace nada

    # New LR curve on resume (issue 88): al reanudar, la curva puede empezar de nuevo en ese paso.
    LR_CURVE = lora_options.LrCurve(OUTPUT_DIR, cfg.get("lr_new_curve", False))
    def lr_at(step):
        return LR * LR_CURVE.factor(LR_SCHEDULER, step, TOTAL_STEPS, WARMUP_STEPS, MIN_LR_RATIO, LR_CYCLES, LR)

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

    # Rejilla XY final: instantáneas del LoRA en los pasos de sus filas (xy_grid.py).
    grid = xy_grid.XYGrid(OUTPUT_DIR, CONFIG_PATH, cfg)
    grid.begin(start_step)

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
        ckpt = os.path.join(OUTPUT_DIR, f"Klein9B_LoRA_step_{current_s}.safetensors")
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
        # lat: objetivo (en _custom no existe: solo aporta el tamaño). En edición además ctrl (latente del antes).
        sample = {}
        for key, suffix in (("lat", "latent"), ("emb", "embed"), ("ctrl", "ctrl")):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                t = torch.load(path, weights_only=True).to(COMPUTE_DTYPE)
                sample[key] = t.pin_memory() if pin else t
        if "ctrl" in sample:
            ctrls = [sample["ctrl"]]
            index = 2
            while os.path.exists(f"{CACHE_DIR}/{nombre}_ctrl_{index}.pt"):
                extra = torch.load(f"{CACHE_DIR}/{nombre}_ctrl_{index}.pt", weights_only=True).to(COMPUTE_DTYPE)
                ctrls.append(extra.pin_memory() if pin else extra)
                index += 1
            if len(ctrls) > 1:
                sample["ctrls"] = ctrls
        return sample

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = load_sample(nombre)
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)
        if pres.active:
            d = torch.load(f"{CACHE_DIR}/{pres.name(nombre)}_embed.pt", weights_only=True).to(COMPUTE_DTYPE)
            cache_data[nombre]["dop"] = d.pin_memory() if pin else d
    pres.begin(start_step)

    edit_mode = any("ctrl" in v for v in cache_data.values())

    # Secuencia más larga del dataset: tokens de imagen (+ los de las imágenes de antes en edición) + 512 de texto.
    def _tokens(sample):
        n = sample["lat"].shape[2] * sample["lat"].shape[3]
        for c in sample.get("ctrls") or ([sample["ctrl"]] if "ctrl" in sample else []):
            n += c.shape[2] * c.shape[3]
        return n + 512
    max_tokens = max(_tokens(v) for v in cache_data.values() if "lat" in v)
    swap = setup_block_swap(transformer, model, card_gb, usable_gb, max_tokens)
    offload = ACTIVATION_OFFLOAD and swap.swapping
    if offload:
        print("[VRAM] " + t("Activation offload ON: the inputs of each block wait in pinned RAM."))

    def load_custom(slot):
        # Cada prompt tiene su embedding y máscaras (<prefix>_*.pt); las imágenes de referencia
        # (ctrl) se codifican una vez como _custom y las comparten todos.
        sample = load_sample(preview_prompts.prefix(slot))
        if slot:
            shared = load_sample("_custom")
            for key in ("ctrl", "ctrls", "ref", "refs", "uncond"):
                if key in shared:
                    sample.setdefault(key, shared[key])
        return sample

    # Custom Prompts: los embeddings se releen cuando el servidor los recodifica en CPU.
    custom = preview_prompts.Embeds(CACHE_DIR, load_custom, prefixed=True)
    custom.refresh(PREVIEW_PROMPTS)
    check_custom_prompt(custom)

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg = torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True).to(COMPUTE_DTYPE)

    all_preview_names = sorted(k for k in cache_data.keys() if not cache_names.is_reserved(k))

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "random":
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

    def make_preview(step, preview_index, preview_slot):
        # Una preview (la de un Custom Prompt o, con slot None, la de Caption Mode). La usan las
        # previews del entrenamiento y la rejilla XY final.
        if preview_slot is None:
            p_name = get_preview_sample(step)
            preview_text = preview_prompts.dataset_caption(cfg, p_name, TRIGGER_WORD)
        else:
            cache_data["_custom"] = custom.items[preview_slot]
            p_name = "_custom"
            preview_text = preview_prompts.custom_text(CACHE_DIR, preview_slot)
        PREVIEW_NOW.update(index=preview_index, prompt=preview_text)
        sample = cache_data[p_name]
        # Tamaño de la preview: el del objetivo; en edición, el de la imagen de antes
        # (como en ComfyUI, la salida hereda el tamaño de la referencia). El prompt manual
        # sin imagen usa el tamaño de la primera del dataset.
        ref = sample.get("ctrl", sample.get("lat", cache_data[all_preview_names[0]]["lat"]))
        print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
        H, W = ref.shape[2], ref.shape[3]
        if PREVIEW_WIDTH > 0 and PREVIEW_HEIGHT > 0 and not edit_mode:
            # Lados explícitos en píxeles: cada token es un parche de 16x16 px.
            H, W = max(1, round(PREVIEW_HEIGHT / 16)), max(1, round(PREVIEW_WIDTH / 16))
        elif PREVIEW_SIZE > 0:
            # Cada token es un parche de 16x16 px.
            s = PREVIEW_SIZE / 16 / math.sqrt(H * W)
            H, W = max(1, round(H * s)), max(1, round(W * s))
        run_preview(model, scheduler, sample, neg, (H, W), step)

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
                if any(c.startswith(("preview_custom", "preview_caption_mode")) for c in changes):
                    custom.refresh(PREVIEW_PROMPTS)
                    check_custom_prompt(custom)
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True)
            # Todos los prompts miden 512 tokens (relleno incluido): se apilan sin máscara.
            embeds = torch.cat([cache_data[n]["emb"] for n in names]).to("cuda", non_blocking=True)

            H, W = size
            latent_packed = pack_latents(latents)
            B = latent_packed.shape[0]

            sigma  = sample_sigma(B, H * W, "cuda")
            noise  = torch.randn_like(latent_packed)
            t_exp  = sigma.view(-1, 1, 1)

            noisy = ((1 - t_exp) * latent_packed + t_exp * noise).to(COMPUTE_DTYPE)
            # El transformer predice la velocidad (ruido - x0), como Flux.1.
            target = noise - latent_packed

            batch = {"emb": embeds}
            if edit_mode:
                batch["ctrl"] = torch.cat([cache_data[n]["ctrl"] for n in names])
            hidden, img_ids, txt_ids = model_inputs(batch, noisy, H, W, "cuda")
            while True:
                try:
                    with (block_swap.SaveOnCpu() if offload else block_swap._null()):
                        pred = model(
                            hidden_states=hidden,
                            encoder_hidden_states=embeds,
                            timestep=sigma.to(COMPUTE_DTYPE),
                            img_ids=img_ids,
                            txt_ids=txt_ids,
                            return_dict=False,
                        )[0][:, :H * W]

                    loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                    break
                except torch.OutOfMemoryError:
                    if swap.resident == 0:
                        raise
                # Sin VRAM (fragmentación, un bucket mayor): como MiniMax-H3, se aparcan 2 bloques más y el paso se
                # repite. Los gradientes a medias de este paso no valen: se descarta la acumulación en curso.
                pred = loss = None
                optimizer.zero_grad(set_to_none=True)
                free_vram()
                parked = swap.park_more(2)
                offload = ACTIVATION_OFFLOAD
                print("\n[VRAM] " + t("Out of VRAM at step {s}: {k} more blocks to RAM ({r}/{n} resident now), and the "
                                      "step is repeated.", s=step, k=parked, r=swap.resident, n=swap.n))
            step_loss = loss.item() * GRAD_ACCUM_STEPS
            running_loss += step_loss

            # Output preservation: mismo latente con ruido y el prompt de clase; la salida con LoRA debe seguir
            # siendo la del modelo sin LoRA. Backward aparte: el grafo de la loss normal ya se ha liberado.
            if pres.due(step):
                c_emb = torch.cat([cache_data[n]["dop"] for n in names]).to("cuda", non_blocking=True)
                c_hidden, c_img_ids, c_txt_ids = model_inputs({"emb": c_emb}, noisy, H, W, "cuda")

                def cls_pred():
                    with (block_swap.SaveOnCpu() if offload else block_swap._null()):
                        return model(hidden_states=c_hidden, encoder_hidden_states=c_emb, timestep=sigma.to(COMPUTE_DTYPE),
                                     img_ids=c_img_ids, txt_ids=c_txt_ids, return_dict=False)[0][:, :H * W]
                pres.step(model, cls_pred, names, scaler, GRAD_ACCUM_STEPS, to_map=lambda x: unpack_latents(x, H, W))
            pres.report(step)

            # La norma solo existe al aplicar el optimizador: entre medias se muestra la última.
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for gparam in optimizer.param_groups:
                    gparam["lr"] = lr_at(step) * gparam.get("lr_mult", 1.0)
                scaler.step(optimizer)      # en FP16 se salta si hubo inf
                scaler.update()
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
                f"lr {lora_options.shown_lr(optimizer, lr_at(step)):.2e} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)
            metrics.log(step, step_loss, lora_options.shown_lr(optimizer, lr_at(step)), grad_norm, t_step, TOTAL_STEPS)
            grid.maybe_snapshot(model, step, TOTAL_STEPS)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                # El servidor codifica en CPU los prompts nuevos: aquí solo se releen los embeddings que cambiaron.
                if custom.refresh(PREVIEW_PROMPTS):
                    check_custom_prompt(custom)
                # Una preview por prompt propio (o una con uno al azar); sin prompts, la de Caption Mode.
                slots = custom.round(PREVIEW_PROMPTS)
                for preview_index, preview_slot in enumerate(slots or [None]):
                    make_preview(step, preview_index, preview_slot)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    print("[VRAM] " + t("Peak: {a:.2f} GB allocated · {r:.2f} GB reserved by PyTorch",
                        a=torch.cuda.max_memory_allocated() / 1e9, r=torch.cuda.max_memory_reserved() / 1e9))
    print(block_swap.ram_report("end"))
    final = os.path.join(OUTPUT_DIR, "Klein9B_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print("✓ " + t("Final LoRA saved to: {path}", path=final))

    # Rejilla XY final: la misma preview que durante el entrenamiento (make_preview), con los Custom
    # Prompts listos o, sin ellos, la primera muestra del dataset.
    globals()["PREVIEW_CAPTION_MODE"] = "first"
    ready = [slot for slot, _ in preview_prompts.prompts(PREVIEW_PROMPTS) if slot in custom.items]
    if ready:
        xy_items = [(preview_prompts.custom_text(CACHE_DIR, s), s) for s in ready]
    else:
        p0 = get_preview_sample(0)
        xy_items = [(preview_prompts.custom_text(CACHE_DIR, 0) if p0 == "_custom"
                     else preview_prompts.dataset_caption(cfg, p0, TRIGGER_WORD), None)]
    grid.run(model, min(last_step_executed, TOTAL_STEPS), [(text, slot, None) for text, slot in xy_items],
             xy_grid.preview_renderer(lambda slot, _size: make_preview(0, 0, slot), globals()),
             random.randint(1, 2147483647) if SEED <= 0 else SEED)


if __name__ == "__main__":
    train_klein9b()
