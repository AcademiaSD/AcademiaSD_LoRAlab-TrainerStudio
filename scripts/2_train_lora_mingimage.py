# -*- coding: utf-8 -*-
"""
2_train_lora_mingimage.py — Entrenamiento LoRA para Ming-Image Design (DiT NF4)
LoRA training for Ming-Image Design (NF4 DiT)

Se entrena solo el DiT (t2i). El MLLM y el connector no se tocan: el texto ya viene
codificado del Pre-Cache (`_embed` = cap_feats 256x2560, `_embed2` = tokens del VLM
3840). El LoRA exportado es un LoRA normal de PEFT, sin tocar el text encoder.

Edición (image-edit): si la caché trae `_ctrl` (el latente del antes), el DiT lo recibe como
`ref_x`, que concatena al ruido como un segundo fotograma; la salida es el primero.

Lee configuración desde train_settings_mingimage.json si existe.
Reads configuration from train_settings_mingimage.json if present.
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
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load
from bitsandbytes.nn import Linear4bit
import cache_names
from i18n import t
import settings_limits
import lora_options
import training_metrics
import preview_prompts
import xy_grid
import output_preservation
import block_swap
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir

# El VAE de Ming (RGBA, 16 canales) tampoco está en diffusers: viene del repo oficial.
from mingimage_model import DiffusionTransformer
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/Ming-Image-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "Models/Ming-Image_NF4",
    "cache_dir": "./Cache_Data/cached_data_mingimage",
    "output_dir": "./Outputs/mingimage_lora_output",
    "total_steps": 1500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 4e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 8,
    "lora_alpha": 8,
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
    "save_every": 50,
    "seed": 42,
    "timestep_sampling": "shift",
    "train_shift": 3.0,
    "preview_every": 0,
    "preview_steps": 12,
    "preview_cfg": 1.0,
    "preview_size": 0,
    "preview_width": 0,
    "preview_height": 0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
    # Memoria (como FLUX.2 Klein 9B): VRAM Profile, Resident Blocks, Block Swap Storage y RAM Limit;
    # sus valores por defecto están en block_swap.SETTINGS_DEFAULTS.
}

CONFIG_PATH = os.environ.get("LORALAB_TRAIN_CONFIG") or "settings/train_settings_mingimage.json"

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
TRAIN_SHIFT       = cfg.get("train_shift",       DEFAULTS["train_shift"])
PREVIEW_EVERY     = int(cfg.get("preview_every", DEFAULTS["preview_every"]) or 0)   # null (campo vacío) = 0, sin previews
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
# Lado de la preview en píxeles (área size², proporción de la muestra); 0 = tamaño de entrenamiento.
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_WIDTH     = int(cfg.get("preview_width",  DEFAULTS["preview_width"]))
PREVIEW_HEIGHT    = int(cfg.get("preview_height", DEFAULTS["preview_height"]))
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()
PREVIEW_PROMPTS = preview_prompts.settings(cfg)   # Custom Prompts (hasta 4), cambian en caliente
PREVIEW_NOW = {"index": 0, "prompt": ""}          # preview en curso: nº dentro de la ronda y su prompt

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./Cache_Data/cached_data_mingimage_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./Outputs/mingimage_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

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
SWAP_CFG = block_swap.settings(cfg)
print(f"  {'VRAM Profile':<22}: {block_swap.describe(SWAP_CFG)}")
print(f"  {'Precision':<22}: {PRECISION.upper()}{' (auto)' if PRECISION_CFG not in ('bf16', 'fp16') else ''}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {'Timesteps / Shift':<22}: {TIMESTEP_SAMPLING} / {TRAIN_SHIFT}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG} | {t('Preview Size')} {PREVIEW_SIZE or t('Training')}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")

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

    from huggingface_hub import snapshot_download

    # El entrenamiento no usa el MLLM (el texto ya está cacheado): se salta sus 9 GB.
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["mllm/*"],
    )

    print("[OK] " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def sample_sigma(batch_size, device):
    # Desplazamiento fijo como el scheduler de Ming (6.0 en inferencia): con 3.0 el
    # entrenamiento cubre también los niveles de ruido bajos, donde se aprenden caras y detalle.
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    sigma = TRAIN_SHIFT * u / (1 + (TRAIN_SHIFT - 1) * u)
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def predict(model, latents, sigma, embeds, embeds2, ctrl=None):
    """
    Ming recibe listas: un latente [C, 1, H, W] por imagen, un texto [256, 2560] y una
    segunda condición [N, 3840] por imagen (N variable, por eso va como lista: el modelo
    rellena por dentro). El tiempo va al revés que sigma (1 = imagen limpia) y la salida
    es x0 - ruido, el opuesto de la velocidad que usa el scheduler.
    En edición `ctrl` es el latente del antes (mismo tamaño): va como ref_x, el DiT lo pega
    como segundo fotograma y devuelve los dos; el que se predice es el primero.
    """
    out = model(x=list(latents.unsqueeze(2).unbind(0)), t=1.0 - sigma,
                # El cache guarda (1, tokens, dim); el DiT espera cada condición como (tokens, dim).
                cap_feats=[e.squeeze(0) if e.dim() == 3 else e for e in embeds],
                cap_feats_2=[e.squeeze(0) if e.dim() == 3 else e for e in embeds2],
                ref_x=list(ctrl.to(latents.dtype).unsqueeze(2).unbind(0)) if ctrl is not None else None,
                return_dict=False)[0]
    return torch.stack([o[:, 0] for o in out])


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab Ming-Image",
        "ss_sd_model_name": "Ming-Image",
        "ss_base_model_version": "Ming-Image",
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

    if EDIT_LORA:
        meta["lora_type"] = "edit"

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "mingimage_lora"

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
    # Formato PEFT/diffusers (une to_q/to_k/to_v en su qkv al cargarlo). Se incluye alpha
    # por capa: sin él, el cargador aplicaría el LoRA con escala 1 en lugar de alpha/rank.
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
    clean.update(lora_options.dora_export(model, lambda name: "transformer." + name))
    save_file(clean, path, metadata=build_lora_metadata(step))


# Lo decide la caché (si trae _ctrl), no los ajustes: así un LoRA no se entrena con el tipo cambiado.
EDIT_LORA = False


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", dtype=(torch.bfloat16 if PRECISION == "bf16" else torch.float32))
        return cls.vae


def denoise(model, scheduler, embed, embed2, H, W, generator, ctrl=None):
    # Mismo muestreo que el pipeline oficial de Ming: sigmas lineales con el shift del scheduler.
    # CFG del pipeline: pred = pos + cfg * (pos - neg) (solo con cfg > 1); el scheduler recibe -pred.
    device = "cuda"
    latents = torch.randn((1, 16, H, W), generator=generator, device=device, dtype=torch.float32)

    # Timesteps del scheduler oficial (shift 6.0). Con sigmas lineales al estilo Z-Image el
    # último paso caería de sigma≈0.35 a 0 y la preview saldría rota.
    scheduler.set_timesteps(PREVIEW_STEPS, device=device)
    scheduler.set_begin_index(0)

    use_cfg = PREVIEW_CFG > 1.0
    texts = [embed.to(device)]
    texts2 = [embed2.to(device)]
    if use_cfg:
        # El negativo oficial es un cero exacto con la MISMA longitud que el positivo
        # (cap_feats_2 tiene N tokens variables, no vale un negativo fijo).
        texts.append(torch.zeros_like(texts[0]))
        texts2.append(torch.zeros_like(texts2[0]))

    for t in scheduler.timesteps:
        sigma = (t / 1000).expand(len(texts))
        # En edición el negativo también ve el antes (el pipeline oficial repite ref_x en el CFG).
        ref = ctrl.to(device).expand(len(texts), -1, -1, -1) if ctrl is not None else None
        out = predict(model, latents.to(COMPUTE_DTYPE).expand(len(texts), -1, -1, -1), sigma, texts, texts2, ref).float()
        pred = out[:1] + PREVIEW_CFG * (out[:1] - out[1:]) if use_cfg else out
        latents = scheduler.step(-pred, t, latents, return_dict=False)[0]

    return latents


def run_preview(model, scheduler, embed, embed2, size, step, ctrl=None):
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(model, scheduler, embed, embed2, H, W, g, ctrl)

            vae = VaeHolder.get().to("cuda")
            lat = latents.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
            try:
                img = vae.decode(lat.unsqueeze(2), return_dict=False)[0]
            except torch.OutOfMemoryError:
                # Con 8 GB y el entrenamiento cargado puede no caber entera: por mosaicos usa mucha menos
                # VRAM, a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(lat.unsqueeze(2), return_dict=False)[0]
            # Ming decodifica RGBA: (1, 4, 1, H, W) -> (H, W, 4). El alfa se guarda tal cual.
            img = img[0, :, 0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, preview_prompts.preview_name(step, PREVIEW_NOW["index"]))
        preview_prompts.save_png(img, out, PREVIEW_NOW["prompt"])
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
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
    Ajustes en caliente: si train_settings_mingimage.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED

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
    SEED                  = fresh("seed",                  int,   SEED)  # solo previews
    preview_prompts.live_update(PREVIEW_PROMPTS, raw, changes)
    return changes


BLOCK_PREFIXES = ("noise_refiner.", "context_refiner.", "layers.")


def swap_blocks(transformer):
    """Bloques en el orden en que corren: refinador de imagen (2), de texto (2) y los 30 principales."""
    return list(transformer.noise_refiner) + list(transformer.context_refiner) + list(transformer.layers)


# Activaciones del paso (GiB) en función de los tokens (imagen + texto), con checkpoint reentrante.
ACT_GB_BASE = 0.6
ACT_GB_PER_TOKEN = 0.00045
DORA_EXTRA_GB = 1.3   # medido a 1024²: el pico sube ~1,3 GB con DoRA


def is_target(name, module):
    if not isinstance(module, (torch.nn.Linear, Linear4bit)):
        return False
    if LORA_TARGETS == "all":
        return True
    # "blocks": atención + MLP de los 30 bloques y de los refinadores de imagen y texto.
    return name.startswith(("layers.", "noise_refiner.", "context_refiner.")) and "adaLN_modulation" not in name


def train_mingimage():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR, USE_DORA  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    global EDIT_LORA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # Output preservation: los prompts de clase se codifican antes de cargar el DiT (VRAM libre). En LoRAs de
    # edición no se usa (cada muestra tiene su _ctrl.pt).
    pres = output_preservation.Preservation(
        cfg, CACHE_DIR, OUTPUT_DIR, TRIGGER_WORD, os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_pre_cache_mingimage.py"),
        edit=any(n.endswith("_ctrl.pt") and not cache_names.is_reserved(n) for n in os.listdir(CACHE_DIR)))

    card_gb, usable_gb = block_swap.apply_vram_profile(SWAP_CFG.vram_profile, log=print, t=t)

    print(t("Loading {name}...", name="Ming-Image DiT (NF4)"))
    t0 = time.time()
    # Los bloques quedan mapeados a sus ficheros: block_swap.setup sube después los que caben.
    transformer, nq, nu = block_swap.load_index_nf4(DiffusionTransformer, os.path.join(MODEL_ID, "transformer"),
                                                    BLOCK_PREFIXES, "cuda", torch.bfloat16)
    print("[OK] " + t("NF4 layers: {a} | BF16 layers: {b}", a=nq, b=nu))
    lora_options.to_compute_dtype(transformer, COMPUTE_DTYPE)
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # "blocks" deja intactas la modulación adaLN, el timestep y las capas de entrada/salida:
    # no hacen falta para una cara o un estilo y así el LoRA combina mejor con otros.
    target_modules = [name for name, m in transformer.named_modules() if is_target(name, m)]
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
    # block swap). El reentrante solo da gradiente si alguna entrada del bloque lo pide: las de los embedders de
    # imagen y de texto (el refinador de texto también lleva LoRA).
    block_swap.enable_reentrant_checkpointing(transformer)
    for emb in transformer.all_x_embedder.values():
        block_swap.inputs_require_grad(emb)
    block_swap.inputs_require_grad(transformer.cap_embedder)
    if PRECISION == "fp16":
        # En FP16 este modelo pasa de 65504 (medido en BF16): la entrada de la atención (attention_norm1 · escala
        # del adaLN) hasta ~75k, q/k/v hasta ~360k, su salida hasta ~1,9M y el producto del MLP (silu(w1)·w3,
        # y w2) hasta ~10M; el flujo residual no pasa de ~13k. Todo lo que va de attention_norm1 a
        # norm_q/norm_k y attention_norm2, y de w3 a ffn_norm2, es lineal y sin bias, y una RMSNorm quita la
        # escala: dividir la salida de attention_norm1 entre 256 y la de w3 entre 1024 da el mismo resultado
        # (LoRA incluido) y deja todo por debajo de ~10000.
        a = lora_options.fp16_rescale(model, r".*\.attention_norm1", 256, on="output")
        b = lora_options.fp16_rescale(model, r".*\.feed_forward\.w3", 1024, on="output")
        # Con ruido puro (previews) la salida de w2 pasa de 65504 aun dividida (medido: >67M sin dividir): w2 va
        # en FP32 y su salida llega así a ffn_norm2.
        c = lora_options.fp32_output(model, r".*\.feed_forward\.w2")
        print(f"[OK] FP16: rescaled {a} attention inputs and {b} MLP gates; {c} MLP outputs in FP32")
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
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(resume_weights):
        print("=" * 65)
        print(t("Checkpoint detected! Restoring state..."))
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(resume_weights, "rb") as f:
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
        ckpt = os.path.join(OUTPUT_DIR, f"MingImage_LoRA_step_{current_s}.safetensors")
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

    def load_tensor(path):
        t = torch.load(path, weights_only=True).to(COMPUTE_DTYPE)
        return t.pin_memory() if pin else t

    cache_data, buckets = {}, defaultdict(list)
    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = {
            "lat": load_tensor(f"{CACHE_DIR}/{f}"),
            "emb": load_tensor(f"{CACHE_DIR}/{nombre}_embed.pt"),
            "emb2": load_tensor(f"{CACHE_DIR}/{nombre}_embed2.pt"),
        }
        # Edición: el latente del antes, con el mismo bucket que el objetivo.
        if os.path.exists(f"{CACHE_DIR}/{nombre}_ctrl.pt"):
            cache_data[nombre]["ctrl"] = load_tensor(f"{CACHE_DIR}/{nombre}_ctrl.pt")
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)
        if pres.active:
            cache_data[nombre]["dop"] = load_tensor(f"{CACHE_DIR}/{pres.name(nombre)}_embed.pt")
            cache_data[nombre]["dop2"] = load_tensor(f"{CACHE_DIR}/{pres.name(nombre)}_embed2.pt")
    pres.begin(start_step)

    EDIT_LORA = any("ctrl" in d for d in cache_data.values())
    if EDIT_LORA and not all("ctrl" in d for d in cache_data.values()):
        print("\n[!] ERROR: " + t("The cache mixes edit pairs and single images: run Pre-Cache again."))
        sys.exit(2)

    custom_ctrl_path = f"{CACHE_DIR}/_custom_ctrl.pt"

    def load_custom_sample(name):
        # Los dos embeddings del prompt (cap_feats y directvlm) y, en edición, la imagen de antes.
        sample = {"emb": load_tensor(f"{CACHE_DIR}/{name}_embed.pt"), "emb2": load_tensor(f"{CACHE_DIR}/{name}_embed2.pt")}
        if EDIT_LORA and os.path.exists(custom_ctrl_path):
            sample["ctrl"] = load_tensor(custom_ctrl_path)
        return sample

    def load_custom(slot):
        # Cada prompt tiene su embedding y máscaras (<prefix>_*.pt); las imágenes de referencia
        # (ctrl) se codifican una vez como _custom y las comparten todos.
        sample = load_custom_sample(preview_prompts.prefix(slot))
        if slot:
            shared = load_custom_sample("_custom")
            for key in ("ctrl", "ctrls", "ref", "refs", "uncond"):
                if key in shared:
                    sample.setdefault(key, shared[key])
        return sample

    # Custom Prompts: los embeddings se releen cuando el servidor los recodifica en CPU.
    custom = preview_prompts.Embeds(CACHE_DIR, load_custom, prefixed=True)
    custom.refresh(PREVIEW_PROMPTS)
    check_custom_prompt(custom)

    all_preview_names = sorted(k for k in cache_data.keys() if not cache_names.is_reserved(k))

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    # Tokens del paso más grande: imagen (parches 2x2; en edición, dos fotogramas) + texto.
    def _tokens(v):
        n = v["lat"].shape[-2] * v["lat"].shape[-1] // 4 * (2 if "ctrl" in v else 1)
        return n + v["emb"].shape[-2] + v["emb2"].shape[-2]
    max_tokens = max(_tokens(v) for v in cache_data.values())
    act_gb = ACT_GB_BASE + max_tokens * ACT_GB_PER_TOKEN + (DORA_EXTRA_GB if USE_DORA else 0.0)
    swap = block_swap.setup(swap_blocks(transformer), model, card_gb, usable_gb, act_gb,
                            resident_blocks=SWAP_CFG.resident_blocks, park=SWAP_CFG.park_mode,
                            ram_limit_gb=SWAP_CFG.ram_limit_gb, ram_reserve_gb=SWAP_CFG.ram_reserve_gb,
                            dora=USE_DORA, dora_margin_gb=0.0, log=print, t=t)
    offload = SWAP_CFG.activation_offload and swap.swapping
    if offload:
        print("[VRAM] " + t("Activation offload ON: the inputs of each block wait in pinned RAM."))

    running_loss, t_step_avg, grad_norm = 0.0, 0.0, 0.0
    kind = t("before/after pairs") if EDIT_LORA else t("images")
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
        # En edición la salida hereda el tamaño del antes (como en el pipeline oficial): el prompt
        # manual usa su propio antes o, si no lo hay, el del primer par.
        ctrl = sample.get("ctrl", cache_data[all_preview_names[0]].get("ctrl")) if EDIT_LORA else None
        # El prompt manual sin imagen usa el tamaño de la primera del dataset.
        ref = ctrl if ctrl is not None else cache_data[p_name if p_name != "_custom" else all_preview_names[0]]["lat"]
        print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
        H, W = ref.shape[2], ref.shape[3]
        if PREVIEW_WIDTH > 0 and PREVIEW_HEIGHT > 0 and ctrl is None:
            # Lados explícitos en píxeles: latente 8x y parches 2x2 (16 px).
            H = max(2, round(PREVIEW_HEIGHT / 8 / 2) * 2)
            W = max(2, round(PREVIEW_WIDTH / 8 / 2) * 2)
        elif PREVIEW_SIZE > 0 and ctrl is None:
            # Latente 8x y parches 2x2: lados múltiplos de 2 en el latente (16 px).
            s = PREVIEW_SIZE / 8 / math.sqrt(H * W)
            H, W = max(2, round(H * s / 2) * 2), max(2, round(W * s / 2) * 2)
        run_preview(model, scheduler, sample["emb"], sample["emb2"], (H, W), step, ctrl)

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
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True).float()
            embeds = [cache_data[n]["emb"].to("cuda", non_blocking=True) for n in names]
            embeds2 = [cache_data[n]["emb2"].to("cuda", non_blocking=True) for n in names]
            ctrl = torch.cat([cache_data[n]["ctrl"] for n in names]).to("cuda", non_blocking=True) if EDIT_LORA else None

            sigma = sample_sigma(len(names), "cuda")
            noise = torch.randn_like(latents)
            t_exp = sigma.view(-1, 1, 1, 1)

            noisy = ((1 - t_exp) * latents + t_exp * noise).to(COMPUTE_DTYPE)
            target = latents - noise

            while True:
                try:
                    with (block_swap.SaveOnCpu() if offload else block_swap._null()):
                        pred = predict(model, noisy, sigma, embeds, embeds2, ctrl)
                    loss = F.mse_loss(pred.float(), target) / GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                    break
                except torch.OutOfMemoryError:
                    pred = loss = None
                    if not block_swap.oom_recover(swap, optimizer, step, log=print, t=t):
                        raise
                    offload = SWAP_CFG.activation_offload
            step_loss = loss.item() * GRAD_ACCUM_STEPS
            running_loss += step_loss

            # Output preservation: mismo latente con ruido y el prompt de clase; la salida con LoRA debe seguir
            # siendo la del modelo sin LoRA. Backward aparte: el grafo de la loss normal ya se ha liberado.
            if pres.due(step):
                cls = [cache_data[n]["dop"].to("cuda", non_blocking=True) for n in names]
                cls2 = [cache_data[n]["dop2"].to("cuda", non_blocking=True) for n in names]

                def cls_pred():
                    with (block_swap.SaveOnCpu() if offload else block_swap._null()):
                        return predict(model, noisy, sigma, cls, cls2, ctrl)
                pres.step(model, cls_pred, names, scaler, GRAD_ACCUM_STEPS)
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
    print(block_swap.peak_report(t))
    final = os.path.join(OUTPUT_DIR, "MingImage_FINAL_LoRA.safetensors")
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
    train_mingimage()
