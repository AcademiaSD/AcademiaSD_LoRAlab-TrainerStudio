# -*- coding: utf-8 -*-
"""
2_train_lora_qwenimage2512.py — Entrenamiento LoRA para Qwen-Image-2512 y Qwen-Image-Edit-2511 (Transformer NF4)
LoRA training for Qwen-Image-2512 and Qwen-Image-Edit-2511 (NF4 Transformer)

Lee configuración desde settings/train_settings_qwenimage2512.json si existe.
Reads configuration from settings/train_settings_qwenimage2512.json if present.
"""
import os
import gc
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
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load
from bitsandbytes.nn import Linear4bit
from i18n import t
import gpu_stats
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import qwen_image2512_model as qm

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/Qwen-Image-2512-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "Models/Qwen-Image-2512-NF4",
    "base_model": "2512",
    "cache_dir": "./Cache_Data/cached_data_qwenimage2512",
    "output_dir": "./Outputs/qwenimage2512_lora_output",
    "total_steps": 500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 2e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 16,
    "lora_alpha": 16,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    "lora_targets": "blocks",
    "blocks_on_gpu": -1,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 50,
    "seed": 42,
    "timestep_sampling": "shift",
    "preview_every": 0,
    "preview_steps": 30,
    "preview_cfg": 4.0,
    "use_turbo": False,
    "turbo_lora_strength": 1.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_qwenimage2512.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID          = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
# "2512" (texto a imagen) o "edit2511" (edición). Un LoRA de edición necesita Edit-2511.
BASE_MODEL        = "edit2511" if cfg.get("base_model", DEFAULTS["base_model"]) == "edit2511" else "2512"
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
USE_RSLORA        = bool(cfg.get("use_rslora",   DEFAULTS["use_rslora"]))
USE_LORAPLUS      = bool(cfg.get("use_loraplus", DEFAULTS["use_loraplus"]))
LORAPLUS_RATIO    = float(cfg.get("loraplus_ratio", DEFAULTS["loraplus_ratio"]))
# Bloques que se quedan en la GPU (de 60); el resto se sube de la RAM bloque a bloque. -1 = automático.
BLOCKS_ON_GPU     = int(cfg.get("blocks_on_gpu", DEFAULTS["blocks_on_gpu"]))
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

# Previews rápidas con Lightning de lightx2v (Apache-2.0): 4 pasos, sin CFG.
USE_TURBO           = cfg.get("use_turbo",           DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"])

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_qwenimage2512_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./qwenimage2512_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

TRANSFORMER_DIR = os.path.join(MODEL_ID, qm.TRANSFORMER_FOLDERS["edit" if BASE_MODEL == "edit2511" else "t2i"])
MODEL_LABEL = "Qwen-Image-Edit-2511" if BASE_MODEL == "edit2511" else "Qwen-Image-2512"
TURBO_LORA_PATH = os.path.join(MODEL_ID, "LoRAs", f"{MODEL_LABEL}-Lightning-4steps-V1.0-bf16.safetensors")

print(f"  {t('Model ID'):<22}: {MODEL_ID} ({MODEL_LABEL})")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Output Dir'):<22}: {OUTPUT_DIR}")
print(f"  {t('Total Steps'):<22}: {TOTAL_STEPS}")
print(f"  {t('Learning Rate'):<22}: {LR}")
print(f"  {t('LR Scheduler'):<22}: {lora_options.lr_scheduler_label(LR_SCHEDULER, WARMUP_STEPS, LR_CYCLES)}")
print(f"  {'LoRA Rank/Alpha':<22}: {LORA_RANK}/{LORA_ALPHA}")
print(f"  {'rsLoRA / LoRA+':<22}: {'on' if USE_RSLORA else 'off'} / {f'x{LORAPLUS_RATIO:g}' if USE_LORAPLUS else 'off'}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {t('Blocks on GPU'):<22}: {t('auto') if BLOCKS_ON_GPU < 0 else BLOCKS_ON_GPU}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")
print(f"  {'Turbo LoRA':<22}: {t('ON (strength {s})', s=TURBO_LORA_STRENGTH) if USE_TURBO else t('OFF')}")

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
                token = json.load(f).get("token", "").strip()
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


def ensure_model_downloaded():
    if os.path.exists(os.path.join(TRANSFORMER_DIR, "index.json")) and os.path.exists(os.path.join(MODEL_ID, "vae", "config.json")):
        print("[OK] " + t("Local model found at: {path}", path=TRANSFORMER_DIR))
        return
    print("⚠ " + t("Local model not found at: {path}", path=TRANSFORMER_DIR))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=HF_REPO_ID))
    enable_hf_file_progress()
    from huggingface_hub import snapshot_download
    # El entrenamiento no usa el text encoder (todo está en el pre-cache) ni el otro transformer.
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MODEL_ID,
        token=get_hf_token(),
        max_workers=2,
        allow_patterns=[os.path.basename(TRANSFORMER_DIR) + "/*", "vae/*", "scheduler/*", "model_index.json", "LICENSE", "README.md",
                        "LoRAs/" + os.path.basename(TURBO_LORA_PATH)],
    )
    print("[OK] " + t("Model downloaded to: {path}", path=MODEL_ID))


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    mu = qm.calculate_shift(image_seq_len, *shift_cfg)
    e_mu = float(np.exp(mu))
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def model_inputs(sample, latents, H, W, device):
    """
    Secuencia de imagen del transformer: el objetivo con ruido delante y, en edición, el antes
    LIMPIO detrás (como QwenImageEditPlusPipeline). Con zero_cond_t (Edit-2511) el transformer le da
    timestep 0 a esos tokens. La predicción es la del objetivo: los primeros (H/2)·(W/2) tokens.
    """
    shapes = [(1, H // 2, W // 2)]
    hidden = latents
    ctrls = sample.get("ctrls") or ([sample["ctrl"]] if "ctrl" in sample else [])
    if ctrls:
        ctrls = [c.to(device, non_blocking=True) for c in ctrls]
        hidden = torch.cat([latents] + [qm.pack_latents(c).expand(latents.shape[0], -1, -1) for c in ctrls], dim=1)
        shapes += [(1, c.shape[2] // 2, c.shape[3] // 2) for c in ctrls]
    return hidden, [shapes] * latents.shape[0]


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": f"AcademiaSD LoRAlab {MODEL_LABEL}",
        "ss_sd_model_name": MODEL_LABEL,
        "ss_base_model_version": MODEL_LABEL,
        "ss_network_module": "peft.LoraModel",
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

    # La GUI lee estas claves al exportar o descargar para poner el sufijo _rs / _plus al nombre.
    if USE_RSLORA:
        meta["rslora"] = "true"
        meta["rslora_train_alpha"] = LORA_ALPHA
    if USE_LORAPLUS:
        meta["loraplus_ratio"] = LORAPLUS_RATIO
    if USE_RSLORA or USE_LORAPLUS:
        meta["ss_network_args"] = json.dumps({"rslora": USE_RSLORA,
                                              "loraplus_lr_ratio": LORAPLUS_RATIO if USE_LORAPLUS else None})

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "qwenimage2512_lora"

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
    # Formato PEFT/diffusers que carga ComfyUI (transformer.transformer_blocks.N...). Se incluye alpha
    # por capa: sin él, ComfyUI aplicaría el LoRA con escala 1 en lugar de alpha/rank.
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        k = "transformer." + k.replace("base_model.model.", "")
        clean[k] = v.to(torch.bfloat16).cpu().contiguous()
        if k.endswith(".lora_A.weight"):
            clean[k[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(export_alpha()))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
        return cls.vae


def denoise(model, scheduler, sample, H, W, generator, neg=None):
    device = "cuda"
    latents = qm.pack_latents(torch.randn((1, 16, H, W), generator=generator, device=device, dtype=torch.bfloat16))
    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    sc = scheduler.config  # el de Lightning fija el shift en 3: base_shift = max_shift = log(3)
    mu = qm.calculate_shift(latents.shape[1], sc.base_image_seq_len, sc.max_image_seq_len, sc.base_shift, sc.max_shift)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    scheduler.set_begin_index(0)

    conds = [sample]
    # En edición el negativo se codificó con la misma imagen de antes (_custom_neg) y lleva su latente.
    use_cfg = neg is not None and PREVIEW_CFG > 1.0
    if use_cfg:
        conds.append({"emb": neg[0], "msk": neg[1],
                      **{k: sample[k] for k in ("ctrl", "ctrls") if k in sample}})

    for t_ in scheduler.timesteps:
        timestep = t_.expand(1).to(torch.bfloat16) / 1000
        preds = []
        for cond in conds:
            msk = cond["msk"].to(device)
            hidden, shapes = model_inputs(cond, latents, H, W, device)
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=cond["emb"].to(device),
                encoder_hidden_states_mask=None if msk.all() else msk,
                timestep=timestep,
                img_shapes=shapes,
                return_dict=False,
            )[0]
            preds.append(out[:, :latents.shape[1]])
        pred = preds[0]
        if use_cfg:
            # CFG real con el reescalado de norma del pipeline oficial.
            comb = preds[1] + PREVIEW_CFG * (preds[0] - preds[1])
            pred = comb * (preds[0].norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))
        latents = scheduler.step(pred, t_, latents, return_dict=False)[0]

    return latents


class TurboHolder:
    # Se lee del disco una sola vez; queda en RAM (no en VRAM) entre previews.
    layers = None

    @classmethod
    def load(cls):
        # Formato de lightx2v: <capa>.lora_down / lora_up / alpha, con los nombres del transformer de diffusers.
        with safe_open(TURBO_LORA_PATH, framework="pt", device="cpu") as f:
            cls.layers = {}
            for k in f.keys():
                if not k.endswith(".lora_down.weight"):
                    continue
                name = k[:-len(".lora_down.weight")]
                down = f.get_tensor(k)
                cls.layers[name] = (down.to(torch.bfloat16).pin_memory(),
                                    f.get_tensor(name + ".lora_up.weight").to(torch.bfloat16).pin_memory(),
                                    f.get_tensor(name + ".alpha").item() / down.shape[0])


def apply_turbo_lora(model):
    """
    Suma el LoRA Lightning a la salida de cada capa durante la preview, sin tocar el LoRA
    que se entrena. Devuelve los hooks para quitarlos al terminar.
    """
    if TurboHolder.layers is None:
        if not os.path.exists(TURBO_LORA_PATH):
            print("  [!] " + t("Turbo LoRA not found: {path}", path=TURBO_LORA_PATH))
            return []
        TurboHolder.load()

    def make_hook(a, b, scale):
        def hook(module, args, output):
            return output + F.linear(F.linear(args[0].to(a.dtype), a), b).to(output.dtype) * scale
        return hook

    hooks = []
    for name, (a, b, alpha_over_rank) in TurboHolder.layers.items():
        a, b = a.to("cuda", non_blocking=True), b.to("cuda", non_blocking=True)
        hooks.append(model.get_submodule("base_model.model." + name).register_forward_hook(
            make_hook(a, b, TURBO_LORA_STRENGTH * alpha_over_rank)))

    print("  🚀 Turbo LoRA: " + t("{n} layers (strength {s})", n=len(hooks), s=TURBO_LORA_STRENGTH))
    return hooks


def run_preview(model, scheduler, sample, neg, size, step):
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    hooks = []
    if USE_TURBO:
        hooks = apply_turbo_lora(model)
        # Lightning se destiló con shift fijo 3 y sin shift_terminal (generate_with_diffusers.py de lightx2v).
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, base_shift=float(np.log(3)), max_shift=float(np.log(3)), shift_terminal=None)

    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(model, scheduler, sample, H, W, g, neg)
            # Los pesos del Turbo LoRA en la GPU ya no hacen falta: se liberan antes del VAE.
            for h in hooks:
                h.remove()
            hooks = []

            vae = VaeHolder.get().to("cuda")
            lat = qm.unpack_latents(latents, H, W).to(vae.dtype).unsqueeze(2)
            mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
            std  = torch.tensor(vae.config.latents_std,  device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
            try:
                img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
            except torch.OutOfMemoryError:
                # Con poca VRAM y el entrenamiento cargado no cabe entera: por mosaicos usa mucho menos,
                # a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out)
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    finally:
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
    Ajustes en caliente: si train_settings_qwenimage2512.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución, bloques en GPU
    y carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED
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


def train_qwenimage2512():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO  # al reanudar se toman del checkpoint
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    edit_cache = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(CACHE_DIR))
    if edit_cache and BASE_MODEL != "edit2511":
        print("\n[!] ERROR: " + t("The cache has before/after pairs: an edit LoRA needs Qwen-Image-Edit-2511 as the base model."))
        sys.exit(1)

    ensure_model_downloaded()

    print(t("Loading {name}...", name=f"{MODEL_LABEL} Transformer (NF4)"))
    t0 = time.time()
    transformer = qm.load_nf4_transformer(TRANSFORMER_DIR)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 8192),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 0.9),
    )

    # "blocks": atención + MLP de los 60 bloques. Las modulaciones (img_mod / txt_mod), el timestep y
    # las capas de entrada/salida se dejan intactas: no hacen falta para una cara o un estilo, y tocarlas
    # rompe la combinación con LoRAs destilados como Lightning (que ajustan justo esas capas).
    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, Linear4bit))
                      and (LORA_TARGETS == "all" or (name.startswith("transformer_blocks.") and "_mod." not in name))]
    print(t("Target LoRA layers: {n} ({kind})", n=len(target_modules), kind=LORA_TARGETS))

    # Reanudar es continuar EL MISMO LoRA: capas, rank y alpha salen del checkpoint, no de la GUI.
    # Si se construyera con otros, el optimizador no encajaría y quedaría una mezcla de pesos.
    # Las capas se leen de los PESOS guardados: adapter_config.json las guarda abreviadas.
    resume_weights = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(resume_weights):
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

    # 20B en NF4 son ~11 GB: los bloques que no caben se suben de la RAM uno a uno (BlockSwap).
    resident = BLOCKS_ON_GPU
    if resident < 0:
        largest = max(int(np.prod(torch.load(os.path.join(CACHE_DIR, f), weights_only=True).shape[-2:]))
                      for f in os.listdir(CACHE_DIR) if f.endswith("_latent.pt") and not f.startswith("_"))
        # VRAM libre de verdad: el escritorio y otros programas también ocupan la GPU.
        resident = qm.auto_resident_blocks(transformer, gpu_stats.mem_get_info()[0] / 1e9,
                                           qm.training_reserve_gb(largest, edit_cache, LORA_RANK))
    swap = qm.BlockSwap(transformer, resident)
    print(t("Blocks on GPU: {n}/60 | from RAM: {gb:.1f} GB ({pin})", n=swap.resident, gb=swap.swapped_gb,
            pin=t("pinned") if swap.pinned else t("not pinned")))
    transformer.enable_gradient_checkpointing(qm.BlockSwap.checkpoint_func)
    # Con el checkpoint reentrante, los LoRA de un bloque solo reciben gradiente si su entrada lo pide.
    for name in ("img_in", "txt_in"):
        getattr(transformer, name).register_forward_hook(lambda m, a, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    free_vram()
    print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0) + f" VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_options.loraplus_groups(model.named_parameters(), LORAPLUS_RATIO) if USE_LORAPLUS else trainable,
        lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        return LR * lora_options.lr_factor(LR_SCHEDULER, step, TOTAL_STEPS, WARMUP_STEPS, MIN_LR_RATIO, LR_CYCLES)

    start_step = 0
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(resume_weights):
        print("=" * 65)
        print(t("Checkpoint detected! Restoring state..."))
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(resume_weights, "rb") as f:
                saved_sd = load(f.read())
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
        with open(os.path.join(RESUME_DIR, "lora_options.json"), "w", encoding="utf-8") as f:
            json.dump({"use_loraplus": USE_LORAPLUS, "loraplus_ratio": LORAPLUS_RATIO}, f)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        ckpt = os.path.join(OUTPUT_DIR, f"QwenImage2512_LoRA_step_{current_s}.safetensors")
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
        for key, suffix, cast in (("lat", "latent", torch.bfloat16), ("emb", "embed", torch.bfloat16),
                                  ("msk", "mask", torch.bool), ("ctrl", "ctrl", torch.bfloat16)):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                tensor = torch.load(path, weights_only=True).to(cast)
                sample[key] = tensor.pin_memory() if pin else tensor
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

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = load_sample(nombre)
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        cache_data["_custom"] = load_sample("_custom")
    if os.path.exists(f"{CACHE_DIR}/_custom_neg_embed.pt"):
        cache_data["_custom_neg"] = load_sample("_custom_neg")
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(f"{CACHE_DIR}/_custom_embed.pt") if "_custom" in cache_data else None]

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg = (torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True).to(torch.bfloat16),
               torch.load(f"{CACHE_DIR}/_neg_mask.pt",  weights_only=True).bool())

    all_preview_names = sorted(k for k in cache_data.keys() if not k.startswith("_"))

    def get_preview_sample(step):
        # Edición: siempre la imagen de antes de la preview, que es donde se ve el efecto y la única
        # con negativo para el CFG (sin CFG, Edit-2511 deja una cuadrícula en la imagen).
        if PREVIEW_CAPTION_MODE == "custom" and "_custom" in cache_data or edit_cache and "_custom_neg" in cache_data:
            return "_custom"
        elif PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    running_loss, t_step_avg, grad_norm = 0.0, 0.0, 0.0
    kind = t("before/after pairs") if edit_cache else t("images")
    print("\n" + t("STARTING TRAINING! {n} {kind} in {b} buckets.", n=len(all_preview_names), kind=kind, b=len(buckets)))

    reload_live_settings()
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
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True)
            embeds, masks = collate_text([cache_data[n]["emb"] for n in names], [cache_data[n]["msk"] for n in names])
            embeds = embeds.to("cuda", non_blocking=True)
            masks = None if masks is None else masks.to("cuda", non_blocking=True)

            H, W = size
            latent_packed = qm.pack_latents(latents)
            B = latent_packed.shape[0]

            sigma  = sample_sigma(B, latent_packed.shape[1], "cuda", shift_cfg)
            noise  = torch.randn_like(latent_packed)
            t_exp  = sigma.view(-1, 1, 1)

            noisy = ((1 - t_exp) * latent_packed + t_exp * noise).to(torch.bfloat16)
            target = noise - latent_packed

            batch = {"emb": embeds}
            if edit_cache:
                batch["ctrl"] = torch.cat([cache_data[n]["ctrl"] for n in names])
            hidden, shapes = model_inputs(batch, noisy, H, W, "cuda")
            pred = model(
                hidden_states=hidden,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=masks,
                timestep=sigma.to(torch.bfloat16),
                img_shapes=shapes,
                return_dict=False,
            )[0][:, :latent_packed.shape[1]]

            loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

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

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                # El servidor codifica en CPU los prompts nuevos: aquí solo se relee el embedding si cambió.
                custom_path = f"{CACHE_DIR}/_custom_embed.pt"
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = load_sample("_custom")
                    if os.path.exists(f"{CACHE_DIR}/_custom_neg_embed.pt"):
                        cache_data["_custom_neg"] = load_sample("_custom_neg")
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                sample = cache_data[p_name]
                # Tamaño de la preview: el del objetivo; en edición, el de la imagen de antes
                # (como en ComfyUI, la salida hereda el tamaño de la referencia).
                ref = sample.get("ctrl", sample.get("lat", next(iter(cache_data.values()))["lat"]))
                print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name}")
                p_neg = neg
                if "ctrl" in sample:
                    p_neg = (cache_data["_custom_neg"]["emb"], cache_data["_custom_neg"]["msk"]) if "_custom_neg" in cache_data else None
                run_preview(model, scheduler, sample, p_neg, (ref.shape[2], ref.shape[3]), step)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    final = os.path.join(OUTPUT_DIR, "QwenImage2512_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print("✓ " + t("Final LoRA saved to: {path}", path=final))


if __name__ == "__main__":
    train_qwenimage2512()
