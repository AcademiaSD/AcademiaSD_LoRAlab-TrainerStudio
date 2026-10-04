# -*- coding: utf-8 -*-
"""
2_train_lora_sdxl.py — Entrenamiento LoRA para SDXL (Base, Pony, Illustrious, NoobAI o un checkpoint propio)
LoRA training for SDXL (Base, Pony, Illustrious, NoobAI or your own checkpoint)

El modelo es el que eligió el Pre-Cache (cached_data_sdxl_<proyecto>/_model.json): los textos se
codificaron con sus text encoders, así que la UNet tiene que ser la del mismo checkpoint.

Lee configuración desde train_settings_sdxl.json si existe.
Reads configuration from train_settings_sdxl.json if present.
"""
import os
import gc
import math
import time
import random
import json
import signal
import sys
from collections import defaultdict

# Buckets de tamaños distintos: sin esto la caché de CUDA se fragmenta y reserva bastante más de lo que se usa.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit, Params4bit
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULTS = {
    "cache_dir": "./cached_data_sdxl",
    "output_dir": "./sdxl_lora_output",
    "total_steps": 1500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_targets": "blocks",
    "precision": "bf16",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "min_snr_gamma": 5.0,
    "save_every": 100,
    "seed": 42,
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 0,
    "preview_size": 0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_sdxl.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = cfg.get("grad_accum_steps",  DEFAULTS["grad_accum_steps"])
LR                = cfg.get("lr",                DEFAULTS["lr"])
MIN_LR_RATIO      = cfg.get("min_lr_ratio",      DEFAULTS["min_lr_ratio"])
WARMUP_STEPS      = cfg.get("warmup_steps",      DEFAULTS["warmup_steps"])
LORA_RANK         = cfg.get("lora_rank",         DEFAULTS["lora_rank"])
LORA_ALPHA        = cfg.get("lora_alpha",        DEFAULTS["lora_alpha"])
LORA_TARGETS      = "all" if cfg.get("lora_targets", DEFAULTS["lora_targets"]) == "all" else "blocks"
# NF4: las capas lineales de los bloques transformer en 4 bits al cargar (para GPUs de portátil de 4-6 GB).
PRECISION         = "nf4" if cfg.get("precision", DEFAULTS["precision"]) == "nf4" else "bf16"
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
MIN_SNR_GAMMA     = cfg.get("min_snr_gamma",     DEFAULTS["min_snr_gamma"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
# 0 = el CFG recomendado para el modelo del preset.
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
# Lado de la preview en píxeles (área size², proporción de la muestra); 0 = tamaño de entrenamiento.
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

if PROJECT_NAME:
    CACHE_DIR  = f"./cached_data_sdxl_{PROJECT_NAME}"
    OUTPUT_DIR = f"./sdxl_lora_output_{PROJECT_NAME}"
else:
    CACHE_DIR  = cfg.get("cache_dir",  DEFAULTS["cache_dir"])
    OUTPUT_DIR = cfg.get("output_dir", DEFAULTS["output_dir"])

MODEL_FILE = os.path.join(CACHE_DIR, "_model.json")
MODEL = json.load(open(MODEL_FILE, encoding="utf-8")) if os.path.exists(MODEL_FILE) else None

print(f"  Model / Modelo           : {MODEL['name'] if MODEL else '(run Pre-Cache first / falta el Pre-Caché)'}")
print(f"  Project / Proyecto       : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra   : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché: {CACHE_DIR}")
print(f"  Output Dir / Salida      : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos      : {TOTAL_STEPS}")
print(f"  Learning Rate / LR       : {LR}")
print(f"  LoRA Rank/Alpha          : {LORA_RANK}/{LORA_ALPHA}")
print(f"  LoRA Targets             : {LORA_TARGETS}")
print(f"  Precision / Precisión    : {PRECISION.upper()}")
print(f"  Batch / Grad Accum       : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Min-SNR gamma            : {MIN_SNR_GAMMA}")
print(f"  Preview Mode / Prompt    : Mode={PREVIEW_CAPTION_MODE} | Custom='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  Preview Every / Steps / CFG / Size: {PREVIEW_EVERY} / {PREVIEW_STEPS} / {PREVIEW_CFG or 'model'} / {PREVIEW_SIZE or 'training'}")
print(f"  Seed Configured / Semilla: {SEED} ({'RANDOM' if SEED <= 0 else 'FIXED'})")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")

if SEED > 0:
    torch.manual_seed(SEED)
    random.seed(SEED)

# El ruido de SDXL: DDPM de 1000 pasos con betas "scaled_linear"; la UNet predice el ruido.
SCHEDULER_CONFIG = dict(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def time_ids(orig, crop, target, device):
    # Condiciones de tamaño de SDXL: (alto, ancho) original, esquina del recorte, (alto, ancho) final.
    return torch.tensor([list(orig) + list(crop) + list(target)], device=device, dtype=torch.bfloat16)


def pad_text(embs, pad_emb):
    """Los textos de un batch se igualan en longitud con bloques vacíos ("" codificado)."""
    n = max(e.shape[0] for e in embs)
    out = []
    for e in embs:
        while e.shape[0] < n:
            e = torch.cat([e, pad_emb[:min(77, n - e.shape[0])]])
        out.append(e)
    return torch.stack(out)


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab SDXL",
        "ss_sd_model_name": MODEL["name"],
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_network_module": "networks.lora",
        "ss_network_dim": LORA_RANK,
        "ss_network_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGETS,
        "ss_learning_rate": LR,
        "ss_lr_scheduler": "cosine_with_warmup",
        "ss_lr_warmup_steps": WARMUP_STEPS,
        "ss_max_train_steps": TOTAL_STEPS,
        "ss_steps": step,
        "ss_batch_size_per_device": BATCH_SIZE,
        "ss_gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "ss_min_snr_gamma": MIN_SNR_GAMMA,
        "ss_seed": SEED,
        "ss_mixed_precision": "bf16",
        "precision": PRECISION,
    }

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "sdxl_lora"

    pc_json = os.path.join(CACHE_DIR, f"pre_cache_settings_{PROJECT_NAME}.json")
    if os.path.exists(pc_json):
        with open(pc_json, "r", encoding="utf-8") as f:
            area = json.load(f).get("target_area")
        if area:
            side = int(round(float(area) ** 0.5))
            meta["ss_resolution"] = f"({side},{side})"

    return {k: str(v) for k, v in meta.items()}


def _export_lora(model, path, step):
    """
    Formato kohya (lora_unet_<capa>.lora_down / lora_up / alpha), el estándar de SDXL: lo cargan
    ComfyUI, Forge, A1111 y CivitAI. Las capas van con sus nombres de diffusers con "_".
    """
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        name, part = k.replace("base_model.model.", "").rsplit(".lora_", 1)
        key = "lora_unet_" + name.replace(".", "_")
        clean[f"{key}.lora_{'down' if part.startswith('A') else 'up'}.weight"] = v.to(torch.bfloat16).cpu().contiguous()
        clean[f"{key}.alpha"] = torch.tensor(float(LORA_ALPHA))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKL.from_single_file(MODEL["checkpoint"], torch_dtype=torch.bfloat16)
        return cls.vae


def run_preview(model, sample, neg, pad_emb, size, step):
    """Euler (el sampler de serie de SDXL) con CFG y el negativo del preset."""
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    cfg_scale = PREVIEW_CFG or MODEL["cfg"]
    print(f"  ↳ Preview Seed used / Semilla utilizada: {actual_seed} | CFG {cfg_scale}")

    try:
        with torch.no_grad():
            sched = EulerDiscreteScheduler(**SCHEDULER_CONFIG, timestep_spacing="leading", steps_offset=1)
            sched.set_timesteps(PREVIEW_STEPS, device="cuda")
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = torch.randn((1, 4, H, W), generator=g, device="cuda") * sched.init_noise_sigma
            emb = pad_text([neg["emb"], sample["emb"]], pad_emb).to("cuda", torch.bfloat16)
            pooled = torch.stack([neg["pooled"], sample["pooled"]]).to("cuda", torch.bfloat16)
            ids = time_ids((H * 8, W * 8), (0, 0), (H * 8, W * 8), "cuda").repeat(2, 1)
            for t in sched.timesteps:
                x = sched.scale_model_input(torch.cat([latents] * 2), t).to(torch.bfloat16)
                out = model(x, t, encoder_hidden_states=emb, added_cond_kwargs={"text_embeds": pooled, "time_ids": ids},
                            return_dict=False)[0].float()
                uncond, cond = out.chunk(2)
                latents = sched.step(uncond + cfg_scale * (cond - uncond), t, latents, return_dict=False)[0]

            vae = VaeHolder.get().to("cuda")
            z = (latents / vae.config.scaling_factor).to(vae.dtype)
            try:
                img = vae.decode(z, return_dict=False)[0]
            except torch.OutOfMemoryError:
                print("  ↳ Low VRAM: tiled VAE decode for previews / Poca VRAM: decodificación por mosaicos en las previews")
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(z, return_dict=False)[0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out)
        print(f"  ↳ Preview saved to / Preview guardada: {out}")
    finally:
        if was_training:
            model.train()
        free_vram()


def check_custom_prompt(has_custom):
    if PREVIEW_CAPTION_MODE != "custom":
        return
    if not has_custom:
        print("\n[PREVIEW] Custom mode but the cache has no encoded prompt: set it, Save JSON and run Pre-Cache "
              "(it only re-encodes the texts). Using the first caption meanwhile.")
        print("[PREVIEW] Modo custom pero la caché no tiene el prompt codificado: escríbelo, Save JSON y lanza el "
              "Pre-Caché (solo vuelve a codificar los textos). Mientras, se usa el primer caption.")
        return

    wanted = PREVIEW_CUSTOM_PROMPT
    if TRIGGER_WORD and wanted and TRIGGER_WORD.lower() not in wanted.lower():
        wanted = f"{TRIGGER_WORD}, {wanted}"
    prompt_file = os.path.join(CACHE_DIR, "_custom_prompt.txt")
    encoded = open(prompt_file, "r", encoding="utf-8").read() if os.path.exists(prompt_file) else ""
    print(f"\n[PREVIEW] Custom prompt / Prompt manual: '{encoded}'")
    if wanted and wanted != encoded:
        print("[PREVIEW] The new prompt is being encoded on CPU (Save JSON); previews switch to it when ready.")
        print("[PREVIEW] El prompt nuevo se está codificando en CPU (Save JSON); las previews lo usarán en cuanto esté.")


_live_mtime = None


def reload_live_settings():
    """
    Ajustes en caliente: si train_settings_sdxl.json ha cambiado (la GUI lo reescribe con Save JSON
    aunque el entrenamiento esté en marcha), aplica la lista de abajo en el paso siguiente. Rank,
    alpha, batch, resolución y carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED

    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return []
    if _live_mtime is None or mtime == _live_mtime:
        _live_mtime = _live_mtime or mtime
        return []

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
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
    SEED                  = fresh("seed",                  int,   SEED)
    return changes


def load_unet():
    """
    UNet del checkpoint en BF16. Con NF4, las capas lineales de los bloques transformer (atención, MLP y
    proyecciones, ~70 % de los pesos) pasan a 4 bits antes de ir a la GPU, así nunca ocupa los 5 GB en
    BF16. Las convoluciones no tienen versión NF4 y se quedan en BF16.

    Leer el checkpoint de un solo archivo carga en RAM el archivo entero y la UNet convertida (~10 GB).
    La primera vez se guarda la UNet ya convertida en SDXL-Models/unet_cache/ (~5 GB en disco); desde
    entonces se carga por mmap y apenas ocupa RAM.
    """
    ckpt = MODEL["checkpoint"]
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    cached = os.path.join("SDXL-Models", "unet_cache", f"{stem}_{os.path.getsize(ckpt)}")
    if not os.path.isfile(os.path.join(cached, "config.json")):
        print("  Converting the UNet once / Convirtiendo la UNet una sola vez (SDXL-Models/unet_cache)...")
        unet = UNet2DConditionModel.from_single_file(ckpt, torch_dtype=torch.bfloat16)
        unet.save_pretrained(cached)
        del unet
        gc.collect()
    unet = UNet2DConditionModel.from_pretrained(cached, torch_dtype=torch.bfloat16)
    unet.requires_grad_(False)
    if PRECISION == "nf4":
        quantized = 0
        for name, module in list(unet.named_modules()):
            if isinstance(module, torch.nn.Linear) and ".attentions." in name:
                layer = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None,
                                   quant_type="nf4", compute_dtype=torch.bfloat16)
                # bitsandbytes cuantiza al mover a la GPU.
                layer.weight = Params4bit(module.weight.data,requires_grad=False, quant_type="nf4")
                if module.bias is not None:
                    layer.bias = torch.nn.Parameter(module.bias.data, requires_grad=False)
                parent, _, child = name.rpartition(".")
                setattr(unet.get_submodule(parent), child, layer)
                quantized += 1
        print(f"[OK] NF4 layers / Capas NF4: {quantized} (transformer blocks / bloques transformer)")
    return unet.to("cuda")


def lora_target_names(unet):
    """
    "blocks": atención, MLP y proyecciones de los bloques transformer (lo que entrena kohya por
    defecto). "all": además las convoluciones de los bloques resnet (LoCon), para estilos.
    """
    linear = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_in", "proj_out")
    names = []
    for name, m in unet.named_modules():
        if ".attentions." in name and name.endswith(linear) and isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            names.append(name)
        elif LORA_TARGETS == "all" and ".resnets." in name and name.endswith(("conv1", "conv2")) and isinstance(m, torch.nn.Conv2d):
            names.append(name)
    return names


def train_sdxl():
    global LORA_RANK, LORA_ALPHA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if MODEL is None or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        sys.exit(2)
    if not os.path.isfile(MODEL["checkpoint"]):
        print(f"[!] Checkpoint not found / No se encuentra el checkpoint: {MODEL['checkpoint']}. Run Pre-Cache / Lanza el Pre-Caché.")
        sys.exit(2)

    print(f"Loading SDXL UNet ({PRECISION.upper()}) from / Cargando UNet de SDXL ({PRECISION.upper()}) de: {MODEL['name']}")
    t0 = time.time()
    unet = load_unet()
    free_vram()
    print(f"UNet loaded in / cargada en {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    unet.enable_gradient_checkpointing()
    noise_scheduler = DDPMScheduler(**SCHEDULER_CONFIG)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")

    target_modules = lora_target_names(unet)
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)} ({LORA_TARGETS})")

    resume_weights = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(resume_weights):
        with safe_open(resume_weights, framework="pt", device="cpu") as f:
            saved_targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1) for k in f.keys()})
        with open(os.path.join(RESUME_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
        if saved != (len(target_modules), LORA_RANK, LORA_ALPHA):
            print(f"\n[!] Resuming with the checkpoint's LoRA: {saved[0]} layers, rank {saved[1]}, alpha {saved[2]} "
                  f"(settings ask for {len(target_modules)} layers, rank {LORA_RANK}, alpha {LORA_ALPHA}; they apply to new trainings).")
            print(f"[!] Se reanuda con el LoRA del checkpoint: {saved[0]} capas, rank {saved[1]}, alpha {saved[2]} "
                  f"(los ajustes piden {len(target_modules)} capas, rank {LORA_RANK}, alpha {LORA_ALPHA}; se aplican a entrenamientos nuevos).")
        target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]

    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                             target_modules=target_modules, init_lora_weights=True)
    model = get_peft_model(unet, lora_config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))

    start_step = 0
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(resume_weights):
        print("=" * 65)
        print("¡Checkpoint detected! Restoring state... / ¡Checkpoint detectado! Restaurando estado...")
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(resume_weights, "rb") as f:
                set_peft_model_state_dict(model, {k: v.float() for k, v in load(f.read()).items()})
            optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
            print(f"Resuming training from step / Reanudando entrenamiento desde el paso {start_step}...")
        except Exception as e:
            print(f"[!] Warning reading checkpoint / Advertencia al leer checkpoint: {e}")
            start_step = 0
        print("=" * 65)

    last_step_executed = start_step

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return
        print(f"\nSaving checkpoint state at step / Guardando estado en paso {current_s}...")
        os.makedirs(RESUME_DIR, exist_ok=True)
        model.save_pretrained(RESUME_DIR)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        ckpt = os.path.join(OUTPUT_DIR, f"SDXL_LoRA_step_{current_s}.safetensors")
        _export_lora(model, ckpt, current_s)
        print(f"✓ Checkpoint saved successfully at step / Checkpoint guardado en paso {current_s}: {ckpt}")

    def handle_signal(sig, frame):
        nonlocal last_step_executed
        print(f"\n[!] Signal received / Señal de detención recibida ({sig}).")
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

    def load_embed(path):
        d = torch.load(path, weights_only=True)
        return {k: (v.pin_memory() if pin else v) for k, v in d.items()}

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        lat = torch.load(f"{CACHE_DIR}/{f}", weights_only=True)
        cache_data[nombre] = {**load_embed(f"{CACHE_DIR}/{nombre}_embed.pt"), "lat": lat["lat"], "orig": lat["orig"], "crop": lat["crop"]}
        buckets[tuple(lat["lat"].shape[-2:])].append(nombre)

    neg = torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True)
    pad_emb = torch.load(f"{CACHE_DIR}/_pad_embed.pt", weights_only=True)["emb"]

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        cache_data["_custom"] = load_embed(f"{CACHE_DIR}/_custom_embed.pt")
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(f"{CACHE_DIR}/_custom_embed.pt") if "_custom" in cache_data else None]

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
    print(f"\nSTARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(all_preview_names)} images / imágenes in {len(buckets)} buckets.")

    reload_live_settings()
    step = start_step
    try:
        while step < TOTAL_STEPS:
            step += 1
            last_step_executed = step

            changes = reload_live_settings()
            if changes:
                print("\n[LIVE] Settings reloaded without stopping / Ajustes recargados sin parar:")
                for c in changes:
                    print(f"[LIVE]   {c}")
                if any(c.startswith(("preview_custom_prompt", "preview_caption_mode")) for c in changes):
                    check_custom_prompt("_custom" in cache_data)
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.stack([cache_data[n]["lat"] for n in names]).to("cuda", torch.float32)
            emb = pad_text([cache_data[n]["emb"] for n in names], pad_emb).to("cuda", torch.bfloat16)
            pooled = torch.stack([cache_data[n]["pooled"] for n in names]).to("cuda", torch.bfloat16)
            H, W = size
            ids = torch.cat([time_ids(cache_data[n]["orig"], cache_data[n]["crop"], (H * 8, W * 8), "cuda") for n in names])

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(names),), device="cuda")
            noisy = noise_scheduler.add_noise(latents, noise, t).to(torch.bfloat16)

            pred = model(noisy, t, encoder_hidden_states=emb,
                         added_cond_kwargs={"text_embeds": pooled, "time_ids": ids}, return_dict=False)[0]

            loss = F.mse_loss(pred.float(), noise, reduction="none").mean(dim=(1, 2, 3))
            if MIN_SNR_GAMMA > 0:
                # Min-SNR: los pasos casi sin ruido dejan de dominar el aprendizaje.
                snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
                loss = loss * torch.clamp(snr, max=MIN_SNR_GAMMA) / snr
            loss = loss.mean() / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for gparam in optimizer.param_groups:
                    gparam["lr"] = lr_at(step)
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
                f"Step/Paso {step:4d}/{TOTAL_STEPS} [{barra}] {pct*100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {lr_at(step):.2e} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                custom_path = f"{CACHE_DIR}/_custom_embed.pt"
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = load_embed(custom_path)
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                ref = cache_data[p_name if p_name != "_custom" else all_preview_names[0]]["lat"]
                print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")
                H, W = ref.shape[-2], ref.shape[-1]
                if PREVIEW_SIZE > 0:
                    # Latente 8x; lados múltiplos de 8 en el latente (64 px).
                    s = PREVIEW_SIZE / 8 / math.sqrt(H * W)
                    H, W = max(8, round(H * s / 8) * 8), max(8, round(W * s / 8) * 8)
                run_preview(model, cache_data[p_name], neg, pad_emb, (H, W), step)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    final = os.path.join(OUTPUT_DIR, "SDXL_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    train_sdxl()
