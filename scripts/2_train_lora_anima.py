# -*- coding: utf-8 -*-
"""
2_train_lora_anima.py — Entrenamiento LoRA para Anima (Transformer BF16 o NF4)
LoRA training for Anima (BF16 or NF4 Transformer)

Se entrena sobre Anima-Base, como recomienda su autor. El transformer (Cosmos-Predict2, 2B) se
carga en BF16, o en NF4 para GPUs de 4 GB: se cuantiza al cargar, solo atención y MLP.

Lee configuración desde train_settings_anima.json si existe.
Reads configuration from train_settings_anima.json if present.
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
from diffusers import AutoencoderKLQwenImage, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load
from bitsandbytes.nn import Linear4bit, Params4bit

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"

DEFAULTS = {
    "model_id": "Anima-Base",
    "cache_dir": "./cached_data_anima",
    "output_dir": "./anima_lora_output",
    "precision": "bf16",
    "total_steps": 1500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 2e-5,
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_targets": "blocks",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 50,
    "seed": 42,
    "timestep_sampling": "shift",
    "train_shift": 3.0,
    "preview_every": 0,
    "preview_steps": 30,
    "preview_cfg": 4.0,
    "preview_size": 0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_anima.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID          = cfg.get("model_id",          DEFAULTS["model_id"])
# "nf4": atención y MLP en 4 bits (~1.2 GB en vez de ~3.6 GB) para GPUs de 4 GB.
PRECISION         = "nf4" if cfg.get("precision", DEFAULTS["precision"]) == "nf4" else "bf16"
TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = cfg.get("grad_accum_steps",  DEFAULTS["grad_accum_steps"])
LR                = cfg.get("lr",                DEFAULTS["lr"])
MIN_LR_RATIO      = cfg.get("min_lr_ratio",      DEFAULTS["min_lr_ratio"])
WARMUP_STEPS      = cfg.get("warmup_steps",      DEFAULTS["warmup_steps"])
LORA_RANK         = cfg.get("lora_rank",         DEFAULTS["lora_rank"])
LORA_ALPHA        = cfg.get("lora_alpha",        DEFAULTS["lora_alpha"])
LORA_TARGETS      = "all" if cfg.get("lora_targets", DEFAULTS["lora_targets"]) == "all" else "blocks"
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
TRAIN_SHIFT       = cfg.get("train_shift",       DEFAULTS["train_shift"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
# Lado de la preview en píxeles (área size², proporción de la muestra); 0 = tamaño de entrenamiento.
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

if PROJECT_NAME:
    CACHE_DIR  = f"./cached_data_anima_{PROJECT_NAME}"
    OUTPUT_DIR = f"./anima_lora_output_{PROJECT_NAME}"
else:
    CACHE_DIR  = cfg.get("cache_dir",  DEFAULTS["cache_dir"])
    OUTPUT_DIR = cfg.get("output_dir", DEFAULTS["output_dir"])

print(f"  Model ID / ID Modelo     : {MODEL_ID}")
print(f"  Precision / Precisión    : {PRECISION.upper()}")
print(f"  Project / Proyecto       : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra   : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché: {CACHE_DIR}")
print(f"  Output Dir / Salida      : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos      : {TOTAL_STEPS}")
print(f"  Learning Rate / LR       : {LR}")
print(f"  LoRA Rank/Alpha          : {LORA_RANK}/{LORA_ALPHA}")
print(f"  LoRA Targets             : {LORA_TARGETS}")
print(f"  Batch / Grad Accum       : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Timesteps / Shift        : {TIMESTEP_SAMPLING} / {TRAIN_SHIFT}")
print(f"  Preview Mode / Prompt    : Mode={PREVIEW_CAPTION_MODE} | Custom='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  Preview Every / Steps / CFG / Size: {PREVIEW_EVERY} / {PREVIEW_STEPS} / {PREVIEW_CFG} / {PREVIEW_SIZE or 'training'}")
print(f"  Seed Configured / Semilla: {SEED} ({'RANDOM' if SEED <= 0 else 'FIXED'})")

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
                    print("✓ Using HF Token / Usando token de HF")
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
    if (os.path.exists(os.path.join(local_path, "transformer", "config.json"))
            and os.path.exists(os.path.join(local_path, "vae", "config.json"))):
        print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
        return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")

    enable_hf_file_progress()

    from huggingface_hub import snapshot_download

    # El entrenamiento no usa el texto: ya está todo en el pre-cache.
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["text_encoder/*", "text_conditioner/*", "tokenizer/*", "t5_tokenizer/*"],
    )

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def sample_sigma(batch_size, device):
    # Desplazamiento fijo, el mismo 3.0 que usa el scheduler de Anima en inferencia.
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    sigma = TRAIN_SHIFT * u / (1 + (TRAIN_SHIFT - 1) * u)
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def predict(model, latents, sigma, embeds):
    """
    Cosmos recibe vídeo [B, C, 1, H, W], el tiempo es sigma y la salida es la velocidad
    (ruido - x0), la misma que usa el scheduler. El texto ya viene del adaptador: [B, 512, 1024].
    """
    H, W = latents.shape[-2:]
    padding_mask = latents.new_zeros(1, 1, H * 8, W * 8)
    out = model(hidden_states=latents.unsqueeze(2), timestep=sigma.to(latents.dtype),
                encoder_hidden_states=embeds, padding_mask=padding_mask, return_dict=False)[0]
    return out.squeeze(2)


def is_quantized(name):
    # Solo atención y MLP de los bloques. Cuantizar también la modulación adaLN, el timestep y la
    # entrada/salida degrada mucho las imágenes de un modelo de 2B (colores lavados, sin detalle).
    return name.startswith("transformer_blocks.") and ".norm" not in name


def load_transformer():
    transformer = CosmosTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    if PRECISION == "nf4":
        quantized = 0
        for name, module in list(transformer.named_modules()):
            if isinstance(module, torch.nn.Linear) and is_quantized(name):
                layer = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None,
                                   quant_type="nf4", compute_dtype=torch.bfloat16)
                # bitsandbytes cuantiza al mover a la GPU.
                layer.weight = Params4bit(module.weight.data.float(), requires_grad=False, quant_type="nf4")
                if module.bias is not None:
                    layer.bias = torch.nn.Parameter(module.bias.data, requires_grad=False)
                parent, _, child = name.rpartition(".")
                setattr(transformer.get_submodule(parent), child, layer)
                quantized += 1
        print(f"[OK] NF4 layers / Capas NF4: {quantized} (attention + MLP / atención + MLP)")
    return transformer


def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab Anima",
        "ss_sd_model_name": "Anima-Base-v1.0",
        "ss_base_model_version": "Anima",
        "ss_network_module": "peft.LoraModel",
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
        "ss_seed": SEED,
        "ss_mixed_precision": "bf16",
    }

    trigger = TRIGGER_WORD.strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or trigger or "anima_lora"

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


# Nombres de diffusers -> nombres nativos de Cosmos/Anima (los del fichero de ComfyUI): la tabla
# TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0 de diffusers al revés. ComfyUI no traduce los de diffusers
# para Anima, así que el LoRA se exporta con los suyos.
NATIVE_PREFIXES = (
    ("transformer_blocks.", "blocks."),
    ("time_embed.t_embedder.", "t_embedder.1."),
    ("patch_embed.proj", "x_embedder.proj.1"),
    ("norm_out.linear_1", "final_layer.adaln_modulation.1"),
    ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
    ("proj_out", "final_layer.linear"),
)
NATIVE_PARTS = (
    (".norm1.linear_1", ".adaln_modulation_self_attn.1"),
    (".norm1.linear_2", ".adaln_modulation_self_attn.2"),
    (".norm2.linear_1", ".adaln_modulation_cross_attn.1"),
    (".norm2.linear_2", ".adaln_modulation_cross_attn.2"),
    (".norm3.linear_1", ".adaln_modulation_mlp.1"),
    (".norm3.linear_2", ".adaln_modulation_mlp.2"),
    (".attn1.", ".self_attn."),
    (".attn2.", ".cross_attn."),
    (".to_q", ".q_proj"),
    (".to_k", ".k_proj"),
    (".to_v", ".v_proj"),
    (".to_out.0", ".output_proj"),
    (".ff.net.0.proj", ".mlp.layer1"),
    (".ff.net.2", ".mlp.layer2"),
)


def native_name(name):
    for old, new in NATIVE_PREFIXES:
        if name.startswith(old):
            name = new + name[len(old):]
            break
    for old, new in NATIVE_PARTS:
        name = name.replace(old, new)
    return name


def _export_lora(model, path, step):
    # Claves diffusion_model.<capa nativa>.lora_A/lora_B, que ComfyUI carga directamente. Se incluye
    # alpha por capa: sin él, ComfyUI aplicaría el LoRA con escala 1 en lugar de alpha/rank.
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        layer, _, rest = k.replace("base_model.model.", "").partition(".lora_")
        k = f"diffusion_model.{native_name(layer)}.lora_{rest}"
        clean[k] = v.to(torch.bfloat16).cpu().contiguous()
        if k.endswith(".lora_A.weight"):
            clean[k[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(LORA_ALPHA))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def denoise(model, scheduler, embed, neg, H, W, generator):
    # Mismo muestreo que AnimaModularPipeline: sigmas lineales con el shift del scheduler y CFG como
    # ComfyUI (1 = sin guía): pred = neg + cfg * (pos - neg), cond y negativo en la misma pasada.
    device = "cuda"
    latents = torch.randn((1, 16, H, W), generator=generator, device=device, dtype=torch.float32)

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    scheduler.set_begin_index(0)

    use_cfg = PREVIEW_CFG > 1.0
    texts = torch.cat([embed] + ([neg] if use_cfg else [])).to(device)

    for t in scheduler.timesteps:
        sigma = (t / 1000).expand(len(texts))
        out = predict(model, latents.to(torch.bfloat16).expand(len(texts), -1, -1, -1), sigma, texts).float()
        pred = out[1:] + PREVIEW_CFG * (out[:1] - out[1:]) if use_cfg else out
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    return latents


def run_preview(model, scheduler, embed, neg, size, step):
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print(f"  ↳ Preview Seed used / Semilla utilizada: {actual_seed}")

    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(model, scheduler, embed, neg, H, W, g)

            vae = VaeHolder.get().to("cuda")
            mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
            lat = (latents.unsqueeze(2) * std + mean).to(vae.dtype)
            try:
                img = vae.decode(lat, return_dict=False)[0][:, :, 0]
            except torch.OutOfMemoryError:
                # Con 8 GB y el entrenamiento cargado puede no caber entera: por mosaicos usa mucha menos
                # VRAM, a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                print("  ↳ Low VRAM: tiled VAE decode for previews / Poca VRAM: decodificación por mosaicos en las previews")
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(lat, return_dict=False)[0][:, :, 0]
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
    """
    El trainer no tiene text encoder (es la razón de ser del pre-cache): el prompt manual
    lo codifica el Pre-Cache, que al relanzarlo salta las imágenes ya cacheadas.
    Aquí solo se avisa si falta o si el de los ajustes ya no es el codificado.
    """
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
    Ajustes en caliente: si train_settings_anima.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SIZE, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED

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
    SEED                  = fresh("seed",                  int,   SEED)  # solo previews
    return changes


def is_target(name, module):
    if not isinstance(module, (torch.nn.Linear, Linear4bit)):
        return False
    if LORA_TARGETS == "all":
        return True
    # "blocks": atención (propia y cruzada con el texto) + MLP de los 28 bloques.
    return name.startswith("transformer_blocks.") and ".norm" not in name


def train_anima():
    global LORA_RANK, LORA_ALPHA  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    print(f"Loading Anima Transformer ({PRECISION.upper()})... / Cargando Transformer de Anima ({PRECISION.upper()})...")
    t0 = time.time()
    transformer = load_transformer()
    transformer.to("cuda")
    free_vram()
    print(f"Transformer 2B loaded in / cargado en {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    transformer.enable_gradient_checkpointing()

    # "blocks" deja intactas la modulación adaLN, el timestep y las capas de entrada/salida:
    # no hacen falta para una cara o un estilo y así el LoRA combina mejor con otros.
    target_modules = [name for name, m in transformer.named_modules() if is_target(name, m)]
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)} ({LORA_TARGETS})")

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
            print(f"\n[!] Resuming with the checkpoint's LoRA: {saved[0]} layers, rank {saved[1]}, alpha {saved[2]} "
                  f"(settings ask for {len(target_modules)} layers, rank {LORA_RANK}, alpha {LORA_ALPHA}; they apply to new trainings).")
            print(f"[!] Se reanuda con el LoRA del checkpoint: {saved[0]} capas, rank {saved[1]}, alpha {saved[2]} "
                  f"(los ajustes piden {len(target_modules)} capas, rank {LORA_RANK}, alpha {LORA_ALPHA}; se aplican a entrenamientos nuevos).")
        target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=target_modules, use_dora=False, init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)
    # LoRA y estados del optimizador en FP32: en 8/16 bits se pierde el detalle fino (caras).
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
        ckpt = os.path.join(OUTPUT_DIR, f"Anima_LoRA_step_{current_s}.safetensors")
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

    def load_tensor(path):
        t = torch.load(path, weights_only=True).to(torch.bfloat16)
        return t.pin_memory() if pin else t

    cache_data, buckets = {}, defaultdict(list)
    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = {"lat": load_tensor(f"{CACHE_DIR}/{f}"), "emb": load_tensor(f"{CACHE_DIR}/{nombre}_embed.pt")}
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)

    custom_path = f"{CACHE_DIR}/_custom_embed.pt"
    if os.path.exists(custom_path):
        cache_data["_custom"] = {"emb": load_tensor(custom_path)}
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(custom_path) if "_custom" in cache_data else None]

    neg = load_tensor(f"{CACHE_DIR}/_neg_embed.pt")

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
        # while y no range(): TOTAL_STEPS puede cambiar en caliente, en ambos sentidos.
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
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True).float()
            embeds = torch.cat([cache_data[n]["emb"] for n in names]).to("cuda", non_blocking=True)

            sigma = sample_sigma(len(names), "cuda")
            noise = torch.randn_like(latents)
            t_exp = sigma.view(-1, 1, 1, 1)

            noisy = ((1 - t_exp) * latents + t_exp * noise).to(torch.bfloat16)
            # Cosmos predice la velocidad (ruido - x0), al revés que Z-Image.
            target = noise - latents

            pred = predict(model, noisy, sigma, embeds)

            loss = F.mse_loss(pred.float(), target) / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            # La norma solo existe al aplicar el optimizador: entre medias se muestra la última.
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
                # El servidor codifica en CPU los prompts nuevos: aquí solo se relee el embedding si cambió.
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = {"emb": load_tensor(custom_path)}
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                # El prompt manual no tiene imagen: la preview usa el tamaño de la primera del dataset.
                ref = cache_data[p_name if p_name != "_custom" else all_preview_names[0]]["lat"]
                print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")
                H, W = ref.shape[2], ref.shape[3]
                if PREVIEW_SIZE > 0:
                    # Latente 8x y parches 2x2: lados múltiplos de 2 en el latente (16 px).
                    s = PREVIEW_SIZE / 8 / math.sqrt(H * W)
                    H, W = max(2, round(H * s / 2) * 2), max(2, round(W * s / 2) * 2)
                run_preview(model, scheduler, cache_data[p_name]["emb"], neg, (H, W), step)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    final = os.path.join(OUTPUT_DIR, "Anima_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    train_anima()
