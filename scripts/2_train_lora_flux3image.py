# -*- coding: utf-8 -*-
"""
2_train_lora_flux3image.py — Entrenamiento LoRA para FLUX.2 [klein] 9B (Transformer NF4)
LoRA training for FLUX.2 [klein] 9B (NF4 Transformer)

Lee configuración desde train_settings_flux3image.json si existe.
Reads configuration from train_settings_flux3image.json if present.
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "FLUX.2-Klein-9B_NF4",
    "cache_dir": "./cached_data_flux3image",
    "output_dir": "./flux3image_lora_output",
    "total_steps": 1000,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 3e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 16,
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
    "use_turbo": False,
    "turbo_lora_strength": 1.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "settings/train_settings_flux3image.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID          = cfg.get("model_id",          DEFAULTS["model_id"])
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
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

# Previews rápidas: LoRA extraído de la resta FLUX.2 [klein] 9B (destilado) - Base 9B, de kalle07
# (FLUX Non-Commercial License). Destila pasos y CFG: 4 pasos sin CFG.
USE_TURBO           = cfg.get("use_turbo",           DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH = cfg.get("turbo_lora_strength", DEFAULTS["turbo_lora_strength"])
TURBO_LORA_REPO     = "kalle07/FLUX.2-klein-9B-turbo-lora-set"
TURBO_LORA_FILE     = "Flux_Klein_9b_Turbo_lora_rank_64_bf16_standard.safetensors"

TRIGGER_WORD     = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

if PROJECT_NAME:
    CACHE_DIR  = f"./cached_data_flux3image_{PROJECT_NAME}"
    OUTPUT_DIR = f"./flux3image_lora_output_{PROJECT_NAME}"
else:
    CACHE_DIR  = cfg.get("cache_dir",  DEFAULTS["cache_dir"])
    OUTPUT_DIR = cfg.get("output_dir", DEFAULTS["output_dir"])

print(f"  Model ID / ID Modelo     : {MODEL_ID}")
print(f"  Project / Proyecto       : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra   : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché: {CACHE_DIR}")
print(f"  Output Dir / Salida      : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos      : {TOTAL_STEPS}")
print(f"  Learning Rate / LR       : {LR}")
print(f"  LoRA Rank/Alpha          : {LORA_RANK}/{LORA_ALPHA}")
print(f"  LoRA Targets             : {LORA_TARGETS}")
print(f"  Batch / Grad Accum       : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Preview Mode / Prompt    : Mode={PREVIEW_CAPTION_MODE} | Custom='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  Preview Every / Steps / CFG / Size: {PREVIEW_EVERY} / {PREVIEW_STEPS} / {PREVIEW_CFG} / {PREVIEW_SIZE or 'training'}")
print(f"  Seed Configured / Semilla: {SEED} ({'RANDOM' if SEED <= 0 else 'FIXED'})")
print(f"  Turbo LoRA Previews      : {'ON (Strength=' + str(TURBO_LORA_STRENGTH) + ')' if USE_TURBO else 'OFF'}")

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
    if (os.path.exists(os.path.join(local_path, "transformer", "index.json"))
            and os.path.exists(os.path.join(local_path, "vae", "config.json"))):
        print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
        return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with pip install huggingface_hub")

    # El entrenamiento no usa el text encoder: ya está todo en el pre-cache.
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["text_encoder/*"],
    )

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
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
    # Con batch > 1 cada muestra lleva su propio antes; en la preview hay uno solo.
    ctrl = sample["ctrl"].to(device, non_blocking=True)
    ctrl_ids = Flux2KleinPipeline._prepare_image_ids([ctrl[:1]]).to(device)
    hidden = torch.cat([latents, pack_latents(ctrl).expand(B, -1, -1).to(latents.dtype)], dim=1)
    return hidden, torch.cat([img_ids, ctrl_ids.expand(B, -1, -1)], dim=1), txt_ids


def load_nf4_transformer(model_dir):
    """
    Construye el Transformer vacío desde transformer/config.json y lo rellena con la
    caché NF4 (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    """
    cache_dir = os.path.join(model_dir, "transformer")
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found in NF4 cache / No existe index.json en caché NF4: {cache_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config(cache_dir))

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

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=False, dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    transformer.load_state_dict(load_file(os.path.join(cache_dir, index["others"])), strict=False, assign=True)

    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(f"NF4 cache incomplete / Caché NF4 incompleta: {missing[:5]}")

    print(f"[OK] NF4 layers / Capas NF4: {len(index['quantized'])} | BF16 layers / Capas BF16: {len(index['unquantized'])}")
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
    meta["ss_output_name"] = PROJECT_NAME or trigger or "flux3image_lora"

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


def _export_lora(model, path, step):
    # Formato PEFT/diffusers que carga ComfyUI. Se incluye alpha por capa: sin él,
    # ComfyUI aplicaría el LoRA con escala 1 en lugar de alpha/rank.
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        k = "transformer." + k.replace("base_model.model.", "")
        clean[k] = v.to(torch.bfloat16).cpu().contiguous()
        if k.endswith(".lora_A.weight"):
            clean[k[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(LORA_ALPHA))
    save_file(clean, path, metadata=build_lora_metadata(step))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def denoise(model, scheduler, sample, H, W, generator, neg=None):
    device = "cuda"
    latents = pack_latents(torch.randn((1, 128, H, W), generator=generator, device=device, dtype=torch.bfloat16))

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=compute_empirical_mu(H * W, PREVIEW_STEPS))
    scheduler.set_begin_index(0)

    # En edición el negativo lleva la misma imagen de antes: solo cambia el texto, como en el pipeline.
    conds = [sample]
    use_cfg = neg is not None and PREVIEW_CFG > 1.0
    if use_cfg:
        conds.append({**sample, "emb": neg})

    for t in scheduler.timesteps:
        timestep = (t / 1000).expand(1).to(torch.bfloat16)
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
            print(f"  Downloading Turbo LoRA / Descargando Turbo LoRA (~350 MB): {TURBO_LORA_REPO}")
            path = hf_hub_download(TURBO_LORA_REPO, TURBO_LORA_FILE, local_dir=os.path.join(MODEL_ID, "LoRAs"))

        # Claves diffusion_model.<capa>.lora_down/lora_up (sin alpha: escala 1) y .diff de las normas q/k.
        cls.layers, cls.norms = {}, {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                name = k[len("diffusion_model."):]
                if name.endswith(".diff"):
                    cls.norms[turbo_targets(name[:-len(".diff")])[0]] = f.get_tensor(k).to(torch.bfloat16)
                    continue
                if not name.endswith(".lora_down.weight"):
                    continue
                name = name[:-len(".lora_down.weight")]
                down = f.get_tensor(k).to(torch.bfloat16).pin_memory()
                up = f.get_tensor(k.replace(".lora_down.", ".lora_up.")).to(torch.bfloat16)
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

    print(f"  🚀 Turbo LoRA: {len(hooks)} layers + {len(TurboHolder.norms)} norms / capas + normas (strength / fuerza {TURBO_LORA_STRENGTH})")
    return undo


def run_preview(model, scheduler, sample, neg, size, step):
    H, W = size
    was_training = model.training
    model.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print(f"  ↳ Preview Seed used / Semilla utilizada: {actual_seed}")

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
                print("  ↳ Low VRAM: tiled VAE decode for previews / Poca VRAM: decodificación por mosaicos en las previews")
                torch.cuda.empty_cache()
                vae.enable_tiling()
                img = vae.decode(lat, return_dict=False)[0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out)
        print(f"  ↳ Preview saved to / Preview guardada: {out}")
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
    Ajustes en caliente: si train_settings_flux3image.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Rank, alpha, batch, resolución y
    carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
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
    USE_TURBO             = fresh("use_turbo",             bool,  USE_TURBO)
    TURBO_LORA_STRENGTH   = fresh("turbo_lora_strength",   float, TURBO_LORA_STRENGTH)
    SEED                 = fresh("seed",                  int,   SEED)  # solo previews
    return changes


def train_flux3image():
    global LORA_RANK, LORA_ALPHA  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    print("Loading FLUX.2 Klein 9B Transformer (NF4)... / Cargando Transformer de FLUX.2 Klein 9B (NF4)...")
    t0 = time.time()
    transformer = load_nf4_transformer(MODEL_ID)
    transformer.to("cuda")
    free_vram()
    print(f"Transformer 9B loaded in / cargado en {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    transformer.enable_gradient_checkpointing()

    # "blocks": atención + MLP de los 8 bloques dobles y los 24 simples. La modulación compartida,
    # el timestep y las capas de entrada/salida se dejan intactas: no hacen falta para una cara
    # o un estilo, y tocarlas rompe la combinación con LoRAs destilados.
    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, Linear4bit))
                      and (LORA_TARGETS == "all" or name.startswith(("transformer_blocks.", "single_transformer_blocks.")))]
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
    lora_weights_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(lora_weights_path):
        print("=" * 65)
        print("¡Checkpoint detected! Restoring state... / ¡Checkpoint detectado! Restaurando estado...")
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(lora_weights_path, "rb") as f:
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
        ckpt = os.path.join(OUTPUT_DIR, f"Flux3Image_LoRA_step_{current_s}.safetensors")
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

    def load_sample(nombre):
        # lat: objetivo (en _custom no existe: solo aporta el tamaño). En edición además ctrl (latente del antes).
        sample = {}
        for key, suffix in (("lat", "latent"), ("emb", "embed"), ("ctrl", "ctrl")):
            path = f"{CACHE_DIR}/{nombre}_{suffix}.pt"
            if os.path.exists(path):
                t = torch.load(path, weights_only=True).to(torch.bfloat16)
                sample[key] = t.pin_memory() if pin else t
        return sample

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
    kind = "before/after pairs / pares antes/después" if edit_mode else "images / imágenes"
    print(f"\nSTARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(all_preview_names)} {kind} in {len(buckets)} buckets.")

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
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True)
            # Todos los prompts miden 512 tokens (relleno incluido): se apilan sin máscara.
            embeds = torch.cat([cache_data[n]["emb"] for n in names]).to("cuda", non_blocking=True)

            H, W = size
            latent_packed = pack_latents(latents)
            B = latent_packed.shape[0]

            sigma  = sample_sigma(B, H * W, "cuda")
            noise  = torch.randn_like(latent_packed)
            t_exp  = sigma.view(-1, 1, 1)

            noisy = ((1 - t_exp) * latent_packed + t_exp * noise).to(torch.bfloat16)
            # El transformer predice la velocidad (ruido - x0), como Flux.1.
            target = noise - latent_packed

            batch = {"emb": embeds}
            if edit_mode:
                batch["ctrl"] = torch.cat([cache_data[n]["ctrl"] for n in names])
            hidden, img_ids, txt_ids = model_inputs(batch, noisy, H, W, "cuda")
            pred = model(
                hidden_states=hidden,
                encoder_hidden_states=embeds,
                timestep=sigma.to(torch.bfloat16),
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0][:, :H * W]

            loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
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
                custom_path = f"{CACHE_DIR}/_custom_embed.pt"
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = load_sample("_custom")
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                sample = cache_data[p_name]
                # Tamaño de la preview: el del objetivo; en edición, el de la imagen de antes
                # (como en ComfyUI, la salida hereda el tamaño de la referencia). El prompt manual
                # sin imagen usa el tamaño de la primera del dataset.
                ref = sample.get("ctrl", sample.get("lat", cache_data[all_preview_names[0]]["lat"]))
                print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")
                H, W = ref.shape[2], ref.shape[3]
                if PREVIEW_SIZE > 0:
                    # Cada token es un parche de 16x16 px.
                    s = PREVIEW_SIZE / 16 / math.sqrt(H * W)
                    H, W = max(1, round(H * s)), max(1, round(W * s))
                run_preview(model, scheduler, sample, neg, (H, W), step)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    final = os.path.join(OUTPUT_DIR, "Flux3Image_FINAL_LoRA.safetensors")
    _export_lora(model, final, min(last_step_executed, TOTAL_STEPS))
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    train_flux3image()
