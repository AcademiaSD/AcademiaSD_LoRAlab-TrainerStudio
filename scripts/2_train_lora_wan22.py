# -*- coding: utf-8 -*-
"""
2_train_lora_wan22.py — Entrenamiento LoRA para Wan 2.2 T2V-A14B (Transformers NF4)
LoRA training for Wan 2.2 T2V-A14B (NF4 Transformers)

Wan 2.2 A14B reparte los pasos entre dos expertos de 14B: el de ruido alto (sigma >= 0.875:
composición y movimiento) y el de ruido bajo (el resto: detalle, caras, estilo). Train Expert
elige cuál se entrena:

    low   solo el de ruido bajo. Basta para personajes, objetos y estilos (recomendado).
    high  solo el de ruido alto: para movimientos, coreografías o encuadres nuevos.
    both  los dos en el mismo entrenamiento, cada uno con su LoRA. Cada grupo de Grad Accum
          sale de un experto según la parte de los niveles de ruido que le toca; el otro
          espera en la RAM (~7.6 GB) y se intercambian en la GPU (~4 s).

Cada experto exporta su propio LoRA (_HIGH / _LOW). En ComfyUI se carga cada uno en su modelo.

Lee configuración desde train_settings_wan22.json si existe.
Reads configuration from train_settings_wan22.json if present.
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

# Antes de importar torch. Imágenes y clips de tamaños distintos fragmentan la VRAM: con 33 fotogramas
# a 480p quedaban reservados 16.3 GB para un pico real de 14.3 GB. Los segmentos ampliables lo evitan
# (también en Windows).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file, load
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import training_metrics
import wan22_model as wan

# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULTS = {
    "model_id": wan.MODEL_DEFAULT,
    "cache_dir": "./Cache_Data/cached_data_wan22",
    "output_dir": "./Outputs/wan22_lora_output",
    "train_expert": "low",
    "total_steps": 1500,
    "batch_size": 1,
    "grad_accum_steps": 1,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 16,
    "lora_alpha": 16,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    "use_lokr": False,
    "lokr_factor": 16,
    "lora_targets": "blocks",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "seed": 42,
    "timestep_sampling": "shift",
    "train_shift": 3.0,
    # Guarda las entradas de cada bloque (las que guarda el gradient checkpointing) en la RAM en
    # vez de en la VRAM. Con vídeo es la diferencia entre caber o no en 16 GB: 33 fotogramas a
    # 480p son ~5.8 GB de activaciones. Con imágenes apenas importa.
    "activation_offload": True,
    "preview_every": 0,
    "preview_steps": 30,
    "preview_cfg": 3.0,
    "preview_cfg_high": 4.0,
    "preview_frames": 1,
    "preview_size": 0,
    "preview_width": 0,
    "preview_height": 0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = os.environ.get("LORALAB_TRAIN_CONFIG") or "settings/train_settings_wan22.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID          = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
TRAIN_EXPERT      = cfg.get("train_expert", DEFAULTS["train_expert"])
if TRAIN_EXPERT not in ("low", "high", "both"):
    TRAIN_EXPERT = DEFAULTS["train_expert"]
TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = max(1, int(cfg.get("grad_accum_steps", DEFAULTS["grad_accum_steps"])))
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
TRAIN_SHIFT       = float(cfg.get("train_shift", DEFAULTS["train_shift"]))
ACTIVATION_OFFLOAD = bool(cfg.get("activation_offload", DEFAULTS["activation_offload"]))
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_CFG_HIGH  = cfg.get("preview_cfg_high",  DEFAULTS["preview_cfg_high"])
PREVIEW_FRAMES    = wan.valid_frames(cfg.get("preview_frames", DEFAULTS["preview_frames"]))
# Lado de la preview en píxeles (área size², proporción de la muestra); 0 = tamaño de entrenamiento.
PREVIEW_SIZE      = cfg.get("preview_size",      DEFAULTS["preview_size"])
PREVIEW_WIDTH     = int(cfg.get("preview_width",  DEFAULTS["preview_width"]))
PREVIEW_HEIGHT    = int(cfg.get("preview_height", DEFAULTS["preview_height"]))
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_wan22_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(f"./wan22_lora_output_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("output_dir"), DEFAULTS["output_dir"]))

EXPERT_LABEL = {"high": "HIGH", "low": "LOW"}

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Output Dir'):<22}: {OUTPUT_DIR}")
print(f"  {t('Train Expert'):<22}: {TRAIN_EXPERT}")
print(f"  {t('Total Steps'):<22}: {TOTAL_STEPS}")
print(f"  {t('Learning Rate'):<22}: {LR}")
print(f"  {t('LR Scheduler'):<22}: {lora_options.lr_scheduler_label(LR_SCHEDULER, WARMUP_STEPS, LR_CYCLES)}")
print(f"  {'LoRA Rank/Alpha':<22}: {LORA_RANK}/{LORA_ALPHA}")
print(f"  {'rsLoRA / LoRA+':<22}: {'on' if USE_RSLORA else 'off'} / {f'x{LORAPLUS_RATIO:g}' if USE_LORAPLUS else 'off'}")
print(f"  {'LoKr':<22}: {f'factor {LOKR_FACTOR}' if USE_LOKR else 'off'}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  {'Timesteps / Shift':<22}: {TIMESTEP_SAMPLING} / {TRAIN_SHIFT}")
print(f"  {t('Activation Offload'):<22}: {t('ON') if ACTIVATION_OFFLOAD else t('OFF')}")
print(f"  {t('Preview'):<22}: {t('Mode')}={PREVIEW_CAPTION_MODE} | Prompt='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  {t('Preview Every'):<22}: {PREVIEW_EVERY} | {t('Preview Steps')} {PREVIEW_STEPS} | CFG {PREVIEW_CFG_HIGH}/{PREVIEW_CFG} | "
      f"{t('Frames')} {PREVIEW_FRAMES} | {t('Preview Size')} {PREVIEW_SIZE or t('Training')}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")
EXPERTS_FILE = os.path.join(RESUME_DIR, "experts.json")

if SEED > 0:
    torch.manual_seed(SEED)
    random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def sigma_range(expert):
    return (wan.BOUNDARY, 1.0) if expert == "high" else (0.0, wan.BOUNDARY)


def shifted(u):
    return TRAIN_SHIFT * u / (1 + (TRAIN_SHIFT - 1) * u)


def high_share():
    """Parte de los niveles de ruido que toca al experto de ruido alto con la distribución de entrenamiento."""
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(20000, generator=torch.Generator().manual_seed(0)))
        return float((shifted(u) >= wan.BOUNDARY).float().mean())
    # shift(u) >= b  <=>  u >= b / (shift - (shift - 1) b)
    return 1.0 - wan.BOUNDARY / (TRAIN_SHIFT - (TRAIN_SHIFT - 1) * wan.BOUNDARY)


def sample_sigma(batch_size, expert, device):
    # La misma distribución desplazada que en las demás trainers, recortada al tramo del experto:
    # se sortea hasta que cae dentro (el de ruido alto se queda con ~30% de los sorteos con shift 3).
    lo, hi = sigma_range(expert)
    out = torch.empty(0)
    while out.numel() < batch_size:
        if TIMESTEP_SAMPLING == "logit_normal":
            u = torch.sigmoid(torch.randn(batch_size * 8))
        else:
            u = torch.rand(batch_size * 8)
        s = shifted(u)
        out = torch.cat([out, s[(s >= lo) & (s < hi)]])
    return out[:batch_size].clamp(1e-4, 1.0 - 1e-4).to(device)


def pad_text(emb):
    """[L, 4096] -> [512, 4096] con ceros detrás, como hace WanPipeline."""
    return F.pad(emb, (0, 0, 0, wan.TEXT_LEN - emb.shape[0]))


def predict(model, latents, sigma, embeds):
    """
    Velocidad (ruido - x0) para latentes [B, 16, F, H, W]; el tiempo va en 0..1000 como sigma * 1000.
    En autocast BF16 PEFT no copia la entrada de cada capa a FP32 para el LoRA (los pesos del LoRA y
    del optimizador siguen en FP32): con vídeo son varios GB menos en cada bloque que se recalcula.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(hidden_states=latents, timestep=sigma * 1000, encoder_hidden_states=embeds, return_dict=False)[0]


def build_lora_metadata(step, expert):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab Wan 2.2",
        "ss_sd_model_name": f"Wan2.2-T2V-A14B ({'high noise' if expert == 'high' else 'low noise'})",
        "ss_base_model_version": "wan_2.2_t2v_a14b",
        "wan_expert": "high_noise" if expert == "high" else "low_noise",
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
    meta["ss_output_name"] = PROJECT_NAME or trigger or "wan22_lora"

    # La resolución la fija el pre-caché del proyecto.
    pc_json = os.path.join(CACHE_DIR, f"pre_cache_settings_{PROJECT_NAME}.json")
    if os.path.exists(pc_json):
        with open(pc_json, "r", encoding="utf-8") as f:
            pc = json.load(f)
        if pc.get("target_area"):
            side = int(round(float(pc["target_area"]) ** 0.5))
            meta["ss_resolution"] = f"({side},{side})"
        if pc.get("num_frames"):
            meta["max_frames"] = pc["num_frames"]

    # safetensors exige que todos los valores sean str.
    return {k: str(v) for k, v in meta.items()}


def export_alpha():
    return lora_options.export_alpha(LORA_ALPHA, LORA_RANK, USE_RSLORA)


def _export_lora(model, path, step, expert):
    # Nombres del repo original de Wan con el prefijo diffusion_model. (ComfyUI, Wan2GP y diffusers
    # los cargan). Se incluye alpha por capa: sin él, ComfyUI aplicaría el LoRA con escala 1.
    if USE_LOKR:
        save_file(lora_options.lokr_export(model, wan.comfy_key, export_alpha()),
                  path, metadata=build_lora_metadata(step, expert))
        return
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        module, _, rest = k.replace("base_model.model.", "").partition(".lora_")
        key = wan.comfy_key(module)
        clean[f"{key}.lora_{rest}"] = v.to(torch.bfloat16).cpu().contiguous()
        if rest == "A.weight":
            clean[key + ".alpha"] = torch.tensor(float(export_alpha()))
    save_file(clean, path, metadata=build_lora_metadata(step, expert))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            cls.vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
        return cls.vae


class Experts:
    """
    Los Transformers cargados, por experto ("high" / "low"). Solo uno está en la GPU a la vez:
    use() mueve el que hace falta y devuelve el otro a la RAM. Los pesos NF4 y los del LoRA se
    mueven en su sitio (los mismos Parameter), así que el optimizador no se entera.
    """

    def __init__(self):
        self.models = {}
        self.on_gpu = None

    def use(self, expert):
        if self.on_gpu != expert:
            if self.on_gpu is not None:
                self.models[self.on_gpu].to("cpu")
                free_vram()
            self.models[expert].to("cuda")
            self.on_gpu = expert
        return self.models[expert]

    def release(self):
        """Deja la GPU sin experto (para el VAE de las previews de vídeo)."""
        if self.on_gpu is not None:
            self.models[self.on_gpu].to("cpu")
            self.on_gpu = None
            free_vram()


def denoise(experts, scheduler, embed, neg, shape, generator):
    """
    Muestreo de WanPipeline (UniPC, flow_shift del scheduler): con t >= 875 el experto de ruido alto
    (CFG high), el resto el de ruido bajo. Sin el otro experto convertido, el entrenado hace todos
    los pasos. Cond y negativo van en la misma pasada.
    """
    device = "cuda"
    latents = torch.randn((1, wan.LATENT_CHANNELS, *shape), generator=generator, device=device, dtype=torch.float32)
    scheduler.set_timesteps(PREVIEW_STEPS, device=device)
    boundary = wan.BOUNDARY * scheduler.config.num_train_timesteps
    texts = torch.stack([pad_text(embed), pad_text(neg)]).to(device)

    for t_ in scheduler.timesteps:
        expert = "high" if t_ >= boundary else "low"
        if expert not in experts.models:
            expert = next(iter(experts.models))
        model = experts.use(expert)
        cfg_scale = PREVIEW_CFG_HIGH if t_ >= boundary else PREVIEW_CFG
        use_cfg = cfg_scale > 1.0
        n = 2 if use_cfg else 1
        out = predict(model, latents.to(torch.bfloat16).expand(n, -1, -1, -1, -1),
                      t_.expand(n) / 1000, texts[:n]).float()
        pred = out[1:] + cfg_scale * (out[:1] - out[1:]) if use_cfg else out
        latents = scheduler.step(pred, t_, latents, return_dict=False)[0]

    return latents


def run_preview(model, experts, scheduler, embed, neg, shape, step, train_expert_now):
    was_training = model.training
    for m in experts.models.values():
        m.eval()

    actual_seed = random.randint(1, 2147483647) if SEED <= 0 else SEED
    print("  ↳ " + t("Preview seed used: {seed}", seed=actual_seed))

    try:
        with torch.no_grad():
            g = torch.Generator(device="cuda").manual_seed(actual_seed)
            latents = denoise(experts, scheduler, embed, neg, shape, g)
            # Con el experto en la GPU (~8 GB) el VAE no cabe al decodificar varios fotogramas,
            # ni por mosaicos. Una imagen sí cabe y se ahorra el ir y venir del experto.
            if shape[0] > 1:
                experts.release()

            vae = VaeHolder.get().to("cuda")
            mean, std = wan.latent_stats(vae, "cuda")
            lat = (latents * std + mean).to(vae.dtype)
            try:
                video = vae.decode(lat, return_dict=False)[0]
            except torch.OutOfMemoryError:
                # Con el entrenamiento cargado puede no caber entero: por mosaicos usa mucha menos VRAM,
                # a cambio de alguna marca leve en las uniones. Sigue así el resto del run.
                print("  ↳ " + t("Low VRAM: tiled VAE decode for previews"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                video = vae.decode(lat, return_dict=False)[0]
            frames = ((video.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy() * 255).astype("uint8")
            vae.to("cpu")

        if frames.shape[0] == 1:
            from PIL import Image
            out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
            Image.fromarray(frames[0]).save(out)
        else:
            out = wan.write_video(frames, os.path.join(OUTPUT_DIR, f"preview_step_{step}.mp4"))
        print("  ↳ " + t("Preview saved to: {path}", path=out))
    except torch.OutOfMemoryError:
        # Una preview que no cabe no debe parar el entrenamiento.
        print("  ↳ [!] " + t("Out of VRAM in the preview: skipped. Lower Preview Frames or the preview size."), flush=True)
        if VaeHolder.vae is not None:
            VaeHolder.vae.to("cpu")
    finally:
        # Vuelve a la GPU el experto con el que sigue el entrenamiento.
        experts.use(train_expert_now)
        for e, m in experts.models.items():
            if e in TRAINED:
                m.train(was_training)
        free_vram()


def check_custom_prompt(has_custom):
    """
    El trainer no tiene text encoder (es la razón de ser del pre-cache): el prompt manual
    lo codifica el Pre-Cache, que al relanzarlo salta las muestras ya cacheadas.
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
    Ajustes en caliente: si train_settings_wan22.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. Experto, rank, alpha, batch, resolución
    y carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_CFG_HIGH, PREVIEW_FRAMES, PREVIEW_SIZE, PREVIEW_WIDTH, PREVIEW_HEIGHT
    global PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT, SEED

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
    PREVIEW_CFG_HIGH      = fresh("preview_cfg_high",      float, PREVIEW_CFG_HIGH)
    PREVIEW_FRAMES        = fresh("preview_frames",        lambda v: wan.valid_frames(int(v)), PREVIEW_FRAMES)
    PREVIEW_SIZE          = fresh("preview_size",          int,   PREVIEW_SIZE)
    PREVIEW_WIDTH         = fresh("preview_width",         int,   PREVIEW_WIDTH)
    PREVIEW_HEIGHT        = fresh("preview_height",        int,   PREVIEW_HEIGHT)
    PREVIEW_CAPTION_MODE  = fresh("preview_caption_mode",  str,   PREVIEW_CAPTION_MODE)
    PREVIEW_CUSTOM_PROMPT = fresh("preview_custom_prompt", lambda v: str(v).strip(), PREVIEW_CUSTOM_PROMPT)
    SEED                  = fresh("seed",                  int,   SEED)  # solo previews
    return changes


# Expertos que se entrenan en este run: los del checkpoint al reanudar, si no los de Train Expert.
TRAINED = []


def read_resume_lora(expert_dir):
    """(capas, rank, alpha, extra) del LoRA o LoKr guardado en expert_dir, o None si no hay."""
    lora_weights = os.path.join(expert_dir, "adapter_model.safetensors")
    lokr_weights = os.path.join(expert_dir, lora_options.LOKR_WEIGHTS)
    if USE_LOKR and os.path.exists(lokr_weights):
        with safe_open(lokr_weights, framework="pt", device="cpu") as f:
            targets = sorted({k.rsplit(".", 1)[0] for k in f.keys()})
        with open(os.path.join(expert_dir, lora_options.LOKR_CONFIG), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        return targets, saved_cfg["r"], saved_cfg["lora_alpha"], saved_cfg
    if not USE_LOKR and os.path.exists(lora_weights):
        with safe_open(lora_weights, framework="pt", device="cpu") as f:
            targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1) for k in f.keys()})
        with open(os.path.join(expert_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        return targets, saved_cfg["r"], saved_cfg["lora_alpha"], saved_cfg
    return None


def train_wan22():
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR  # al reanudar se toman del checkpoint (el alpha se exporta con el LoRA)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR))
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    # Reanudar es continuar EL MISMO entrenamiento: expertos, método, capas, rank y alpha salen del
    # checkpoint, no de la GUI. Si se construyera con otros, el optimizador no encajaría.
    resuming = os.path.exists(STEP_FILE) and os.path.exists(EXPERTS_FILE)
    if resuming:
        with open(EXPERTS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        wanted = ["high", "low"] if TRAIN_EXPERT == "both" else [TRAIN_EXPERT]
        if saved["experts"] != wanted:
            print("\n[!] " + t("Resuming with the checkpoint's experts: {e} (settings ask for {e2}; they apply to new trainings).",
                              e="+".join(saved["experts"]), e2="+".join(wanted)))
        if saved["lokr"] != USE_LOKR:
            print("\n[!] " + t("Resuming with the checkpoint's method: {m} (settings ask for {m2}; it applies to new trainings).",
                              m="LoKr" if saved["lokr"] else "LoRA", m2="LoKr" if USE_LOKR else "LoRA"))
        TRAINED[:] = saved["experts"]
        USE_LOKR = saved["lokr"]
    else:
        TRAINED[:] = ["high", "low"] if TRAIN_EXPERT == "both" else [TRAIN_EXPERT]
    if USE_LOKR:
        USE_RSLORA = USE_LORAPLUS = False

    wan.ensure_base(MODEL_ID)
    for e in TRAINED:
        wan.ensure_expert(MODEL_ID, e)

    # En Windows el driver de NVIDIA no da "out of memory": cuando la VRAM se llena pasa a la RAM
    # compartida y cada paso tarda 30 veces más sin avisar (un clip de 33 fotogramas a 480p en 16 GB:
    # 544 s en vez de ~30). Con el asignador limitado a la VRAM libre, una muestra que no cabe da
    # OutOfMemoryError al momento y el bucle la aparta (ver más abajo).
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.5, (free - 512 * 1024 ** 2) / total))

    experts = Experts()
    lora_models = {}
    for e in TRAINED:
        print(t("Loading {name}...", name=f"Wan 2.2 Transformer ({'high' if e == 'high' else 'low'} noise, NF4)"))
        t0 = time.time()
        transformer = wan.load_expert(MODEL_ID, e)
        print(t("Transformer loaded in {s:.1f}s.", s=time.time() - t0), flush=True)

        # Checkpoint reentrante en dos mitades por bloque (ver wan.split_block_checkpoint): guarda la entrada
        # de cada mitad con save_for_backward, que es lo que save_on_cpu manda a la RAM con Activation
        # Offload. Necesita que la entrada de los bloques pida gradiente.
        transformer.enable_gradient_checkpointing(gradient_checkpointing_func=wan.split_block_checkpoint)
        transformer.patch_embedding.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

        # "blocks" deja intactas las capas de entrada/salida, del timestep y del texto:
        # no hacen falta para una cara o un estilo y así el LoRA combina mejor con otros.
        target_modules = [name for name, m in transformer.named_modules() if wan.is_lora_target(name, m, LORA_TARGETS)]

        expert_dir = os.path.join(RESUME_DIR, e)
        saved = read_resume_lora(expert_dir) if resuming else None
        if saved:
            s_targets, s_rank, s_alpha, s_cfg = saved
            if (len(s_targets), s_rank, s_alpha) != (len(target_modules), LORA_RANK, LORA_ALPHA):
                print("\n[!] " + t("Resuming with the checkpoint's LoRA: {n} layers, rank {r}, alpha {a} "
                                  "(settings ask for {n2} layers, rank {r2}, alpha {a2}; they apply to new trainings).",
                                  n=len(s_targets), r=s_rank, a=s_alpha, n2=len(target_modules), r2=LORA_RANK, a2=LORA_ALPHA))
            target_modules, LORA_RANK, LORA_ALPHA = s_targets, s_rank, s_alpha
            if USE_LOKR:
                LOKR_FACTOR = s_cfg["lokr_factor"]
            else:
                # rsLoRA lo guarda PEFT en adapter_config.json; LoRA+ va en lora_options.json.
                opts_file = os.path.join(RESUME_DIR, "lora_options.json")
                opts = {}
                if os.path.exists(opts_file):
                    with open(opts_file, "r", encoding="utf-8") as f:
                        opts = json.load(f)
                saved_opts = (bool(s_cfg.get("use_rslora", False)), bool(opts.get("use_loraplus", False)),
                              float(opts.get("loraplus_ratio", LORAPLUS_RATIO)))
                if saved_opts[:2] != (USE_RSLORA, USE_LORAPLUS):
                    print("\n[!] " + t("Resuming with the checkpoint's options: rsLoRA {a}, LoRA+ {b} "
                                      "(settings ask for rsLoRA {a2}, LoRA+ {b2}; they apply to new trainings).",
                                      a=saved_opts[0], b=saved_opts[1], a2=USE_RSLORA, b2=USE_LORAPLUS))
                USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO = saved_opts
        print(t("Target LoRA layers: {n} ({kind})", n=len(target_modules), kind=LORA_TARGETS))

        if USE_LOKR:
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
        lora_models[e] = model
        experts.models[e] = model

    # Para las previews hace falta también el experto que no se entrena (sin LoRA). Solo se usa si
    # ya está convertido: convertirlo son otros ~57 GB de descarga.
    other = "high" if TRAINED == ["low"] else "low" if TRAINED == ["high"] else None
    preview_note = None
    if other and PREVIEW_EVERY > 0:
        if wan.expert_ready(MODEL_ID, other):
            print(t("Loading {name}...", name=f"Wan 2.2 Transformer ({other} noise, NF4, previews)"))
            experts.models[other] = wan.load_expert(MODEL_ID, other).eval()
        else:
            preview_note = t("The {other} noise expert is not converted: previews use the trained expert for every step "
                             "(rougher than the final result in ComfyUI).", other=other)
            print("[PREVIEW] " + preview_note)

    current = TRAINED[0]
    experts.use(current)
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    named = [(n, p) for e in TRAINED for n, p in lora_models[e].named_parameters() if p.requires_grad]
    trainable = [p for _, p in named]
    # LoRA+: lora_B en su propio grupo, con el LR multiplicado (lr_mult se aplica en cada paso).
    if USE_LORAPLUS:
        optimizer = torch.optim.AdamW(lora_options.loraplus_groups(named, LORAPLUS_RATIO), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        return LR * lora_options.lr_factor(LR_SCHEDULER, step, TOTAL_STEPS, WARMUP_STEPS, MIN_LR_RATIO, LR_CYCLES)

    start_step = 0
    if resuming and os.path.exists(OPT_FILE):
        print("=" * 65)
        print(t("Checkpoint detected! Restoring state..."))
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            for e in TRAINED:
                d = os.path.join(RESUME_DIR, e)
                with open(os.path.join(d, lora_options.LOKR_WEIGHTS if USE_LOKR else "adapter_model.safetensors"), "rb") as f:
                    saved_sd = load(f.read())
                if USE_LOKR:
                    lora_models[e].load_lokr_state_dict(saved_sd)
                else:
                    set_peft_model_state_dict(lora_models[e], {k: v.float() for k, v in saved_sd.items()})
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
        for e in TRAINED:
            d = os.path.join(RESUME_DIR, e)
            os.makedirs(d, exist_ok=True)
            lora_models[e].save_pretrained(d)
            # Solo puede quedar uno: los pesos del otro método serían de un entrenamiento anterior.
            stale = os.path.join(d, "adapter_model.safetensors" if USE_LOKR else lora_options.LOKR_WEIGHTS)
            if os.path.exists(stale):
                os.remove(stale)
        with open(os.path.join(RESUME_DIR, "lora_options.json"), "w", encoding="utf-8") as f:
            json.dump({"use_loraplus": USE_LORAPLUS, "loraplus_ratio": LORAPLUS_RATIO}, f)
        with open(EXPERTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"experts": TRAINED, "lokr": USE_LOKR}, f)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        for e in TRAINED:
            ckpt = os.path.join(OUTPUT_DIR, f"Wan22_{EXPERT_LABEL[e]}_LoRA_step_{current_s}.safetensors")
            _export_lora(lora_models[e], ckpt, current_s, e)
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

    for e in TRAINED:
        lora_models[e].train()
    optimizer.zero_grad(set_to_none=True)

    pin = torch.cuda.is_available()

    def load_tensor(path):
        t_ = torch.load(path, weights_only=True).to(torch.bfloat16)
        return t_.pin_memory() if pin else t_

    cache_data, buckets = {}, defaultdict(list)
    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        if not os.path.exists(f"{CACHE_DIR}/{nombre}_embed.pt"):
            continue
        cache_data[nombre] = {"lat": load_tensor(f"{CACHE_DIR}/{f}"), "emb": load_tensor(f"{CACHE_DIR}/{nombre}_embed.pt")}
        buckets[tuple(cache_data[nombre]["lat"].shape[2:])].append(nombre)

    custom_path = f"{CACHE_DIR}/_custom_embed.pt"
    if os.path.exists(custom_path):
        cache_data["_custom"] = {"emb": load_tensor(custom_path)}
    check_custom_prompt("_custom" in cache_data)
    custom_mtime = [os.path.getmtime(custom_path) if "_custom" in cache_data else None]

    neg = load_tensor(f"{CACHE_DIR}/_neg_embed.pt")

    all_preview_names = sorted(k for k in cache_data.keys() if not k.startswith("_"))
    n_video = sum(1 for k in all_preview_names if cache_data[k]["lat"].shape[2] > 1)

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

    share = high_share()
    if len(TRAINED) == 2:
        print(t("Both experts: {p:.0f}% of the steps go to the high noise one (shift {s:g}).", p=share * 100, s=TRAIN_SHIFT))

    running_loss, t_step_avg, grad_norm = 0.0, 0.0, 0.0
    print("\n" + t("STARTING TRAINING! {n} {kind} in {b} buckets.", n=len(all_preview_names),
                   kind=f"{t('samples')} ({len(all_preview_names) - n_video} {t('images')}, {n_video} {t('videos')})", b=len(buckets)))

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

            # Cada grupo de acumulación es de un solo experto: el optimizador se aplica sobre gradientes
            # de un mismo modelo y el intercambio de GPU se paga una vez por grupo.
            if (step - 1) % GRAD_ACCUM_STEPS == 0 and len(TRAINED) == 2:
                current = "high" if random.random() < share else "low"
            model = experts.use(current)

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n]["lat"] for n in names]).to("cuda", non_blocking=True).float()
            embeds = torch.stack([pad_text(cache_data[n]["emb"]) for n in names]).to("cuda", non_blocking=True)

            sigma = sample_sigma(len(names), current, "cuda")
            noise = torch.randn_like(latents)
            t_exp = sigma.view(-1, 1, 1, 1, 1)

            noisy = ((1 - t_exp) * latents + t_exp * noise).to(torch.bfloat16)
            target = noise - latents

            try:
                if ACTIVATION_OFFLOAD:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        pred = predict(model, noisy, sigma, embeds)
                else:
                    pred = predict(model, noisy, sigma, embeds)

                loss = F.mse_loss(pred.float(), target) / GRAD_ACCUM_STEPS
                loss.backward()
            except torch.OutOfMemoryError:
                # El tamaño de este bucket no cabe: se aparta y el paso se repite con otro. Se tiran los
                # gradientes que llevaba acumulados el grupo.
                pred = loss = None
                optimizer.zero_grad(set_to_none=True)
                free_vram()
                f_, h_, w_ = size
                print("\n[!] " + t("Out of VRAM with {n} samples of {f} frames at {w}x{h}: they are left out of this training. "
                                  "Lower Max Frames or the resolution in Pre-Cache, or turn on Activation Offload.",
                                  n=len(buckets[size]), f=(f_ - 1) * 4 + 1, w=w_ * wan.SPATIAL, h=h_ * wan.SPATIAL))
                del buckets[size]
                if not buckets:
                    print("[!] " + t("No sample fits in VRAM."))
                    sys.exit(1)
                step -= 1
                continue
            step_loss = loss.item() * GRAD_ACCUM_STEPS
            running_loss += step_loss
            del pred, loss, noisy, target, latents, noise

            # La norma solo existe al aplicar el optimizador: entre medias se muestra la última.
            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in trainable if p.grad is not None], MAX_GRAD_NORM).item()
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
                f"{STEP_WORD} {step:4d}/{TOTAL_STEPS} [{barra}] {pct*100:5.1f}% | {current} | "
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
                if os.path.exists(custom_path) and os.path.getmtime(custom_path) != custom_mtime[0]:
                    custom_mtime[0] = os.path.getmtime(custom_path)
                    cache_data["_custom"] = {"emb": load_tensor(custom_path)}
                    check_custom_prompt(True)
                p_name = get_preview_sample(step)
                # El prompt manual no tiene muestra: la preview usa el tamaño de la primera del dataset.
                ref = cache_data[p_name if p_name != "_custom" else all_preview_names[0]]["lat"]
                print(f"\n  [Preview] {t('Mode')}: {PREVIEW_CAPTION_MODE} | {t('Sample')}: {p_name} | {t('Frames')}: {PREVIEW_FRAMES}")
                if preview_note:
                    print("  ↳ " + preview_note)
                H, W = ref.shape[3], ref.shape[4]
                if PREVIEW_WIDTH > 0 and PREVIEW_HEIGHT > 0:
                    # Lados explícitos en píxeles: latente 8x y parches 2x2 (16 px).
                    H = max(2, round(PREVIEW_HEIGHT / 8 / 2) * 2)
                    W = max(2, round(PREVIEW_WIDTH / 8 / 2) * 2)
                elif PREVIEW_SIZE > 0:
                    s = PREVIEW_SIZE / 8 / math.sqrt(H * W)
                    H, W = max(2, round(H * s / 2) * 2), max(2, round(W * s / 2) * 2)
                shape = ((PREVIEW_FRAMES - 1) // 4 + 1, H, W)
                run_preview(model, experts, scheduler, cache_data[p_name]["emb"], neg, shape, step, current)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\n" + t("Training completed!"))
    for e in TRAINED:
        final = os.path.join(OUTPUT_DIR, f"Wan22_{EXPERT_LABEL[e]}_FINAL_LoRA.safetensors")
        _export_lora(lora_models[e], final, min(last_step_executed, TOTAL_STEPS), e)
        print("✓ " + t("Final LoRA saved to: {path}", path=final))


if __name__ == "__main__":
    train_wan22()
