# -*- coding: utf-8 -*-
"""
2_train_lora_yue2.py — Entrenamiento LoRA para YuE2-3B (etapas NAR y AR separadas)
LoRA training for YuE2-3B (separate NAR and AR stages)

YuE2 es un Mixture-of-Transformers: cada bloque tiene DOS rutas completas (AR:
self_attn+mlp para los códigos semánticos; NAR: nar_self_attn+nar_mlp para el
flow matching del latente). Un LoRA que mezclase las dos pondría a entrenar
mecanismos para papeles distintos: este trainer entrena UNA etapa por run y
exporta un .safetensors POR ETAPA. Un cargador de LoRA solo inyecta lo que el
fichero contiene, así que los dos ficheros son independientes y combinables.

  stage = "nar"  ->  YuE2_NAR_LoRA_step_<N>.safetensors / YuE2_FINAL_NAR_LoRA.safetensors
                     letra y fraseo: flow matching del latente VAE (MSE de
                     velocidad), condicionado con el prefijo de texto + los
                     códigos reales del clip. En las pruebas reales NO clona
                     la voz. (Antes: YuE2_LoRA_step_<N> / YuE2_FINAL_LoRA.)
  stage = "ar"   ->  YuE2_AR_LoRA_step_<N>.safetensors / YuE2_FINAL_AR_LoRA.safetensors
                     voz, timbre y sonido (la etapa que clona la voz): CE
                     sobre los códigos codec. Requiere códigos POR CLIP: se muestrean UNA
                     vez con el modelo base (iguales parámetros de sampling que
                     el release) y quedan en <cache>/_codes_<clip>.pt. Si
                     añades clips, relanza el trainer: solo muestrea los que
                     falten.
  stage = "both" ->  NAR entera primero y AR después, cada una con total_steps
                     y checkpoints propios (resume_checkpoint[_ar]).

Protocolo del wheel yue2_infer-0.1.5, SIN trust_remote_code: el prefijo sale
del pre-caché (<name>_prompt.pt, ya lleva el plan ABC horneado), el estado NAR
entra igual que el CachedNAR oficial (pad START/END con x=0, time_embedder con
t=logit(t) shift θ, posiciones de latente locales clampadas) y la velocidad
diana es eps - x0 con x_t = (1-t)·x0 + t·eps (t ~ U(0,1) por defecto). El MSE
inicial esperable es ~2.2 (var(eps) + var(x0), latente sin normalizar,
std ~1.07); con un dataset pequeño baja hacia ~1.5-2.0, no a las cifras de
modelos con latente unitario.

Lee settings/train_settings_yue2.json; la GUI pinta el progreso con "Step N/Total".
"""
import os
import gc
import math
import time
import random
import json
import zlib
import signal
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"torch\.nn\.utils\.weight_norm is deprecated in favor of torch\.nn\.utils\.parametrizations\.weight_norm\.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`?use_return_dict`? is deprecated! Use `?return_dict`? instead!",
    category=Warning,
)
import wave

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file, load, safe_open
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import training_metrics

STEP_WORD = t("Step")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Los modelos viven en Models/ desde la migracion del layout.
# The models live in Models/ since the layout migration.
NF4_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_NF4", STUDIO_ROOT))
RAW_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_Raw", STUDIO_ROOT))
RAW_VAE_DIR = str(resolve_model_dir(None, "Models/YuE2-Vae", STUDIO_ROOT))
try:
    with open(os.path.join(STUDIO_ROOT, "settings", "train_settings_yue2.json"), "r", encoding="utf-8") as _f:
        _precision_hint = str(json.load(_f).get("precision", "bf16")).lower()
except (OSError, ValueError):
    _precision_hint = "bf16"
MODEL_DIR = NF4_DIR if _precision_hint == "nf4" else RAW_DIR
VAE_DIR = os.path.join(NF4_DIR, "vae") if _precision_hint == "nf4" else RAW_VAE_DIR
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, RAW_DIR)
sys.path.insert(0, NF4_DIR)
sys.path.insert(0, VAE_DIR)

# Protocolo del wheel vendorizado en scripts/yue2/ (Apache-2.0)
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer
from modeling_yue2 import sdpa  # el sdpa del modeling (native GQA), mismas matemáticas que el wheel

DEFAULTS = {
    "model_id": "Models/YuE2-3B_Raw",
    "cache_dir": "./Cache_Data/cached_data_yue2",
    "output_dir": "./Outputs/yue2_lora_output",
    "stage": "ar",               # ar | nar | both (ar: la etapa que clona la voz en las pruebas reales)
    "precision": "bf16",         # bf16 | nf4 (el NF4 se valida contra BF16 en Fase 6)
    "total_steps": 600,          # columna AR del README (la etapa por defecto)
    "batch_size": 1,             # reservado: los clips cambian de longitud; usa grad_accum_steps
    "grad_accum_steps": 4,
    "lr": 2e-4,                  # plan: 2e-4..4e-4; AR 2e-4, NAR 3e-4
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,
    "lr_scheduler": "cosine",
    "lr_cycles": 3,
    "lora_rank": 8,              # AR 8 (por defecto); NAR 16
    "lora_alpha": 8,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    "use_lokr": False,
    "lokr_factor": 16,
    "lora_targets": "blocks",    # blocks: atención+MLP de la etapa; all: también llm2vae/vae2llm/time_embedder
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 50,
    "seed": 42,
    "timestep_sampling": "logit_normal",  # uniform | logit_normal (t del flow matching)
    "regularizer": True,            # mezcla el minted pack en la etapa AR
    "regularizer_ratio": 0.5,       # proporcion de pasos AR con cancion minted
    "preview_every": 0,          # 0 = sin previews de audio (cada N pasos)
    "preview_caption_mode": "first",
    "preview_ode_steps": 16,     # el release usa 32; 16 para previews rápidas
    "preview_seconds": 20.0,
    "preview_compare_base": False,
    "preview_cfg": 0.0,          # 0 = sin CFG simbólico en las previews (futuro)
    "preview_custom_prompt": "", # letra de la preview; vacía = letra del clip elegido
    "project_name": "",
}

CONFIG_PATH = "settings/train_settings_yue2.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))


def _g(key):
    return cfg.get(key, DEFAULTS[key])


STAGE            = str(_g("stage")).strip().lower()
if STAGE not in ("nar", "ar", "both"):
    print("[!] " + t("Unknown stage '{s}': must be nar, ar or both. Using ar.", s=STAGE))
    STAGE = "ar"
PRECISION        = "nf4" if _g("precision") == "nf4" else "bf16"
TOTAL_STEPS      = int(_g("total_steps"))
BATCH_SIZE       = max(1, int(_g("batch_size")))
GRAD_ACCUM_STEPS = max(1, int(_g("grad_accum_steps")))
LR               = float(_g("lr"))
MIN_LR_RATIO     = float(_g("min_lr_ratio"))
WARMUP_STEPS     = int(_g("warmup_steps"))
LR_SCHEDULER     = str(_g("lr_scheduler"))
LR_CYCLES        = int(_g("lr_cycles"))
LORA_RANK        = int(_g("lora_rank"))
LORA_ALPHA       = int(_g("lora_alpha"))
LORA_TARGETS     = "all" if _g("lora_targets") == "all" else "blocks"
USE_RSLORA       = bool(_g("use_rslora"))
USE_LORAPLUS     = bool(_g("use_loraplus"))
LORAPLUS_RATIO   = float(_g("loraplus_ratio"))
USE_LOKR         = bool(_g("use_lokr"))
LOKR_FACTOR      = max(1, int(_g("lokr_factor")))
WEIGHT_DECAY     = float(_g("weight_decay"))
MAX_GRAD_NORM    = float(_g("max_grad_norm"))
SAVE_EVERY       = int(_g("save_every"))
SEED             = int(_g("seed"))
T_SAMPLING       = str(_g("timestep_sampling"))
REGULARIZER_ON   = bool(_g("regularizer"))
REGULARIZER_RATIO = float(_g("regularizer_ratio"))
PREVIEW_EVERY    = int(_g("preview_every"))
PREVIEW_CAPTION_MODE = str(_g("preview_caption_mode")).strip().lower()
if PREVIEW_CAPTION_MODE not in ("first", "random", "rotate4", "custom"):
    PREVIEW_CAPTION_MODE = "first"
PREVIEW_ODE      = max(1, int(_g("preview_ode_steps")))
PREVIEW_SECONDS  = float(_g("preview_seconds"))
PREVIEW_COMPARE_BASE = bool(_g("preview_compare_base"))
PREVIEW_CFG      = float(_g("preview_cfg"))
PREVIEW_CUSTOM   = str(_g("preview_custom_prompt")).strip()
PROJECT_NAME     = str(_g("project_name")).strip()

if PROJECT_NAME:
    CACHE_DIR  = str(resolve_cache_dir(f"./Cache_Data/cached_data_yue2_{PROJECT_NAME}",
                                      DEFAULTS["cache_dir"]))
    OUTPUT_DIR = str(resolve_output_dir(f"./Outputs/yue2_lora_output_{PROJECT_NAME}",
                                       DEFAULTS["output_dir"]))
else:
    CACHE_DIR  = str(resolve_cache_dir(_g("cache_dir"), DEFAULTS["cache_dir"]))
    OUTPUT_DIR = str(resolve_output_dir(_g("output_dir"), DEFAULTS["output_dir"]))

# Parámetros de sampling del release para LOS CÓDIGOS y previews (fase semántica)
with open(os.path.join(MODEL_DIR, "yue2_generation_config.json"), encoding="utf-8") as _gf:
    _gen_cfg = json.load(_gf)
SEMANTIC_CFG = _gen_cfg["semantic"] if "semantic" in _gen_cfg else _gen_cfg

STAGES_TO_RUN = ["nar", "ar"] if STAGE == "both" else [STAGE]

_os = os.path.basename(MODEL_DIR)
print(f"  {t('Model ID'):<22}: {_os}")
print(f"  {t('Precision'):<22}: {PRECISION.upper()}")
_stages_note = " + ".join(STAGES_TO_RUN) if STAGE == "both" else STAGE
print(f"  {t('Stage'):<22}: {_stages_note}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Output Dir'):<22}: {OUTPUT_DIR}")
_steps_note = f"{TOTAL_STEPS} per stage" if STAGE == "both" else str(TOTAL_STEPS)
print(f"  {t('Total Steps'):<22}: {_steps_note}")
print(f"  {t('Learning Rate'):<22}: {LR}")
print(f"  {t('LR Scheduler'):<22}: {lora_options.lr_scheduler_label(LR_SCHEDULER, WARMUP_STEPS, LR_CYCLES)}")
print(f"  {'LoRA Rank/Alpha':<22}: {LORA_RANK}/{LORA_ALPHA}")
print(f"  {'rsLoRA / LoRA+':<22}: {'on' if USE_RSLORA else 'off'}"
      f" / {'x' + format(LORAPLUS_RATIO, 'g') if USE_LORAPLUS else 'off'}")
print(f"  {'LoKr':<22}: {'factor ' + str(LOKR_FACTOR) if USE_LOKR else 'off'}")
print(f"  {t('LoRA Targets'):<22}: {LORA_TARGETS}")
print(f"  {'Batch / Grad Accum':<22}: {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
_ts_note = T_SAMPLING + " (x_t=(1-t)x0+t*eps, target eps-x0)" if STAGE != "ar" else T_SAMPLING
print(f"  {'T sampling':<22}: {_ts_note}")
_pv_note = f"every {PREVIEW_EVERY} | ode {PREVIEW_ODE} | {PREVIEW_SECONDS}s | cfg {PREVIEW_CFG if PREVIEW_CFG > 0 else 1.0:g}*"
print(f"  {t('Preview'):<22}: {_pv_note}")
print(f"  {t('Seed'):<22}: {SEED} ({t('random') if SEED <= 0 else t('fixed')})")


os.makedirs(OUTPUT_DIR, exist_ok=True)


def stage_paths(stage):
    suffix = "" if stage == "nar" else f"_{stage}"
    return (os.path.join(OUTPUT_DIR, f"resume_checkpoint{suffix}"),
            os.path.join(OUTPUT_DIR, f"current_step{suffix}.txt"))


def stage_names(stage):
    """Nombres de fichero por etapa. El server escanea *_step_N con el regex
    "step_(\\d+)", así que las dos variantes pisan el mismo patrón; el prefijo
    y el identificador de etapa distinguen nar/ar en las listas de la GUI."""
    if stage == "nar":
        return "YuE2_NAR_LoRA_step_{}.safetensors", "YuE2_FINAL_NAR_LoRA.safetensors"
    return "YuE2_AR_LoRA_step_{}.safetensors", "YuE2_FINAL_AR_LoRA.safetensors"


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def export_alpha():
    return lora_options.export_alpha(LORA_ALPHA, LORA_RANK, USE_RSLORA)


def build_lora_metadata(step, stage):
    """Cabecera ss_* de kohya-ss (lo que leen los gestores en la ficha) + la
    etapa: los ficheros NAR y AR se aplican por separado en la inferencia."""
    meta = {
        "format": "pt",
        "trained_with": "AcademiaSD LoRAlab YuE2",
        "ss_sd_model_name": "m-a-p/YuE2-3B",
        "ss_base_model_version": "YuE2-3B",
        "ss_network_module": "lycoris.kohya" if USE_LOKR else "peft.LoraModel",
        "ss_network_dim": LORA_RANK,
        "ss_network_alpha": export_alpha(),
        "lora_targets": LORA_TARGETS,
        "yue2_stage": stage,
        "ss_learning_rate": LR,
        "ss_lr_scheduler": LR_SCHEDULER,
        "ss_lr_warmup_steps": WARMUP_STEPS,
        "ss_max_train_steps": TOTAL_STEPS,
        "ss_steps": step,
        "ss_batch_size_per_device": BATCH_SIZE,
        "ss_gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "ss_seed": SEED,
        "ss_mixed_precision": "bf16",
        "yue2_protocol": "yue2-native-v1",
    }
    if USE_LOKR:
        meta["ss_network_args"] = json.dumps({"algo": "lokr", "factor": LOKR_FACTOR})
    elif USE_RSLORA or USE_LORAPLUS:
        meta["ss_network_args"] = json.dumps({"rslora": USE_RSLORA,
                                              "loraplus_lr_ratio": LORAPLUS_RATIO if USE_LORAPLUS else None})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or "yue2_lora"
    return {k: str(v) for k, v in meta.items()}


# ComfyUI nativo separa YuE2 en dos: la rama NAR es el MODEL (diffusion_model.*, nombres self_attn/mlp) y la
# AR va en el CLIP (text_encoders.*). Las dos fusionan q/k/v (qkv_proj) y gate/up (gate_up_proj), y
# comfy/lora.py no traduce las proyecciones separadas para YuE2.
COMFY_FUSED = (("self_attn", ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
               ("mlp", ("gate_proj", "up_proj"), "gate_up_proj"))


def _comfy_name(name, stage):
    if stage == "ar":
        return "text_encoders." + name
    return "diffusion_model." + name.replace(".nar_self_attn.", ".self_attn.").replace(".nar_mlp.", ".mlp.")


def comfy_lora(sd, alpha, stage):
    """LoRA de una etapa (nombres PEFT nativos) -> formato de ComfyUI, sin pérdida: A se concatena por rank
    y B va en diagonal por bloques, así B·A da cada delta en su tramo de la capa fusionada. El rank
    fusionado es N·r, por eso alpha se multiplica por N: la escala alpha/rank no cambia. alpha=None: B ya
    lleva la escala, así que alpha = rank (factor 1)."""
    prefix = "nar_" if stage == "nar" else ""
    out, fused = {}, {}
    for name in sorted({k.rsplit(".lora_", 1)[0] for k in sd}):
        a, b = sd[name + ".lora_A.weight"], sd[name + ".lora_B.weight"]
        head, _, leaf = name.rpartition(".")
        group = next((g for g in COMFY_FUSED if head.endswith("." + prefix + g[0]) and leaf in g[1]), None)
        if group is None:
            key = _comfy_name(name, stage)
            out.update({key + ".lora_A.weight": a, key + ".lora_B.weight": b,
                        key + ".alpha": torch.tensor(float(a.shape[0]) if alpha is None else alpha)})
        else:
            fused.setdefault((head, group), {})[leaf] = (a, b)
    for (head, (_, parts, target)), pieces in fused.items():
        if set(pieces) != set(parts):
            raise ValueError(f"LoRA incompleto para fusionar {head}.{target}: falta {set(parts) - set(pieces)}")
        key = _comfy_name(f"{head}.{target}", stage)
        out[key + ".lora_A.weight"] = torch.cat([pieces[p][0] for p in parts])
        out[key + ".lora_B.weight"] = torch.block_diag(*[pieces[p][1] for p in parts])
        out[key + ".alpha"] = torch.tensor(float(out[key + ".lora_A.weight"].shape[0]) if alpha is None
                                           else alpha * len(parts))
    return out


def comfy_lokr(model, stage):
    """LoKr -> ComfyUI. Las capas que ComfyUI no fusiona quedan como LoKr con su nombre de ComfyUI. Las
    fusionadas (q/k/v, gate/up) no admiten un LoKr por tramo, así que se escriben como el LoRA exacto
    equivalente: w1 ⊗ (w2_a·w2_b) = (w1 ⊗ w2_a)·(I ⊗ w2_b), rank c·r, con la escala dentro de B."""
    prefix = "nar_" if stage == "nar" else ""
    out, pairs = {}, {}
    for name, layer in model.lokr_layers().items():
        head, _, leaf = name.rpartition(".")
        if any(head.endswith("." + prefix + g[0]) and leaf in g[1] for g in COMFY_FUSED):
            w1, w2a, w2b = (getattr(layer, k).detach().float().cpu() for k in ("lokr_w1", "lokr_w2_a", "lokr_w2_b"))
            pairs[name + ".lora_B.weight"] = (layer.scale * torch.kron(w1, w2a)).to(torch.bfloat16).contiguous()
            pairs[name + ".lora_A.weight"] = torch.kron(torch.eye(layer.c), w2b).to(torch.bfloat16).contiguous()
        else:
            key = _comfy_name(name, stage)
            out.update({f"{key}.{k}": v for k, v in layer.export_tensors().items()})
            out[key + ".alpha"] = torch.tensor(float(export_alpha()))
    out.update(comfy_lora(pairs, None, stage))
    return out


def _export_lora(model, path, step, stage):
    """PEFT estándar con los nombres NATIVOS del modeling: las claves quedan
    model.layers.N.nar_self_attn.q_proj.lora_A.weight etc. (un cargador PEFT
    sobre YuE2ForCausalLM las encuentra tal cual), con alpha por capa porque
    un alpha global no basta cuando el checkpoint fue entrenado con rsLoRA."""
    if USE_LOKR:
        save_file(comfy_lokr(model, stage), path, metadata=build_lora_metadata(step, stage))
        return
    clean = {}
    for k, v in get_peft_model_state_dict(model).items():
        layer, _, rest = k.replace("base_model.model.", "").partition(".lora_")
        clean[f"{layer}.lora_{rest}"] = v.to(torch.bfloat16).cpu().contiguous()
        if f"{layer}.lora_{rest}".endswith(".lora_A.weight"):
            clean[f"{layer}.alpha"] = torch.tensor(float(export_alpha()))
    # Para Load LoRA de ComfyUI (YuE2 nativo): NAR en el lado MODEL, AR en el lado CLIP.
    clean = comfy_lora({k: v for k, v in clean.items() if not k.endswith(".alpha")}, float(export_alpha()), stage)
    save_file(clean, path, metadata=build_lora_metadata(step, stage))


# ══════════════════════════════════════════════════════════════════════════════
# MODELO: cargas + nombres de capas LoRA
# ══════════════════════════════════════════════════════════════════════════════

# NAR: solo las rutas exclusivas del flow matching; AR: solo sus bloques. Un
# LoRA de cada fichero NUNCA toca la otra rama: el MoT mantiene sus papeles.
# "all" amplía con los puentes del latente (vae2llm/llm2vae/time_embedder):
# puede adaptar timbres muy lejanos a la base pero mueve la escala del latente.
NAR_TARGET_PARTS = ("nar_self_attn.q_proj", "nar_self_attn.k_proj", "nar_self_attn.v_proj",
                    "nar_self_attn.o_proj", "nar_mlp.gate_proj", "nar_mlp.up_proj",
                    "nar_mlp.down_proj")
AR_TARGET_PARTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def target_names(model, stage):
    """Nombres COMPLETOS de módulo (model.layers.0.nar_self_attn.q_proj...):
    PEFT velaría por sufijo y 'mlp.gate_proj' casaría TAMBIÉN con nar_mlp
    (nar_mlp.gate_proj termina igual), así que la etapa se disambigua por
    nombre completo."""
    parts = NAR_TARGET_PARTS if stage == "nar" else AR_TARGET_PARTS
    # Con ancla "." delante: "nar_mlp.gate_proj" NO casa "mlp.gate_proj" porque
    # el carácter previo es "_" — sin este punto el LoRA AR secuestraría la
    # rama NAR (y viceversa no: "nar_self_attn…" solo es sufijo de sí mismo).
    names = {n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear)
             and any(n.endswith("." + p) for p in parts)}
    if LORA_TARGETS == "all" and stage == "nar":
        aux = ("llm2vae", "vae2llm", "time_embedder")
        names |= {n for n, m in model.named_modules()
                  if isinstance(m, torch.nn.Linear)
                  and any(n == a or n.startswith(a + ".") for a in aux)}
    return sorted(names)


def is_quantized(name):
    """NF4 solo de atención+MLP de CADA etapa. Los puentes del latente
    (llm2vae/vae2llm/time_embedder), normas, embed y cabezas quedan en BF16:
    la escala del latente del VAE y el flow matching son la parte sensible."""
    return (".norm" not in name
            and not name.startswith(("llm2vae", "vae2llm", "time_embedder", "lm_head",
                                     "model.embed_tokens", "latent_pos_embed")))


NF4_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_NF4", STUDIO_ROOT))   # caché del conversor NF4 (Fase 5)


def load_nf4_backbone():
    """Caché NF4 pre-cuantizada (tools/yue2/5_conversor_YuE2_NF4.py): index.json
    + weights/ + others.safetensors, formato studio (Qwen-Image 2.1). Reproduce
    la cuantización del conversor sin cuantizar en vivo (25 s en vez de ~2 min
    y sin pasar dos veces por los pesos BF16)."""
    from bitsandbytes.nn import Linear4bit, Params4bit
    from bitsandbytes.functional import QuantState
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from modeling_yue2 import YuE2ForCausalLM

    cache_dir = os.path.join(NF4_DIR, "model")
    with open(os.path.join(cache_dir, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        cfg = YuE2ForCausalLM.config_class.from_pretrained(cache_dir)
        base = YuE2ForCausalLM(cfg)
    base.to_empty(device="cpu")

    def set_module(name, module):
        parent, _, child = name.rpartition(".")
        setattr(base.get_submodule(parent) if parent else base, child, module)

    weights_dir = os.path.join(cache_dir, "weights")
    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt",
                       device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias = f.get_tensor("bias") if info["bias"] else None
            meta = f.metadata()
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k)
                       for k in f.keys() if k.startswith("quant_state.")}
        for k, v in (meta or {}).items():          # campos no tensoriales del QuantState
            if k.startswith("qs_"):
                try:
                    qs_dict[k[3:]] = json.loads(v)
                except (TypeError, ValueError):
                    pass

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4",
                            quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device="cpu")
        weight.bnb_quantized = True
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"],
                               quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"],
                                    bias=info["bias"], dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    base.load_state_dict(load_file(os.path.join(cache_dir, index["others"])),
                         strict=False, assign=True)
    missing = [n for n, t in list(base.named_parameters()) + list(base.named_buffers())
               if t.is_meta]
    if missing:
        raise RuntimeError("Caché NF4 incompleta: " + ", ".join(missing[:6]))
    base.requires_grad_(False)
    return base


def load_model():
    """Carga el backbone BF16 RAW o el cache NF4 según la precisión configurada."""
    from modeling_yue2 import YuE2ForCausalLM
    if PRECISION == "nf4" and os.path.isfile(os.path.join(NF4_DIR, "model", "index.json")):
        print("[OK] " + t("Loading pre-quantized NF4 cache from {path}", path=NF4_DIR))
        return load_nf4_backbone()
    if not os.path.isfile(os.path.join(RAW_DIR, "config.json")):
        if PRECISION == "nf4":
            raise FileNotFoundError(
                f"Falta el modelo NF4 en {NF4_DIR} y tampoco existe el RAW en {RAW_DIR}. Ejecuta Pre-Cache."
            )
        raise FileNotFoundError(
            f"No se encuentra el modelo BF16 RAW en {RAW_DIR}. Ejecuta Pre-Cache con precisión BF16."
        )
    base = YuE2ForCausalLM.from_pretrained(RAW_DIR, local_files_only=True,
                                          torch_dtype=torch.bfloat16)
    base.requires_grad_(False)
    if PRECISION == "nf4":
        print("[i] " + t("No NF4 cache found ({path}); quantizing on the fly.", path=NF4_DIR))
        from bitsandbytes.nn import Linear4bit, Params4bit
        quantized = 0
        for name, module in list(base.named_modules()):
            if isinstance(module, torch.nn.Linear) and is_quantized(name):
                layer = Linear4bit(module.in_features, module.out_features,
                                   bias=module.bias is not None,
                                   quant_type="nf4", compute_dtype=torch.bfloat16)
                layer.weight = Params4bit(module.weight.data.float(), requires_grad=False,
                                          quant_type="nf4")
                if module.bias is not None:
                    layer.bias = torch.nn.Parameter(module.bias.data, requires_grad=False)
                parent, _, child = name.rpartition(".")
                setattr(base.get_submodule(parent), child, layer)
                quantized += 1
        print("[OK] " + t("NF4 layers: {n} (attention + MLP, both stages)", n=quantized))
    return base


# ══════════════════════════════════════════════════════════════════════════════
# NAR: espejo del CachedNAR del wheel, con GRADIENTE en las capas nar_*
# ══════════════════════════════════════════════════════════════════════════════

def unwrap(model):
    """El modeling de YuE2ForCausalLM bajo el wrapper: PeftModel →
    LoraModel(YuE2ForCausalLM); LoKrModel → base_model.model. Los módulos
    con LoRA (nar_self_attn.q_proj…) están envueltos EN SITIO, así que el
    gradiente del LoRA llega igual."""
    base = getattr(model, "base_model", None)
    if base is None:
        return model
    inner = getattr(base, "model", None)
    return inner if inner is not None else base


def nar_context(model, base_ids, frames, device):
    """K/V de la ruta AR del prefijo bajo no_grad (para el LoRA NAR son
    constantes; su gradiente no pasa por la capa AR) más cos/sin en las
    posiciones NAR (ar_length..ar_length+nar_length), como el CachedNAR.
    Con no_grad (NO inference_mode): los tensores se usan luego en autograd."""
    core = unwrap(model)
    backbone = core.model
    cos_ar, sin_ar = backbone.rotary_emb(
        torch.arange(len(base_ids), device=device)[None])
    cos_nar, sin_nar = backbone.rotary_emb(
        torch.arange(len(base_ids), len(base_ids) + frames + 2, device=device)[None])
    with torch.no_grad():
        ids = torch.tensor([base_ids], dtype=torch.long, device=device)
        x = backbone.embed_tokens(ids)
        cache = []
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_ar, sin_ar)
            h = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + layer.self_attn.o_proj(h.transpose(1, 2).reshape(1, x.shape[1], -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            cache.append((k[0], v[0]))            # [L, kv_heads, hd]
    return cache, cos_nar, sin_nar


def nar_velocity(model, state, raw_t, kv_cache, cos, sin):
    """v_theta(x_t, t) con gradiente en las ramas nar_*. state [T,64] en el
    dtype del modelo (bf16); raw_t = logit(t) ya clampado. Devuelve T frames:
    el pad START/END se descarta con [0, 1:-1] como el wheel (el latente de
    entrenamiento también entra así: x_nar[1:T+1] = x_t)."""
    core = unwrap(model)
    frames = state.shape[0]
    nar_length = frames + 2
    x_nar = F.pad(state, (0, 0, 1, 1))                       # START/END con x=0
    shifted = core._shift_t_value(raw_t, state.device, state.dtype)
    x = core.vae2llm(x_nar[None])
    x = x + core.time_embedder(shifted.expand(nar_length))[None]
    pos = torch.arange(nar_length, device=state.device).clamp(max=core.config.max_latent_frames - 1)
    x = x + core.latent_pos_embed(pos)[None]
    for layer, (ar_k, ar_v) in zip(core.model.layers, kv_cache):
        q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), cos, sin)
        k = torch.cat((ar_k, k[0]), dim=0)        # [ar_len+T+2, kv, hd] NAR ve el texto
        v = torch.cat((ar_v, v[0]), dim=0)
        h = sdpa(q[0].transpose(0, 1).unsqueeze(0),
                 k.transpose(0, 1).unsqueeze(0),
                 v.transpose(0, 1).unsqueeze(0))       # bidireccional sobre [1,H,S,hd] (GQA)
        x = x + layer.nar_self_attn.o_proj(h.transpose(1, 2).reshape(1, nar_length, -1))
        x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
    return core.llm2vae(core.model.norm(x))[0, 1:-1]


def logit(t_value):
    return float(torch.logit(torch.tensor(float(t_value), dtype=torch.float64)).clamp(-20, 20).item())


def sample_t():
    """~U(0,1) con borde seguro: t=0 (estado limpio) o t=1 (ruido puro) dan
    logit() infinito; margen 1e-4 como el release hace con sus t de solver."""
    if T_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(1))
    else:
        u = torch.rand(1)
    return float(u.clamp(1e-4, 1.0 - 1e-4).item())


def nar_loss_step(model, x0, t_value, base_ids):
    """x0 [T,64] float32 (latente cacheado tal cual, SIN normalizar), t
    escalar. El estado entra bf16 (el NAR del release es exacto en bf16) pero
    el TARGET y el MSE van en float32: en bf16 el zumbido del gradiente supera
    el paso de aprendizaje en el régimen de pérdida baja."""
    core = unwrap(model)
    device = next(core.vae2llm.parameters()).device
    dtype = next(core.vae2llm.parameters()).dtype
    frames = x0.shape[0]
    kv_cache, cos_nar, sin_nar = nar_context(model, base_ids, frames, device)
    eps = torch.randn((frames, 64), dtype=torch.float32, device=device)
    x_t = ((1.0 - t_value) * x0 + t_value * eps).to(dtype)
    target = eps - x0                              # float32, var ~2.15 de partida
    v = nar_velocity(model, x_t, logit(t_value), kv_cache, cos_nar, sin_nar)
    return F.mse_loss(v.float(), target)


# ══════════════════════════════════════════════════════════════════════════════
# AR: muestreo de códigos (sampling EXACTO del release) + CE de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def _window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or not recent_ids:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty ** freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _distribution(logits, history, step):
    """Copia del sampling.distribution del release para la fase semántica:
    logits FP32; permitidos = rango codec + MUSIC_END; min_tokens cierra el
    end; luego ventana de repetición; temperatura; top_k; top_p con
    cumsum-cancel (conserva el 1º)."""
    scores = logits.float().clone()
    allowed = torch.full_like(scores, float("-inf"))
    allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., MUSIC_END] = 0
    scores = scores + allowed
    if step < int(SEMANTIC_CFG["min_tokens"]):
        scores[..., MUSIC_END] = -torch.inf
    scores = _window_penalty(scores, history[-int(SEMANTIC_CFG["penalty_window"]):],
                             float(SEMANTIC_CFG["repetition_penalty"]))
    temperature = float(SEMANTIC_CFG["temperature"])
    if temperature == 0:
        return scores
    if temperature != 1:
        scores = scores / temperature
    top_k = int(SEMANTIC_CFG["top_k"])
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    top_p = float(SEMANTIC_CFG["top_p"])
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


@torch.inference_mode()
def generate_codes(model, base_ids, frames, seed):
    """Códigos semánticos del modelo actual con parámetros del release
    (semántica del wheel generate_tokens, sin cuda-graph: eager). Devuelve
    (ids_sin_offset, eos). Cada clip limita su presupuesto a sus frames."""
    device = next(model.parameters()).device
    if len(base_ids) + frames > CONTEXT:
        raise RuntimeError(f"Clip de {frames} frames + prefijo no cabe en CONTEXT={CONTEXT}")
    generator = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
    out = model(torch.tensor([base_ids], dtype=torch.long, device=device),
                use_cache=True, logits_to_keep=1)
    logits, cache = out.logits[:, -1, :], out.past_key_values
    history, eos = [], False
    for step in range(frames):
        scores = _distribution(logits, history, step)
        if float(SEMANTIC_CFG["temperature"]) == 0:
            next_id = int(scores.argmax(-1).item())
        else:
            probabilities = scores.softmax(-1)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
        if next_id == MUSIC_END:
            eos = True
            break
        history.append(next_id)
        out = model(torch.tensor([[next_id]], dtype=torch.long, device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :]
    return [h - CODEC_OFFSET for h in history], eos


def codes_path(name):
    # Códigos REALES del clip, del Pre-Cache (MERT-v2 + head de la comunidad, yue2/real_codes.py).
    return os.path.join(CACHE_DIR, f"_realcodes_{name}.pt")


def ar_loss_step(model, base_ids, codes):
    """Teacher forcing completo: prefijo + códigos(+OFFSET) + MUSIC_END; CE
    SOLO sobre las predicciones desde la última posición del prefijo. Los
    logits se recortan a la cola: pesan (códigos+1) posiciones, no las ~1550
    de la secuencia entera."""
    core = unwrap(model)
    device = next(core.parameters()).device
    seq = list(base_ids) + [int(c) + CODEC_OFFSET for c in codes] + [MUSIC_END]
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    hidden, _ = core.model(input_ids=ids,
                           position_ids=torch.arange(len(seq), device=device)[None],
                           use_cache=False)
    tail = hidden[:, -(len(codes) + 2):-1, :]      # predice codes + MUSIC_END
    logits = core.lm_head(tail).float()            # [1, K, V] -> aplanado para el CE
    targets = torch.tensor(seq[-(len(codes) + 1):], dtype=torch.long, device=device)
    return F.cross_entropy(logits.reshape(len(targets), -1), targets)


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ: una sola lectura; latente [1,64,T] fp32 (SIN normalizar) + prefijo
# ══════════════════════════════════════════════════════════════════════════════

SAMPLES = {}
META = {}
REGULARIZER = None


def load_cache():
    meta_file = os.path.join(CACHE_DIR, "_meta.json")
    if not os.path.isfile(meta_file):
        print(f"\n[!] ERROR: " + t("Cache directory '{path}' has not been pre-cached "
                                  "(no _meta.json). Run Pre-Cache first!", path=CACHE_DIR))
        sys.exit(2)
    with open(meta_file, "r", encoding="utf-8") as f:
        META.update(json.load(f))
    if int(META.get("cache_format_version", 0)) < 1:
        print("[!] " + t("Cache format {v} is older than this trainer (v1). Re-run Pre-Cache.",
                         v=META.get("cache_format_version")))
        sys.exit(2)
    names = {f[:-len("_latent.pt")] for f in os.listdir(CACHE_DIR)
             if f.endswith("_latent.pt") and not f.startswith("_")}
    if not names:
        print(f"\n[!] ERROR: " + t("No latents in '{path}': run Pre-Cache first!", path=CACHE_DIR))
        sys.exit(2)
    for name in sorted(names):
        prompt_file = os.path.join(CACHE_DIR, f"{name}_prompt.pt")
        if not os.path.isfile(prompt_file):
            print("[!] " + t("{name}: no cached prompt, skipping clip (re-run Pre-Cache)", name=name))
            continue
        lat = torch.load(os.path.join(CACHE_DIR, f"{name}_latent.pt"), weights_only=True)
        entry = torch.load(prompt_file, weights_only=True)
        SAMPLES[name] = {"lat": lat,
                         "base_ids": entry["base_ids"].tolist(),
                         "lyrics_len": int(entry.get("lyrics_len", 0)),
                         "codes": torch.load(codes_path(name), weights_only=True).tolist()
                         if os.path.isfile(codes_path(name)) else None}
    print(f"  {t('Samples'):<20}: {len(SAMPLES)} | cot={META.get('cot')} | "
          f"style='{str(META.get('style_prompt', ''))[:40]}'")
    missing = [n for n, s in SAMPLES.items() if s["codes"] is None]
    if missing:
        print("\n[!] ERROR: " + t("Cache directory '{path}' is empty or does not exist.", path=CACHE_DIR)
              + f" ({len(missing)}: _realcodes_*.pt)")
        print("[!] " + t("Please run Pre-Cache first!"))
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"


# ══════════════════════════════════════════════════════════════════════════════
# PREVIEWS DE AUDIO: codes (modelo actual) -> ODE midpoint -> decoder VAE
# ══════════════════════════════════════════════════════════════════════════════

TOKENIZER = None
DATASET_PATH = "./dataset"
if os.path.exists("settings/pre_cache_settings_yue2.json"):
    try:
        with open("settings/pre_cache_settings_yue2.json", "r", encoding="utf-8") as _f:
            DATASET_PATH = str(json.load(_f).get("dataset_path", DATASET_PATH))
    except (OSError, ValueError):
        pass


def preview_sample_for_step(step, fallback=None):
    names = sorted(SAMPLES)
    if not names:
        return fallback
    if PREVIEW_CAPTION_MODE == "random":
        return random.choice(names)
    if PREVIEW_CAPTION_MODE == "rotate4":
        choices = names[:min(4, len(names))]
        interval = max(1, PREVIEW_EVERY)
        return choices[((step - 1) // interval) % len(choices)]
    if PREVIEW_CAPTION_MODE == "custom" and fallback in SAMPLES:
        return fallback
    return names[0]


def _preview_letra(sample_name):
    """La letra seleccionada para el preview o el prompt custom actual."""
    if PREVIEW_CAPTION_MODE == "custom":
        return PREVIEW_CUSTOM
    if sample_name:
        path = os.path.join(DATASET_PATH, sample_name + ".txt")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    return ""


@torch.inference_mode()
def nar_solve(model, base_ids, frames, generator):
    """ODE midpoint exacta del wheel (t 1→0, raw_t logit clampado, ruido CPU
    FP32 sembrado). PREVIEW_ODE en vez de 32. Devuelve [frames,64] FP32 CPU."""
    core = unwrap(model)
    device = next(core.vae2llm.parameters()).device
    dtype = next(core.vae2llm.parameters()).dtype
    state = torch.randn((frames, 64), dtype=torch.float32, device="cpu",
                        generator=generator).to(device=device, dtype=dtype)
    kv_cache, cos, sin = nar_context(model, base_ids, frames, device)
    dt = 1.0 / PREVIEW_ODE
    for step in range(PREVIEW_ODE):
        tt = 1.0 - step * dt
        first = nar_velocity(model, state, logit(tt), kv_cache, cos, sin)
        mid = state - first * (dt / 2)
        state = state - nar_velocity(model, mid, logit(tt - dt / 2), kv_cache, cos, sin) * dt
    result = state.float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("previews: latentes no finitos; revisa el LoRA")
    return result


def run_preview(model, step, stage, sample_name=None):
    """Genera preview con LoRA y, opcionalmente, también con los adapters desactivados."""
    model.eval()
    seed = SEED if SEED > 0 else random.randint(1, 2147483647)
    print("  ↳ " + t("Preview seed used: {seed}", seed=seed))
    try:
        frames = max(8, int(PREVIEW_SECONDS * 25))
        estilo = str(META.get("style_prompt", ""))
        request = SongRequest(style=estilo, lyrics=_preview_letra(sample_name),
                              cot=str(META.get("cot", "off")), id="preview")
        prefix = token_prefixes(request, TOKENIZER)

        def make_latents():
            codes, eos = generate_codes(model, prefix, frames, seed)
            if len(codes) < 4:
                return None
            seq = list(prefix) + [c + CODEC_OFFSET for c in codes] + ([] if eos else [MUSIC_END])
            generator = torch.Generator(device="cpu").manual_seed(seed & 0x7FFFFFFF)
            return nar_solve(model, seq, len(codes), generator)  # [T,64] FP32 CPU

        previews = []
        if PREVIEW_COMPARE_BASE:
            print("  ↳ " + t("Generating preview without LoRA..."))
            base_oom = False
            try:
                with model.disable_adapter():
                    z_base = make_latents()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                z_base = None
                base_oom = True
                print("  ↳ [Preview] " + t("base comparison skipped: out of VRAM"))
            if z_base is not None:
                previews.append(("without_lora", z_base))
            elif not base_oom:
                print("  ↳ [Preview] " + t("semantic stage produced too few codes; base preview skipped"))
            free_vram()

        print("  ↳ " + t("Generating preview with LoRA..."))
        lora_oom = False
        try:
            z_lora = make_latents()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            z_lora = None
            lora_oom = True
            print("  ↳ [Preview] " + t("LoRA comparison skipped: out of VRAM"))
        if z_lora is not None:
            previews.append(("with_lora" if PREVIEW_COMPARE_BASE else "", z_lora))
        elif not lora_oom:
            print("  ↳ [Preview] " + t("semantic stage produced too few codes; LoRA preview skipped"))
        if not previews:
            return

        from modeling_vae import YuE2VAE
        vae = YuE2VAE.from_pretrained(VAE_DIR, device="cuda", local_files_only=True)
        for suffix, z in previews:
            try:
                audio = vae.decode(z[None].transpose(1, 2).to("cuda"))   # [1,64,T] -> [1,2,N]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  ↳ [Preview] " + t("audio decode skipped: out of VRAM"))
                continue
            # Con stage=both las dos etapas pasan por los mismos pasos: la AR lleva su marca para no pisar a la NAR.
            tag = f"preview_step_{step}_ar" if stage == "ar" else f"preview_step_{step}"
            name = f"{tag}_{suffix}.wav" if suffix else f"{tag}.wav"
            out = os.path.join(OUTPUT_DIR, name)
            muestras = audio[0].clamp(-1, 1).T.contiguous().cpu().numpy()   # [N, 2]
            with wave.open(out, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes((muestras * 32767.0).astype("<i2").tobytes())
            print("  ↳ " + t("Preview saved to: {path}", path=out))
            del audio
        del vae
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("  ↳ [Preview] " + t("skipped: out of VRAM (reduce preview_seconds)"))
    finally:
        model.train()
        free_vram()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO DE UNA ETAPA (resume/checkpoints/export por etapa)
# ══════════════════════════════════════════════════════════════════════════════

_live_mtime = None


def reload_live_settings():
    """Aplica al siguiente paso los ajustes compatibles con el modelo cargado."""
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_ODE, PREVIEW_SECONDS, PREVIEW_COMPARE_BASE, PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM, SEED
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return []
    if _live_mtime is not None and mtime == _live_mtime:
        return []
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []  # escritura incompleta: se vuelve a intentar en el paso siguiente
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

    TOTAL_STEPS = max(1, fresh("total_steps", int, TOTAL_STEPS))
    SAVE_EVERY = max(0, fresh("save_every", int, SAVE_EVERY))
    LR = max(0.0, fresh("lr", float, LR))
    MAX_GRAD_NORM = max(0.0, fresh("max_grad_norm", float, MAX_GRAD_NORM))
    PREVIEW_EVERY = max(0, fresh("preview_every", int, PREVIEW_EVERY))
    PREVIEW_ODE = max(1, fresh("preview_ode_steps", int, PREVIEW_ODE))
    PREVIEW_SECONDS = max(1.0, fresh("preview_seconds", float, PREVIEW_SECONDS))
    PREVIEW_COMPARE_BASE = fresh("preview_compare_base", bool, PREVIEW_COMPARE_BASE)
    PREVIEW_CUSTOM = fresh("preview_custom_prompt", lambda value: str(value).strip(), PREVIEW_CUSTOM)
    mode = fresh("preview_caption_mode", lambda value: str(value).strip().lower(), PREVIEW_CAPTION_MODE)
    if mode in ("first", "random", "rotate4", "custom"):
        PREVIEW_CAPTION_MODE = mode
    elif mode != PREVIEW_CAPTION_MODE:
        changes.append(f"preview_caption_mode: {mode} ignored (unsupported)")
    SEED = fresh("seed", int, SEED)  # affects previews only after startup
    return changes


def train_stage(stage, base):
    """Una etapa con su PEFT/LoKr propio. El resume guarda pesos, optimizador
    y paso POR ETAPA; al reanudar manda el checkpoint (capas/rank/alpha/opciones)."""
    global LORA_RANK, LORA_ALPHA, USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO, USE_LOKR, LOKR_FACTOR
    resume_dir, step_file = stage_paths(stage)
    step_name, final_name = stage_names(stage)
    opt_file = os.path.join(OUTPUT_DIR, f"optimizer{'' if stage == 'nar' else '_' + stage}.pt")

    target_modules = target_names(base, stage)
    print(t("Target LoRA layers ({stage}): {n} ({kind})",
            stage=stage.upper(), n=len(target_modules), kind=LORA_TARGETS))

    # Reanudar es continuar EL MISMO LoRA: método, capas, rank y alpha salen
    # del checkpoint (las capas se leen de los PESOS, no de adapter_config,
    # que las guarda abreviadas).
    lora_weights = os.path.join(resume_dir, "adapter_model.safetensors")
    lokr_weights = os.path.join(resume_dir, lora_options.LOKR_WEIGHTS)
    if os.path.exists(step_file) and (os.path.exists(lora_weights) or os.path.exists(lokr_weights)):
        saved_lokr = not os.path.exists(lora_weights)
        if saved_lokr != USE_LOKR:
            print("\n[!] " + t("Resuming with the checkpoint's method: {m} (settings ask for {m2}; applies to new trainings).",
                              m="LoKr" if saved_lokr else "LoRA", m2="LoKr" if USE_LOKR else "LoRA"))
        USE_LOKR = saved_lokr
        if stage == "ar":
            # LoKr comparte implementación con Anima y no distingue etapas;
            # se registra para saber de qué fichero salen las claves lokr_*.
            pass
    resume_weights = lokr_weights if USE_LOKR else lora_weights
    if os.path.exists(step_file) and os.path.exists(resume_weights):
        with safe_open(resume_weights, framework="pt", device="cpu") as f:
            if USE_LOKR:
                saved_targets = sorted({k.rsplit(".lokr_")[0] for k in f.keys()})
                with open(os.path.join(resume_dir, lora_options.LOKR_CONFIG), "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
                saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
                if saved != (len(target_modules), LORA_RANK, LORA_ALPHA) or saved_cfg["lokr_factor"] != LOKR_FACTOR:
                    print("\n[!] " + t("Resuming with the checkpoint's LoKr: {n} layers, rank {r}, alpha {a}, factor {f}",
                                       n=saved[0], r=saved[1], a=saved[2], f=saved_cfg["lokr_factor"]))
                target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]
                LOKR_FACTOR = saved_cfg["lokr_factor"]
            else:
                saved_targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1)
                                        for k in f.keys()})
                with open(os.path.join(resume_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
                saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
                if saved != (len(target_modules), LORA_RANK, LORA_ALPHA):
                    print("\n[!] " + t("Resuming with the checkpoint's LoRA: {n} layers, rank {r}, alpha {a} "
                                       "(settings ask for {n2} layers, rank {r2}, alpha {a2}; applies to new trainings).",
                                       n=saved[0], r=saved[1], a=saved[2],
                                       n2=len(target_modules), r2=LORA_RANK, a2=LORA_ALPHA))
                target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]
                opts_file = os.path.join(resume_dir, "lora_options.json")
                opts = {}
                if os.path.exists(opts_file):
                    with open(opts_file, "r", encoding="utf-8") as f:
                        opts = json.load(f)
                saved_opts = (bool(saved_cfg.get("use_rslora", False)), bool(opts.get("use_loraplus", False)),
                              float(opts.get("loraplus_ratio", LORAPLUS_RATIO)))
                if saved_opts[:2] != (USE_RSLORA, USE_LORAPLUS):
                    print("\n[!] " + t("Resuming with the checkpoint's options: rsLoRA {a}, LoRA+ {b}",
                                       a=saved_opts[0], b=saved_opts[1]))
                USE_RSLORA, USE_LORAPLUS, LORAPLUS_RATIO = saved_opts

    if USE_LOKR:
        USE_RSLORA = USE_LORAPLUS = False
        model = lora_options.LoKrModel(base, target_modules, LORA_RANK, LORA_ALPHA, LOKR_FACTOR)
    else:
        lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                                 target_modules=target_modules, use_dora=False,
                                 init_lora_weights=True, use_rslora=USE_RSLORA)
        model = get_peft_model(base, lora_config)
    # LoRA A/B en FP32: en bf16 el ruido del gradiente acumulado compite con
    # la tasa de aprendizaje (mismo criterio que Anima; peft 0.21 castea el
    # input al dtype del LoRA en el forward).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if USE_LORAPLUS:
        optimizer = torch.optim.AdamW(
            lora_options.loraplus_groups(model.named_parameters(), LORAPLUS_RATIO),
            lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(idx):
        return LR * lora_options.lr_factor(LR_SCHEDULER, idx, TOTAL_STEPS,
                                           WARMUP_STEPS, MIN_LR_RATIO, LR_CYCLES)

    start_step = 0
    if os.path.exists(step_file) and os.path.exists(resume_weights) and os.path.exists(opt_file):
        print("=" * 65)
        print(t("Checkpoint detected! Restoring state..."))
        try:
            with open(step_file, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(resume_weights, "rb") as f:
                saved_sd = load(f.read())
            if USE_LOKR:
                model.load_lokr_state_dict(saved_sd)
            else:
                set_peft_model_state_dict(model, {k: v.float() for k, v in saved_sd.items()})
            optimizer.load_state_dict(torch.load(opt_file, weights_only=False))
            print(t("Resuming training from step {n}...", n=start_step))
        except Exception as e:
            print("[!] " + t("Warning reading checkpoint: {error}", error=e))
            start_step = 0
        print("=" * 65)

    last_done = start_step

    def save_checkpoint_now(current):
        if current <= 0:
            return
        print("\n" + t("Saving checkpoint state at step {n}...", n=current))
        os.makedirs(resume_dir, exist_ok=True)
        if USE_LOKR:
            model.save_pretrained(resume_dir)
            stale = os.path.join(resume_dir, "adapter_model.safetensors")
        else:
            model.save_pretrained(resume_dir)
            stale = os.path.join(resume_dir, lora_options.LOKR_WEIGHTS)
        if os.path.exists(stale):
            os.remove(stale)
        with open(os.path.join(resume_dir, "lora_options.json"), "w", encoding="utf-8") as f:
            json.dump({"use_loraplus": bool(USE_LORAPLUS),
                       "loraplus_ratio": LORAPLUS_RATIO}, f)
        torch.save(optimizer.state_dict(), opt_file)
        with open(step_file, "w", encoding="utf-8") as f:
            f.write(str(current))
        ckpt = os.path.join(OUTPUT_DIR, step_name.format(current))
        _export_lora(model, ckpt, current, stage)
        print("✓ " + t("Checkpoint saved at step {n}: {path}", n=current, path=ckpt))

    def handle_signal(sig, _frame):
        # Deja el guardado al except para hacerlo una sola vez y detener también
        # el bucle exterior cuando stage=both.
        print("\n[!] " + t("Stop signal received ({sig}).", sig=sig))
        raise SystemExit(0)

    for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
        try:
            signal.signal(getattr(signal, sig), handle_signal)
        except (AttributeError, OSError, ValueError):
            pass

    def release_signals():
        """Los handlers retienen el wrapper (y el modelo) por cierre: crucial
        en stage=both, donde la etapa siguiente carga su modelo fresco. Sin
        esto, la etapa AR arrastraba 7,3 GB del NAR en VRAM."""
        for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
            try:
                signal.signal(getattr(signal, sig), signal.SIG_DFL)
            except (AttributeError, OSError, ValueError):
                pass

    if start_step >= TOTAL_STEPS and start_step > 0:
        print("\n[OK] " + t("'{s}' already has {n} steps (>= total_steps): the stage skips "
                            "ahead to its export. To train again, raise total_steps "
                            "or delete {f}.", s=stage.upper(), n=start_step, f=step_file))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Cubetas por longitud T: cada micro-paso pilla un clip al azar de la
    # cubeta al azar; el gradiente se acumula como Anima.
    buckets = {}
    for n in SAMPLES:
        buckets.setdefault(SAMPLES[n]["lat"].shape[-1], []).append(n)

    running_loss, avg_step_time, grad_norm = 0.0, 0.0, 0.0
    print("\n" + t("STARTING TRAINING! {n} audio clips, stage {s}.",
                   n=len(SAMPLES), s=stage.upper()))
    reload_live_settings()
    metrics = training_metrics.MetricsLog(OUTPUT_DIR, start_step, f"metrics_{stage}.jsonl")
    step = start_step
    try:
        while step < TOTAL_STEPS:
            step += 1
            last_done = step
            changes = reload_live_settings()
            if changes:
                print("\n[LIVE] " + t("Settings reloaded without stopping:"))
                for c in changes:
                    print(f"[LIVE]   {c}")
                if step > TOTAL_STEPS:
                    break

            t0 = time.time()
            name = random.choice(buckets[random.choice(list(buckets))])
            info = SAMPLES[name]

            if stage == "nar":
                # El NAR ve texto + códigos reales del clip, como cuando genera en ComfyUI.
                x0 = info["lat"][0].permute(1, 0).contiguous().float().cuda()   # [T, 64]
                context = info["base_ids"] + [c + CODEC_OFFSET for c in info["codes"]] + [MUSIC_END]
                loss = nar_loss_step(model, x0, sample_t(), context)
            else:
                if REGULARIZER is not None and random.random() < REGULARIZER_RATIO:
                    reg_ids, reg_codes = REGULARIZER.sample()
                    loss = ar_loss_step(model, reg_ids, reg_codes)
                else:
                    loss = ar_loss_step(model, info["base_ids"], info["codes"])

            (loss / GRAD_ACCUM_STEPS).backward()
            step_loss = loss.item()
            running_loss += step_loss

            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step) * group.get("lr_mult", 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            elapsed = time.time() - t0
            avg_step_time = elapsed if avg_step_time == 0 else 0.1 * elapsed + 0.9 * avg_step_time
            eta_s = (TOTAL_STEPS - step) * avg_step_time
            eta = f"{int(eta_s // 3600):02d}:{int((eta_s % 3600) // 60):02d}:{int(eta_s % 60):02d}"
            pct = step / TOTAL_STEPS
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            seen = max(1, step - start_step)
            print("\r"
                  f"{STEP_WORD} {step:4d}/{TOTAL_STEPS} [{bar}] {pct * 100:5.1f}% | "
                  f"Loss {running_loss / seen:.4f} | gnorm {grad_norm:.3f} | "
                  f"lr {lr_at(step):.2e} | {avg_step_time:.2f}s/it | ETA {eta}",
                  end="", flush=True)
            metrics.log(step, step_loss, lr_at(step), grad_norm, elapsed, TOTAL_STEPS)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                print()
                run_preview(model, step, stage, sample_name=preview_sample_for_step(step, name))

    except (KeyboardInterrupt, SystemExit):
        print()
        release_signals()
        save_checkpoint_now(last_done)
        raise SystemExit(0)

    print("\n\n" + t("Stage {s} completed!", s=stage.upper()))
    release_signals()
    final = os.path.join(OUTPUT_DIR, final_name)
    _export_lora(model, final, min(last_done, TOTAL_STEPS), stage)
    print("✓ " + t("Final LoRA saved to: {path}", path=final))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global TOKENIZER
    if SEED > 0:
        torch.manual_seed(SEED)
        random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        print("[!] " + t("CUDA is required for YuE2 training."))
        sys.exit(2)

    load_cache()
    TOKENIZER = YuE2TextTokenizer.from_pretrained(MODEL_DIR)

    # El regularizador del AR: canciones "minted" de la comunidad para que un set
    # pequeno no colapse la gramatica de tokens. Solo lo usa la etapa AR.
    global REGULARIZER
    REGULARIZER = None
    if REGULARIZER_ON and "ar" in STAGES_TO_RUN:
        from yue2 import regularizer as yue2_regularizer
        path = str(META.get("regularizer_path") or "") or yue2_regularizer.pack_path(STUDIO_ROOT)
        if not os.path.isfile(path):
            print("  [!] " + t("Regularizer pack not found: {path}. Run Pre-Cache to download it.",
                            path=path))
        else:
            # La longitud se acota a la de los clips del artista: los logits del AR son
            # [n_codigos, 184704] y una cancion de 4 minutos se va de la VRAM.
            largos = [len(s.get("codes", [])) for s in SAMPLES.values()]
            max_codes = (max(largos) + 255) // 256 * 256 if largos else 0
            REGULARIZER = yue2_regularizer.Regularizer(
                path, TOKENIZER, cot=str(META.get("cot", "off")),
                max_codes=max_codes)
            print(f"  {t('Regularizer')}: {len(REGULARIZER)} canciones minted, "
                  f"{REGULARIZER_RATIO*100:.0f}% de los pasos AR")

    # 'both' recarga el modelo por etapa: PEFT envuelve los Linear del propio
    # objeto base; partir de pesos frescos evita mezclas de adapters.
    for stage in STAGES_TO_RUN:
        print("\n" + "=" * 65)
        print(t("Loading {name}...", name=f"YuE2-3B backbone ({PRECISION.upper()})"))
        t0 = time.time()
        base = load_model()
        base.cuda()
        free_vram()
        print(t("Model loaded in {s:.1f}s.", s=time.time() - t0)
              + f" VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

        train_stage(stage, base)
        del base
        free_vram()

    print("\n" + t("YuE2 training finished!"))


if __name__ == "__main__":
    main()
