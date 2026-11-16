# -*- coding: utf-8 -*-
"""
1_pre_cache_LTX25.py
Pre-cache de LTX-2.5 para entrenamiento LoRA (dataset de imágenes).

OPTIMIZACIÓN DE VRAM:
El text encoder de LTX-2.5 es enorme y NO cabe en GPUs de 16 GB.
Antes se hacía pipe.to("cuda") y se saturaba la VRAM (swap -> lentísimo).
Ahora el pipeline se carga en CPU y se aplica offload según
`precache_offload`:

"none"      : todo en VRAM (solo si tienes VRAM de sobra).
"model"     : 1 componente en VRAM cada vez (solo si cada uno cabe).
"sequential": capa por capa en VRAM (RECOMENDADO en 16 GB).
"cpu"       : text encoder en CPU, VAE en VRAM (lento pero 0 VRAM texto).

Y con `text_encoder_4bit: true` se intenta cuantizar el text encoder a
4-bit para que quepa entero en VRAM (experimental, con fallback).

CAMBIOS PARA BAJA RAM:
- Detección de RAM física.
- Modo LOW_RAM automático si la RAM es menor o igual a low_ram_threshold_gb.
- En LOW_RAM se fuerza sequential offload y text encoder 4-bit.
- Se limita la CPU que puede usar Accelerate.
- Se usa offload_folder para permitir apoyo en disco.
- Se evita el fallback BF16 completo en CPU como primera opción.
- Si el 4-bit falla, se intenta BF16 con device_map auto + disk offload.
"""

import os
import gc
import json
import math
import sys
import traceback
import importlib
import shutil
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DiffusionPipeline
from i18n import t
from text_io import read_text_file
import gpu_stats
import ltx25_model
from project_layout import resolve_cache_dir, resolve_model_dir


# ============================================================================
# DETECCIÓN DE RAM FÍSICA
# ============================================================================
def _detect_system_ram_gb():
    """
    Intenta detectar la RAM física total.
    Usa psutil si está instalado; si no, usa /proc/meminfo o Windows API.
    """
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass

    return None


SYSTEM_RAM_GB = _detect_system_ram_gb()


# ============================================================================
# CONFIG
# ============================================================================
DEFAULTS = {
    "model_id": "Models/LTX25-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_ltx25",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 32,
    "max_seq_len": 1024,
    "frame_rate": 24.0,
    "num_frames": 1,
    # Clips de vídeo (.mp4 .mov .mkv .webm): se leen a 24 fps, con 8n+1 fotogramas (1, 9, 17, ..., 121),
    # escalados a su propia área. Por defecto 384x384 y 49 fotogramas (~2 s): ~1000 tokens, cabe en 8 GB.
    "video_target_area": 384 * 384,
    "max_frames": 49,
    # Pista de audio de los clips (Fase 3): se codifica con el VAE de audio. Un clip mudo lleva el latente de un
    # silencio real y queda marcado sin audio (no entra en la pérdida de audio).
    "use_audio": True,
    # Fase 4, IC-LoRA de edición: pares nombre_before / nombre_after (imágenes o clips) + nombre.txt con la
    # instrucción. El "after" es el objetivo; el "before" se guarda como referencia limpia. reference_downscale 2
    # codifica la referencia a la mitad de lado (1/4 de tokens), como los IC-LoRA oficiales de LTX.
    "lora_type": "normal",
    "reference_downscale": 1,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",

    # --- Gestión de VRAM del pre-cache ---
    "precache_offload": "sequential",   # none | model | sequential | cpu
    "text_encoder_4bit": True,          # experimental: cuantiza text encoder a 4-bit

    # --- Baja RAM ---
    # Si la RAM física detectada es <= este valor, activa modo conservador.
    # 48 GB es recomendable si el pico actual está cerca de 64 GB.
    # Si solo quieres activarlo en sistemas de 32 GB o menos, pon 32.
    "low_ram_threshold_gb": 48.0,

    # Último recurso en LOW_RAM si falla disk offload:
    # cargar BF16 completo en CPU. Puede usar pagefile/swap, pero puede dar OOM.
    "low_ram_allow_cpu_fallback": False,
}

CONFIG_PATH = os.environ.get("LORALAB_PRECACHE_CONFIG") or "settings/pre_cache_settings_ltx25.json"

HF_BASE_REPO_ID = "Lightricks/LTX-2.5-Diffusers"  # gated; solo lo usa tools/ltx25
HF_NF4_REPO_ID = "AcademiaSD/LTX25-NF4-for-LoRA-Training"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("[!] " + t("{path} not found, using default values.", path=CONFIG_PATH))


def cfg_get(key, default):
    if not isinstance(cfg, dict):
        return default

    candidates = [
        key,
        key + " ",
        " " + key,
        " " + key + " ",
    ]

    for candidate in candidates:
        if candidate in cfg:
            return cfg[candidate]

    return default


def _cfg_bool(key, default):
    value = cfg_get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    return bool(value)


MODEL_ID = str(resolve_model_dir(cfg_get("model_id", DEFAULTS["model_id"]), DEFAULTS["model_id"]))
DATASET_PATH = str(cfg_get("dataset_path", DEFAULTS["dataset_path"])).strip()
TARGET_AREA = int(cfg_get("target_area", DEFAULTS["target_area"]))
MAX_SIDE = int(cfg_get("max_side", DEFAULTS["max_side"]))
MULTIPLE = int(cfg_get("multiple", DEFAULTS["multiple"]))
MAX_SEQ_LEN = int(cfg_get("max_seq_len", DEFAULTS["max_seq_len"]))
FRAME_RATE = float(cfg_get("frame_rate", DEFAULTS["frame_rate"]))
NUM_FRAMES = int(cfg_get("num_frames", DEFAULTS["num_frames"]))
VIDEO_TARGET_AREA = int(cfg_get("video_target_area", DEFAULTS["video_target_area"]))
MAX_FRAMES = int(cfg_get("max_frames", DEFAULTS["max_frames"]))
USE_AUDIO = _cfg_bool("use_audio", DEFAULTS["use_audio"])
LORA_TYPE = str(cfg_get("lora_type", DEFAULTS["lora_type"])).strip().lower()
REF_DOWNSCALE = max(1, int(cfg_get("reference_downscale", DEFAULTS["reference_downscale"])))
TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()
PREVIEW_CUSTOM_PROMPT = str(cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()

PRECACHE_OFFLOAD = str(
    cfg_get("precache_offload", DEFAULTS["precache_offload"])
).strip().lower()

TEXT_ENCODER_4BIT = _cfg_bool(
    "text_encoder_4bit",
    DEFAULTS["text_encoder_4bit"]
)

LOW_RAM_THRESHOLD_GB = float(
    cfg_get("low_ram_threshold_gb", DEFAULTS["low_ram_threshold_gb"])
)

LOW_RAM_ALLOW_CPU_FALLBACK = _cfg_bool(
    "low_ram_allow_cpu_fallback",
    DEFAULTS["low_ram_allow_cpu_fallback"]
)

LOW_RAM_MODE = bool(
    LOW_RAM_THRESHOLD_GB > 0
    and SYSTEM_RAM_GB is not None
    and SYSTEM_RAM_GB <= LOW_RAM_THRESHOLD_GB
)

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_ltx25_{PROJECT_NAME}" if PROJECT_NAME else cfg_get("cache_dir"), DEFAULTS["cache_dir"]))

# LTX-2.5 exige dimensiones divisibles por 32.
MULTIPLE = max(32, MULTIPLE)

if LOW_RAM_MODE:
    PRECACHE_OFFLOAD = "sequential"
    TEXT_ENCODER_4BIT = True
    MAX_SEQ_LEN = min(MAX_SEQ_LEN, 512)

    # Limitar threads puede reducir picos secundarios en CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    print()
    print("=" * 80)
    print("[LOW-RAM] " + t("Conservative mode on"))
    print("[LOW-RAM] " + t("RAM detected: {gb:.1f} GB", gb=SYSTEM_RAM_GB))
    print("[LOW-RAM] " + t("Threshold: {gb:.1f} GB", gb=LOW_RAM_THRESHOLD_GB))
    print("[LOW-RAM] " + t("Forcing sequential offload + CPU limit + disk offload."))
    print("=" * 80)


# ============================================================================
# DESCARGA DESDE HUGGING FACE
# ============================================================================
def get_hf_token():
    if os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token_data = json.load(f)

            token = token_data.get("token", "").strip()
            if token:
                return token
        except Exception:
            pass

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    return None


def enable_hf_file_progress():
    # Lanzado desde la GUI la salida no es una terminal y tqdm se desactiva: una descarga de varios GB
    # parecería colgada. Se fuerzan las barras de progreso por fichero (en bytes).
    import logging
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


def ensure_ltx25_model_downloaded(local_path):
    """
    LTX-2.5 viene entero del repo NF4 de AcademiaSD (el oficial tiene gate): transformer NF4 + Gemma 4 NF4 +
    connectors, VAEs, vocoder, scheduler y tokenizer. Con tools/ltx25 se puede generar en local.
    LTX-2.5 comes whole from AcademiaSD's NF4 repo (the official one is gated).
    """
    local_path = str(local_path or "./Models/LTX25-NF4").strip() or "./Models/LTX25-NF4"
    needed = ("model_index.json", "index.json", "others.safetensors", os.path.join("text_encoder_NF4", "config.json"))
    if all(os.path.exists(os.path.join(local_path, n)) for n in needed):
        print("[OK] " + t("Local model found at: {path}", path=local_path))
        return local_path

    print()
    print("=" * 80)
    print(t("WARNING"))
    print("=" * 80)
    print(t("This will download about {gb} GB. This may take several minutes.", gb=28))
    print("=" * 80)

    from huggingface_hub import snapshot_download

    token = get_hf_token()
    if token:
        print(t("Using HF Token"))
    enable_hf_file_progress()
    os.makedirs(local_path, exist_ok=True)
    print(t("Downloading from Hugging Face: {repo}", repo=HF_NF4_REPO_ID))
    snapshot_download(repo_id=HF_NF4_REPO_ID, local_dir=local_path, token=token, max_workers=4)
    for n in needed:
        if not os.path.exists(os.path.join(local_path, n)):
            raise RuntimeError(t("Incomplete download: {name} missing in {path}", name=n, path=local_path))
    print()
    print("[OK] " + t("Model downloaded to: {path}", path=local_path))
    return local_path


# ============================================================================
# UTILIDADES
# ============================================================================
def free_vram(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def vram_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def read_audio_channels(model_id, default=128):
    """
    Lee audio_in_channels del config.json del transformer en disco (0 VRAM).
    El del repo NF4 (raíz) es el mismo que el de transformer/, que ya no se descarga.
    """
    for rel in ("config.json", os.path.join("transformer", "config.json")):
        p = os.path.join(model_id, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                return int(c.get("audio_in_channels", default))
            except Exception:
                pass

    return default


def read_model_index_components(model_id):
    """
    Lee model_index.json y devuelve los componentes del pipeline.
    """
    path = os.path.join(model_id, "model_index.json")

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out = {}

        if isinstance(data, dict):
            for key, value in data.items():
                if key.startswith("_"):
                    continue

                if isinstance(value, (list, tuple)) and len(value) == 2:
                    out[key] = value

        return out

    except Exception as exc:
        print("[LIGHT] " + t("Could not read model_index.json: {error}", error=exc))
        return {}


def resolve_component_class(model_id, component_name, current_object=None):
    """
    Resuelve la clase de un componente del pipeline a partir de model_index.json.
    """
    info = read_model_index_components(model_id).get(component_name)

    if info is not None:
        module_name, class_name = info

        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            print("[LIGHT] " + t("Could not import {name}: {error}", name=f"{module_name}.{class_name}", error=exc))

    if current_object is not None:
        return type(current_object)

    return None


def list_text_encoder_components(pipe):
    """
    Detecta todos los text encoders presentes en el pipeline.
    """
    names = []

    # Desde model_index.json
    for key in read_model_index_components(MODEL_ID).keys():
        if key.lower().startswith("text_encoder") and key not in names:
            names.append(key)

    # Por si existe alguna instancia directa en el pipeline
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if hasattr(pipe, attr) and attr not in names:
            names.append(attr)

    return names


def unload_pipeline_component(pipe, name):
    """
    Libera de RAM un componente del pipeline antes de recargarlo.
    """
    obj = getattr(pipe, name, None)

    if obj is not None:
        try:
            setattr(pipe, name, None)
        except Exception:
            pass

        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cpu_ram_limit_gb():
    """
    Límite de RAM que dejamos usar a Accelerate para offload.
    En modo baja RAM hay que ser mucho más conservador.
    """
    total = SYSTEM_RAM_GB or 32.0

    if LOW_RAM_MODE:
        # Para 32 GB o menos: dejar ~35% como máximo, con tope 12 GB.
        # Si tienes 16 GB, esto dejaría ~5-6 GB.
        return max(2.0, min(12.0, total * 0.35))

    # Para máquinas con más RAM.
    return max(8.0, min(24.0, total * 0.45))


def auto_max_memory_for_cuda():
    """
    Limita la memoria máxima usada por accelerate en modo auto.
    En baja RAM también limita agresivamente la CPU para favorecer
    que accelerate/transformers puedan usar offload a disco.
    """
    try:
        free, total = gpu_stats.mem_get_info()
        free_gb = free / (1024**3)

        # Dejamos margen para CUDA, VAE, activaciones y el propio Python.
        gpu_limit_gb = max(1.0, free_gb - 1.5)
    except Exception:
        gpu_limit_gb = 6.0

    cpu_limit_gb = _cpu_ram_limit_gb()

    return {
        0: f"{gpu_limit_gb:.1f}GiB",
        "cpu": f"{cpu_limit_gb:.1f}GiB",
    }


def load_precache_pipeline_light(model_id, skip_text_encoders=True):
    """
    Carga el pipeline en CPU evitando cargar de golpe los text encoders.
    Esto reduce muchísimo el primer pico de RAM.
    """
    overrides = {
        "transformer": None,
    }

    components = read_model_index_components(model_id)

    # No cargar text encoders originales si vamos a cuantizarlos después
    if skip_text_encoders:
        for name in components.keys():
            if name.lower().startswith("text_encoder"):
                overrides[name] = None

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **overrides
        )
        return pipe

    except TypeError as exc:
        print("[LIGHT] " + t("Overrides not accepted by the pipeline ({error}).", error=exc))
        print("[LIGHT] " + t("Fallback: loading without the transformer only."))
        return DiffusionPipeline.from_pretrained(
            model_id,
            transformer=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )


def quantize_one_text_encoder_4bit(pipe, component_name):
    """
    Cuantiza UN text encoder concreto:
    - libera antes el original de RAM
    - intenta cargarlo 4-bit directo a VRAM si no estamos en LOW_RAM
    - si falla, intenta auto con offload
    - en LOW_RAM evita el fallback BF16 completo en CPU como primera opción
    """
    current_object = getattr(pipe, component_name, None)

    component_class = resolve_component_class(
        MODEL_ID,
        component_name,
        current_object
    )

    if component_class is None:
        print("[4bit] " + t("Could not resolve the class of {name}.", name=component_name))
        return False, None

    # CLAVE: liberar el original antes de cargar el cuantizado
    print("[4bit] " + t("Freeing {name} from RAM before quantizing...", name=component_name))
    unload_pipeline_component(pipe, component_name)

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        print("[4bit] " + t("BitsAndBytesConfig not available: {error}", error=exc))
        return False, None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    offload_dir = os.path.join(".", "_offload_tmp_{}".format(component_name))
    os.makedirs(offload_dir, exist_ok=True)

    strategies = []

    # En baja RAM evitamos intentar meter todo directamente en CUDA de golpe.
    # Es preferible device_map auto con límites de CPU/GPU.
    if not LOW_RAM_MODE:
        strategies.append(
            (
                "cuda:0 directo",
                {
                    "device_map": "cuda:0",
                },
            )
        )

    strategies.append(
        (
            "auto 4bit con offload",
            {
                "device_map": "auto",
                "max_memory": auto_max_memory_for_cuda(),
                "offload_folder": offload_dir,
            },
        )
    )

    # text_encoder_NF4/ es este mismo text encoder ya cuantizado con bnb_cfg (salidas idénticas):
    # se carga tal cual, sin leer ni descargar el FP32 de ~49 GB.
    prequantized = component_name + "_NF4"
    if os.path.exists(os.path.join(MODEL_ID, prequantized, "config.json")):
        # LTX-2.5: Gemma 4 con solo las capas de texto en VRAM (~5.6 GB): cabe en una tarjeta de 8 GB.
        try:
            import ltx25_model
            setattr(pipe, component_name, ltx25_model.load_text_encoder_lean(os.path.join(MODEL_ID, prequantized)))
            print("[4bit] " + t("Current VRAM: {gb:.2f} GB", gb=vram_gb()))
            return True, "lean NF4"
        except Exception as exc:
            print("[4bit] " + t("{name} failed with {label}: {error}", name=component_name, label="lean NF4", error=exc))
            gc.collect()
            torch.cuda.empty_cache()
        base_kwargs = {"subfolder": prequantized}
    else:
        base_kwargs = {"subfolder": component_name, "quantization_config": bnb_cfg}

    for label, extra_kwargs in strategies:
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            **base_kwargs,
        }
        kwargs.update(extra_kwargs)

        try:
            print("[4bit] " + t("{name} -> trying {label}...", name=component_name, label=label))

            new_object = component_class.from_pretrained(
                MODEL_ID,
                **kwargs
            )

            setattr(pipe, component_name, new_object)

            print("[4bit] " + t("{name} quantized OK with {label}.", name=component_name, label=label))
            print("[4bit] " + t("Current VRAM: {gb:.2f} GB", gb=vram_gb()))

            # No borramos la carpeta si la estrategia usó offload,
            # porque podría contener datos necesarios durante la ejecución.
            if os.path.exists(offload_dir) and "offload" not in label.lower():
                try:
                    shutil.rmtree(offload_dir, ignore_errors=True)
                except Exception:
                    pass

            return True, label

        except Exception as exc:
            print("[4bit] " + t("{name} failed with {label}: {error}", name=component_name, label=label, error=exc))
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------------
    # En baja RAM NO cargamos directamente el text encoder BF16 completo en CPU,
    # porque eso puede pedir decenas de GB y causar OOM.
    #
    # En su lugar intentamos device_map="auto" con límite de CPU y offload a disco.
    # Será más lento, pero evita el pico de RAM.
    # ------------------------------------------------------------------------
    if LOW_RAM_MODE:
        try:
            print("[4bit] " + t("{name} -> LOW_RAM fallback: bf16 with device_map auto + disk offload...", name=component_name))

            fallback_object = component_class.from_pretrained(
                MODEL_ID,
                subfolder=component_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=auto_max_memory_for_cuda(),
                offload_folder=offload_dir,
            )

            setattr(pipe, component_name, fallback_object)

            print("[4bit] " + t("{name} loaded in bf16 auto/disk offload mode (slow but low on RAM).", name=component_name))

            return True, "bf16 auto/disk offload"

        except Exception as exc:
            print("[4bit] " + t("LOW_RAM fallback failed for {name}: {error}", name=component_name, error=exc))

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Último recurso opcional. Puede volver a causar pico alto de RAM,
            # pero si el sistema tiene pagefile/swap suficiente podría funcionar.
            if LOW_RAM_ALLOW_CPU_FALLBACK:
                try:
                    print("[4bit] " + t("{name} -> LOW_RAM last resort: full bf16 on CPU...", name=component_name))

                    fallback_object = component_class.from_pretrained(
                        MODEL_ID,
                        subfolder=component_name,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )

                    setattr(pipe, component_name, fallback_object)

                    print("[4bit] " + t("{name} loaded on CPU as a last resort.", name=component_name))

                    return True, "bf16 CPU fallback"

                except Exception as exc2:
                    print("[4bit] " + t("CRITICAL: CPU last resort failed for {name}: {error}", name=component_name, error=exc2))

            return False, None

    # En máquinas con suficiente RAM sí podemos mantener el fallback original.
    try:
        print("[4bit] " + t("{name} -> reloading the original bf16 on CPU as a fallback...", name=component_name))

        fallback_object = component_class.from_pretrained(
            MODEL_ID,
            subfolder=component_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        setattr(pipe, component_name, fallback_object)

    except Exception as exc:
        print("[4bit] " + t("CRITICAL: could not restore {name}: {error}", name=component_name, error=exc))

    return False, None


def try_quantize_text_encoders_4bit(pipe):
    """
    Cuantiza TODOS los text encoders detectados, uno detrás de otro.
    Devuelve:
    - True/False si todos se cuantizaron correctamente
    - text_device recomendado
    """
    names = list_text_encoder_components(pipe)

    if not names:
        print("[4bit] " + t("No text encoders detected; skipped."))
        return False, "cpu"

    print("[4bit] " + t("Text encoders detected: {names}", names=", ".join(names)))

    all_ok = True

    for name in names:
        ok, label = quantize_one_text_encoder_4bit(pipe, name)

        if not ok:
            all_ok = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_ok:
        print("[4bit] " + t("Some text encoder could not be quantized; fallback/offload will be used."))
        return False, "cpu"

    return True, "cuda"


def _patch_module_to_noop_device(module):
    """
    Neutraliza module.to(...) para cambios de DEVICE (no de dtype).
    Evita que encode_prompt haga text_encoder.to("cuda") y dispare OOM
    cuando usamos offload secuencial (cuyos hooks ya mueven las hojas).
    """
    if module is None:
        return

    if getattr(module, "_ltx_to_patched", False):
        return

    orig_to = module.to

    def _to(*args, **kwargs):
        dtype = kwargs.get("dtype", None)

        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a

        # Si piden dtype, lo aplicamos SIN tocar el device.
        if dtype is not None:
            try:
                return orig_to(dtype=dtype)
            except Exception:
                return module

        # Cambio de device -> ignorado (los hooks de offload lo gestionan).
        return module

    module.to = _to
    module._ltx_to_patched = True


def setup_offload(pipe, mode):
    """
    Configura la ubicación de los componentes y devuelve el device de texto
    que debe usarse en encode_prompt ("cuda" o "cpu").
    """
    mode = (mode or "sequential").lower()

    vae = getattr(pipe, "vae", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    connectors = getattr(pipe, "connectors", None)

    if mode == "none":
        # Comportamiento original: todo en VRAM (solo con VRAM de sobra).
        pipe.to("cuda")
        print("[OFFLOAD] " + t("none -> whole pipeline in VRAM."))
        print("[VRAM] " + t("after pipe.to(cuda): {gb:.2f} GB", gb=vram_gb()))
        return "cuda"

    if mode == "cpu":
        # Text encoder en CPU (0 VRAM de texto), VAE en VRAM (cabe).
        if vae is not None:
            vae.to("cuda")

        print("[OFFLOAD] " + t("cpu -> text encoder on CPU, VAE in VRAM."))
        print("[VRAM] " + t("after moving the VAE: {gb:.2f} GB", gb=vram_gb()))
        return "cpu"

    if mode == "model":
        try:
            pipe.enable_model_cpu_offload(device="cuda")
            print("[OFFLOAD] " + t("model -> one component in VRAM at a time."))
            print("[OFFLOAD] " + t("(only valid if each component fits in VRAM on its own)"))
            return "cuda"
        except Exception as e:
            print("[OFFLOAD] " + t("model failed, falling back to sequential:"), e)
            mode = "sequential"

    # mode == "sequential"
    try:
        pipe.enable_sequential_cpu_offload(device="cuda")

        # Blindar todos los text encoders y connectors contra .to("cuda") internos.
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))

        return "cuda"

    except Exception as e:
        print("[OFFLOAD] " + t("sequential failed, falling back to cpu:"), e)

        if vae is not None:
            vae.to("cuda")

        return "cpu"


def json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype)
        }

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def atomic_json(data, path):
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def bucket_size(width, height, area=None):
    ar = width / height

    bh = math.sqrt((area or TARGET_AREA) / ar)
    bw = ar * bh

    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)

    if max(bw, bh) > MAX_SIDE:
        scale = MAX_SIDE / max(bw, bh)

        bw = max(MULTIPLE, int(bw * scale) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * scale) // MULTIPLE * MULTIPLE)

    return int(bw), int(bh)


def read_prompt(base_name):
    path = os.path.join(DATASET_PATH, base_name + ".txt")

    prompt = ""

    if os.path.exists(path):
        prompt = read_text_file(path).strip()

    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")

    return prompt


def save_prompt_result(result, prefix):
    def recurse(obj, path):
        if torch.is_tensor(obj):
            safe_path = path.replace(".", "_").replace("/", "_").replace("\\", "_")
            filename = f"{prefix}_{safe_path}.pt"

            torch.save(obj.detach().cpu(), os.path.join(CACHE_DIR, filename))

            return {
                "type": "tensor",
                "file": filename,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }

        if isinstance(obj, dict):
            return {
                "type": "dict",
                "items": {
                    str(k): recurse(v, f"{path}_{k}")
                    for k, v in obj.items()
                }
            }

        if isinstance(obj, (tuple, list)):
            return {
                "type": "tuple" if isinstance(obj, tuple) else "list",
                "items": [
                    recurse(v, f"{path}_{i}")
                    for i, v in enumerate(obj)
                ]
            }

        return {
            "type": "value",
            "value": json_safe(obj)
        }

    structure = recurse(result, "root")
    atomic_json(structure, os.path.join(CACHE_DIR, f"{prefix}_structure.json"))

    return structure


def extract_prompt_tensors(result):
    found = []

    def recurse(obj, path="root"):
        if torch.is_tensor(obj):
            found.append((path, obj.detach().cpu()))
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, f"{path}.{k}")
            return

        if isinstance(obj, (tuple, list)):
            for i, v in enumerate(obj):
                recurse(v, f"{path}.{i}")
            return

    recurse(result)
    return found


def encode_prompt(pipe, prompt, text_device):
    result = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        max_sequence_length=MAX_SEQ_LEN,
        device=torch.device(text_device),
        dtype=torch.bfloat16,
    )

    return result


def encode_video_latent(vae, image):
    image_tensor = F_vision.pil_to_tensor(image).float() / 127.5 - 1.0
    return encode_pixels(vae, image_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1))


def encode_pixels(vae, pixels):
    """[1, 3, F, H, W] en [-1, 1] -> latente normalizado [1, 128, (F-1)/8+1, H/32, W/32] en CPU."""
    image_tensor = pixels.to("cuda", dtype=torch.bfloat16)

    encoded = vae.encode(image_tensor)

    if hasattr(encoded, "latent_dist"):
        latent = encoded.latent_dist.sample()
    elif torch.is_tensor(encoded):
        latent = encoded
    elif isinstance(encoded, tuple):
        latent = encoded[0]
    else:
        raise RuntimeError(t("Unknown output of VAE.encode(): {name}", name=type(encoded)))

    latent = latent.detach()

    # IMPORTANTE: el VAE de LTX-2.5 (AutoencoderKLLTX2) no produce un
    # espacio latente de varianza ~1. El pipeline oficial SIEMPRE
    # normaliza el latente crudo antes de pasarlo al transformer:
    #
    #   latents = (latents - latents_mean) * scaling_factor / latents_std
    #
    # Sin este paso, el transformer (preentrenado sobre latentes YA
    # normalizados) recibe un "clean" con escala/varianza equivocada.
    latents_mean = getattr(vae, "latents_mean", None)
    latents_std = getattr(vae, "latents_std", None)

    if latents_mean is not None and latents_std is not None:
        latents_mean = latents_mean.to(
            device=latent.device,
            dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)

        latents_std = latents_std.to(
            device=latent.device,
            dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)

        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

        latent = (latent - latents_mean) * scaling_factor / latents_std

    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def make_audio_latent(video_latent, audio_channels):
    return torch.zeros(
        (video_latent.shape[0], audio_channels, 1),
        dtype=torch.bfloat16
    )


# ============================================================================
# MAIN
# ============================================================================
def preprocess_ltx25():
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError(t("CUDA is not available."))

    # ------------------------------------------------------------------
    # Descargar modelo antes de buscar la carpeta local
    # ------------------------------------------------------------------
    MODEL_ID = ensure_ltx25_model_downloaded(MODEL_ID)

    # Perfil de VRAM (el mismo del entrenamiento): simula una tarjeta menor con un tope duro.
    try:
        prof = str(cfg_get("vram_profile", "auto")).strip().lower()
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if prof not in ("", "auto") and float(prof) < total:
            usable = float(prof) - 1.0
            torch.cuda.set_per_process_memory_fraction(usable / total, 0)
            print("[VRAM] " + t("Profile: {cap:.0f} GB card simulated on a {gb:.1f} GB GPU (hard cap {u:.2f} GB for PyTorch).",
                                cap=float(prof), gb=total, u=usable))
    except ValueError:
        pass

    if not os.path.isdir(MODEL_ID):
        raise FileNotFoundError(t("Model not found: {path}", path=MODEL_ID))

    if not os.path.exists(os.path.join(MODEL_ID, "model_index.json")):
        raise FileNotFoundError(t("{name} missing in: {path}. The download of the {model} model may be incomplete.", name="model_index.json", path=MODEL_ID, model="base"))

    if not os.path.exists(os.path.join(MODEL_ID, "index.json")):
        raise FileNotFoundError(t("{name} missing in: {path}. The download of the {model} model may be incomplete.", name="index.json", path=MODEL_ID, model="NF4"))

    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # "images" son todas las muestras: imágenes y clips de vídeo.
    images = sorted(
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp") + ltx25_model.VIDEO_EXTS)
    )

    # Edición: cada muestra es un par; se recorre el "after" (objetivo) y la referencia sale de REF_FILES.
    REF_FILES = {}
    if LORA_TYPE == "edit":
        stems = {os.path.splitext(f)[0]: f for f in images}
        pairs = [(s[:-len("_after")], f) for s, f in stems.items()
                 if s.endswith("_after") and s[:-len("_after")] + "_before" in stems]
        if not pairs:
            raise RuntimeError(t("Edit LoRA but there are no name_before / name_after pairs in '{path}'.", path=DATASET_PATH))
        REF_FILES = {name: stems[name + "_before"] for name, _ in pairs}
        loose = len(images) - 2 * len(pairs)
        images = sorted(f for _, f in pairs)
        print(t("Edit LoRA: {n} before/after pairs", n=len(images)) +
              (" · " + t("{n} files outside any pair are ignored", n=loose) if loose else "") +
              " · reference downscale x{}".format(REF_DOWNSCALE))

    def sample_base(filename):
        stem = os.path.splitext(filename)[0]
        return stem[:-len("_after")] if LORA_TYPE == "edit" and stem.endswith("_after") else stem

    if not images:
        raise RuntimeError(t("No images found in '{path}'. Please add images.", path=DATASET_PATH))

    print()
    print("=" * 80)
    print(" LTX-2.5 PRE-CACHE")
    print("=" * 80)
    print(f"{t('Model'):<13}:", os.path.abspath(MODEL_ID))
    print("Dataset      :", os.path.abspath(DATASET_PATH))
    print("Cache        :", os.path.abspath(CACHE_DIR))
    print(f"{t('Target Area'):<13}:", TARGET_AREA)
    print(f"{t('Multiple'):<13}:", MULTIPLE)
    print("Frames       :", NUM_FRAMES)
    print("FPS          :", FRAME_RATE)
    print("Max seq len  :", MAX_SEQ_LEN)
    print("Offload mode :", PRECACHE_OFFLOAD)
    print("TextEnc 4bit :", TEXT_ENCODER_4BIT)
    print("Low RAM mode :", LOW_RAM_MODE)

    if SYSTEM_RAM_GB is not None:
        print("System RAM   : {:.1f} GB".format(SYSTEM_RAM_GB))
    else:
        print("System RAM   :", t("unknown"))

    print("=" * 80)

    # Canales de audio desde disco (sin cargar el transformer).
    audio_channels = read_audio_channels(MODEL_ID, 128)
    print("Audio latent channels:", audio_channels)

    # ------------------------------------------------------------------
    # Cargar pipeline en CPU (SIN .to("cuda")).
    # ------------------------------------------------------------------
    print()

    if TEXT_ENCODER_4BIT:
        print(t("Loading LTX-2.5 on CPU (without the transformer and WITHOUT text encoders)..."))
    else:
        print(t("Loading LTX-2.5 on CPU (without the transformer)..."))

    pipe = load_precache_pipeline_light(
        MODEL_ID,
        skip_text_encoders=TEXT_ENCODER_4BIT,
    )

    print("Pipeline:", type(pipe).__name__)
    print("VAE:", type(getattr(pipe, "vae", None)).__name__)

    for te_name in list_text_encoder_components(pipe):
        te_obj = getattr(pipe, te_name, None)

        if te_obj is None:
            print("{}: None ({})".format(te_name, t("will be loaded quantized")))
        else:
            print("{}: {}".format(te_name, type(te_obj).__name__))

    print("Audio VAE:", type(getattr(pipe, "audio_vae", None)).__name__)

    # ------------------------------------------------------------------
    # Opcional: cuantizar TODOS los text encoders a 4-bit.
    # ------------------------------------------------------------------
    used_4bit = False
    text_device = "cpu"

    if TEXT_ENCODER_4BIT:
        used_4bit, text_device = try_quantize_text_encoders_4bit(pipe)

    # ------------------------------------------------------------------
    # Offload.
    # ------------------------------------------------------------------
    if used_4bit:
        # El VAE NO sube todavía: textos y latentes van en dos pasadas para que el pico quepa en 8 GB
        # (Gemma 4 NF4 ~7.7 GB y VAE ~1.5 GB no caben juntos).

        # Evitar que el pipeline intente moverlos de golpe a CUDA más adelante
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))

        text_device = "cuda"

        print("[OFFLOAD] " + t("4bit on -> 4-bit text encoder in VRAM; the VAE comes after the texts."))
        print("[VRAM] " + t("after 4bit: {gb:.2f} GB", gb=vram_gb()))

    else:
        text_device = setup_offload(pipe, PRECACHE_OFFLOAD)

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # NEGATIVE PROMPT
    # ------------------------------------------------------------------
    print("\n" + t("Encoding negative/empty prompt..."))

    with torch.inference_mode():
        neg_result = encode_prompt(pipe, "", text_device)

    save_prompt_result(neg_result, "_neg")

    print("Negative prompt tensors:")
    for path, tensor in extract_prompt_tensors(neg_result):
        print(" ", path, tuple(tensor.shape), tensor.dtype)

    print("[VRAM] " + t("peak after the negative prompt: {gb:.2f} GB", gb=vram_peak_gb()))

    free_vram(neg_result)

    # ------------------------------------------------------------------
    # CUSTOM PROMPT
    # ------------------------------------------------------------------
    if PREVIEW_CUSTOM_PROMPT:
        custom_prompt = PREVIEW_CUSTOM_PROMPT

        if TRIGGER_WORD and TRIGGER_WORD.lower() not in custom_prompt.lower():
            custom_prompt = f"{TRIGGER_WORD}, {custom_prompt}".strip(", ")

        print("\n" + t("Encoding: '{prompt}'", prompt=custom_prompt))

        with torch.inference_mode():
            custom_result = encode_prompt(pipe, custom_prompt, text_device)

        save_prompt_result(custom_result, "_custom")
        free_vram(custom_result)

    # ------------------------------------------------------------------
    # DATASET, pasada 1: textos (Gemma 4 en VRAM, VAE fuera)
    # ------------------------------------------------------------------
    structures = {}
    for idx, filename in enumerate(images, start=1):
        base = sample_base(filename)
        prompt_structure_path = os.path.join(CACHE_DIR, f"{base}_prompt_structure.json")
        prompt = read_prompt(base)
        if os.path.exists(prompt_structure_path):
            with open(prompt_structure_path, "r", encoding="utf-8") as f:
                structures[base] = (prompt, json.load(f))
            continue
        print(f"[{idx}/{len(images)}] " + t("Text") + f": {filename}")
        print("Prompt:", prompt)
        with torch.inference_mode():
            prompt_result = encode_prompt(pipe, prompt, text_device)
        structures[base] = (prompt, save_prompt_result(prompt_result, f"{base}_prompt"))
        for path, tensor in extract_prompt_tensors(prompt_result):
            print(" ", path, tuple(tensor.shape), tensor.dtype)
        free_vram(prompt_result)
    print("[VRAM] " + t("peak after the texts: {gb:.2f} GB", gb=vram_peak_gb()))

    # Fuera el text encoder antes de subir el VAE.
    for attr in list_text_encoder_components(pipe):
        unload_pipeline_component(pipe, attr)
    free_vram()
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        vae.to("cuda")
    audio_vae = getattr(pipe, "audio_vae", None)
    if audio_vae is not None:
        audio_vae.to("cuda", dtype=torch.float32)   # 0.1 GB; en FP32 como ltx-trainer
    print("[VRAM] " + t("VAE in VRAM: {gb:.2f} GB", gb=vram_gb()))

    # ------------------------------------------------------------------
    # DATASET, pasada 2: latentes (VAE)
    # ------------------------------------------------------------------
    for idx, filename in enumerate(images, start=1):
        base = sample_base(filename)

        video_path = os.path.join(CACHE_DIR, f"{base}_video_latent.pt")
        audio_path = os.path.join(CACHE_DIR, f"{base}_audio_latent.pt")
        info_path = os.path.join(CACHE_DIR, f"{base}_info.json")

        if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(info_path):
            print(f"[{idx}/{len(images)}] {t('SKIP')} {filename}")
            continue

        print()
        print("=" * 80)
        print(f"[{idx}/{len(images)}] {filename}")

        src = os.path.join(DATASET_PATH, filename)
        is_video = filename.lower().endswith(ltx25_model.VIDEO_EXTS)
        if is_video:
            info = ltx25_model.video_info(src)
            if info is None:
                print("[!] " + t("Unreadable video, skipped: {name}", name=filename))
                continue
            frames = ltx25_model.clip_frames(info["duration"], MAX_FRAMES)
            bw, bh = bucket_size(info["width"], info["height"], VIDEO_TARGET_AREA)
            print("Bucket:", f"{bw}x{bh}", "·", t("{n} frames at 24 fps ({s:.2f} s of {d:.2f} s)",
                                                   n=frames, s=frames / 24.0, d=info["duration"]))
            pixels = ltx25_model.read_video(src, frames, bw, bh)
            pixels = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
            print(t("Encoding video latent..."))
            # Tiles en el VAE solo para clips (en 8 GB un clip entero no cabe de una vez); las imágenes como antes.
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            try:
                with torch.inference_mode():
                    video_latent = encode_pixels(pipe.vae, pixels)
            finally:
                if hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
            del pixels
        else:
            frames = NUM_FRAMES
            image = Image.open(src).convert("RGB")
            bw, bh = bucket_size(image.width, image.height)
            scale = max(bw / image.width, bh / image.height)
            image = image.resize(
                (math.ceil(image.width * scale), math.ceil(image.height * scale)),
                Image.LANCZOS,
            )
            left = (image.width - bw) // 2
            top = (image.height - bh) // 2
            image = image.crop((left, top, left + bw, top + bh))
            print("Bucket:", f"{bw}x{bh}")

            print(t("Encoding video latent..."))
            with torch.inference_mode():
                video_latent = encode_video_latent(pipe.vae, image)
        torch.save(video_latent, video_path)

        # Referencia del par (edición): el "before" al tamaño del objetivo (o 1/REF_DOWNSCALE de lado).
        ref_info = None
        if base in REF_FILES:
            ref_src = os.path.join(DATASET_PATH, REF_FILES[base])
            rw = max(32, (bw // REF_DOWNSCALE) // 32 * 32)
            rh = max(32, (bh // REF_DOWNSCALE) // 32 * 32)
            if ref_src.lower().endswith(ltx25_model.VIDEO_EXTS):
                rinfo = ltx25_model.video_info(ref_src)
                rframes = min(frames, ltx25_model.clip_frames(rinfo["duration"], MAX_FRAMES)) if rinfo else 1
                rpix = ltx25_model.read_video(ref_src, rframes, rw, rh)
                rpix = torch.from_numpy(rpix.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
            else:
                rimg = Image.open(ref_src).convert("RGB")
                sc = max(rw / rimg.width, rh / rimg.height)
                rimg = rimg.resize((math.ceil(rimg.width * sc), math.ceil(rimg.height * sc)), Image.LANCZOS)
                l, tp = (rimg.width - rw) // 2, (rimg.height - rh) // 2
                rimg = rimg.crop((l, tp, l + rw, tp + rh))
                rpix = (F_vision.pil_to_tensor(rimg).float() / 127.5 - 1.0).unsqueeze(0).unsqueeze(2)
            if hasattr(pipe.vae, "enable_tiling") and rpix.shape[2] > 1:
                pipe.vae.enable_tiling()
            try:
                with torch.inference_mode():
                    ref_latent = encode_pixels(pipe.vae, rpix)
            finally:
                if hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
            torch.save(ref_latent, os.path.join(CACHE_DIR, f"{base}_ref_latent.pt"))
            ref_info = {"file": REF_FILES[base], "downscale": REF_DOWNSCALE, "shape": list(ref_latent.shape)}
            print("Reference:", REF_FILES[base], f"{rw}x{rh}", tuple(ref_latent.shape))
            del rpix
        print("Video latent:", tuple(video_latent.shape))

        # Audio: latente de la duración del clip (25 por segundo, como LTX2Pipeline). Con pista se codifica la pista;
        # un clip mudo lleva el latente de un silencio de verdad. Las imágenes, un latente vacío de 1.
        has_audio = False
        if is_video and audio_vae is not None:
            seconds = frames / 24.0
            n_lat = ltx25_model.audio_latent_length(frames)
            wave = ltx25_model.read_audio(src, 0.0, seconds) if (USE_AUDIO and info.get("has_audio")) else None
            has_audio = wave is not None and float(abs(wave).max()) > 1e-4
            if not has_audio:
                wave = torch.zeros((2, int(round(seconds * ltx25_model.AUDIO_SR)))).numpy()
            audio_latent = ltx25_model.encode_audio(audio_vae, wave)
            if audio_latent.shape[-1] < n_lat:
                audio_latent = torch.nn.functional.pad(audio_latent, (0, n_lat - audio_latent.shape[-1]))
            audio_latent = audio_latent[..., :n_lat].contiguous()
            print(t("Audio") + ":", t("clip track") if has_audio else t("silence (no track)"))
        else:
            audio_latent = torch.zeros((1, audio_channels, ltx25_model.audio_latent_length(frames) if is_video else 1),
                                       dtype=torch.bfloat16)
        torch.save(audio_latent, audio_path)
        print("Audio latent:", tuple(audio_latent.shape))
        print("[VRAM] " + t("accumulated peak: {gb:.2f} GB", gb=vram_peak_gb()))

        prompt, structure = structures[base]
        atomic_json(
            {
                "filename": filename,
                "width": bw,
                "height": bh,
                "num_frames": frames,
                "media": "video" if is_video else "image",
                "has_audio": bool(has_audio),
                "reference": ref_info,
                "frame_rate": FRAME_RATE,
                "prompt": prompt,
                "video_latent": os.path.basename(video_path),
                "audio_latent": os.path.basename(audio_path),
                "prompt_structure": structure,
            },
            info_path,
        )

        free_vram(audio_latent, video_latent)

    # ------------------------------------------------------------------
    # CACHE INFO
    # ------------------------------------------------------------------
    cache_info = {
        "format": "LTX25-LoRA-Precache",
        "version": 1,
        "model_id": MODEL_ID,
        "dataset_path": DATASET_PATH,
        "cache_dir": CACHE_DIR,
        "target_area": TARGET_AREA,
        "multiple": MULTIPLE,
        "frame_rate": FRAME_RATE,
        "num_frames": NUM_FRAMES,
        "max_sequence_length": MAX_SEQ_LEN,
        "trigger_word": TRIGGER_WORD,
        "audio_latent_channels": audio_channels,
        "precache_offload": PRECACHE_OFFLOAD,
        "text_encoder_4bit": bool(used_4bit),
        "low_ram_mode": LOW_RAM_MODE,
        "low_ram_threshold_gb": LOW_RAM_THRESHOLD_GB,
        "system_ram_gb": SYSTEM_RAM_GB,
        "prompt_encoding": "LTX2Pipeline.encode_prompt",
        "video_target_area": VIDEO_TARGET_AREA,
        "max_frames": MAX_FRAMES,
        "use_audio": USE_AUDIO,
        "lora_type": LORA_TYPE,
        "reference_downscale": REF_DOWNSCALE,
        "note": "Images and video clips (24 fps, 8n+1 frames). Clip audio encoded with the audio VAE (ltx-core mel).",
    }

    atomic_json(cache_info, os.path.join(CACHE_DIR, "cache_info.json"))

    free_vram(pipe)

    print()
    print("=" * 80)
    print(t("LTX-2.5 PRE-CACHE COMPLETED"))
    print("=" * 80)
    print("Cache:", os.path.abspath(CACHE_DIR))
    print(t("Images") + ":", len(images))
    print(t("Total VRAM peak: {gb:.2f} GB", gb=vram_peak_gb()))
    print("=" * 80)


if __name__ == "__main__":
    try:
        preprocess_ltx25()
    except Exception:
        print()
        print("=" * 80)
        print(t("ERROR IN LTX-2.5 PRE-CACHE"))
        print("=" * 80)
        traceback.print_exc()
        raise