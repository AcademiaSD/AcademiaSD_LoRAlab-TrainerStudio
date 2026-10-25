# -*- coding: utf-8 -*-
"""Local Boogu model download and differentiable NF4 cache for LoRAlab."""
import contextlib
import copy
import gc
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

from i18n import t
from project_layout import resolve_model_dir

# Los modelos de Boogu siempre se cargan de NF4_DIR: DiffSynth no debe intentar bajarlos de ModelScope.
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
# Aviso de bitsandbytes en cada capa NF4 de 3360 de ancho: no es accionable y llena la consola.
warnings.filterwarnings("ignore", message="inner dimension")


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_REPO = "Boogu/Boogu-Image-0.1-Edit"
NF4_REPO = "AcademiaSD/Boogu-Image-Edit-NF4-for-LoRA-Training"
RAW_DIR = resolve_model_dir("Boogu-image-edit-Raw", "Boogu-image-edit-Raw")
NF4_DIR = resolve_model_dir("Boogu-image-edit-NF4", "Boogu-image-edit-NF4")
NF4_TRANSFORMER = NF4_DIR / "transformer" / "model.safetensors"
NF4_TEXT_ENCODER = NF4_DIR / "mllm" / "model.safetensors"
NF4_METADATA = NF4_DIR / "boogu_nf4_metadata.json"
COMPONENTS = (
    ("boogu_image_dit", "transformer", NF4_TRANSFORMER),
    ("qwen_image_21_text_encoder", "mllm", NF4_TEXT_ENCODER),
)
REPOSITORY_DIRS = ("assets", "mllm", "processor", "scheduler", "transformer", "vae")
REPOSITORY_FILES = (".gitattributes", "README.md", "README_CN.md", "model_index.json")


def quiet():
    # DiffSynth imprime su cuadro de descargas y cada modelo que carga; en la consola de la GUI solo es ruido.
    return contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8"))


def _read_token():
    path = BASE_DIR / "settings" / "HF_token.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("token", "").strip() or None
    except (OSError, json.JSONDecodeError):
        return None


def _configure_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _weight_files(folder):
    files = sorted(Path(folder).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No se encontraron pesos .safetensors en {folder}")
    return [str(path) for path in files]


def ensure_raw_model(log=print):
    from huggingface_hub import snapshot_download

    log(f"[Boogu] Descargando el modelo raw completo en {RAW_DIR}", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(RAW_DIR),
        token=_read_token(),
        max_workers=4,
    )
    required = [RAW_DIR / name for name in REPOSITORY_DIRS + REPOSITORY_FILES]
    required.extend((RAW_DIR / "transformer" / "config.json", RAW_DIR / "mllm" / "config.json"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("La descarga raw de Boogu no coincide con el repositorio completo; faltan: "
                           + ", ".join(missing))
    for folder in (RAW_DIR / "transformer", RAW_DIR / "mllm", RAW_DIR / "vae"):
        _weight_files(folder)
    # `snapshot_download(local_dir=...)` stages unfinished transfers under this
    # helper directory. Remove it only after every repository path and weight is
    # present so the completed raw folder mirrors Boogu's published file tree.
    download_cache = RAW_DIR / ".cache"
    if download_cache.exists():
        shutil.rmtree(download_cache)
    return RAW_DIR


def _raw_registry_hashes():
    from diffsynth.core.loader import hash_model_file
    from diffsynth.models import model_loader

    hashes = {}
    configs = list(model_loader.MODEL_CONFIGS)
    for model_name, raw_folder, _ in COMPONENTS:
        raw_hash = hash_model_file(_weight_files(RAW_DIR / raw_folder))
        if not any(item.get("model_hash") == raw_hash and item.get("model_name") == model_name
                   for item in configs):
            raise RuntimeError(
                f"DiffSynth no reconoce los pesos raw de {model_name}; no se iniciará la conversión NF4."
            )
        hashes[model_name] = raw_hash
    return hashes


def _convert_component(raw_files, model_name, output_file, log=print):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from diffsynth.core import ModelConfig
    from diffsynth.core.quant import QuantizeConfig
    from diffsynth.pipelines.boogu_image import BooguImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("La conversión NF4 local de Boogu necesita una GPU CUDA disponible.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"[Boogu NF4] Cargando {model_name} desde los pesos raw locales", flush=True)
    pipe = BooguImagePipeline(device="cpu", torch_dtype=torch.bfloat16)
    pool = pipe.download_and_load_models([ModelConfig(path=raw_files, skip_download=True)])
    model = pool.fetch_model(model_name)
    if model is None:
        raise RuntimeError(f"DiffSynth no reconoció el componente Boogu {model_name}.")
    quant = QuantizeConfig(method="bitsandbytes_nf4")
    log(f"[Boogu NF4] Cuantizando {model_name} a NF4; las capas se procesan en CUDA y se guardan en RAM", flush=True)
    quant.quantize_model(model, compute_device="cuda", model_device="cpu")
    tensors, metadata = quant.flatten_state_dict(model.state_dict())
    temp = output_file.with_suffix(".safetensors.tmp")
    save_file(tensors, str(temp), metadata=metadata)
    os.replace(temp, output_file)
    with safe_open(str(output_file), framework="pt", device="cpu") as converted:
        if not converted.keys():
            raise RuntimeError(f"La conversión de {model_name} generó un safetensors vacío.")
    del tensors, metadata, model, pool, pipe, quant
    gc.collect()
    torch.cuda.empty_cache()
    log(f"[Boogu NF4] Guardado: {output_file}", flush=True)


def _nf4_ready():
    return (NF4_TRANSFORMER.is_file() and NF4_TEXT_ENCODER.is_file() and any((NF4_DIR / "vae").glob("*.safetensors"))
            and (NF4_DIR / "model_index.json").is_file() and NF4_METADATA.is_file())


def ensure_nf4_model(log=print):
    """Baja la carpeta NF4 ya convertida (~11 GB) en vez del modelo raw (~37 GB) más la conversión en GPU."""
    _configure_console()
    if _nf4_ready():
        return NF4_DIR
    from huggingface_hub import snapshot_download

    log(t("Downloading from Hugging Face: {repo}", repo=NF4_REPO), flush=True)
    snapshot_download(repo_id=NF4_REPO, local_dir=str(NF4_DIR), token=_read_token(), max_workers=2)
    if not _nf4_ready():
        raise RuntimeError(f"La descarga de {NF4_REPO} está incompleta: {NF4_DIR}")
    log(t("Model downloaded to: {path}", path=NF4_DIR), flush=True)
    return NF4_DIR


def convert_nf4_model(log=print):
    """Genera Boogu-image-edit-NF4 desde el modelo raw; solo para preparar el repo NF4_REPO."""
    _configure_console()
    if _nf4_ready():
        # Recover cleanly if a previous run completed conversion but was interrupted
        # before deleting its temporary source tree.
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
            log(f"[Boogu] Carpeta raw temporal eliminada: {RAW_DIR}", flush=True)
        return NF4_DIR

    ensure_raw_model(log)
    raw_hashes = _raw_registry_hashes()
    NF4_DIR.mkdir(parents=True, exist_ok=True)
    log("[Boogu] Preparando la carpeta local NF4 con VAE, processor y configuración raw", flush=True)
    def ignore_raw_weights(directory, names):
        relative = Path(directory).resolve().relative_to(RAW_DIR.resolve())
        # Fuera lo que no carga ningún script: assets y README del repo, código remoto y los índices de shards raw.
        ignored = {name for name in names if name in {".cache", ".gitattributes", "assets", "README.md", "README_CN.md"}
                   or name.endswith((".index.json", ".py"))}
        # Replace only the two large components with their NF4 weights. Keep the
        # VAE safetensors and other runtime assets in the local model folder.
        if relative.parts and relative.parts[0] in {"transformer", "mllm"}:
            ignored.update(name for name in names if name.endswith(".safetensors"))
        return ignored

    shutil.copytree(RAW_DIR, NF4_DIR, dirs_exist_ok=True, ignore=ignore_raw_weights)
    for model_name, raw_folder, output_file in COMPONENTS:
        _convert_component(_weight_files(RAW_DIR / raw_folder), model_name, output_file, log)

    required = (
        NF4_TRANSFORMER,
        NF4_TEXT_ENCODER,
        NF4_DIR / "transformer" / "config.json",
        NF4_DIR / "mllm" / "config.json",
        NF4_DIR / "model_index.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    vae_files = list((NF4_DIR / "vae").glob("*.safetensors"))
    if missing or not vae_files:
        raise RuntimeError("Conversión Boogu incompleta; se conserva la carpeta raw. Faltan: "
                           + (", ".join(missing) if missing else "pesos del VAE"))
    for path in (NF4_TRANSFORMER, NF4_TEXT_ENCODER, *vae_files):
        if path.stat().st_size == 0:
            raise RuntimeError(f"Conversión Boogu incompleta; se conserva la carpeta raw. Archivo vacío: {path}")

    metadata = {
        "format": "LoRAlab-Boogu-NF4",
        "version": 1,
        "repository": MODEL_REPO,
        "components": {
            model_name: {"raw_hash": raw_hashes[model_name], "file": str(output_file.relative_to(NF4_DIR))}
            for model_name, _, output_file in COMPONENTS
        },
    }
    temporary_metadata = NF4_METADATA.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_metadata, NF4_METADATA)

    shutil.rmtree(RAW_DIR)
    log(f"[Boogu] Conversión NF4 verificada; carpeta raw temporal eliminada: {RAW_DIR}", flush=True)
    return NF4_DIR


def register_nf4_models():
    """Teach this DiffSynth process to load the two locally prequantized files."""
    from diffsynth.core.loader import hash_model_file
    from diffsynth.models import model_loader

    metadata = json.loads(NF4_METADATA.read_text(encoding="utf-8"))
    configs = list(model_loader.MODEL_CONFIGS)
    additions = []
    for expected_name, _, nf4_file in COMPONENTS:
        raw_hash = metadata["components"][expected_name]["raw_hash"]
        raw_config = next((item for item in configs if item.get("model_hash") == raw_hash), None)
        if raw_config is None or raw_config.get("model_name") != expected_name:
            raise RuntimeError(
                f"DiffSynth no tiene un registro compatible para {expected_name}; no se cargará un cache NF4 desconocido."
            )
        entry = copy.deepcopy(raw_config)
        entry["model_hash"] = hash_model_file(str(nf4_file))
        entry.pop("state_dict_converter", None)
        entry["quant_config"] = {"method": "bitsandbytes_nf4", "load_prequantized": True}
        if not any(item.get("model_hash") == entry["model_hash"] for item in configs + additions):
            additions.append(entry)
    model_loader.MODEL_CONFIGS = tuple(configs + additions)


if __name__ == "__main__":
    convert_nf4_model()
