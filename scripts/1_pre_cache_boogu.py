# -*- coding: utf-8 -*-
"""Boogu NF4 prompt-embedding and VAE-latent pre-cache."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from boogu_dataset import CACHE_VERSION, build_metadata, cache_path, resolve_path
from boogu_model import (
    NF4_DIR, NF4_TEXT_ENCODER, NF4_TRANSFORMER, ensure_nf4_model,
    quiet, register_nf4_models,
)
from project_layout import resolve_cache_dir

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = BASE_DIR / "settings" / "pre_cache_settings_boogu.json"
MODEL_ID = "Boogu-Image-0.1-Edit"
SEQUENCE_LENGTH = 1280


def _configure_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _signature(path):
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _prompt_hash(prompt, edit_signature, mode):
    value = json.dumps([CACHE_VERSION, mode, prompt, edit_signature, SEQUENCE_LENGTH], ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_hash(signature, config):
    value = json.dumps([CACHE_VERSION, signature, config], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError):
        return {}


def _with_trigger(prompt, trigger):
    trigger = str(trigger or "").strip()
    prompt = str(prompt or "").strip()
    if trigger and trigger.casefold() not in prompt.casefold():
        return f"{trigger}, {prompt}" if prompt else trigger
    return prompt


def _resolve_edit_path(dataset, value):
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = dataset / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen de referencia de edición: {path}")
    return path


def _fit_target(image, target_area, max_side, multiple):
    ratio = min(1.0, (target_area / (image.width * image.height)) ** 0.5,
                max_side / max(image.width, image.height))
    width = max(multiple, int(image.width * ratio) // multiple * multiple)
    height = max(multiple, int(image.height * ratio) // multiple * multiple)
    while width * height > target_area and (width > multiple or height > multiple):
        if width / image.width >= height / image.height and width > multiple:
            width -= multiple
        elif height > multiple:
            height -= multiple
        else:
            break
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, width, height


def _load_cache_settings():
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig")) if CONFIG.exists() else {}
    dataset = resolve_path(config.get("dataset_path", "./dataset"), BASE_DIR)
    cache = resolve_cache_dir(config.get("cache_dir"), "./Cache_Data/cached_data_boogu", BASE_DIR)
    mode = str(config.get("lora_type", "edit")).lower()
    if mode not in ("normal", "edit"):
        raise ValueError("lora_type debe ser 'normal' o 'edit'.")
    area = max(262144, min(1048576, int(config.get("target_area", 589824))))
    multiple = max(16, int(config.get("multiple", 32)))
    max_side = max(multiple, int(config.get("max_side", 1280)))
    return config, dataset, cache, mode, area, multiple, max_side


def _encode_negative(pipe):
    from diffsynth.pipelines.boogu_image import SYSTEM_PROMPT_TI2I

    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_TI2I}]},
                {"role": "user", "content": [{"type": "text", "text": ""}]}]
    inputs = pipe.processor.apply_chat_template([messages], padding="longest", max_length=SEQUENCE_LENGTH,
                                                padding_side="right", return_tensors="pt", tokenize=True,
                                                return_dict=True)
    inputs = {k: v.to(pipe.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    hidden = pipe.text_encoder.model.model(**inputs)[0]
    return hidden.to(dtype=torch.bfloat16), inputs["attention_mask"]


def _encode_text(rows, dataset, cache, mode, trigger, device="cuda"):
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.boogu_image import BooguImagePipeline, BooguImageUnit_PromptEmbedder

    register_nf4_models()
    with quiet():
        pipe = BooguImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(path=str(NF4_TEXT_ENCODER))],
            processor_config=ModelConfig(path=str(NF4_DIR / "processor")),
        )
    pipe.load_models_to_device(["text_encoder"])
    unit = BooguImageUnit_PromptEmbedder()
    try:
        with torch.inference_mode():
            negative, negative_mask = _encode_negative(pipe)
            for index, row in enumerate(rows, 1):
                target_name = row["image"]
                reference = _resolve_edit_path(dataset, row["edit_image"]) if mode == "edit" and row.get("edit_image") else None
                reference_sig = _signature(reference) if reference else ""
                prompt = _with_trigger(row.get("prompt", ""), trigger)
                wanted_hash = _prompt_hash(prompt, reference_sig, mode)
                path = cache_path(cache, row)
                saved = _load(path)
                text_keys = {"prompt_embeds", "prompt_mask", "negative_embeds", "negative_mask"}
                if saved.get("prompt_hash") == wanted_hash and text_keys.issubset(saved):
                    print(f"[Texto {index}/{len(rows)}] {Path(target_name).name}: ya cacheado", flush=True)
                    continue
                edit_image = None
                if reference:
                    with Image.open(reference) as source:
                        edit_image = source.convert("RGB").copy()
                positive, mask = unit.encode_prompt(
                    pipe, prompt, SEQUENCE_LENGTH, edit_image=edit_image,
                )
                saved.update({
                    "cache_version": CACHE_VERSION,
                    "mode": mode,
                    "prompt_hash": wanted_hash,
                    "prompt_embeds": positive.detach().to(device="cpu", dtype=torch.bfloat16),
                    "prompt_mask": mask.detach().to(device="cpu", dtype=torch.bool),
                    "negative_embeds": negative.detach().to(device="cpu", dtype=torch.bfloat16),
                    "negative_mask": negative_mask.detach().to(device="cpu", dtype=torch.bool),
                })
                _save(path, saved)
                print(f"[Texto {index}/{len(rows)}] {Path(target_name).name}: embeddings NF4 guardados", flush=True)
                del positive, mask, edit_image
    finally:
        del unit, pipe
        gc_collect_cuda()


def gc_collect_cuda():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _encode_latents(rows, dataset, cache, mode, target_area, multiple, max_side):
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.boogu_image import BooguImagePipeline

    vae_files = sorted((NF4_DIR / "vae").glob("*.safetensors"))
    if not vae_files:
        raise FileNotFoundError(f"No se encontró el VAE local en {NF4_DIR / 'vae'}")
    with quiet():
        pipe = BooguImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(path=[str(path) for path in vae_files])],
        )
    pipe.load_models_to_device(["vae_encoder"])
    latent_config = [target_area, multiple, max_side]
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                target = dataset / row["image"]
                reference = _resolve_edit_path(dataset, row["edit_image"]) if mode == "edit" and row.get("edit_image") else None
                target_hash = _image_hash(_signature(target), latent_config)
                reference_hash = _image_hash(_signature(reference), latent_config) if reference else ""
                path = cache_path(cache, row)
                saved = _load(path)
                need_target = saved.get("target_hash") != target_hash or "input_latents" not in saved
                need_reference = bool(reference) and (
                    saved.get("reference_hash") != reference_hash or "reference_latents" not in saved
                )
                if not need_target and not need_reference:
                    print(f"[Latentes {index}/{len(rows)}] {target.name}: ya cacheados", flush=True)
                    continue
                if need_target:
                    with Image.open(target) as source:
                        image, width, height = _fit_target(
                            source.convert("RGB"), target_area, max_side, multiple,
                        )
                        pixels = pipe.preprocess_image(image, device="cuda", torch_dtype=torch.bfloat16)
                    latent = pipe.vae_encoder(pixels).detach().to(device="cpu", dtype=torch.bfloat16)
                    saved.update({
                        "input_latents": latent,
                        "width": width,
                        "height": height,
                        "target_hash": target_hash,
                    })
                    del pixels, latent
                if need_reference:
                    with Image.open(reference) as source:
                        image, _, _ = _fit_target(source.convert("RGB"), target_area, max_side, 16)
                        pixels = pipe.preprocess_image(image, device="cuda", torch_dtype=torch.bfloat16)
                    latent = pipe.vae_encoder(pixels).squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16)
                    saved.update({"reference_latents": latent, "reference_hash": reference_hash})
                    del pixels, latent
                saved.update({"cache_version": CACHE_VERSION, "mode": mode})
                _save(path, saved)
                print(f"[Latentes {index}/{len(rows)}] {target.name}: guardados", flush=True)
    finally:
        del pipe
        gc_collect_cuda()


def _encode_custom_reference(dataset, cache, reference_value, target_area, max_side, device="cuda"):
    if not reference_value:
        return
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.boogu_image import BooguImagePipeline

    reference = _resolve_edit_path(dataset, reference_value)
    reference_hash = _image_hash(_signature(reference), [target_area, max_side])
    custom_path = cache_path(cache, {"image": "_custom"})
    saved = _load(custom_path)
    if saved.get("custom_reference_hash") == reference_hash and "reference_latents" in saved:
        return
    vae_files = sorted((NF4_DIR / "vae").glob("*.safetensors"))
    with quiet():
        pipe = BooguImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[ModelConfig(path=[str(path) for path in vae_files])],
        )
    pipe.load_models_to_device(["vae_encoder"])
    try:
        with Image.open(reference) as source:
            # Multiplo 32: la preview sale del tamaño de la referencia y el latente va en parches 2x2.
            image, width, height = _fit_target(source.convert("RGB"), target_area, max_side, 32)
            pixels = pipe.preprocess_image(image, device=device, torch_dtype=torch.bfloat16)
        with torch.inference_mode():
            latent = pipe.vae_encoder(pixels).squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16)
        saved.update({
            "reference_latents": latent,
            "custom_reference_hash": reference_hash,
            "width": width,
            "height": height,
        })
        _save(custom_path, saved)
        print(f"[Preview] Latente de referencia cacheado: {reference.name}", flush=True)
    finally:
        del pipe
        gc_collect_cuda()


def _custom_preview(config, mode, first_row):
    # Prompt de la preview personalizada (con el trigger) y, en edición, su imagen de antes:
    # la elegida o, si está vacía, la del primer par.
    prompt = _with_trigger(config.get("preview_custom_prompt", ""), config.get("trigger_word", ""))
    reference = ""
    if mode == "edit" and prompt:
        reference = str(config.get("preview_edit_image", "")).strip() or first_row["edit_image"]
    return prompt, reference


def _custom_size(cache, first_row):
    # Sin imagen de antes, la preview personalizada sale del tamaño de la primera muestra.
    custom_path = cache_path(cache, {"image": "_custom"})
    custom_saved = _load(custom_path)
    first_saved = _load(cache_path(cache, first_row))
    custom_saved.update({"width": first_saved["width"], "height": first_saved["height"]})
    _save(custom_path, custom_saved)


def prompt_only(device):
    """Codifica solo el prompt (y la imagen) de la preview personalizada en una caché ya hecha.

    Lo lanza el servidor al guardar los ajustes: en CPU si hay un entrenamiento en marcha, para no
    tocar su VRAM. El trainer relee el manifest antes de cada preview y cambia al prompt nuevo.
    """
    _configure_console()
    try:
        config, dataset, cache, mode, area, _, max_side = _load_cache_settings()
        manifest_path = cache / "cache_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with (cache / "metadata.jsonl").open(encoding="utf-8") as stream:
            first_row = json.loads(stream.readline())
        prompt, reference = _custom_preview(config, mode, first_row)
        if not prompt:
            return 0
        row = {"image": "_custom", "prompt": prompt, "custom": True}
        if reference:
            row["edit_image"] = reference
        print(f"[Custom Prompt] Prompt: {prompt}", flush=True)
        _encode_text([row], dataset, cache, mode, config.get("trigger_word", ""), device)
        if reference:
            print(f"[Custom Prompt] Image: {reference}", flush=True)
            _encode_custom_reference(dataset, cache, reference, area, max_side, device)
        else:
            _custom_size(cache, first_row)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"preview_custom_prompt": prompt, "preview_edit_image": reference})
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
        print("[Custom Prompt] Done", flush=True)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1


def main():
    _configure_console()
    try:
        config, dataset, cache, mode, area, multiple, max_side = _load_cache_settings()
        rows = build_metadata(dataset, cache / "metadata.jsonl", mode)
        incomplete = {
            "cache_version": CACHE_VERSION,
            "model": MODEL_ID,
            "mode": mode,
            "complete": False,
        }
        temporary = cache / "cache_manifest.json.tmp"
        cache.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(incomplete, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, cache / "cache_manifest.json")
        ensure_nf4_model()
        if not torch.cuda.is_available():
            raise RuntimeError("La precaché de Boogu necesita CUDA para codificar con los modelos NF4 locales.")
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Boogu {mode}: {len(rows)} muestras | caché: embeddings + latentes", flush=True)
        text_rows = list(rows)
        custom_prompt, custom_reference = _custom_preview(config, mode, rows[0])
        if custom_prompt:
            custom_row = {"image": "_custom", "prompt": custom_prompt, "custom": True}
            if custom_reference:
                custom_row["edit_image"] = custom_reference
            text_rows.append(custom_row)
        _encode_text(text_rows, dataset, cache, mode, config.get("trigger_word", ""))
        gc_collect_cuda()
        _encode_latents(rows, dataset, cache, mode, area, multiple, max_side)
        if mode == "edit" and custom_prompt:
            _encode_custom_reference(dataset, cache, custom_reference, area, max_side)
        if custom_prompt and mode == "normal":
            _custom_size(cache, rows[0])

        manifest = {
            "cache_version": CACHE_VERSION,
            "model": MODEL_ID,
            "mode": mode,
            "complete": True,
            "sample_count": len(rows),
            "metadata_sha256": _file_hash(cache / "metadata.jsonl"),
            "trigger_word": str(config.get("trigger_word", "")).strip(),
            "preview_custom_prompt": custom_prompt,
            "preview_edit_image": custom_reference,
            "target_area": area,
            "multiple": multiple,
            "max_side": max_side,
            "samples": [cache_path(cache, row).name for row in rows],
        }
        temporary = cache / "cache_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, cache / "cache_manifest.json")
        print(f"✓ Precaché Boogu lista: {len(rows)} embeddings y latentes.", flush=True)
        print(f"El entrenamiento cargará solo el transformer NF4: {NF4_TRANSFORMER}", flush=True)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(prompt_only(sys.argv[2]) if sys.argv[1:2] == ["--prompt-only"] else main())
