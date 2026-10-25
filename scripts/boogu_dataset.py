# -*- coding: utf-8 -*-
"""Dataset preparation for Boogu Image Edit's DiffSynth training recipe."""
import json
import hashlib
import os
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BEFORE_SUFFIX = "_before"
AFTER_SUFFIX = "_after"
CACHE_VERSION = 2


def cache_path(cache_dir, row):
    """Stable per-target cache filename, independent of the absolute dataset path."""
    key = str(row["image"]).replace("\\", "/").casefold()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / "samples" / f"{digest}.pt"


def resolve_path(value, base_dir):
    path = Path(os.path.expandvars(os.path.expanduser(str(value or "./dataset"))))
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    return path


def build_metadata(dataset_dir, metadata_path, mode="edit"):
    dataset_dir = Path(dataset_dir)
    metadata_path = Path(metadata_path)
    if not dataset_dir.is_dir():
        raise ValueError(f"No existe la carpeta del dataset: {dataset_dir}")

    images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    rows, errors = [], []
    mode = "edit" if str(mode).lower() == "edit" else "normal"
    by_stem = {}
    for image in images:
        key = image.stem.casefold()
        if key in by_stem:
            errors.append(f"{image.stem}: hay más de una imagen con el mismo nombre base")
        else:
            by_stem[key] = image

    if mode == "normal":
        # Normal LoRAs use one image plus its own same-stem caption. Pair
        # members are reserved for the edit workflow and are skipped here.
        for image in sorted(images, key=lambda p: p.name.casefold()):
            if image.stem.casefold().endswith((BEFORE_SUFFIX, AFTER_SUFFIX)):
                continue
            caption = dataset_dir / f"{image.stem}.txt"
            if not caption.is_file():
                errors.append(f"{image.name}: falta {caption.name} con el caption")
                continue
            try:
                prompt = caption.read_text(encoding="utf-8-sig").strip()
            except UnicodeError:
                errors.append(f"{caption.name}: no es texto UTF-8 válido")
                continue
            if not prompt:
                errors.append(f"{caption.name}: el caption está vacío")
                continue
            rows.append({"image": image.name, "prompt": prompt})
        if errors:
            sample = "\n".join(f"  - {item}" for item in errors[:20])
            extra = f"\n  ... y {len(errors) - 20} más" if len(errors) > 20 else ""
            raise ValueError(f"Hay imágenes sin caption válido ({len(errors)}):\n{sample}{extra}")
        if not rows:
            raise ValueError("No se encontraron imágenes normales con captions nombre.txt.")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_path, metadata_path)
        return rows

    before_bases = {
        p.stem[:-len(BEFORE_SUFFIX)].casefold()
        for p in images if p.stem.casefold().endswith(BEFORE_SUFFIX)
    }
    for before in sorted(images, key=lambda p: p.name.casefold()):
        if not before.stem.casefold().endswith(BEFORE_SUFFIX):
            continue
        base = before.stem[:-len(BEFORE_SUFFIX)]
        after = by_stem.get((base + AFTER_SUFFIX).casefold())
        if after is None:
            errors.append(f"{before.name}: falta {base}{AFTER_SUFFIX}.*")
            continue

        caption = dataset_dir / f"{base}.txt"
        if not caption.is_file():
            errors.append(f"{before.name}: falta {caption.name} con la instrucción")
            continue
        try:
            prompt = caption.read_text(encoding="utf-8-sig").strip()
        except UnicodeError:
            errors.append(f"{caption.name}: no es texto UTF-8 válido")
            continue
        if not prompt:
            errors.append(f"{caption.name}: la instrucción está vacía")
            continue

        # DiffSynth maps "image" to the edited target and extra_input "edit_image"
        # to the source/reference image, matching its official Boogu Edit recipe.
        rows.append({
            "image": after.relative_to(dataset_dir).as_posix(),
            "edit_image": before.relative_to(dataset_dir).as_posix(),
            "prompt": prompt,
        })

    for after in images:
        if after.stem.casefold().endswith(AFTER_SUFFIX):
            base = after.stem[:-len(AFTER_SUFFIX)].casefold()
            if base not in before_bases:
                errors.append(f"{after.name}: falta {base}{BEFORE_SUFFIX}.*")

    if errors:
        sample = "\n".join(f"  - {item}" for item in errors[:20])
        extra = f"\n  ... y {len(errors) - 20} más" if len(errors) > 20 else ""
        raise ValueError(f"Hay parejas incompletas o instrucciones inválidas ({len(errors)}):\n{sample}{extra}")
    if not rows:
        raise ValueError("No se encontraron parejas nombre_before.ext + nombre_after.ext con nombre.txt.")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, metadata_path)
    return rows
