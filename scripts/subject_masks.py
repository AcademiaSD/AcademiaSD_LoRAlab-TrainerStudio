# -*- coding: utf-8 -*-
"""
subject_masks.py — Pérdida con máscara del sujeto (masked loss).
Masked loss: the LoRA learns the subject, not the backgrounds of the dataset.

Un segmentador recorta el sujeto de cada imagen del dataset (persona, objeto...) y la pérdida pesa 1 dentro
de la máscara y «mask_background» (0.1 por defecto) fuera. El fondo no se ignora del todo: con 0 el LoRA puede
dejar halos o fondos raros. La media se normaliza con los pesos, así que el learning rate significa lo mismo.

Segmentador: BiRefNet lite (ZhengPeng7/BiRefNet_lite, MIT, ~0.2 GB en Models/BiRefNet-lite), que separa el
sujeto principal del fondo sin indicarle qué es. Las máscaras se calculan antes de cargar el modelo, con el
mismo recorte que la pre-caché (cubrir el bucket y recortar al centro), y se guardan en la caché a la
resolución del latente: <muestra>_submask.pt, con _submask.json para recalcular solo las imágenes que cambien.
Al calcularlas se guarda una hoja (masked_loss_preview.png en la carpeta del entrenamiento) para revisarlas.
"""
import json
import os

REPO_ID = "ZhengPeng7/BiRefNet_lite"
MODEL_DIR = "Models/BiRefNet-lite"
FILES = ["*.py", "*.json", "*.safetensors"]
SUFFIX = "_submask.pt"
INDEX = "_submask.json"
PREVIEW = "masked_loss_preview.png"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
SIZE = 1024                      # resolución a la que trabaja BiRefNet


def settings(cfg):
    """(activa, peso del fondo)."""
    try:
        bg = max(0.0, min(1.0, float(cfg.get("mask_background", 0.1))))
    except (TypeError, ValueError):
        bg = 0.1
    return bool(cfg.get("masked_loss")), bg


def _model_dir():
    try:
        from project_layout import resolve_model_dir
        return str(resolve_model_dir(MODEL_DIR, MODEL_DIR))
    except Exception:
        return MODEL_DIR


class Segmenter:
    def __init__(self, device="cuda", log=print):
        import torch
        from transformers import AutoModelForImageSegmentation
        path = _model_dir()
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            from huggingface_hub import snapshot_download
            log("[MASK] Downloading BiRefNet lite (~0.2 GB, MIT) to {}...".format(path))
            snapshot_download(REPO_ID, local_dir=path, allow_patterns=FILES)
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
        self.model = self.model.to(self.device, self.dtype).eval()

    def __call__(self, image):
        """Máscara del sujeto (numpy float32 HxW en 0..1) de una imagen PIL RGB, del tamaño de la imagen."""
        import numpy as np
        import torch
        from PIL import Image
        x = image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        a = (np.asarray(x, dtype=np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(self.device, self.dtype)
        with torch.no_grad():
            m = self.model(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
        return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR),
                          dtype=np.float32) / 255

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


def _find_image(folder, name):
    for ext in IMAGE_EXTS:
        for e in (ext, ext.upper()):
            p = os.path.join(folder, name + e)
            if os.path.isfile(p):
                return p
    return None


def _fit(img, bw, bh):
    """Como la pre-caché: escala hasta cubrir el bucket y recorta al centro."""
    import math
    from PIL import Image
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def _latent_hw(cache_dir, name):
    import torch
    t = torch.load(os.path.join(cache_dir, name + "_latent.pt"), weights_only=True, map_location="cpu")
    return int(t.shape[-2]), int(t.shape[-1])


def prepare(cfg, cache_dir, output_dir, log=print, samples=None, latent_hw=None, scale=8):
    """
    Máscaras del sujeto de cada muestra, a la resolución de su latente: {muestra: tensor [1, H, W]} o None si
    no se usa (o no se pudo: el entrenamiento sigue sin ella). latent_hw(muestra) -> (H, W); scale: píxeles
    por posición del latente (para el recorte, igual que la pre-caché).
    """
    on, bg = settings(cfg)
    if not on:
        return None
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import cache_names
    if samples is None:
        samples = sorted(f[:-len("_latent.pt")] for f in os.listdir(cache_dir)
                         if f.endswith("_latent.pt") and not cache_names.is_reserved(f))
    latent_hw = latent_hw or (lambda n: _latent_hw(cache_dir, n))
    folder = cfg.get("dataset_path") or "./dataset"
    index_path = os.path.join(cache_dir, INDEX)
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = {}
    todo, info = [], {}
    for s in samples:
        src = _find_image(folder, s)
        if src is None:
            continue
        try:
            h, w = latent_hw(s)
        except Exception:
            continue                  # su latente no se puede leer: esa muestra va sin máscara
        key = [os.path.basename(src), int(os.path.getmtime(src)), h, w]
        info[s] = (src, h, w, key)
        if index.get(s) != key or not os.path.isfile(os.path.join(cache_dir, s + SUFFIX)):
            todo.append(s)
    if not info:
        log("[!] Masked loss: no dataset image found in {}: training without it.".format(folder))
        return None
    if todo:
        log("[MASK] Subject masks with BiRefNet for {} images...".format(len(todo)))
        try:
            seg = Segmenter("cuda", log=log)
            sheet = []
            for s in todo:
                src, h, w, key = info[s]
                img = _fit(Image.open(src).convert("RGB"), w * scale, h * scale)
                m = seg(img)
                t = torch.from_numpy(m)[None, None]
                small = F.interpolate(t, size=(h, w), mode="area")[0].to(torch.float16)
                torch.save(small, os.path.join(cache_dir, s + SUFFIX + ".tmp"))
                os.replace(os.path.join(cache_dir, s + SUFFIX + ".tmp"), os.path.join(cache_dir, s + SUFFIX))
                index[s] = key
                if len(sheet) < 24:
                    sheet.append((img, m))
            seg.close()
        except Exception as e:
            log("[!] Masked loss: the masks could not be made ({}): training without it.".format(e))
            return None
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
        try:
            _save_sheet(sheet, os.path.join(str(output_dir), PREVIEW))
            log("[MASK] Preview of the masks: {}".format(os.path.join(str(output_dir), PREVIEW)))
        except Exception as e:
            log("[!] Masked loss: preview not saved ({})".format(e))
    masks = {s: torch.load(os.path.join(cache_dir, s + SUFFIX), weights_only=True).float() for s in info}
    cover = sum(float(m.mean()) for m in masks.values()) / len(masks)
    log("[MASK] Masked loss on: {} masks, subject {:.0f}% of the image on average, background weight {}.".format(
        len(masks), cover * 100, bg))
    return masks


def _save_sheet(items, path):
    """Cada imagen con el fondo oscurecido y teñido: lo que queda claro es lo que aprende el LoRA."""
    import numpy as np
    from PIL import Image
    if not items:
        return
    cell = 220
    cols = min(6, len(items))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (11, 18, 34))
    for i, (img, m) in enumerate(items):
        a = np.asarray(img, dtype=np.float32) / 255
        k = m[..., None]
        tint = np.array([0.10, 0.12, 0.30], dtype=np.float32)
        out = a * k + (a * 0.25 + tint * 0.75) * (1 - k)
        tile = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))
        tile.thumbnail((cell - 8, cell - 8))
        x = (i % cols) * cell + (cell - tile.width) // 2
        y = (i // cols) * cell + (cell - tile.height) // 2
        sheet.paste(tile, (x, y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path + ".tmp.png")
    os.replace(path + ".tmp.png", path)


def masked_mse(pred, target, weight):
    """MSE con pesos por posición (weight [B, 1, H, W], se expande a los canales), normalizada por los pesos."""
    se = (pred - target) ** 2
    w = weight.to(se.dtype).expand_as(se)
    return (se * w).sum() / w.sum().clamp_min(1e-8)


def weights(masks, background):
    """Máscaras [B, 1, H, W] en 0..1 -> pesos: 1 en el sujeto, background en el fondo."""
    return background + (1.0 - background) * masks


def register(app, get_output_dir):
    """Sirve la hoja de máscaras del proyecto (la GUI la enseña encima de las previews)."""
    from pathlib import Path
    from flask import jsonify, send_from_directory, abort

    @app.route("/api/masked-loss-preview")
    def masked_loss_preview_info():
        p = Path(get_output_dir()) / PREVIEW
        return jsonify({"exists": p.is_file(), "version": int(p.stat().st_mtime * 1000) if p.is_file() else 0})

    @app.route("/api/masked-loss-preview.png")
    def masked_loss_preview_png():
        p = Path(get_output_dir()) / PREVIEW
        if not p.is_file():
            abort(404)
        return send_from_directory(str(p.parent), p.name, max_age=0)


# ── Lo que usa cada trainer ──
class Subject:
    """
    Uso en un trainer:

        subject = subject_masks.Subject(cfg, CACHE_DIR, OUTPUT_DIR)      # antes de cargar el modelo
        ...
        loss = subject_masks.loss(pred.float(), target, subject.weight(names, grid)) / GRAD_ACCUM_STEPS

    weight() da None si no se usa (y loss() es entonces la MSE de siempre). Con la predicción como latente
    [B, C, H, W], grid=None; empaquetada en tokens [B, L, C], grid=(alto, ancho) de la rejilla de tokens (la
    máscara se reduce por áreas a esa rejilla y se aplana por filas, como los tokens).
    latent_hw(muestra) -> (H, W) del latente guardado, si no son sus dos últimas dimensiones.
    """

    def __init__(self, cfg, cache_dir, output_dir, log=print, latent_hw=None, samples=None):
        self.on, self.bg = settings(cfg)
        self.masks = prepare(cfg, cache_dir, output_dir, log=log, samples=samples, latent_hw=latent_hw) if self.on else None
        self.active = bool(self.masks)

    def weight(self, names, grid=None, device="cuda"):
        if not self.active or any(n not in self.masks for n in names):
            return None
        import torch
        import torch.nn.functional as F
        m = torch.stack([self.masks[n] for n in names]).to(device, non_blocking=True)   # [B, 1, H, W]
        if grid is not None:
            m = F.interpolate(m, size=tuple(int(v) for v in grid), mode="area").flatten(2).transpose(1, 2)
        return weights(m, self.bg)


def loss(pred, target, weight=None, per_sample=False):
    """MSE con los pesos de la máscara (o la de siempre sin ellos). per_sample: una por muestra del lote."""
    import torch.nn.functional as F
    if weight is None:
        if per_sample:
            return F.mse_loss(pred, target, reduction="none").mean(dim=tuple(range(1, pred.dim())))
        return F.mse_loss(pred, target)
    se = (pred - target) ** 2
    w = weight.to(se.dtype).expand_as(se)
    if per_sample:
        dims = tuple(range(1, se.dim()))
        return (se * w).sum(dims) / w.sum(dims).clamp_min(1e-8)
    return (se * w).sum() / w.sum().clamp_min(1e-8)
