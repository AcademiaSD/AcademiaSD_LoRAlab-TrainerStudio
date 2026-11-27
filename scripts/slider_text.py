# -*- coding: utf-8 -*-
"""
slider_text.py — Slider LoRA por texto (común a los trainers)
Text slider LoRA (shared by the trainers)

Un slider por texto aprende un concepto que va de un extremo a otro ("muy delgado" ↔ "muy gordo") sin
imágenes de antes y después. El modelo base ya sabe dibujar los dos extremos: en cada paso, sobre una
imagen del dataset con ruido, se calcula su predicción con el caption, con caption + extremo positivo y
con caption + extremo negativo (sin LoRA). El LoRA a multiplicador m aprende

    objetivo = pred(caption) + ETA · (m / PUSH) · (pred(caption + positivo) − pred(caption + negativo))

Es lineal en m: con fuerza 0 sale el modelo base tal cual y cada punto de la fuerza mueve la imagen la
misma distancia, así que +2 es un poco y +5 mucho, y lo mismo hacia el lado negativo (método de
others devs, concept sliders). Medido en Z-Image con el peso corporal: 8 % de desvío de una recta,
frente al 24 % del slider de pares, que solo cambiaba en los extremos.

A text slider learns a concept that goes from one end to the other ("very thin" ↔ "very fat") without
before/after images. The base model can already draw both ends: at each step, on a noisy dataset image,
its prediction is computed with the caption, caption + positive end and caption + negative end (LoRA off).
The LoRA at multiplier m learns base + ETA · (m / PUSH) · (pos − neg): linear in m, so strength 0 is the
base model and every point of strength moves the image by the same amount.
"""
DEFAULTS = {
    "slider_text": False,          # activa el modo slider por texto
    "slider_pos": "",              # extremo positivo, p. ej. "a very fat obese person"
    "slider_neg": "",              # extremo negativo, p. ej. "a very thin skinny person"
    "slider_eta": 1.5,             # intensidad del concepto a la fuerza máxima
    "slider_push": 3.0,            # multiplicador interno del entrenamiento (no se muestra)
    "slider_strength_100": 5.0,    # fuerza en ComfyUI que corresponde al extremo
    "slider_preview_strengths": [-5, -2.5, 0, 2.5, 5],
    # Preservación: cada paso añade una de estas frases (al azar) al caption y a los dos extremos, y la
    # dirección se promedia sobre ellas (p. ej. etnias). Con encuadres en Krea 2 empeoró los saltos de cámara.
    # Vacío = sin preservación.
    "slider_preserve": [],
    # Ultra: el LoRA solo en los bloques de composición (si el trainer los define).
    "slider_ultra": False,
    # Imágenes de práctica: >0 = las genera el modelo base con los captions antes de entrenar y el slider aprende
    # sobre ellas en vez de sobre las del dataset (others devs); 0 = las del dataset.
    "slider_bank": 0,
}
MANIFEST = "_slider.json"


def settings(cfg):
    s = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    s["slider_pos"] = str(s["slider_pos"] or "").strip()
    s["slider_neg"] = str(s["slider_neg"] or "").strip()
    s["active"] = bool(s["slider_text"]) and bool(s["slider_pos"]) and bool(s["slider_neg"])
    return s


def cache_name(kind, caption_id, variant):
    """Nombre en la caché de un caption (kind "c"), o con un extremo ("pos" / "neg"), con su variante."""
    return f"_slider_{kind}_{caption_id}_{variant}"


def plan(captions, s):
    """
    captions: {muestra: caption}. Devuelve (manifiesto, [(nombre en la caché, prompt)]). Los prompts van
    por caption distinto, no por imagen: con muchas imágenes del mismo caption ocupan mucho menos.
    """
    unique = sorted(set(captions.values()))
    ids = {c: i for i, c in enumerate(unique)}
    variants = [str(v).strip() for v in s["slider_preserve"] if str(v).strip()] or [""]
    jobs = []
    for c, i in ids.items():
        for v, phrase in enumerate(variants):
            base = prompt(c, phrase) if phrase else c
            jobs.append((cache_name("c", i, v), base))
            for side in ("pos", "neg"):
                jobs.append((cache_name(side, i, v), prompt(base, s["slider_" + side])))
    manifest = {"samples": {n: ids[c] for n, c in captions.items()}, "variants": len(variants),
                "ends": ends(s)}
    return manifest, jobs


def ends(s):
    """Lo que el Pre-Cache codificó: si cambia, hay que volver a lanzarlo."""
    return [s["slider_pos"], s["slider_neg"], [str(v).strip() for v in s["slider_preserve"] if str(v).strip()]]


def stale(manifest, s, samples):
    """True si la caché del slider no sirve: no existe, le faltan muestras o los extremos cambiaron."""
    return (not manifest or manifest.get("ends") != ends(s)
            or any(n not in manifest["samples"] for n in samples))


def write_manifest(cache_dir, manifest):
    import json
    import os
    with open(os.path.join(cache_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def read_manifest(cache_dir):
    import json
    import os
    path = os.path.join(cache_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt(caption, phrase):
    return f"{caption}, {phrase}" if caption else phrase


def target(base, pos, neg, mult, s):
    """Objetivo del LoRA a multiplicador `mult` (predicciones del modelo base, sin gradiente)."""
    return base + s["slider_eta"] * (mult / s["slider_push"]) * (pos - neg)


def random_multiplier(s, rng):
    return s["slider_push"] * rng.uniform(-1.0, 1.0)


def export_factor(s):
    """
    Factor de lora_B al exportar: con él, la fuerza slider_strength_100 en ComfyUI da el multiplicador
    PUSH del entrenamiento (el extremo) y la fuerza 1 un quinto del camino.
    """
    return s["slider_push"] / s["slider_strength_100"]


def strength_to_multiplier(strength, s):
    return strength * export_factor(s)


class Dial:
    """Multiplicador de todas las capas LoRA (PEFT): 0 = modelo base, 1 = LoRA normal."""

    def __init__(self, model, adapter="default"):
        from peft.tuners.lora import LoraLayer
        self.layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
        self.adapter = adapter
        for layer in self.layers:
            layer.__dict__.setdefault("_slider_base_scaling", dict(layer.scaling))

    def set(self, mult):
        for layer in self.layers:
            layer.scaling[self.adapter] = layer._slider_base_scaling[self.adapter] * mult


def strip(images, strengths):
    """Tira horizontal de previews con la fuerza encima de cada una (arrays HxWx3 uint8 o PIL)."""
    import numpy as np
    from PIL import Image, ImageDraw
    ims = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]
    w, h = ims[0].size
    band = max(24, h // 24)
    out = Image.new("RGB", (w * len(ims), h + band), "white")
    d = ImageDraw.Draw(out)
    for i, (im, s) in enumerate(zip(ims, strengths)):
        out.paste(im, (i * w, band))
        d.text((i * w + 8, band // 4), f"{s:+g}", fill="black")
    return np.array(out)


def describe(s):
    return (f"'{s['slider_neg']}' ↔ '{s['slider_pos']}' | ETA {s['slider_eta']:g} | "
            f"strength at the end / fuerza en el extremo {s['slider_strength_100']:g}")
