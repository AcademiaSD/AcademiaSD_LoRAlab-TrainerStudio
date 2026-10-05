# -*- coding: utf-8 -*-
"""
slider_core.py — Núcleo común de los LoRAs Slider / Shared core of Slider LoRAs

Todo lo que no depende del modelo: nombres del dataset (grupo_posición, grupo_anc), validación,
pares, multiplicador de cada paso, escala del LoRA exportado y fuerzas de las previews.
Lo usan el pre-caché, el trainer y el servidor de cada modelo. Diseño: docs/SLIDERS.md.

Everything that does not depend on the model: dataset names (group_position, group_anc),
validation, pairs, the multiplier of each step, the export scale and the preview strengths.
Used by every model's pre-cache, trainer and server. Design: docs/SLIDERS.md.

Basado en el prototipo de experiments/slider. La idea de entrenar en las dos direcciones es de
Concept Sliders (Gandikota et al.); el push, el rank bajo y "ultra" siguen lo que describe Fizgig
(@shootthesound) para sus sliders. No se usa código de ninguno de los dos.
"""
import json
import os
import random
import re

# Caption de los grupos sin .txt. En edición, la instrucción neutra del prototipo: el texto no da
# pistas y el efecto tiene que vivir en el LoRA. En texto a imagen, sin texto.
EDIT_CAPTION = "Keep the photo exactly as it is."

DEFAULTS = {
    "slider_mode": "edit",        # "edit": modifica una foto | "t2i": modifica la generación
    "slider_strength_100": 5.0,   # fuerza en ComfyUI que corresponde a la posición 100
    "slider_push": 3.0,           # fuerza interna del entrenamiento (no se muestra en la GUI)
    "slider_ultra": True,         # LoRA solo en los bloques de composición (si el modelo lo define)
}

# Pares de más de 100 posiciones (-100 -> 100) piden el doble de push: solo se usan si el grupo
# no tiene otros.
MAX_SPAN = 100
MANIFEST = "_slider.json"
_NAME = re.compile(r"^(.+)_(-?\d+|anc)$")


def parse_name(stem):
    """'boca1_25' -> ('boca1', 25); 'fondo1_anc' -> ('fondo1', None); otro nombre -> None."""
    m = _NAME.match(stem)
    if not m:
        return None
    group, pos = m.groups()
    return (group, None) if pos == "anc" else (group, int(pos))


def caption_stem(stem):
    """Nombre del .txt de una imagen: el del grupo ('boca1_25' -> 'boca1')."""
    parsed = parse_name(stem)
    return parsed[0] if parsed else stem


def default_caption(mode):
    return EDIT_CAPTION if mode == "edit" else ""


def read_settings(cfg):
    """Ajustes del slider con sus valores por defecto, corregidos si no son válidos."""
    s = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    if s["slider_mode"] not in ("edit", "t2i"):
        s["slider_mode"] = DEFAULTS["slider_mode"]
    for key in ("slider_strength_100", "slider_push"):
        try:
            s[key] = float(s[key])
        except (TypeError, ValueError):
            s[key] = DEFAULTS[key]
        if s[key] <= 0:
            s[key] = DEFAULTS[key]
    s["slider_ultra"] = bool(s["slider_ultra"])
    return s


def scan(stems):
    """
    Agrupa los nombres del dataset. Devuelve (groups, anchors, problems):
      groups   {grupo: {posición: stem}}, solo los grupos con 2 o más posiciones
      anchors  [stem]
      problems [(en, es)] para mostrar en consola
    """
    groups, anchors, problems = {}, [], []
    for stem in sorted(stems):
        parsed = parse_name(stem)
        if parsed is None:
            problems.append((f"'{stem}' ignored: name it group_position (boca1_50) or group_anc",
                             f"'{stem}' ignorado: llámalo grupo_posición (boca1_50) o grupo_anc"))
        elif parsed[1] is None:
            anchors.append(stem)
        elif not -100 <= parsed[1] <= 100:
            problems.append((f"'{stem}' ignored: positions go from -100 to 100",
                             f"'{stem}' ignorado: las posiciones van de -100 a 100"))
        elif parsed[1] in groups.setdefault(parsed[0], {}):
            problems.append((f"'{stem}' ignored: group '{parsed[0]}' already has position {parsed[1]}",
                             f"'{stem}' ignorado: el grupo '{parsed[0]}' ya tiene la posición {parsed[1]}"))
        else:
            groups[parsed[0]][parsed[1]] = stem

    for group in [g for g, pos in groups.items() if len(pos) < 2]:
        if groups[group]:
            problems.append((f"Group '{group}' ignored: it needs at least 2 positions",
                             f"Grupo '{group}' ignorado: necesita al menos 2 posiciones"))
        del groups[group]
    if groups and not anchors:
        problems.append(("No anchors (group_anc): recommended, they keep framing and light in place",
                         "Sin anchors (grupo_anc): recomendados, mantienen el encuadre y la luz"))
    return groups, anchors, problems


def group_pairs(positions):
    """Pares (p_a, p_b) con p_a < p_b de un grupo, sin los de más de MAX_SPAN si hay otros."""
    pos = sorted(positions)
    pairs = [(a, b) for i, a in enumerate(pos) for b in pos[i + 1:]]
    short = [(a, b) for a, b in pairs if b - a <= MAX_SPAN]
    return short or pairs


class Sampler:
    """
    Elige las muestras de cada paso y su multiplicador del LoRA. Cada elemento es un dict:
      lat   stem de la imagen objetivo
      ctrl  stem de la imagen de entrada (solo edición) o None
      emb   grupo cuyo caption se usa
      mult  multiplicador del LoRA (se aplica con set_multiplier)
    Edición: un elemento, un par ordenado p_i -> p_j a PUSH * (p_j - p_i) / 100.
    Texto a imagen: dos elementos del mismo grupo, a -/+ PUSH * (p_b - p_a) / 200, que van en el mismo
    paso con el mismo ruido y timestep. Anchors: la imagen hacia sí misma a +-PUSH al azar.
    Cada grupo pesa lo mismo tenga las imágenes que tenga; un anchor sale con probabilidad A / (A + 2G).
    """

    def __init__(self, groups, anchors, mode, push, rng=random):
        self.groups = {g: (pos, group_pairs(pos)) for g, pos in groups.items()}
        self.names = sorted(self.groups)
        self.anchors = list(anchors)
        self.mode, self.push, self.rng = mode, push, rng

    def p_anchor(self):
        return len(self.anchors) / max(1, len(self.anchors) + 2 * len(self.names))

    def next(self):
        if self.anchors and (not self.names or self.rng.random() < self.p_anchor()):
            stem = self.rng.choice(self.anchors)
            return [{"lat": stem, "ctrl": stem if self.mode == "edit" else None,
                     "emb": caption_stem(stem), "mult": self.push * self.rng.choice((-1.0, 1.0))}]
        group = self.rng.choice(self.names)
        pos, pairs = self.groups[group]
        a, b = self.rng.choice(pairs)
        if self.mode == "edit":
            if self.rng.random() < 0.5:
                a, b = b, a
            return [{"lat": pos[b], "ctrl": pos[a], "emb": group, "mult": self.push * (b - a) / 100}]
        half = self.push * (b - a) / 200
        return [{"lat": pos[a], "ctrl": None, "emb": group, "mult": -half},
                {"lat": pos[b], "ctrl": None, "emb": group, "mult": half}]


def lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def set_multiplier(layers, mult, adapter="default"):
    """Escala de cada capa LoRA = alpha / rank * mult (mult = 1 es el LoRA normal)."""
    for layer in layers:
        layer.scaling[adapter] = layer.lora_alpha[adapter] / layer.r[adapter] * mult


def export_factor(settings):
    """
    Factor de los pesos lora_B al exportar: con él, la fuerza slider_strength_100 en ComfyUI da el
    multiplicador PUSH del entrenamiento, es decir, la posición 100.
    """
    return settings["slider_push"] / settings["slider_strength_100"]


def preview_strengths(settings):
    s = settings["slider_strength_100"]
    return [-s, -s / 2, 0.0, s / 2, s]


def strength_to_multiplier(strength, settings):
    return strength * export_factor(settings)


def write_manifest(cache_dir, groups, anchors, mode, default_prompt=""):
    # default_prompt: el de las previews sin prompt manual (caption del primer grupo o el neutro).
    data = {"mode": mode, "anchors": anchors, "default_prompt": default_prompt,
            "groups": {g: {str(p): stem for p, stem in sorted(pos.items())} for g, pos in groups.items()}}
    tmp = os.path.join(cache_dir, MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(cache_dir, MANIFEST))


def read_manifest(cache_dir):
    """(groups, anchors, mode) del pre-caché, o None si la caché no es de un slider."""
    path = os.path.join(cache_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups = {g: {int(p): stem for p, stem in pos.items()} for g, pos in data["groups"].items()}
    return groups, data["anchors"], data.get("mode", DEFAULTS["slider_mode"])


def default_prompt(cache_dir):
    """Prompt de las previews sin prompt manual, o None si la caché no es de un slider."""
    path = os.path.join(cache_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("default_prompt", "")


def describe(groups, anchors):
    images = sum(len(p) for p in groups.values())
    return (f"{len(groups)} groups, {images} images, {len(anchors)} anchors / "
            f"{len(groups)} grupos, {images} imágenes, {len(anchors)} anchors")
