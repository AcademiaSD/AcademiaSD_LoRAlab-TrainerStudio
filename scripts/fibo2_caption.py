# -*- coding: utf-8 -*-
"""
fibo2_caption.py — Utilidades de caption de Fibo 2.0, compartidas por el pre-caché,
el entrenador y el servidor.

Fibo 2.0 lee captions estructurados (JSON de VGL). El trigger word no se puede pegar
delante de un JSON (lo rompería): va dentro de su short_description. Estas funciones
son las mismas en los tres sitios, para que el caption que codifica el pre-caché y el
que el servidor compara con él sean siempre el mismo.

Un caption puede ser:
  - <nombre>.json  : el caption estructurado (gana sobre el .txt).
  - <nombre>.txt   : una frase suelta, que el pipeline lee como descripción corta.
"""
import json
import os
import re

JSON_EXT = ".json"
TXT_EXT = ".txt"


def compact(caption):
    """El caption en la forma en que se entrenó: JSON sin espacios."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def with_trigger(prompt, trigger, is_edit=False):
    """
    Añade el trigger word al caption sin romperlo: en un caption JSON va dentro de su
    short_description; en una frase suelta, delante. Un prompt de edición (el que lleva
    "edit_instruction") se queda como está: pegarle delante rompería la instrucción.
    En modo edición, una frase suelta ES la instrucción, así que tampoco se toca.
    """
    trigger = (trigger or "").strip()
    if not trigger or not prompt:
        return prompt
    if trigger.lower() in prompt.lower():
        return prompt
    try:
        caption = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        caption = None
    if isinstance(caption, dict):
        target = caption
        if "structured_caption" in caption and isinstance(caption["structured_caption"], dict):
            target = caption["structured_caption"]
        if isinstance(target.get("short_description"), str):
            target["short_description"] = f"{trigger}, {target['short_description']}"
            return compact(caption)
        if "edit_instruction" in caption:
            return prompt
        target["short_description"] = trigger
        return compact(caption)
    if is_edit:
        # Una frase suelta en modo edición es la INSTRUCCIÓN de la edición: pegarle el
        # trigger delante la corrompe (así lo hace el pipeline oficial).
        return prompt
    return f"{trigger}, {prompt}".strip(", ")


def caption_path(stem_path, prefer_json=True):
    """
    La ruta del caption de una muestra: el <nombre>.json si existe, si no el <nombre>.txt.
    `stem_path` es la ruta sin extensión (absoluta o relativa).
    """
    stem = str(stem_path)
    if prefer_json:
        as_json = stem + JSON_EXT
        if os.path.exists(as_json):
            return as_json
    return stem + TXT_EXT


def read_caption(stem_path, prefer_json=True):
    """
    El caption de una muestra como texto: el JSON serializado compacto si hay un .json,
    o el texto del .txt. "" si no hay ninguno.
    """
    path = caption_path(stem_path, prefer_json)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return ""
    if path.endswith(JSON_EXT):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        return compact(data) if isinstance(data, dict) else text
    return text


def _apply_common_to_text(text, trigger, common, mode, is_edit=False):
    """El texto común sobre un caption de texto plano (la regla del resto del estudio)."""
    if common and mode == "remove":
        text = re.sub(re.escape(common), "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"^[\s,;:.]+", "", text)
        text = re.sub(r"[,;:]\s*$", "", text).strip()
    elif common and mode == "replace":
        text = common
    elif common and common.lower() not in text.lower():
        base = text if not text or text[-1] in ".!?,;:" else text + "."
        text = f"{base} {common}"
    if (not is_edit and mode != "remove" and trigger
            and trigger.lower() not in text.lower()):
        text = f"{trigger}, {text}".strip(", ")
    return text


def apply_common(caption_text, trigger, common, mode, is_edit=False):
    """
    El trigger word y el texto común sobre un caption, sea JSON o texto plano.
    En un caption JSON el texto va dentro del short_description (o en uno nuevo si no
    hay), para no romper el JSON. En modo edición, un caption de texto plano es la
    instrucción de la edición y no se le pega el trigger.
    """
    common = (common or "").strip()
    trigger = (trigger or "").strip()
    if not common and not trigger:
        return caption_text
    try:
        caption = json.loads(caption_text)
    except (json.JSONDecodeError, TypeError):
        return _apply_common_to_text(caption_text, trigger, common, mode, is_edit)
    if not isinstance(caption, dict):
        return _apply_common_to_text(caption_text, trigger, common, mode, is_edit)

    target = caption
    if "structured_caption" in caption and isinstance(caption["structured_caption"], dict):
        target = caption["structured_caption"]
    description = target.get("short_description")
    if not isinstance(description, str):
        description = ""
    updated = _apply_common_to_text(description, trigger, common, mode, is_edit)
    if updated != description or "short_description" not in target:
        target["short_description"] = updated
    return compact(caption)
