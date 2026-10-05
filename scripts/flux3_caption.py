# -*- coding: utf-8 -*-
"""
flux3_caption.py — Captions con cajas en el formato de FLUX 3 Image (layout prompt)
Captions with boxes in the FLUX 3 Image format (layout prompt)

Formato (docs.bfl.ai/flux_3/flux3_image_bounding_boxes, guides/prompting_layout):
  un párrafo (el caption) que cita cada elemento por su id entre <>, un espacio, y una lista JSON
  de filas {"id", "bbox", "desc"}. bbox = [top, left, bottom, right], enteros de 0 a 1000 desde la
  esquina superior izquierda (el mismo orden y escala que Ideogram 4). El texto a pintar va entre
  comillas dentro del desc de su fila, con \\n para los saltos de línea; el fondo puede ser una fila.

  a paragraph (the caption) citing every element by its id in <>, a space, and a JSON list of rows
  {"id", "bbox", "desc"}, bbox = [top, left, bottom, right] on a 0-1000 grid.

Lo usan el captioner de Ideogram 4 (estilo "flux3") y la herramienta de conversión; será la base
del captioner del entrenador de FLUX 3 cuando haya pesos abiertos. Sin dependencias de torch.
"""
import json
import re

# Separadores de las filas: los de JSON.stringify, como el "Copy request body" de la documentación.
# (Su ejemplo en Python usa json.dumps por defecto, con espacios; no se sabe con cuál se entrenó.)
ROW_SEPARATORS = (",", ":")

PROMPT = (
    "Write a layout caption of this image for the FLUX 3 image model. Output ONE JSON object on a single line "
    "and nothing else (no markdown, no comments): "
    '{"caption":"...","elements":[{"id":"...","bbox_2d":[x1,y1,x2,y2],"desc":"..."}, ...]}\n\n'
    "caption: one paragraph of 80 to 250 words. Start with the medium and the look (photograph, illustration, "
    "painting, 3D render, graphic design...; film, lens, camera angle and framing for photographs). Then introduce "
    "the subjects, say where each one sits in the frame (left foreground, in her lap, upper right corner...) and how "
    "they relate, and end with the light, the colors and the background. Every element of the list is cited exactly "
    "once in the caption by its id in angle brackets, right after the words that name it, where it first appears: "
    '"a woman <person_1> sits in the left foreground holding a small dog <animal_1>". Describe only what is visible; '
    "no words like stunning, beautiful, iconic or masterpiece, and no \"this image shows\".\n\n"
    "elements: the main subject first, then every other person, animal and object that matters, every block of "
    "legible text, and the background areas. At most 40 elements. A dense group of similar things (a distant crowd, "
    "a shelf of books, a pile of fruit) is ONE element. A person's hair, clothes and what they hold go in that "
    "person's desc. The background can be one element covering the whole frame, or one element per area (sky, "
    "wall, floor).\n"
    "- id: a short lowercase noun with a number, unique: person_1, person_2, animal_1, cup_1, background_1, sky_1. "
    "For text use the language and Text: En_Text_1, Ja_Text_1, Fr_Text_1, Unknown_Text_1.\n"
    "- desc: 15 to 60 words about appearance: material, color, shape, pose, expression and visible details. For a "
    'text element, quote the exact characters in double quotes in their own script, with \\n for line breaks, then '
    'the type: case, weight, typeface style, color and orientation, e.g. Centered text reading \\"OPEN\\nLATE\\" in '
    "a bold white sans-serif typeface. Only for flat, uniform colors of graphic designs, an exact color may be given "
    "as an uppercase hex code tied to that element, e.g. a solid #E01075 background.\n"
    "- bbox_2d: the tight box around the element in THIS image, [x1, y1, x2, y2] with x from left (0) to right (1000) "
    "and y from top (0) to bottom (1000). Boxes overlap when the objects overlap: a dog in its owner's lap has its "
    "box inside the owner's. Every box must match where the element really is."
)

_ID = re.compile(r"[^a-z0-9_]+")
_TOKEN = re.compile(r"<([A-Za-z][^<>\n]{0,40})>")


def clean_id(raw, used):
    """Id válido y único: minúsculas, letras/números/_, terminado en _N (los Xx_Text_N conservan su forma)."""
    raw = str(raw or "").strip().strip("<>")
    m = re.fullmatch(r"([A-Za-z]+)_Text_(\d+)", raw)
    if m:
        base, n = f"{m.group(1).capitalize()}_Text", int(m.group(2))
    else:
        s = _ID.sub("_", raw.lower()).strip("_") or "element"
        m = re.fullmatch(r"(.*?)_?(\d+)", s)
        base, n = (m.group(1) or "element", int(m.group(2))) if m else (s, 1)
    while f"{base}_{n}" in used:
        n += 1
    used.add(f"{base}_{n}")
    return f"{base}_{n}"


def to_bbox(box_2d):
    """[x1, y1, x2, y2] de Qwen3-VL -> [top, left, bottom, right] enteros en 0-1000, o None."""
    if not (isinstance(box_2d, list) and len(box_2d) == 4 and all(isinstance(v, (int, float)) for v in box_2d)):
        return None
    x1, y1, x2, y2 = (max(0, min(1000, int(round(v)))) for v in box_2d)
    top, left, bottom, right = min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)
    return [top, left, bottom, right] if bottom > top and right > left else None


def text_language(text):
    """Prefijo de idioma de un id de texto, por la escritura de sus caracteres."""
    if re.search(r"[぀-ヿ]", text):
        return "Ja"
    if re.search(r"[가-힯]", text):
        return "Ko"
    if re.search(r"[一-鿿]", text):
        return "Zh"
    if re.search(r"[؀-ۿ]", text):
        return "Ar"
    if re.search(r"[Ѐ-ӿ]", text):
        return "Ru"
    return "En" if re.search(r"[A-Za-z]", text) else "Unknown"


def first_clause(desc, limit=12):
    """El principio de un desc, para citar en el caption un elemento que no se citaba."""
    words = re.split(r"[.;:,]", desc, 1)[0].split()
    return " ".join(words[:limit]).rstrip(",") or "an element"


def iou(a, b):
    """IoU de dos cajas [top, left, bottom, right]."""
    it = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    il = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda q: max(0, q[2] - q[0]) * max(0, q[3] - q[1])
    inter = it * il
    return inter / max(1, area(a) + area(b) - inter)


def normalize(caption, rows):
    """
    Deja caption y filas coherentes: ids válidos y únicos, cajas válidas, sin filas repetidas, cada fila
    citada como <id> en el caption (si falta, se añade una frase al final) y ningún <token> sin fila.
    rows: [{"id", "bbox" ([top, left, bottom, right]), "desc"}]. Devuelve (caption, rows).
    """
    used, out, renamed, seen = set(), [], {}, set()
    for r in rows:
        if not isinstance(r, dict) or r.get("bbox") is None:
            continue
        desc = " ".join(str(r.get("desc", "")).split())
        key = (tuple(r["bbox"]), desc[:60].lower())
        if not desc or key in seen:
            continue
        seen.add(key)
        new = clean_id(r.get("id"), used)
        renamed.setdefault(str(r.get("id", "")).strip().strip("<>"), new)
        out.append({"id": new, "bbox": list(r["bbox"]), "desc": desc})

    ids = {r["id"] for r in out}

    def fix(m):
        name = renamed.get(m.group(1).strip(), m.group(1).strip())
        return f"<{name}>" if name in ids else ""
    caption = " ".join(_TOKEN.sub(fix, str(caption or "")).split())
    caption = re.sub(r"\s+([,.;:])", r"\1", caption)
    # Cada id una sola vez: las citas repetidas se quitan.
    for i in ids:
        first = caption.find(f"<{i}>")
        if first >= 0:
            caption = caption[:first + len(i) + 2] + caption[first + len(i) + 2:].replace(f" <{i}>", "").replace(f"<{i}>", "")
    missing = [r for r in out if f"<{r['id']}>" not in caption]
    if missing:
        caption = (caption.rstrip(". ") + ". " if caption else "") + " ".join(
            f"{first_clause(r['desc'])[:1].upper()}{first_clause(r['desc'])[1:]} <{r['id']}>." for r in missing)
    return caption.strip(), out


def merge_ocr(rows, texts):
    """
    Los textos del OCR ([{"bbox_2d", "text"}]) corrigen las palabras citadas en la fila de texto con
    la que se solapan; los que no se solapan con ninguna se añaden como filas nuevas.
    """
    for x in texts:
        box = to_bbox(x.get("bbox_2d"))
        text = str(x.get("text", "")).strip()
        if not box or not text:
            continue
        quoted = text.replace("\n", "\\n")
        texty = [r for r in rows if "_Text_" in r.get("id", "") or '"' in r.get("desc", "")]
        match = max(texty, key=lambda r: iou(r["bbox"], box), default=None)
        if match is not None and iou(match["bbox"], box) > 0.3:
            if '"' in match["desc"]:
                match["desc"] = re.sub(r'"[^"]*"', lambda m: f'"{quoted}"', match["desc"], count=1)
            else:
                match["desc"] = f'Text reading "{quoted}". {match["desc"]}'
        else:
            rows.append({"id": f"{text_language(text)}_Text_1", "bbox": box,
                         "desc": f'Text reading "{quoted}", in the same type style, color and placement as in the scene.'})
    return rows


def build(caption, rows):
    """El prompt de FLUX 3: caption, un espacio y las filas en JSON."""
    return f"{caption} {json.dumps(rows, ensure_ascii=False, separators=ROW_SEPARATORS)}"


def from_model(data, texts=()):
    """Respuesta de Qwen3-VL ({"caption", "elements": [{"id", "bbox_2d", "desc"}]}) + OCR -> prompt."""
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        raise ValueError("JSON without caption/elements / JSON sin caption/elements")
    rows = [{"id": e.get("id"), "bbox": to_bbox(e.get("bbox_2d")), "desc": str(e.get("desc", ""))}
            for e in data["elements"] if isinstance(e, dict)]
    rows = merge_ocr([r for r in rows if r["bbox"]], list(texts))
    caption, rows = normalize(str(data.get("caption", "")), rows)
    if not caption or not rows:
        raise ValueError("empty caption / caption vacío")
    return build(caption, rows)


def parse(prompt):
    """Prompt de FLUX 3 -> (caption, filas). Lanza ValueError si no tiene la lista de filas al final."""
    start = prompt.rfind(" [")
    if start < 0 or not prompt.rstrip().endswith("]"):
        raise ValueError("no element table / sin tabla de elementos")
    return prompt[:start].strip(), json.loads(prompt[start + 1:])


# ── Conversión desde un caption JSON de Ideogram 4 ───────────────────────────

_NOUNS = [
    ("person", r"\b(woman|man|girl|boy|child|kid|person|lady|gentleman|baby|player|dancer|figure|people|crowd)\b"),
    ("animal", r"\b(dog|cat|bird|horse|cow|animal|puppy|kitten|fox|deer|lion|tiger|bear|rabbit|fish|duck|retriever|terrier|parrot|owl)\b"),
]


def ideogram_id(desc, is_text, text):
    """
    Id semántico para un elemento de Ideogram: person/animal si lo es, si no el sustantivo principal
    del desc (la última palabra antes de "with", "on", "in"...: "A worn red skateboard with..." -> skateboard).
    """
    if is_text:
        return f"{text_language(text)}_Text_1"
    low = desc.lower()
    for name, pattern in _NOUNS:
        if re.search(pattern, low[:80]):
            return f"{name}_1"
    head = re.split(r"[,.;:]|\b(?:with|on|in|of|at|and|that|which|who|holding|wearing|standing|sitting|lying|from|under|over|near|against)\b", low, 1)[0]
    words = [w for w in re.findall(r"[a-z]+", head) if len(w) > 2]
    return f"{words[-1] if words else 'object'}_1"


def from_ideogram(caption_json):
    """
    Caption JSON de Ideogram 4 -> prompt de FLUX 3. Una conversión aproximada: el estilo, la luz y la
    paleta pasan a frases del caption, el fondo a una fila de pantalla completa, cada elemento a una fila
    (texto entre comillas en su desc, paleta del elemento como hex en su desc) y se citan por su id.
    """
    data = json.loads(caption_json) if isinstance(caption_json, str) else caption_json
    sd = data.get("style_description") or {}
    cd = data.get("compositional_deconstruction") or {}
    parts = [str(data.get("high_level_description", "")).strip().rstrip(".") + "."]
    look = [sd.get("photo") or sd.get("art_style"), sd.get("aesthetics")]
    look = ", ".join(str(v).strip().rstrip(".") for v in look if v)
    if look:
        parts.append(f"{look[:1].upper()}{look[1:]}.")
    if sd.get("lighting"):
        parts.append(f"Lighting: {str(sd['lighting']).strip().rstrip('.')}.")

    rows, mentions = [], []
    if str(cd.get("background", "")).strip():
        rows.append({"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": str(cd["background"]).strip()})
        mentions.append(("The background", "background_1"))
    for e in cd.get("elements") or []:
        if not isinstance(e, dict) or not isinstance(e.get("bbox"), list) or len(e["bbox"]) != 4:
            continue
        is_text = e.get("type") == "text"
        text = str(e.get("text", ""))
        desc = str(e.get("desc", "")).strip()
        if is_text:
            desc = f'Text reading "{text.replace(chr(10), chr(92) + "n")}". {desc}'.strip()
        palette = [c for c in e.get("color_palette") or [] if re.fullmatch(r"#[0-9A-Fa-f]{6}", str(c))]
        if palette:
            desc = f"{desc.rstrip('.')}, in {', '.join(c.upper() for c in palette)}."
        rid = ideogram_id(str(e.get("desc", "")), is_text, text)
        rows.append({"id": rid, "bbox": [int(v) for v in e["bbox"]], "desc": desc})
        mentions.append((first_clause(str(e.get("desc", "")) or text), rid))

    # Los ids definitivos (únicos) salen de normalize; se citan con un marcador que normalize reconoce.
    used, final = set(), []
    for (label, rid), row in zip(mentions, rows):
        new = clean_id(rid, used)
        row["id"] = new
        final.append(f"{label[:1].upper()}{label[1:]} <{new}>.")
    palette = [c for c in sd.get("color_palette") or [] if re.fullmatch(r"#[0-9A-Fa-f]{6}", str(c))]
    if palette:
        final.append(f"The dominant colors are {', '.join(c.upper() for c in palette[:8])}.")
    caption, rows = normalize(" ".join(parts + final), rows)
    return build(caption, rows)
