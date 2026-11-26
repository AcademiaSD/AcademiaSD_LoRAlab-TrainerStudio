# -*- coding: utf-8 -*-
"""
dataset_analyzer.py — Analizador del dataset antes de entrenar.
Dataset analyzer: what can spoil the LoRA, before spending an hour training.

Botón «🔍 Analyze» en la barra del Dataset Manager (GUI/dataset_analyzer.js). Todo en CPU y sin modelos grandes:

- Imágenes duplicadas o casi iguales (hash perceptual de diferencias, 64 bits).
- Imágenes borrosas (varianza del laplaciano, comparada con la del resto del dataset) o pequeñas.
- Caras (con Face likeness instalado y un dataset de personaje): sin cara, con varias, o que se parecen poco
  al resto (puede ser otra persona).
- Captions: que faltan, sin el trigger, demasiado largos, casi iguales entre sí, y frases repetidas en casi
  todos (rasgos fijos como «long wavy blonde hair»: el trigger no aprende la identidad -> captions Character).
- Proporciones: cuántas imágenes caen en cada forma (vertical, cuadrada, horizontal...).

Servidor: register(app, get_dataset_dir) sirve /dataset_analyzer.js y POST /api/dataset-analyze {trigger}.
"""
import os
import re
from collections import Counter
from pathlib import Path

from i18n import t

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"
# Palabras de rasgos fijos de una persona: una frase repetida con ellas es lo que el trigger debería aprender.
TRAITS = re.compile(r"\b(hair|haired|eyes?|eyed|skin|complexion|blonde|brunette|redhead|ginger|freckles?|beard|"
                    r"young|old|aged?|years?|teen|face|facial|lips|nose|eyebrows?|cheekbones|jawline|tattoos?|"
                    r"woman|man|girl|boy)\b")
STOP = set("a an the and or of with in on at to by for from her his their its is are was be as this that "
           "she he they it there which while into over under near behind front back up down out".split())


def _dhash(img):
    import numpy as np
    g = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _sharpness(img):
    import numpy as np
    g = img.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def _shape(w, h):
    r = w / h
    if r < 0.7:
        return "portrait (≤ 2:3)"
    if r < 0.9:
        return "portrait (3:4)"
    if r <= 1.12:
        return "square"
    if r <= 1.45:
        return "landscape (4:3)"
    return "landscape (≥ 3:2)"


def _words(text):
    return re.findall(r"[a-z][a-z'-]*", (text or "").lower())


def analyze(folder, trigger="", face_check=True):
    from PIL import Image
    folder = Path(folder)
    if not folder.is_dir():
        return {"error": t("Dataset folder not found: {path}", path=str(folder))}
    trigger = (trigger or "").strip()
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    edit = any(re.search(r"_(before|after)$", os.path.splitext(f)[0], re.I) for f in files)
    items, issues = [], []

    def issue(level, kind, msg, names, **extra):
        if names:
            issues.append(dict(level=level, kind=kind, msg=msg, files=sorted(set(names)), **extra))

    for f in files:
        p = folder / f
        rec = {"file": f, "caption": None}
        try:
            with Image.open(p) as im:
                im.load()
                rec["w"], rec["h"] = im.size
                rgb = im.convert("RGB")
                rec["hash"] = _dhash(rgb)
                rec["sharp"] = _sharpness(rgb)
        except Exception as e:
            issue("warn", "broken", t("These images cannot be opened ({error}).", error=type(e).__name__), [f])
            continue
        txt = p.with_suffix(".txt")
        if txt.is_file():
            try:
                from text_io import read_text_file
                rec["caption"] = read_text_file(str(txt)).strip()
            except Exception:
                rec["caption"] = ""
        items.append(rec)
    n = len(items)
    if not n:
        return {"count": 0, "issues": [dict(level="warn", kind="empty", msg=t("No images in the dataset folder."), files=[])],
                "folder": str(folder)}

    # ── Duplicados ──
    dups = []
    for i in range(n):
        for j in range(i + 1, n):
            if bin(items[i]["hash"] ^ items[j]["hash"]).count("1") <= 4:
                dups.append((items[i]["file"], items[j]["file"]))
    if dups:
        issue("warn", "duplicates", t("Duplicate or almost identical images: the LoRA sees them twice as often. Keep one of each pair."), [x for p in dups for x in p], pairs=dups)

    # ── Tamaño y nitidez ──
    small = [r["file"] for r in items if min(r["w"], r["h"]) < 512]
    issue("warn", "small", t("Smaller than 512 px on the short side: they are upscaled, and look soft in the LoRA."), small)
    medium = [r["file"] for r in items if 512 <= min(r["w"], r["h"]) < 768]
    issue("info", "medium", t("Under 768 px on the short side: fine for 512-768 training, soft at 1024."), medium)
    sharp = sorted(r["sharp"] for r in items)
    med = sharp[len(sharp) // 2] if sharp else 0
    blurry = [r["file"] for r in items if med > 0 and r["sharp"] < med * 0.2 and r["sharp"] < 150]
    issue("warn", "blurry", t("Much blurrier than the rest of the dataset: the LoRA learns the blur too."), blurry)

    # ── Proporciones ──
    shapes = Counter(_shape(r["w"], r["h"]) for r in items)

    # ── Caras (Face likeness) ──
    faces = None
    if face_check and not edit:
        try:
            import face_match
            if face_match.installed():
                faces = _faces(folder, items, issue)
        except Exception:
            faces = None

    # ── Captions ──
    caps = [r for r in items if r["caption"] is not None]
    missing = [r["file"] for r in items if not r["caption"]]
    issue("warn", "no_caption", t("Without a caption (or empty): run Create Captions, or write them."), missing)
    if caps and not edit:
        if trigger:
            no_trig = [r["file"] for r in caps if r["caption"] and trigger.lower() not in r["caption"].lower()]
            issue("info", "no_trigger", t("The caption does not have the trigger «{trigger}»: the pre-cache puts it in front.", trigger=trigger),
                  no_trig)
        character = bool(faces and faces.get("character"))
        if character:
            long_caps = [r["file"] for r in caps if len(_words(r["caption"])) > 80]
            issue("info", "long", t("Long captions (more than 80 words): for a character, short captions with only what changes between photos usually give more likeness (Character caption type)."), long_caps)
        _repeated(caps, trigger, issue, character)
        near = []
        for i in range(len(caps)):
            a = set(_words(caps[i]["caption"])) - STOP
            for j in range(i + 1, len(caps)):
                b = set(_words(caps[j]["caption"])) - STOP
                if a and b and len(a & b) / len(a | b) > 0.8:
                    near.append((caps[i]["file"], caps[j]["file"]))
        if near:
            issue("info", "same_captions", t("Almost the same caption for different images: describe what changes in each one."),
                  [x for p in near for x in p], pairs=near)

    order = {"warn": 0, "info": 1}
    issues.sort(key=lambda i: (order.get(i["level"], 2), -len(i["files"])))
    return {"count": n, "captions": len(caps), "edit": edit, "folder": str(folder), "issues": issues,
            "shapes": shapes.most_common(), "faces": faces,
            "sizes": {"min": min(min(r["w"], r["h"]) for r in items), "max": max(max(r["w"], r["h"]) for r in items)},
            "captions_text": {r["file"]: r["caption"] or "" for r in items}}


def _repeated(caps, trigger, issue, character=False):
    """Frases de 2-4 palabras que salen en casi todos los captions: suelen ser rasgos fijos (pelo, ojos...)."""
    trig = set(_words(trigger))
    seen = Counter()
    for r in caps:
        w = [x for x in _words(r["caption"]) if x not in trig]
        grams = set()
        for k in (2, 3, 4):
            for i in range(len(w) - k + 1):
                g = w[i:i + k]
                if g[0] in STOP or g[-1] in STOP or sum(x not in STOP for x in g) < 2:
                    continue
                grams.add(" ".join(g))
        seen.update(grams)
    need = max(3, int(len(caps) * 0.6 + 0.999))
    common = [(g, c) for g, c in seen.items() if c >= need]
    # Las más largas primero; una frase contenida en otra ya mostrada no se repite.
    common.sort(key=lambda gc: (-len(gc[0].split()), -gc[1]))
    shown = []
    for g, c in common:
        if not any(g in s for s, _ in shown):
            shown.append((g, c))
    shown = sorted(shown, key=lambda gc: -gc[1])[:8]
    if shown and len(caps) >= 4:
        fmt = lambda items: ", ".join("«{}» ({:.0f}%)".format(g, 100 * c / len(caps)) for g, c in items)
        traits = [gc for gc in shown if TRAITS.search(gc[0])] if character else []
        rest = [gc for gc in shown if gc not in traits]
        if traits:
            issue("warn", "repeated", t("Repeated in almost every caption: {phrases}. If these are fixed traits of the person (hair, eyes, skin, age...), the model ties them to the words and not to the trigger, and the LoRA looks less like the person with other prompts. Use the Character caption type.", phrases=fmt(traits)),
                  [r["file"] for r in caps], phrases=traits)
        if rest:
            issue("info", "repeated", t("Repeated in almost every caption: {phrases}. What repeats in the captions is tied to those words: if it is the style or subject the trigger should learn, leave it out of the captions.", phrases=fmt(rest)),
                  [r["file"] for r in caps], phrases=rest)


def _faces(folder, items, issue):
    """Sin cara, varias caras, o poco parecido al resto (con InsightFace, en CPU)."""
    import numpy as np
    import face_match
    eng = face_match._get_engine()
    embs, none, many = {}, [], []
    for r in items:
        found = eng.faces(str(folder / r["file"]))
        if not found:
            none.append(r["file"])
        else:
            if len(found) > 1:
                many.append(r["file"])
            embs[r["file"]] = found[0]
    if len(embs) < max(3, len(items) // 2):
        return {"with_face": len(embs), "character": False}
    ref = np.mean(list(embs.values()), axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    sims = {f: float(e @ ref) for f, e in embs.items()}
    vals = sorted(sims.values())
    med = vals[len(vals) // 2]
    odd = [f for f, s in sims.items() if s < min(0.35, med - 0.2)]
    issue("warn", "no_face", t("No face found, in a character dataset: fine for body shots, but check them."), none)
    issue("info", "many_faces", t("More than one face: the LoRA can mix the people."), many)
    issue("warn", "other_person", t("Look much less like the rest: maybe another person, or an unusual angle."), odd,
          scores={f: round(sims[f] * 100) for f in odd})
    return {"with_face": len(embs), "character": True, "median": round(med * 100),
            "scores": {f: round(s * 100) for f, s in sims.items()}}


def register(app, get_dataset_dir):
    from flask import jsonify, request, send_from_directory

    @app.route("/dataset_analyzer.js")
    def dataset_analyzer_js():
        return send_from_directory(str(GUI_DIR), "dataset_analyzer.js")

    @app.route("/api/dataset-analyze", methods=["POST"])
    def dataset_analyze():
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(analyze(get_dataset_dir(), data.get("trigger", ""), bool(data.get("faces", True))))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
