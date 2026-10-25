# -*- coding: utf-8 -*-
"""
0_caption_flux3klein.py — Captions automáticos del dataset con Qwen3-VL-8B, en el formato de FLUX.3
Dataset auto-captioning with Qwen3-VL-8B, in the FLUX.3 caption format

Cuatro modos, adaptados de las instrucciones de sistema del prompt upsampler de BFL (diffusers,
modular_pipelines/flux3/prompt_upsampler.py: t2i, t2i_bbox, i2i, i2i_bbox) para describir una imagen
que ya existe en vez de inventarla / Four modes, adapted from BFL's prompt upsampler system prompts:
  normal        un párrafo de prosa detallado por imagen (medio, escena, luz, posiciones, sujetos).
  normal_boxes  el párrafo citando cada elemento como <person_1>... + tabla JSON [{"id","bbox","desc"}].
  edit          pares nombre_before + nombre_after: la instrucción detallada que convierte el antes
                en el después, cerrando con que el resto queda igual.
  edit_boxes    la instrucción citando <ref_image_0> y los <id> + tabla JSON
                [{"id","from","src_bbox","tgt_bbox","desc"}]: el formato de anotación con el que BFL
                entrenó la edición. El captioner ve el antes Y el después, así que las cajas salen de
                lo que cambió de verdad.
Las cajas de FLUX.3 son [y_min, x_min, y_max, x_max] en una cuadrícula 0-1000. A Qwen se le piden en su
formato nativo de grounding ([x1, y1, x2, y2] en 0-1000, el que mejor sabe dar) y aquí se convierten.
Con cajas va en dos pasos: primero la tabla en JSON y después el párrafo, que cita los ids de esa tabla.

La palabra trigger va siempre al principio. / The trigger word always goes first.

Usa el mismo captioner que LTX-2.3, Z-Image y Anima: el text encoder NF4 del repo de Qwen-Image 2.1
(Qwen3-VL-8B-Instruct, ~5.5 GB de VRAM). La primera vez se descargan solo text_encoder_NF4/ y
processor/ (~5 GB) a Models/Captioner-Qwen3-VL-8B/, compartido con los demás entrenadores.

Lee caption_settings_flux3klein.json; la ruta del dataset y el trigger salen de
pre_cache_settings_flux3klein.json.
"""
import os
import re
import sys
import json
import time
import warnings

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from i18n import t
from project_layout import resolve_model_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas de visión (4304) no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

# Partes comunes de las instrucciones, tomadas de las de BFL.
_PROSE = (
    "Lead with the medium and style (\"Full-color outdoor photograph, ...\", \"Full-color indoor photograph, ...\", "
    "\"Studio photograph, ...\", \"Close-up photograph, ...\", or the art style and medium of a non-photographic image), "
    "then the scene, lighting and camera, then the subjects with concrete colors, materials and state. Every person in "
    "focus gets an observable profile: ethnicity, gender, age range, skin tone (\"fair skin\", \"brown skin\"), hair, "
    "clothing, pose and expression. Make the layout reconstructable from words: frame-relative positions (frame-left, "
    "frame-right, upper, lower, foreground, midground, background), sentence-initial anchors (\"On the left, ...\", "
    "\"In the background, ...\") and relations (holds, faces, behind). Out-of-focus or distant planes are shapes, colors "
    "and silhouettes, and the background takes about a sixth of the caption. Transcribe every legible text verbatim "
    "inside double quotes with its typography, color and placement. Be definite: never \"this image shows\", \"we can "
    "see\" or \"is visible\", no hedging, no quality or emotion judgments, no closing sentence about mood. Do not "
    "invent names. All prose is English."
)
_ROWS = (
    "id: the bare class noun plus an index numbered left to right, then top to bottom, from 1 (person_1, dog_2); a text "
    "block is its two-letter language code plus _Text_ plus an index (En_Text_1); ids are unique. Boxes are "
    "[x1, y1, x2, y2] integers from 0 to 1000, x from the left edge and y from the top edge, tight around the element. "
    "desc: one self-contained paragraph opening with the element's identity (never \"the X\"): shape, material, colors, "
    "state and markings, but no position. People get the richest desc: ethnicity, gender, age range, skin tone, build, "
    "hair, face, each garment, accessories, pose and expression. A text row quotes its wording verbatim plus its typography."
)

DEFAULTS = {
    "captioner_dir": "Models/Captioner-Qwen3-VL-8B",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "caption_mode": "normal",
    "caption_prompt": (
        "Write the caption a FLUX.3 text-to-image model needs to generate exactly this image: ONE continuous prose "
        "block in the generative voice, 100 to 300 words. " + _PROSE + " Output only the caption: no Markdown, JSON, "
        "coordinates, ids or angle-bracket tokens."
    ),
    # Con cajas va en dos pasos: primero la tabla (TABLE_PROMPTS, JSON, donde Qwen es fiable) y luego
    # este párrafo, al que se le añade la tabla para que cite sus ids.
    "caption_boxes_prompt": (
        "Write the global caption a FLUX.3 text-to-image model needs to generate exactly this image: one flowing "
        "paragraph in the generative voice, 80 to 250 words (medium and style first, then scene, lighting, camera and "
        "frame-relative layout). Name each element of the table below at its first mention and cite it with its id in "
        "angle brackets: \"a wicker table <table_1> holding a tiered stand <stand_1>\". Cite every table id exactly "
        "once and no other id; appearance detail lives in the table, not in the paragraph; no coordinates. Be "
        "definite, no quality judgments, do not invent names. Plain prose only: no labels such as \"Medium:\" or "
        "\"Scene:\", no headings, no lists. All prose is English. Output only the paragraph."
    ),
    "caption_edit_prompt": (
        "The first image is the source and the second image is the edited result. Write the instruction an "
        "image-editing model receives together with the source to produce exactly this result: ONE detailed English "
        "prose block, 40 to 150 words, in instruction voice (\"Change the ...\", \"Remove the ...\", \"Replace ...\", "
        "\"Add ...\"), never observer phrasing, never a comparison with the result and never \"first image\" or "
        "\"second image\". Name what changes with frame-relative positions and land every end state on a concrete "
        "value (new colors, materials, text in double quotes, pose, degree); include visible knock-on effects such as "
        "shadows, reflections and lighting. Close with one compact clause that everything else stays identical to the "
        "source. Output only the instruction: no Markdown, JSON, coordinates, ids or angle-bracket tokens."
    ),
    "caption_edit_boxes_prompt": (
        "The first image is the source, <ref_image_0>; the second image is the edited result. Write the instruction "
        "an image-editing model receives together with the source to produce exactly this result: ONE English "
        "paragraph, 40 to 150 words, instruction voice (\"Change the ...\", \"Remove the ...\", \"Add ...\"), never "
        "observer phrasing and never a comparison with the result. Refer to the source as <ref_image_0>. Cite every "
        "item of the table below at its first mention with its id in angle brackets (\"remove the delivery van "
        "<van_1>\"), every id exactly once and no other id. End states (new color, new text in double quotes, pose, "
        "degree) live in the paragraph; static appearance lives in the table; no coordinates. Close with one compact "
        "clause that everything else stays identical to the source. The table lists everything the edit involves "
        "(items with from \"target\" are new): describe only those changes and invent no other. Output only the paragraph."
    ),
    # Tope de cada generación (párrafo o tabla); para sola al terminar.
    "max_new_tokens": 1500,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    # Más píxeles = más tokens de visión y un prefill más lento. 0 = sin reescalar.
    "max_image_side": 1024,
    # False = no toca las muestras que ya tienen un .txt con contenido.
    "overwrite": True,
}

TABLE_PROMPTS = {
    "normal_boxes": (
        "List the elements of this image as ONE JSON array and nothing else: one row per distinct object, person, "
        "animal, building or salient region and one per coherent text block; no rows for style, lighting, atmosphere "
        "or a plain backdrop; more than about ten interchangeable items become one row giving the exact count. At "
        "most 15 rows. Row keys, in order: \"id\", \"bbox_2d\", \"desc\". " + _ROWS
    ),
    "edit_boxes": (
        "The first image is the source and the second image is the edited result. List what the edit involves as ONE "
        "JSON array and nothing else: one row per edited, removed, added, moved or anchoring item, plus the background "
        "when it is itself changed; no rows for untouched scenery. Row keys, in order: \"id\", \"from\", "
        "\"src_bbox_2d\", \"tgt_bbox_2d\", \"desc\". from: \"ref_image_0\" when the item already exists in the "
        "source, \"target\" when it only exists in the edited image (an added item). src_bbox_2d: its box in the "
        "source image, null when it is not there. tgt_bbox_2d: its box in the edited image, null when the edit removed "
        "it. desc: its static appearance in the source (for added items, in the edited image). " + _ROWS
    ),
}

MODES = ("normal", "normal_boxes", "edit", "edit_boxes")
PROMPT_KEYS = {"normal": "caption_prompt", "normal_boxes": "caption_boxes_prompt",
               "edit": "caption_edit_prompt", "edit_boxes": "caption_edit_boxes_prompt"}
# Claves de caja de Qwen -> claves de FLUX.3, en el orden de las filas de BFL.
BOX_KEYS = {"normal_boxes": (("bbox_2d", "bbox"),),
            "edit_boxes": (("src_bbox_2d", "src_bbox"), ("tgt_bbox_2d", "tgt_bbox"))}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
EDIT_SUFFIXES = ("_before", "_after")

PRECACHE_CONFIG = "settings/pre_cache_settings_flux3klein.json"
CONFIG_PATH = "settings/caption_settings_flux3klein.json"
HF_REPO_ID = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_config():
    cfg = dict(DEFAULTS)
    precache = read_json(PRECACHE_CONFIG)
    for key in ("dataset_path", "trigger_word"):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})
    return cfg


def find_samples(dataset, mode):
    """
    Devuelve [(nombre, [rutas de imagen])]. En los modos de edición, el antes va primero.
    En los normales se ignoran las imágenes _before/_after.
    """
    images = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    stems = {os.path.splitext(f)[0]: f for f in images}

    if mode.startswith("edit"):
        samples = []
        for stem, name in stems.items():
            if stem.endswith("_before") and stem[:-len("_before")] + "_after" in stems:
                base = stem[:-len("_before")]
                samples.append((base, [os.path.join(dataset, name), os.path.join(dataset, stems[base + "_after"])]))
        return samples

    return [(stem, [os.path.join(dataset, name)]) for stem, name in stems.items() if not stem.endswith(EDIT_SUFFIXES)]


def load_captioner(captioner_dir):
    captioner_dir = str(resolve_model_dir(captioner_dir, "Models/Captioner-Qwen3-VL-8B"))
    path = os.path.join(captioner_dir, "text_encoder_NF4")
    if not os.path.exists(os.path.join(path, "config.json")):
        print(t("Downloading the captioner from Hugging Face: {repo} (~5 GB)", repo=HF_REPO_ID), flush=True)
        token = read_json("settings/HF_token.json").get("token", "").strip() or None
        snapshot_download(repo_id=HF_REPO_ID, local_dir=captioner_dir, token=token, max_workers=2,
                          allow_patterns=["text_encoder_NF4/*", "processor/*"])

    print(t("Loading {name}...", name="Qwen3-VL-8B (NF4)"), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(os.path.join(captioner_dir, "processor"))
    print(f"{t('Ready.')} VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    return model, processor


def shrink(image, max_side):
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def clean_caption(text):
    text = " ".join(re.sub(r"```(?:json)?", "", text).split())
    for prefix in ("The image shows ", "This image shows ", "The image depicts ", "This image depicts ",
                   "The photo shows ", "In this image, ", "In the image, ", "Sure, ", "Certainly, ", "Here is a ",
                   "Instruction: ", "Caption: "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:]
            break
    return text.strip().strip('"')


def close_json(text):
    """Cierra la cadena, las listas y los objetos que hayan quedado abiertos (tabla cortada)."""
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    text = re.sub(r"[,:\s]+$", "", text + ('"' if in_str else ""))
    return text + "".join(reversed(stack))


def flux3_box(box):
    """[x1, y1, x2, y2] de Qwen -> [y_min, x_min, y_max, x_max] de FLUX.3, en 0-1000."""
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return None
    x1, y1, x2, y2 = (max(0, min(1000, int(round(v)))) for v in box)
    return [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]


def read_table(text, mode):
    """La tabla JSON de Qwen -> filas con las claves, el orden y las cajas de FLUX.3. Si llegó
    cortada, se cierra."""
    start = re.search(r"\[\s*\{", text)
    if not start:
        return []
    text = text[start.start():]
    try:
        rows = json.loads(text[:text.rfind("]") + 1])
    except ValueError:
        try:
            rows = json.loads(close_json(text))
        except ValueError:
            return []

    table = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        out = {"id": str(row["id"]).strip("<>")}
        if mode == "edit_boxes":
            out["from"] = "target" if str(row.get("from", "")).strip() == "target" else "ref_image_0"
        for qwen_key, key in BOX_KEYS[mode]:
            out[key] = flux3_box(row.get(qwen_key, row.get(key)))
        if mode == "edit_boxes":
            # Coherencia de BFL: lo nuevo no tiene caja en el antes; lo que no estaba en el antes es nuevo.
            if out["src_bbox"] is None:
                out["from"] = "target"
            if out["from"] == "target":
                out["src_bbox"] = None
        out["desc"] = str(row.get("desc", ""))
        table.append(out)
    return table


def changed_region(before, after):
    """
    Pista para los modos de edición: qué zona cambió, comparando los píxeles del antes y el después
    (Qwen sola compara mal dos imágenes). Devuelve una frase para el prompt, o "" si no son del mismo
    encuadre y no se pueden comparar.
    """
    if abs(before.width / before.height - after.width / after.height) > 0.02:
        return ""
    size = (256, max(1, round(256 * before.height / before.width)))
    a = np.asarray(before.resize(size, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(after.resize(size, Image.BILINEAR), dtype=np.float32)
    changed = np.abs(a - b).mean(axis=2) > 24
    share = changed.mean()
    if share < 0.002:
        return "Pixel comparison: the two images are almost identical; the edit is subtle."
    if share > 0.5:
        return "Pixel comparison: the edit changes most of the image (a global change such as color, style or lighting)."
    ys, xs = np.nonzero(changed)
    # Percentiles: unos pocos píxeles sueltos no estiran la caja.
    x1, x2 = np.percentile(xs, [1, 99]) / size[0] * 1000
    y1, y2 = np.percentile(ys, [1, 99]) / size[1] * 1000
    return (f"Pixel comparison: the edit changed the region [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] "
            f"([x1, y1, x2, y2] in 0-1000); everything outside it is unchanged.")


def caption_sample(model, processor, images, prompt, cfg, clean=True, sample=True):
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")

    temperature = float(cfg["temperature"]) if sample else 0.0
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            repetition_penalty=1.05,
        )
    text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_caption(text) if clean else text


def with_trigger(trigger, caption):
    trigger = trigger.strip().rstrip(",")
    if not trigger or caption.lower().startswith(trigger.lower()):
        return caption
    return f"{trigger}, {caption}"


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]
    mode = cfg["caption_mode"] if cfg["caption_mode"] in MODES else "normal"
    prompt = cfg[PROMPT_KEYS[mode]]

    if not os.path.isdir(dataset):
        print("[!] " + t("Dataset folder does not exist: {path}", path=dataset))
        return 1

    samples = find_samples(dataset, mode)
    pending = []
    for name, paths in samples:
        txt = os.path.join(dataset, name + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append((name, paths))

    kind = t("before/after pairs") if mode.startswith("edit") else t("images")
    print(f"  {t('Dataset'):<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Mode'):<22}: {mode}")
    print(f"  {t('Samples'):<22}: {len(samples)} {kind}, {t('{n} to caption', n=len(pending))}")
    print(f"  {t('Trigger Word'):<22}: {cfg['trigger_word'] or t('(none)')}")
    print(f"  {t('Overwrite'):<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

    if not samples and mode.startswith("edit"):
        print("[!] " + t("No pairs found. Name them name_before.ext and name_after.ext."))
        return 1
    if not pending:
        print(t("Nothing to do: every sample already has a caption."))
        return 0

    model, processor = load_captioner(cfg["captioner_dir"])

    started = time.time()
    done, failed = 0, []
    for i, (name, paths) in enumerate(pending, 1):
        t0 = time.time()
        try:
            images = []
            for p in paths:
                with Image.open(p) as img:
                    images.append(shrink(img.convert("RGB"), int(cfg["max_image_side"])))
            note = ""
            hint = changed_region(*images) if mode.startswith("edit") else ""
            if hint:
                hint += " Use it only to locate the change; never write coordinates or numbers from it."
            ask = prompt + ("\n" + hint if hint else "")
            if mode in BOX_KEYS:
                table_ask = TABLE_PROMPTS[mode] + ("\n" + hint if hint else "")
                table = read_table(caption_sample(model, processor, images, table_ask, cfg, clean=False, sample=False), mode)
                if not table:
                    raise RuntimeError(t("no valid JSON table"))
                # El párrafo solo necesita los ids y qué es cada uno: sin cajas, para que no las copie.
                listed = [{k: v for k, v in row.items() if not k.endswith("bbox")} for row in table]
                base = ask + "\nTable: " + json.dumps(listed, ensure_ascii=False)
                ask = base
                # Un reintento si el párrafo no cita todos los ids: sin cita, la tabla no se asocia al texto.
                for attempt in range(2):
                    paragraph = caption_sample(model, processor, images, ask, cfg)
                    cited = set(re.findall(r"<([^<>\s]+)>", paragraph))
                    missing = [row["id"] for row in table if row["id"] not in cited]
                    if not missing:
                        break
                    ask = (base + "\nYou MUST write each of these "
                           "ids in angle brackets right after the words that name it: " + ", ".join(f"<{m}>" for m in missing))
                if missing:
                    note = t("ids not cited in the paragraph: {ids}", ids=", ".join(missing[:5]))
                caption = f"{paragraph} {json.dumps(table, ensure_ascii=False)}"
            else:
                caption = caption_sample(model, processor, images, ask, cfg)
            if not caption:
                raise RuntimeError(t("empty caption"))

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
            if note:
                print(f"    [!] {note}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} {t('FAILED')}: {e}", flush=True)

    total = time.time() - started
    print("\n✓ " + t("{n} captions in {time} ({per:.1f}s per sample).", n=done, time=f"{int(total // 60)}m{int(total % 60):02d}s", per=total / max(done, 1)))
    if failed:
        print("[!] " + t("{n} failed: {names}", n=len(failed), names=', '.join(failed[:10])))
    print(t("Review them in the Dataset Manager before pre-caching."))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
