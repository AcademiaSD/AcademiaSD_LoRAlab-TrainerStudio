# -*- coding: utf-8 -*-
"""
0_caption_mingimage.py — Captions automáticos del dataset con Qwen3-VL-8B
Dataset auto-captioning with Qwen3-VL-8B

Un .txt por imagen en el formato nativo de Ming-Image: un JSON estilo Figma con
`canvas_settings` (aspect_ratio, ambient_lighting, image_style) y una lista de
`layers` (description, coordinates, hierarchy_and_relation, color_specs).
One .txt per image in Ming-Image's native format: a Figma-style JSON caption.

La palabra trigger se añade al principio de `canvas_settings.image_style`.
The trigger word is prepended to `canvas_settings.image_style`.

Modo edit (pares nombre_before + nombre_after): el modelo ve las dos imágenes y escribe en
nombre.txt UNA instrucción en texto libre, que es lo que recibe la tarea image-edit de Ming
(no el JSON). Edit mode writes one free-text instruction per pair to name.txt.

Usa el mismo captioner que los demás entrenadores: el text encoder NF4 del repo de
Qwen-Image 2.1 (Qwen3-VL-8B-Instruct, ~5.5 GB de VRAM). La primera vez se descargan
solo text_encoder_NF4/ y processor/ (~5 GB) a Captioner-Qwen3-VL-8B/, compartido.
The shared Qwen3-VL-8B captioner is used, the same one as the other trainers.

Lee caption_settings_mingimage.json; la ruta del dataset y el trigger salen de
pre_cache_settings_mingimage.json.
"""
import os
import re
import sys
import json
import time
import warnings
from fractions import Fraction

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
from i18n import t
from project_layout import resolve_model_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas de visión (4304) no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

# Prompt oficial de Ming-Image (assets/t2i_rewriter_system_prompt.txt) adaptado a describir una
# imagen en vez de expandir una petición. Debe coincidir con DEFAULT_CAPTION_PROMPT del HTML.
CAPTION_PROMPT = (
    "Describe this image as one precise, high-resolution Figma-style caption. Return only one JSON object.\n\n"
    "Use exactly two top-level keys. `canvas_settings` contains exactly `aspect_ratio`, `ambient_lighting` and "
    "`image_style`. `layers` lists the visible groups from background to topmost overlay. Every layer contains "
    "exactly `description`, `coordinates`, `hierarchy_and_relation` and `color_specs`; `color_specs` is an array "
    "of hex colors.\n\n"
    "`coordinates` MUST be one string, never an object or array, in exactly this form: "
    "\"cx: 0.500, cy: 0.500, w: 1.000, h: 1.000\". Values are normalized 0-1; each bbox encloses its complete "
    "owned object and stays inside the canvas.\n\n"
    "A layer is one selectable visible semantic group: background, full person, coherent object, panel, card, "
    "row, or text block. Prefer the fewest groups that preserve the layout. Keep people and objects intact. "
    "Never create invisible parents, guides, placeholders, empty layers, duplicate summaries, or multiple "
    "owners for one element.\n\n"
    "Copy every legible rendered string character-for-character and as one contiguous string, in its own "
    "script. Enumerate intended copy and never hide content behind \"other text\", \"remaining labels\" or "
    "\"etc.\".\n\n"
    "Describe concrete composition, typography, materials, texture, lighting, pose and camera treatment "
    "without literary filler. Use `hierarchy_and_relation` only for ownership, alignment, containment, "
    "stacking and occlusion.\n\n"
    "Infer structured layouts first. Use one complete layer per card and state its row and column. A compact "
    "secondary table may be one layer only if every header and cell is listed; otherwise use one complete "
    "header and one complete layer per body row, binding values to columns and stating blanks. Enumerate "
    "sequences, schedules, spans, gaps and vacant tracks in visual order. Do not mistake ordinary alignment "
    "for a table.\n\n"
    "Silently verify schema, string coordinates, Z-order, exact-text counts, geometry, bbox validity and "
    "completeness."
)

CAPTION_EDIT_PROMPT = (
    "The first image is the original and the second image is the edited result. Write ONE "
    "imperative instruction, in at most 40 words, that turns the first image into the second. "
    "Start with a verb such as Colorize, Remove, Replace, Change, Add or Make. Say what changes "
    "and, briefly, what must stay the same. Do not describe content that does not change, do "
    "not mention 'first image' or 'second image', and write no preamble."
)

# Una instrucción de 40 palabras son ~60 tokens: el límite del JSON (3072) solo alargaría un bucle.
EDIT_MAX_NEW_TOKENS = 300

DEFAULTS = {
    "captioner_dir": "Models/Captioner-Qwen3-VL-8B",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "caption_mode": "normal",
    "caption_prompt": CAPTION_PROMPT,
    # El mismo que Qwen-Image 2512: la tarea image-edit de Ming recibe la instrucción tal cual.
    # Debe coincidir con DEFAULT_CAPTION_PROMPTS.edit del HTML.
    "caption_edit_prompt": CAPTION_EDIT_PROMPT,
    # Un caption de 5 capas ronda 1.200 tokens; el margen evita JSON cortados.
    "max_new_tokens": 3072,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    # 1024 hace legibles los textos de las maquetas; más píxeles = más tokens de visión.
    "max_image_side": 1024,
    # False = no toca las imágenes que ya tienen un .txt con contenido.
    "overwrite": True,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PRECACHE_CONFIG = "settings/pre_cache_settings_mingimage.json"
CONFIG_PATH = "settings/caption_settings_mingimage.json"
HF_REPO_ID = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"

# El aviso de transparencia va SOLO cuando el fichero tiene alpha de verdad: si se deja en el prompt
# base, el modelo llama "transparent" a fondos opacos (6 de 30 en las pruebas).
TRANSPARENT_HINT = (
    "The background of this image is transparent (it has an alpha channel). "
    "Describe the canvas as transparent instead of inventing a scene behind it.\n\n"
)
OPAQUE_HINT = "This canvas is fully opaque: do not describe the background as transparent.\n\n"

COORD_RE = re.compile(
    r"cx\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*cy\s*:\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*w\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*h\s*:\s*(-?\d+(?:\.\d+)?)"
)


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


def prepare_image(path, max_side):
    """Devuelve (imagen RGB para el modelo, tiene_alpha, tamaño original). El alpha se compone sobre
    blanco para que el modelo no vea manchas negras, y se le avisa de que el fondo real es transparente."""
    with Image.open(path) as img:
        size = img.size
        has_alpha = False
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            lo, _ = rgba.getchannel("A").getextrema()
            has_alpha = lo < 255
            if has_alpha:
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, rgba)
        return shrink(img.convert("RGB"), max_side), has_alpha, size


class StopOnLoop(StoppingCriteria):
    """En maquetas cargadas el modelo a veces repite una capa sin fin: se corta cuando la misma
    descripción aparece tres veces. El JSON se cierra después y lo repetido se quita."""
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] % 32:
            return False
        text = self.tokenizer.decode(input_ids[0][self.start:], skip_special_tokens=True)
        descs = re.findall(r'"(?:description|hierarchy_and_relation)"\s*:\s*"([^"]{12,})"', text[-8000:])
        return len(descs) >= 3 and descs.count(descs[-1]) >= 3


def generate(model, processor, images, prompt, max_new_tokens, temperature):
    images = images if isinstance(images, list) else [images]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
    stop = StoppingCriteriaList([StopOnLoop(processor.tokenizer, inputs["input_ids"].shape[1])])
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-4), top_p=0.9, repetition_penalty=1.05,
                             stopping_criteria=stop)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Lectura tolerante del JSON ───────────────────────────────────────────────

def close_json(text):
    """Cierra la cadena, las listas y los objetos que hayan quedado abiertos."""
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


def repair_tail(fragment):
    """Un JSON cortado: se cierra, quitando elementos del final hasta que sea válido."""
    while True:
        try:
            return json.loads(close_json(fragment))
        except ValueError:
            k = max(fragment.rfind("},"), fragment.rfind("}]"))
            if k <= 0:
                return None
            fragment = fragment[:k + 1]


def loads_caption(text):
    """El caption llega a veces partido en varios objetos, sin cerrar o las dos cosas: se juntan los
    trozos y el último se cierra."""
    text = re.sub(r"```(?:json)?", "", text)
    dec, merged, i = json.JSONDecoder(), {}, text.find("{")
    while i >= 0:
        try:
            obj, i = dec.raw_decode(text, i)
        except ValueError:
            obj, i = repair_tail(text[i:]), -1
        if isinstance(obj, dict):
            merged.update(obj)
        if i >= 0:
            i = text.find("{", i)
    if not isinstance(merged.get("layers"), list):
        raise ValueError(t("JSON without the Ming-Image caption keys"))
    return merged


# ── Normalización del caption ────────────────────────────────────────────────

def text_of(value):
    return " ".join(str(value if value is not None else "").split())


def aspect_ratio(size):
    w, h = size
    frac = Fraction(w, h).limit_denominator(64)
    return f"{frac.numerator}:{frac.denominator}, {w} × {h} px"


def coordinates_of(value):
    """Siempre el string oficial; admite que el modelo devuelva un objeto o una lista."""
    if isinstance(value, dict):
        value = f"cx: {value.get('cx')}, cy: {value.get('cy')}, w: {value.get('w')}, h: {value.get('h')}"
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        value = f"cx: {value[0]}, cy: {value[1]}, w: {value[2]}, h: {value[3]}"
    m = COORD_RE.search(str(value or ""))
    if not m:
        return None
    cx, cy, w, h = (float(v) for v in m.groups())
    w = min(max(w, 0.001), 1.0)
    h = min(max(h, 0.001), 1.0)
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return f"cx: {cx:.3f}, cy: {cy:.3f}, w: {w:.3f}, h: {h:.3f}"


def hexes(colors, limit):
    out, seen = [], set()
    for c in colors if isinstance(colors, list) else []:
        m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(c).strip())
        if m and m.group(1).upper() not in seen:
            seen.add(m.group(1).upper())
            out.append("#" + m.group(1).upper())
    return out[:limit]


def ming_caption(data, size, trigger):
    """Formato nativo: dos claves, capas de fondo a frente, coordenadas como string normalizado,
    colores hex en mayúsculas y sin capas repetidas."""
    cs = data.get("canvas_settings") if isinstance(data.get("canvas_settings"), dict) else {}
    style = text_of(cs.get("image_style"))
    trigger = trigger.strip().rstrip(",")
    if trigger and trigger.lower() not in style.lower():
        style = f"{trigger}, {style}".strip(", ")

    canvas = {
        "aspect_ratio": aspect_ratio(size),
        "ambient_lighting": text_of(cs.get("ambient_lighting")),
        "image_style": style,
    }

    layers, seen = [], set()
    for layer in data["layers"]:
        if not isinstance(layer, dict):
            continue
        desc = text_of(layer.get("description"))
        if not desc or desc.lower() in seen:
            continue
        coords = coordinates_of(layer.get("coordinates"))
        if coords is None:
            # Sin caja el esquema no vale, pero la descripción sí: se deja a lienzo completo.
            coords = "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000"
            print("[!] " + t("Layer without valid coordinates, using the full canvas: {desc}", desc=desc[:60]), flush=True)
        seen.add(desc.lower())
        layers.append({
            "description": desc,
            "coordinates": coords,
            "hierarchy_and_relation": text_of(layer.get("hierarchy_and_relation")),
            "color_specs": hexes(layer.get("color_specs"), 10),
        })

    if not layers:
        raise ValueError(t("JSON without layers"))

    return json.dumps({"canvas_settings": canvas, "layers": layers}, ensure_ascii=False, separators=(",", ":"))


def caption_image(model, processor, image, size, has_alpha, cfg):
    prompt = (TRANSPARENT_HINT if has_alpha else OPAQUE_HINT) + cfg["caption_prompt"]
    data = loads_caption(generate(model, processor, image, prompt, int(cfg["max_new_tokens"]),
                                  float(cfg["temperature"])))
    return ming_caption(data, size, cfg["trigger_word"])


def edit_instruction(model, processor, images, cfg):
    """Instrucción de edición: una frase en prosa, sin comillas ni preámbulo, con el trigger delante."""
    text = generate(model, processor, images, cfg["caption_edit_prompt"],
                    min(int(cfg["max_new_tokens"]), EDIT_MAX_NEW_TOKENS), float(cfg["temperature"]))
    text = " ".join(text.split())
    for prefix in ("Instruction: ", "Sure, ", "Certainly, "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
    text = text.strip().strip('"').strip()
    trigger = cfg["trigger_word"].strip().rstrip(",")
    if trigger and not text.lower().startswith(trigger.lower()):
        text = f"{trigger}, {text}"
    return text


def find_pairs(dataset):
    """[(nombre, ruta _before, ruta _after)] de los pares completos."""
    names = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    stems = {os.path.splitext(n)[0]: n for n in names}
    return [(stem[:-len("_before")], os.path.join(dataset, name), os.path.join(dataset, stems[stem[:-len("_before")] + "_after"]))
            for stem, name in stems.items() if stem.endswith("_before") and stem[:-len("_before")] + "_after" in stems]


def find_images(dataset):
    """Para un modelo texto→imagen se captiona el objetivo de cada par antes/después: en
    nombre_before + nombre_after, la imagen _before es la ENTRADA de una edición, no el resultado
    que se quiere aprender, así que se ignora. Un _before sin pareja se captiona como una imagen más."""
    names = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    stems = {os.path.splitext(n)[0] for n in names}
    out = []
    for name in names:
        stem = os.path.splitext(name)[0]
        if stem.endswith("_before") and stem[:-len("_before")] + "_after" in stems:
            continue
        out.append(name)
    return names, out


def main_edit(cfg, dataset):
    pairs = find_pairs(dataset)
    pending = []
    for name, before, after in pairs:
        txt = os.path.join(dataset, name + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append((name, before, after))

    print(f"  {t('Dataset'):<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Mode'):<22}: edit")
    print(f"  {t('Samples'):<22}: {len(pairs)} {t('before/after pairs')}, {t('{n} to caption', n=len(pending))}")
    print(f"  {t('Style'):<22}: {t('one edit instruction per pair (name.txt)')}")
    print(f"  {t('Trigger Word'):<22}: {cfg['trigger_word'] or t('(none)')}")
    print(f"  {t('Overwrite'):<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

    if not pairs:
        print("[!] " + t("No pairs found. Name them name_before.ext and name_after.ext."))
        return 1
    if not pending:
        print(t("Nothing to do: every sample already has a caption."))
        return 0

    model, processor = load_captioner(cfg["captioner_dir"])

    started = time.time()
    done, failed = 0, []
    for i, (name, before, after) in enumerate(pending, 1):
        t0 = time.time()
        try:
            # Las dos con el alpha compuesto sobre blanco, como en el modo normal.
            images = [prepare_image(p, int(cfg["max_image_side"]))[0] for p in (before, after)]
            caption = edit_instruction(model, processor, images, cfg)
            if not caption:
                raise RuntimeError(t("empty caption"))
            with open(os.path.join(dataset, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {caption[:200]}{'...' if len(caption) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} {t('FAILED')}: {e}", flush=True)

    total = time.time() - started
    print("\n✓ " + t("{n} captions in {time} ({per:.1f}s per sample).", n=done, time=f"{int(total // 60)}m{int(total % 60):02d}s", per=total / max(done, 1)))
    if failed:
        print("[!] " + t("{n} failed: {names}", n=len(failed), names=', '.join(failed[:10])))
    print(t("Review them in the Dataset Manager before pre-caching."))
    return 0 if done else 1


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]

    if not os.path.isdir(dataset):
        print("[!] " + t("Dataset folder does not exist: {path}", path=dataset))
        return 1

    if cfg["caption_mode"] == "edit":
        return main_edit(cfg, dataset)

    all_images, images = find_images(dataset)
    pending = []
    for name in images:
        txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append(name)

    print(f"  {t('Dataset'):<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Images'):<22}: {len(images)}, {t('{n} to caption', n=len(pending))}")
    if len(all_images) != len(images):
        print(f"  {'Pairs':<22}: {len(all_images) - len(images)} _before skipped (input of an edit, not the target)")
    print(f"  {t('Style'):<22}: Ming-Image JSON (Figma-style)")
    print(f"  {t('Trigger Word'):<22}: {cfg['trigger_word'] or t('(none)')}")
    print(f"  {t('Overwrite'):<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

    if not pending:
        print(t("Nothing to do: every image already has a caption."))
        return 0

    model, processor = load_captioner(cfg["captioner_dir"])

    started = time.time()
    done, failed = 0, []
    for i, name in enumerate(pending, 1):
        t0 = time.time()
        stem = os.path.splitext(name)[0]
        try:
            image, has_alpha, size = prepare_image(os.path.join(dataset, name), int(cfg["max_image_side"]))
            if image.size[0] < 2 or image.size[1] < 2:
                raise RuntimeError(t("image too small"))
            caption = caption_image(model, processor, image, size, has_alpha, cfg)
            if not caption:
                raise RuntimeError(t("empty caption"))

            with open(os.path.join(dataset, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption)
            done += 1

            layers = json.loads(caption)["layers"]
            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | {len(layers)} layers"
                  f"{' | alpha' if has_alpha else ''} | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {caption[:200]}{'...' if len(caption) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} {t('FAILED')}: {e}", flush=True)

    total = time.time() - started
    print("\n✓ " + t("{n} captions in {time} ({per:.1f}s per image).", n=done, time=f"{int(total // 60)}m{int(total % 60):02d}s", per=total / max(done, 1)))
    if failed:
        print("[!] " + t("{n} failed: {names}", n=len(failed), names=', '.join(failed[:10])))
    print(t("Review them in the Dataset Manager before pre-caching."))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
