# -*- coding: utf-8 -*-
"""
0_caption_ideogram4.py — Captions automáticos del dataset con Qwen3-VL-8B
Dataset auto-captioning with Qwen3-VL-8B

Un .txt por imagen. La palabra trigger va siempre al principio.
One .txt per image. The trigger word always goes first.

Tres estilos / Three styles:
  json           el caption estructurado nativo de Ideogram 4 (el formato de Ideogram 4.5 y FLUX.3
                 Image), con una caja por cada sujeto, objeto y texto.
  json_detailed  lo mismo, y además una caja por cada parte de cada persona o animal (cara, nariz,
                 orejas, manos, piernas...), prenda y accesorio: para LoRAs de edición.
  natural        un párrafo de descripción.

Los captions JSON se hacen en dos pasadas: el caption con sus cajas y un OCR que lee cada rótulo con
su caja (en una sola pasada el modelo se deja muchos textos en escenas cargadas). El resultado se
guarda en el formato oficial: claves en su orden, cajas [y1, x1, y2, x2] en 0-1000, sin repetidos.
Qwen3-VL localiza en su formato nativo [x1, y1, x2, y2]; aquí se le da la vuelta.

Usa el captioner compartido con los demás entrenadores: el text encoder NF4 del repo de Qwen-Image
2.1 (Qwen3-VL-8B-Instruct, ~5.5 GB de VRAM). El text encoder de Ideogram 4 es el mismo modelo sin
lm_head, así que no puede escribir. La primera vez se descargan solo text_encoder_NF4/ y processor/
(~5 GB) a Captioner-Qwen3-VL-8B/.

Lee caption_settings_ideogram4.json; la ruta del dataset y el trigger salen de
pre_cache_settings_ideogram4.json.
"""
import os
import re
import sys
import json
import time
import warnings

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas de visión (4304) no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

DEFAULTS = {
    "captioner_dir": "Captioner-Qwen3-VL-8B",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "caption_style": "json",
    "caption_prompt": (
        "Describe this image for training a text-to-image model, in at most 150 words. "
        "One single paragraph of plain prose, no lists and no preamble. Cover the main "
        "subject and what they are doing, their appearance, hair and clothing, the pose, "
        "the framing and camera angle, the setting and background, the lighting and the "
        "colors. Do not invent a name and do not mention that this is an image or a photo."
    ),
    # Estilo natural: 150 palabras son ~200 tokens. Los JSON con cajas usan su propio límite.
    "max_new_tokens": 300,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    # Más píxeles = más tokens de visión y un prefill más lento. 0 = sin reescalar.
    "max_image_side": 1024,
    # False = no toca las imágenes que ya tienen un .txt con contenido.
    "overwrite": True,
}

JSON_MAX_TOKENS = 4096

OCR_PROMPT = (
    "Read every piece of legible text in this image: signs, shop names, posters, labels, screens, prices, logos "
    "with letters, license plates. Go through the whole image from left to right and from top to bottom. Each "
    "complete sign or text block is ONE item, with all its lines together and \\n between lines; never one item per "
    "word. Copy the characters exactly in their own script (Korean, Chinese, Japanese and Arabic stay as they are). "
    "At most 40 items. If there is no legible text, output [].\n"
    'Output a JSON list and nothing else: [{"bbox_2d":[x1,y1,x2,y2],"text":"..."}, ...] with x from the left edge (0) '
    "to the right edge (1000) and y from the top edge (0) to the bottom edge (1000)."
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PRECACHE_CONFIG = "settings/pre_cache_settings_ideogram4.json"
CONFIG_PATH = "settings/caption_settings_ideogram4.json"
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


def load_captioner(captioner_dir):
    path = os.path.join(captioner_dir, "text_encoder_NF4")
    if not os.path.exists(os.path.join(path, "config.json")):
        print(f"Downloading captioner from Hugging Face / Descargando el captioner desde Hugging Face: {HF_REPO_ID} (~5 GB)", flush=True)
        token = read_json("settings/HF_token.json").get("token", "").strip() or None
        snapshot_download(repo_id=HF_REPO_ID, local_dir=captioner_dir, token=token, max_workers=2,
                          allow_patterns=["text_encoder_NF4/*", "processor/*"])

    print("Loading Qwen3-VL-8B (NF4)... / Cargando Qwen3-VL-8B (NF4)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(os.path.join(captioner_dir, "processor"))
    print(f"Ready / Listo. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    return model, processor


def shrink(image, max_side):
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def clean_caption(text):
    text = " ".join(text.split())
    for prefix in ("The image shows ", "This image shows ", "The image depicts ", "This image depicts ",
                   "The photo shows ", "In this image, ", "In the image, ", "Sure, ", "Certainly, ", "Here is a "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:]
            break
    return text.strip().strip('"')


class StopOnLoop(StoppingCriteria):
    """En escenas cargadas el modelo a veces repite un elemento sin fin: se corta cuando el mismo
    "desc" aparece tres veces. El JSON se cierra después y lo repetido se quita."""
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] % 32:
            return False
        text = self.tokenizer.decode(input_ids[0][self.start:], skip_special_tokens=True)
        descs = re.findall(r'"(?:desc|text)"\s*:\s*"([^"]{12,})"', text[-6000:])
        return len(descs) >= 3 and descs.count(descs[-1]) >= 3


def generate(model, processor, image, prompt, max_new_tokens, temperature):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
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
    if not isinstance(merged.get("compositional_deconstruction"), dict):
        raise ValueError("JSON without the Ideogram caption keys / JSON sin las claves del caption de Ideogram")
    return merged


def loads_list(text):
    start = text.find("[")
    if start < 0:
        return []
    try:
        items = json.loads(text[start:text.rfind("]") + 1])
    except ValueError:
        items = repair_tail(text[start:]) or []
    return [it for it in items if isinstance(it, dict) and isinstance(it.get("bbox_2d"), list) and len(it["bbox_2d"]) == 4]


# ── Caption JSON ─────────────────────────────────────────────────────────────

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda q: max(0, q[2] - q[0]) * max(0, q[3] - q[1])
    inter = ix * iy
    return inter / max(1, area(a) + area(b) - inter)


def merge_ocr(elements, texts):
    """Los textos del OCR corrigen los caracteres del elemento de texto con el que se solapan; los que
    no se solapan con ninguno se añaden detrás del sujeto principal."""
    added = []
    for x in texts:
        boxed = [e for e in elements if e.get("type") == "text" and isinstance(e.get("bbox_2d"), list) and len(e["bbox_2d"]) == 4]
        match = max(boxed, key=lambda e: iou(e["bbox_2d"], x["bbox_2d"]), default=None)
        if match is not None and iou(match["bbox_2d"], x["bbox_2d"]) > 0.3:
            match["text"] = str(x.get("text", match.get("text", "")))
        else:
            added.append({"type": "text", "bbox_2d": x["bbox_2d"], "text": str(x.get("text", "")),
                          "desc": "Legible text in the same style, color and placement as it appears in the scene."})
    return elements[:1] + added + elements[1:]


def hexes(colors, limit):
    out = []
    for c in colors if isinstance(colors, list) else []:
        m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(c).strip())
        if m:
            out.append("#" + m.group(1).upper())
    return out[:limit]


def ideogram_caption(data, texts):
    """Formato oficial de Ideogram 4: orden de claves, cajas [y1, x1, y2, x2] en 0-1000, hex en
    mayúsculas y sin elementos repetidos."""
    sd_in = data.get("style_description") if isinstance(data.get("style_description"), dict) else {}
    sd = {"aesthetics": str(sd_in.get("aesthetics", "")), "lighting": str(sd_in.get("lighting", ""))}
    if str(sd_in.get("medium", "")).lower().startswith("photo"):
        sd["photo"] = str(sd_in.get("photo", ""))
        sd["medium"] = "photograph"
    else:
        sd["medium"] = str(sd_in.get("medium", ""))
        sd["art_style"] = str(sd_in.get("art_style", ""))
    palette = hexes(sd_in.get("color_palette"), 16)
    if palette:
        sd["color_palette"] = palette

    cd = data["compositional_deconstruction"]
    raw_elements = [e for e in cd.get("elements") or [] if isinstance(e, dict)]
    elements, seen = [], set()
    for e in merge_ocr(raw_elements, texts):
        el = {"type": "text" if e.get("type") == "text" else "obj"}
        box = e.get("bbox_2d")
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = (max(0, min(1000, int(round(v)))) for v in box)
            el["bbox"] = [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
        if el["type"] == "text":
            el["text"] = str(e.get("text", ""))
        el["desc"] = str(e.get("desc", ""))
        keys = {("desc", el.get("text", ""), el["desc"][:60].lower())}
        if "bbox" in el:
            keys.add(("box", el["type"], tuple(el["bbox"])))
        if keys & seen:
            continue
        seen |= keys
        elements.append(el)

    caption = {"high_level_description": str(data.get("high_level_description", "")), "style_description": sd,
               "compositional_deconstruction": {"background": str(cd.get("background", "")), "elements": elements}}
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def caption_image(model, processor, image, prompt, cfg):
    temperature = float(cfg["temperature"])
    if not cfg["caption_style"].startswith("json"):
        return clean_caption(generate(model, processor, image, prompt, int(cfg["max_new_tokens"]), temperature))
    data = loads_caption(generate(model, processor, image, prompt, JSON_MAX_TOKENS, temperature))
    texts = loads_list(generate(model, processor, image, OCR_PROMPT, 2048, temperature))
    return ideogram_caption(data, texts)


def with_trigger(trigger, caption):
    trigger = trigger.strip().rstrip(",")
    if not trigger or trigger.lower() in caption.lower():
        return caption
    if caption.startswith("{"):
        data = json.loads(caption)
        data["high_level_description"] = f"{trigger}, {data['high_level_description']}"
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"{trigger}, {caption}"


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]

    if not os.path.isdir(dataset):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {dataset}")
        return 1

    images = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    pending = []
    for name in images:
        txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append(name)

    print(f"  Dataset / Ruta Dataset : {os.path.abspath(dataset)}")
    print(f"  Images / Imágenes      : {len(images)}, {len(pending)} to caption / por describir")
    print(f"  Style / Estilo         : {cfg['caption_style']}")
    print(f"  Trigger Word / Palabra : {cfg['trigger_word'] or '(none / ninguna)'}")
    print(f"  Overwrite / Rehacer    : {'yes / sí' if cfg['overwrite'] else 'only missing / solo los que faltan'}")

    if not pending:
        print("Nothing to do: every image already has a caption. / Nada que hacer: todas las imágenes ya tienen caption.")
        return 0

    model, processor = load_captioner(cfg["captioner_dir"])

    started = time.time()
    done, failed = 0, []
    for i, name in enumerate(pending, 1):
        t0 = time.time()
        stem = os.path.splitext(name)[0]
        try:
            with Image.open(os.path.join(dataset, name)) as img:
                image = shrink(img.convert("RGB"), int(cfg["max_image_side"]))
            caption = caption_image(model, processor, image, cfg["caption_prompt"], cfg)
            if not caption:
                raise RuntimeError("empty caption / caption vacío")

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            extra = ""
            if final.startswith("{"):
                els = json.loads(final)["compositional_deconstruction"]["elements"]
                extra = f" | {len(els)} elements, {sum(e['type'] == 'text' for e in els)} texts"
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s{extra} | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} FAILED / FALLÓ: {e}", flush=True)

    total = time.time() - started
    print(f"\n✓ {done} captions in / en {int(total // 60)}m{int(total % 60):02d}s ({total / max(done, 1):.1f}s per image / por imagen).")
    if failed:
        print(f"[!] {len(failed)} failed / fallaron: {', '.join(failed[:10])}")
    print("Review them in the Dataset Manager before pre-caching. / Revísalos en el Dataset Manager antes del pre-caché.")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
