# -*- coding: utf-8 -*-
"""
0_caption_qwen_image21.py — Captions automáticos del dataset con Qwen3-VL-8B
Dataset auto-captioning with Qwen3-VL-8B

Dos modos / Two modes:
  normal  personajes, objetos, estilos: un .txt por imagen con una descripción.
          characters, objects, styles: one .txt per image with a description.
  edit    pares nombre_before.ext + nombre_after.ext: un nombre.txt con la INSTRUCCIÓN
          que convierte el antes en el después (imperativo, como los ejemplos de edición
          de Qwen: "Change the background to a sunset beach"). El modelo ya ve el antes,
          así que el texto dice qué hacer, no qué hay.
          before/after pairs: one name.txt with the imperative INSTRUCTION that turns the
          before into the after.

La palabra trigger va siempre al principio. / The trigger word always goes first.

No descarga ningún modelo: el text encoder de Qwen-Image 2.1 es exactamente
Qwen3-VL-8B-Instruct (comprobado tensor a tensor contra el repo oficial), así que
se usa la variante text_encoder_NF4 del propio modelo (~5.5 GB de VRAM). Es la
única que conserva lm_head: el pre-cache solo lee hidden states y en BF16/INT8
se quitó para ahorrar memoria.

Lee caption_settings_qwenimage21.json; la ruta del dataset y el trigger salen de
pre_cache_settings_qwenimage21.json.
"""
import os
import sys
import json
import time
import warnings

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas de visión (4304) no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

DEFAULTS = {
    "model_id": "Qwen-Image21-NF4",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "caption_mode": "normal",
    "caption_prompt": (
        "Describe this image for training a text-to-image model, in at most 150 words. "
        "One single paragraph of plain prose, no lists and no preamble. Cover the main "
        "subject and what they are doing, their appearance, hair and clothing, the pose, "
        "the framing and camera angle, the setting and background, the lighting and the "
        "colors. Do not invent a name and do not mention that this is an image or a photo."
    ),
    "caption_edit_prompt": (
        "The first image is the original and the second image is the edited result. Write ONE "
        "imperative instruction, in at most 40 words, that turns the first image into the second. "
        "Start with a verb such as Colorize, Remove, Replace, Change, Add or Make. Say what changes "
        "and, briefly, what must stay the same. Do not describe content that does not change, do "
        "not mention 'first image' or 'second image', and write no preamble."
    ),
    # 150 palabras son ~200 tokens; el margen evita captions cortados a media frase.
    "max_new_tokens": 300,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    # Más píxeles = más tokens de visión y un prefill más lento. 0 = sin reescalar.
    "max_image_side": 768,
    # False = no toca las muestras que ya tienen un .txt con contenido.
    "overwrite": True,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
EDIT_SUFFIXES = ("_before", "_after")

PRECACHE_CONFIG = "pre_cache_settings_qwenimage21.json"
CONFIG_PATH = "caption_settings_qwenimage21.json"
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
    for key in ("model_id", "dataset_path", "trigger_word"):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})
    return cfg


def find_samples(dataset, mode):
    """
    Devuelve [(nombre, [rutas de imagen])]. En modo edit, el antes va primero.
    En modo normal se ignoran las imágenes _before/_after.
    """
    images = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    stems = {os.path.splitext(f)[0]: f for f in images}

    if mode == "edit":
        samples = []
        for stem, name in stems.items():
            if stem.endswith("_before") and stem[:-len("_before")] + "_after" in stems:
                base = stem[:-len("_before")]
                samples.append((base, [os.path.join(dataset, name), os.path.join(dataset, stems[base + "_after"])]))
        return samples

    return [(stem, [os.path.join(dataset, name)]) for stem, name in stems.items() if not stem.endswith(EDIT_SUFFIXES)]


def load_captioner(model_id):
    path = os.path.join(model_id, "text_encoder_NF4")
    if not os.path.exists(os.path.join(path, "config.json")):
        # Instalación limpia: solo hace falta el text encoder NF4 y el processor (~5 GB), no el modelo entero.
        print(f"Downloading captioner from Hugging Face / Descargando el captioner desde Hugging Face: {HF_REPO_ID}", flush=True)
        token = read_json("HF_token.json").get("token", "").strip() or None
        snapshot_download(repo_id=HF_REPO_ID, local_dir=model_id, token=token, max_workers=2,
                          allow_patterns=["text_encoder_NF4/*", "processor/*"])

    print(f"Loading Qwen3-VL-8B (NF4)... / Cargando Qwen3-VL-8B (NF4)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    if model.config.tie_word_embeddings:
        raise RuntimeError(
            "text_encoder_NF4 has no lm_head and cannot write text. Re-run 5_conversor_QwenImage21_NF4.py for NF4. / "
            "text_encoder_NF4 no tiene lm_head y no puede escribir texto. Vuelve a convertir NF4 con 5_conversor_QwenImage21_NF4.py."
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(os.path.join(model_id, "processor"))
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
                   "The photo shows ", "In this image, ", "In the image, ", "Sure, ", "Certainly, ", "Here is a ",
                   "Instruction: "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:]
            break
    return text.strip().strip('"')


def caption_sample(model, processor, images, prompt, cfg):
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")

    temperature = float(cfg["temperature"])
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
        )
    return clean_caption(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def with_trigger(trigger, caption):
    trigger = trigger.strip().rstrip(",")
    if not trigger or caption.lower().startswith(trigger.lower()):
        return caption
    return f"{trigger}, {caption}"


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]
    mode = "edit" if cfg["caption_mode"] == "edit" else "normal"
    prompt = cfg["caption_edit_prompt"] if mode == "edit" else cfg["caption_prompt"]

    if not os.path.isdir(dataset):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {dataset}")
        return 1

    samples = find_samples(dataset, mode)
    pending = []
    for name, paths in samples:
        txt = os.path.join(dataset, name + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append((name, paths))

    kind = "before/after pairs / pares antes/después" if mode == "edit" else "images / imágenes"
    print(f"  Dataset / Ruta Dataset : {os.path.abspath(dataset)}")
    print(f"  Mode / Modo            : {mode}")
    print(f"  Samples / Muestras     : {len(samples)} {kind}, {len(pending)} to caption / por describir")
    print(f"  Trigger Word / Palabra : {cfg['trigger_word'] or '(none / ninguna)'}")
    print(f"  Overwrite / Rehacer    : {'yes / sí' if cfg['overwrite'] else 'only missing / solo los que faltan'}")

    if not samples and mode == "edit":
        print("[!] No pairs found. Name them name_before.ext and name_after.ext. / "
              "No hay pares. Nómbralos nombre_before.ext y nombre_after.ext.")
        return 1
    if not pending:
        print("Nothing to do: every sample already has a caption. / Nada que hacer: todas las muestras ya tienen caption.")
        return 0

    model, processor = load_captioner(cfg["model_id"])

    started = time.time()
    done, failed = 0, []
    for i, (name, paths) in enumerate(pending, 1):
        t0 = time.time()
        try:
            images = []
            for p in paths:
                with Image.open(p) as img:
                    images.append(shrink(img.convert("RGB"), int(cfg["max_image_side"])))
            caption = caption_sample(model, processor, images, prompt, cfg)
            if not caption:
                raise RuntimeError("empty caption / caption vacío")

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} FAILED / FALLÓ: {e}", flush=True)

    total = time.time() - started
    print(f"\n✓ {done} captions in / en {int(total // 60)}m{int(total % 60):02d}s ({total / max(done, 1):.1f}s per sample / por muestra).")
    if failed:
        print(f"[!] {len(failed)} failed / fallaron: {', '.join(failed[:10])}")
    print("Review them in the Dataset Manager before pre-caching. / Revísalos en el Dataset Manager antes del pre-caché.")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
