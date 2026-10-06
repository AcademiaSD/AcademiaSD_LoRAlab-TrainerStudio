# -*- coding: utf-8 -*-
"""
0_caption_ernie.py — Generate one independent text-to-image caption per image.
Uses the shared Qwen3-VL-8B NF4 captioner and ERNIE-specific settings.
The optional trigger word is placed first in each caption.
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
from i18n import t

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
    "caption_prompt": (
        "Describe this image for training a text-to-image model, in at most 150 words. "
        "One single paragraph of plain prose, no lists and no preamble. Cover the main "
        "subject and what they are doing, their appearance, hair and clothing, the pose, "
        "the framing and camera angle, the setting and background, the lighting and the "
        "colors. Do not invent a name and do not mention that this is an image or a photo."
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

PRECACHE_CONFIG = "settings/pre_cache_settings_ernie.json"
CONFIG_PATH = "settings/caption_settings_ernie.json"
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


def find_samples(dataset):
    """Return one independent text-to-image sample per image."""
    images = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    return [(os.path.splitext(name)[0], [os.path.join(dataset, name)]) for name in images]


def load_captioner(captioner_dir):
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
    prompt = cfg["caption_prompt"]

    if not os.path.isdir(dataset):
        print("[!] " + t("Dataset folder does not exist: {path}", path=dataset))
        return 1

    samples = find_samples(dataset)
    pending = []
    for name, paths in samples:
        txt = os.path.join(dataset, name + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append((name, paths))

    kind = t("images")
    print(f"  {t('Dataset'):<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Samples'):<22}: {len(samples)} {kind}, {t('{n} to caption', n=len(pending))}")
    print(f"  {t('Trigger Word'):<22}: {cfg['trigger_word'] or t('(none)')}")
    print(f"  {t('Overwrite'):<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

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
            caption = caption_sample(model, processor, images, prompt, cfg)
            if not caption:
                raise RuntimeError(t("empty caption"))

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
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
