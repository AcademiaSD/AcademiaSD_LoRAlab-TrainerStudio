# -*- coding: utf-8 -*-
"""
0_caption_wan22.py — Captions automáticos del dataset con Qwen3-VL-8B
Dataset auto-captioning with Qwen3-VL-8B

Un .txt por imagen o vídeo con una descripción. La palabra trigger va siempre al principio.
One .txt per image or video with a description. The trigger word always goes first.

Usa el mismo captioner que Z-Image, LTX-2.3 y Qwen-Image 2.1 (Models/Captioner-Qwen3-VL-8B,
~5.5 GB de VRAM). De cada vídeo ve solo el tramo que se entrena (los Max Frames primeros
fotogramas a 16 fps): un caption que describiera el final del clip enseñaría al LoRA un texto
que no corresponde a sus píxeles.

Lee caption_settings_wan22.json; la ruta del dataset, el trigger y Max Frames salen de
pre_cache_settings_wan22.json.
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
from project_layout import resolve_model_dir
import wan22_model as wan

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas de visión (4304) no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

DEFAULTS = {
    "captioner_dir": "Models/Captioner-Qwen3-VL-8B",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "num_frames": 17,
    "caption_prompt": (
        "Describe this image for training a text-to-video model, in at most 120 words. "
        "One single paragraph of plain prose, no lists and no preamble. Cover the main "
        "subject, their appearance, hair and clothing, the pose, the framing and camera angle, "
        "the setting and background, the lighting and the colors. Do not invent a name and do "
        "not mention that this is an image or a photo."
    ),
    "caption_video_prompt": (
        "Describe this video clip for training a text-to-video model, in at most 120 words. "
        "One single paragraph of plain prose, no lists and no preamble. Describe the main "
        "subject and their appearance, what they do and how they move, how the camera moves, "
        "the setting, the lighting and the colors. Do not invent a name and do not mention "
        "that this is a video or a clip."
    ),
    # Fotogramas que ve el captioner, repartidos por el tramo entrenable del clip.
    "caption_video_frames": 8,
    "max_new_tokens": 300,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    "max_image_side": 768,
    "max_video_side": 448,
    # False = no toca lo que ya tiene un .txt con contenido.
    "overwrite": True,
}

PRECACHE_CONFIG = "settings/pre_cache_settings_wan22.json"
CONFIG_PATH = "settings/caption_settings_wan22.json"
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
    for key in ("dataset_path", "trigger_word", "num_frames"):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})
    return cfg


def load_captioner(captioner_dir):
    captioner_dir = str(resolve_model_dir(captioner_dir, "Models/Captioner-Qwen3-VL-8B"))
    path = os.path.join(captioner_dir, "text_encoder_NF4")
    if not os.path.exists(os.path.join(path, "config.json")):
        print(t("Downloading the captioner from Hugging Face: {repo} (~5 GB)", repo=HF_REPO_ID), flush=True)
        wan.enable_hf_file_progress()
        snapshot_download(repo_id=HF_REPO_ID, local_dir=captioner_dir, token=wan.get_hf_token(), max_workers=2,
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
                   "The video shows ", "This video shows ", "The video depicts ", "In this video, ",
                   "The photo shows ", "In this image, ", "In the image, ", "Sure, ", "Certainly, ", "Here is a "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:]
            break
    return text.strip().strip('"')


def generate(model, processor, inputs, cfg):
    temperature = float(cfg["temperature"])
    with torch.inference_mode():
        out = model.generate(
            **inputs.to("cuda"),
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
        )
    return clean_caption(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def caption_image(model, processor, image, cfg):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": cfg["caption_prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, processor, processor(text=[text], images=[image], return_tensors="pt"), cfg)


def caption_video(model, processor, path, cfg):
    info = wan.video_info(path)
    if info is None:
        raise RuntimeError(t("ffprobe could not read the video"))
    w, h, _, duration = info
    # El tramo que entrena el pre-caché: sus fotogramas a 16 fps, sacados ya a un tamaño pequeño.
    window = wan.clip_frames(duration, cfg["num_frames"])
    side = int(cfg["max_video_side"])
    scale = min(1.0, side / max(w, h))
    vw, vh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    clip = wan.read_video(path, window, vw, vh)

    count = max(2, min(int(cfg["caption_video_frames"]), window))
    picks = sorted({round(i * (window - 1) / (count - 1)) for i in range(count)})
    frames = [Image.fromarray(clip[i]) for i in picks]
    if len(frames) % 2:
        # Qwen3-VL agrupa los fotogramas de dos en dos (temporal_patch_size 2).
        frames.append(frames[-1])

    messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": cfg["caption_video_prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Los metadatos describen lo que se entrega (los fotogramas elegidos), no el clip original.
    seconds = max(window / wan.FPS, 1e-3)
    metadata = [{"fps": len(frames) / seconds, "total_num_frames": len(frames), "duration": seconds}]
    inputs = processor(text=[text], videos=[frames], return_tensors="pt", video_metadata=metadata)
    return generate(model, processor, inputs, cfg)


def with_trigger(trigger, caption):
    trigger = trigger.strip().rstrip(",")
    if not trigger or caption.lower().startswith(trigger.lower()):
        return caption
    return f"{trigger}, {caption}"


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]

    if not os.path.isdir(dataset):
        print("[!] " + t("Dataset folder does not exist: {path}", path=dataset))
        return 1

    files = sorted(f for f in os.listdir(dataset) if f.lower().endswith(wan.IMAGE_EXTS + wan.VIDEO_EXTS))
    pending = []
    for name in files:
        txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append(name)

    print(f"  {t('Dataset'):<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Files'):<22}: {len(files)}, {t('{n} to caption', n=len(pending))}")
    print(f"  {t('Trigger Word'):<22}: {cfg['trigger_word'] or t('(none)')}")
    print(f"  {t('Overwrite'):<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

    if not pending:
        print(t("Nothing to do: every file already has a caption."))
        return 0

    model, processor = load_captioner(cfg["captioner_dir"])

    started = time.time()
    done, failed = 0, []
    for i, name in enumerate(pending, 1):
        t0 = time.time()
        stem = os.path.splitext(name)[0]
        path = os.path.join(dataset, name)
        try:
            if name.lower().endswith(wan.VIDEO_EXTS):
                caption = caption_video(model, processor, path, cfg)
            else:
                with Image.open(path) as img:
                    image = shrink(img.convert("RGB"), int(cfg["max_image_side"]))
                caption = caption_image(model, processor, image, cfg)
            if not caption:
                raise RuntimeError(t("empty caption"))

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} {t('FAILED')}: {e}", flush=True)

    total = time.time() - started
    print("\n✓ " + t("{n} captions in {time} ({per:.1f}s per file).", n=done, time=f"{int(total // 60)}m{int(total % 60):02d}s", per=total / max(done, 1)))
    if failed:
        print("[!] " + t("{n} failed: {names}", n=len(failed), names=', '.join(failed[:10])))
    print(t("Review them in the Dataset Manager before pre-caching."))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
