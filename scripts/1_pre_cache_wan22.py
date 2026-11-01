# -*- coding: utf-8 -*-
"""
1_pre_cache_wan22.py — Pre-caché de latentes + embeddings para Wan 2.2 T2V-A14B
Pre-cache of latents + embeddings for Wan 2.2 T2V-A14B

El dataset puede mezclar imágenes y vídeos. Una imagen es un vídeo de 1 fotograma; de cada
vídeo se toman los Max Frames primeros fotogramas a 16 fps, en la rejilla 4k+1 del VAE
(1, 5, 9, ..., 33, ..., 81). Cada muestra va a un bucket por resolución y duración.

Lee configuración desde pre_cache_settings_wan22.json si existe.
Reads configuration from pre_cache_settings_wan22.json if present.
"""
import os
import gc
import math
import json
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir
import wan22_model as wan

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": wan.MODEL_DEFAULT,
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_wan22",
    # 480p de Wan (832 x 480). 720p (1280 x 720) multiplica por 2,3 el coste de cada paso.
    "target_area": 832 * 480,
    "max_side": 1280,
    "multiple": 16,
    # Fotogramas por clip como máximo (4k+1, a 16 fps). Medido en 16 GB con Activation Offload: 17 a 480p
    # caben (13.2 GB de pico, ~17 s/paso); 33 a 480p no (16.6 GB). 81 son los 5 s nativos de Wan.
    "num_frames": 17,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = os.environ.get("LORALAB_PRECACHE_CONFIG") or "settings/pre_cache_settings_wan22.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("✓ " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("⚠ " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID     = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = int(cfg.get("target_area",  DEFAULTS["target_area"]))
MAX_SIDE     = int(cfg.get("max_side",     DEFAULTS["max_side"]))
MULTIPLE     = int(cfg.get("multiple",     DEFAULTS["multiple"]))
NUM_FRAMES   = wan.valid_frames(cfg.get("num_frames", DEFAULTS["num_frames"]))
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_wan22_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2 en el transformer: lados múltiplos de 16 px como mínimo.
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using {d}.", n=MULTIPLE, d=16))
    MULTIPLE = 16

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Target Area'):<22}: {TARGET_AREA} px²")
print(f"  {t('Max Side'):<22}: {MAX_SIDE}")
print(f"  {t('Multiple'):<22}: {MULTIPLE}")
print(f"  {t('Max Frames'):<22}: {NUM_FRAMES} ({NUM_FRAMES / wan.FPS:.1f} s @ {wan.FPS} fps)")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def bucket_size(w: int, h: int):
    ar = w / h
    bh = math.sqrt(TARGET_AREA / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        s = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * s) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * s) // MULTIPLE * MULTIPLE)
    return bw, bh


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_encoder(device):
    """UMT5-XXL en BF16 (~11 GB). Con menos de 14 GB de VRAM se cuantiza a NF4 al cargar (~3.5 GB)."""
    path = os.path.join(MODEL_ID, "text_encoder")
    kwargs = {"dtype": torch.bfloat16}
    if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 14 * 1024 ** 3:
        from transformers import BitsAndBytesConfig
        print("  " + t("Less than 14 GB of VRAM: text encoder quantized to NF4 on load"))
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    te = UMT5EncoderModel.from_pretrained(path, device_map=device, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_ID, "tokenizer"))
    return te, tok


def encode_text(text_encoder, tokenizer, prompt, device):
    # Como WanPipeline: 512 tokens como máximo y solo los reales; el trainer rellena con ceros hasta 512.
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    ids = tokenizer([prompt_clean(prompt)], padding="max_length", max_length=wan.TEXT_LEN, truncation=True,
                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(ids.attention_mask.sum())
    with torch.inference_mode():
        out = text_encoder(ids.input_ids.to(device), ids.attention_mask.to(device)).last_hidden_state
    return out[0, :n].to(torch.bfloat16).cpu()


def find_samples():
    """[(nombre, ruta, es_vídeo)]: cada imagen o vídeo del dataset es una muestra."""
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith(wan.IMAGE_EXTS + wan.VIDEO_EXTS))
    return [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f), f.lower().endswith(wan.VIDEO_EXTS)) for f in files]


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def sample_geometry(src, is_video):
    """(ancho, alto, fotogramas) con los que se cachea la muestra, o None si no se puede leer."""
    if not is_video:
        with Image.open(src) as im:
            return (*bucket_size(*im.size), 1)
    info = wan.video_info(src)
    if info is None:
        return None
    w, h, _, duration = info
    return (*bucket_size(w, h), wan.clip_frames(duration, NUM_FRAMES))


def encode_latents(vae, jobs):
    mean, std = wan.latent_stats(vae, "cuda")
    with torch.inference_mode():
        for idx, (src, is_video, out, (bw, bh, frames)) in enumerate(jobs, 1):
            if is_video:
                pixels = torch.from_numpy(wan.read_video(src, frames, bw, bh).copy()).permute(3, 0, 1, 2)
            else:
                pixels = torch.from_numpy(np.asarray(fit(Image.open(src).convert("RGB"), bw, bh)).copy())
                pixels = pixels.permute(2, 0, 1).unsqueeze(1)
            # [1, 3, F, H, W] en [-1, 1]
            x = (pixels.unsqueeze(0).float() / 127.5 - 1.0).to("cuda", dtype=vae.dtype)
            try:
                z = vae.encode(x).latent_dist.mode().float()
            except torch.OutOfMemoryError:
                print("  ↳ " + t("Low VRAM: tiled VAE encode"))
                torch.cuda.empty_cache()
                vae.enable_tiling()
                z = vae.encode(x).latent_dist.mode().float()
            latent = ((z - mean) / std).to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            kind = t("Video") if is_video else t("Image")
            print(f"[{idx}/{len(jobs)}] {kind}: {os.path.basename(src)} -> {out} | {bw}x{bh}x{frames} -> {tuple(latent.shape[2:])}", flush=True)
            del x, z, latent, pixels


def preprocess_wan22():
    if not os.path.exists(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)

    samples = find_samples()
    if not samples:
        print("[!] " + t("No images or videos found in '{path}'.", path=DATASET_PATH))
        sys.exit(1)
    n_video = sum(1 for s in samples if s[2])
    print(f"  {t('Samples'):<22}: {len(samples)} ({len(samples) - n_video} {t('images')}, {n_video} {t('videos')})")

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el dataset.
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in ("latent", "embed")}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    wan.ensure_base(MODEL_ID, text_encoder=True)

    # ── FASE 1: TEXT ENCODER (UMT5-XXL) ─────────────────────────────────────
    # El text encoder y el VAE nunca coinciden en VRAM.
    print("\n" + t("Loading {name}...", name="Text Encoder (UMT5-XXL)"))
    te, tok = load_text_encoder("cuda")

    torch.save(encode_text(te, tok, wan.NEGATIVE_PROMPT, "cuda"), os.path.join(CACHE_DIR, "_neg_embed.pt"))

    if PREVIEW_CUSTOM_PROMPT:
        c_prompt = with_trigger(PREVIEW_CUSTOM_PROMPT)
        print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt))
        torch.save(encode_text(te, tok, c_prompt, "cuda"), os.path.join(CACHE_DIR, "_custom_embed.pt"))
        # El trainer y el servidor comparan con este texto para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
    else:
        for f in ("_custom_embed.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _, _) in enumerate(samples, 1):
        embed = encode_text(te, tok, with_trigger(read_caption(name)), "cuda")
        torch.save(embed, os.path.join(CACHE_DIR, f"{name}_embed.pt"))
        print(f"[{idx}/{len(samples)}] {t('Text')}: {name} | {embed.shape[0]} tokens")

    del te, tok
    free_vram()

    # ── FASE 2: VAE (Wan 2.1: 16 canales, 8x en espacio, 4x en tiempo) ──────
    # Un latente ya cacheado con la geometría actual y más nuevo que su fichero no se vuelve a
    # codificar: relanzar el pre-caché para cambiar el prompt manual o los captions solo cuesta la
    # fase de texto.
    jobs = []
    for name, src, is_video in samples:
        geo = sample_geometry(src, is_video)
        if geo is None:
            print("[!] " + t("Skipped (ffprobe could not read it): {name}", name=os.path.basename(src)))
            continue
        bw, bh, frames = geo
        shape = ((frames - 1) // 4 + 1, bh // wan.SPATIAL, bw // wan.SPATIAL)
        lat_path = os.path.join(CACHE_DIR, f"{name}_latent.pt")
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-3:]) == shape):
            continue
        jobs.append((src, is_video, f"{name}_latent.pt", geo))

    if len(jobs) < len(samples):
        print("\n" + t("{n} latents already cached, skipped.", n=len(samples) - len(jobs)))
    if not jobs:
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (Wan 2.1)"))
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, jobs)
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def encode_preview_prompt(cache_dir, prompt, device):
    """
    Codifica solo el prompt manual de las previews. Lo lanza el servidor al guardar un prompt nuevo.
    Sin entrenamiento en marcha va en GPU; con el entrenamiento en marcha va en CPU para no tocar
    su VRAM (UMT5-XXL en CPU: ~11 GB de RAM y un minuto), y el trainer relee el embedding antes de
    la siguiente preview.
    """
    where = device.upper().replace("CUDA", "GPU")
    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    te, tok = load_text_encoder(device)
    embed = encode_text(te, tok, prompt, device)
    del te, tok
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_embed.pt", lambda f: torch.save(embed, f))
    print("[Custom Prompt] " + t("Ready."), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_wan22()
