# -*- coding: utf-8 -*-
"""
1_pre_cache_flux1.py — Pre-caché de latentes + embeddings para FLUX.1 Dev
Pre-cache of latents + embeddings for FLUX.1 Dev

Lee configuración desde pre_cache_settings_flux1.json si existe.
Reads configuration from pre_cache_settings_flux1.json if present.
"""
import os
import gc
import math
import json
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import AutoencoderKL, FluxPipeline
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
import logging
import sys
from i18n import t
from text_io import read_text_file
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4 de diffusers, sin acceso restringido. Fijado a un commit: el repo no es nuestro.
HF_REPO_ID = "diffusers/FLUX.1-dev-bnb-4bit"
HF_REVISION = "27114c0933fb58aea21f707e113c25e8c6558487"

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Models/FLUX.1-Dev_NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_flux1",
    "target_area": 512 * 512,
    "max_side": 768,
    "multiple": 32,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = os.environ.get("LORALAB_PRECACHE_CONFIG") or "settings/pre_cache_settings_flux1.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("✓ " + t("Configuration loaded from {path}", path=CONFIG_PATH))
else:
    cfg = {}
    print("⚠ " + t("{path} not found, using default values.", path=CONFIG_PATH))

MODEL_ID     = str(resolve_model_dir(cfg.get("model_id"), DEFAULTS["model_id"]))
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()

# Formato automático de carpeta de caché según nombre del proyecto
CACHE_DIR = str(resolve_cache_dir(f"./cached_data_flux1_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2 en el transformer: lados múltiplos de 16 px como mínimo.
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using {d}.", n=MULTIPLE, d=32))
    MULTIPLE = 32

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Target Area'):<22}: {TARGET_AREA} px²")
print(f"  {t('Max Side'):<22}: {MAX_SIDE}")
print(f"  {t('Multiple'):<22}: {MULTIPLE}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_hf_token():
    if os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token_data = json.load(f)
                token = token_data.get("token", "").strip()
                if token:
                    print("✓ " + t("Using HF Token"))
                    return token
        except Exception:
            pass
    return None


def enable_hf_file_progress():
    try:
        import tqdm
        import tqdm.auto
        import tqdm.std

        def patch_tqdm(cls):
            orig_init = cls.__init__
            def new_init(self, *args, **kwargs):
                kwargs['disable'] = False
                kwargs['mininterval'] = 0.5
                orig_init(self, *args, **kwargs)
            cls.__init__ = new_init

        patch_tqdm(tqdm.std.tqdm)
        patch_tqdm(tqdm.auto.tqdm)
        if hasattr(tqdm, 'tqdm'):
            patch_tqdm(tqdm.tqdm)

        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    except Exception:
        pass

    os.environ["TQDM_DISABLE"] = "0"
    os.environ["TQDM_MININTERVAL"] = "0.5"


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


def ensure_model_downloaded(local_path, repo_id):
    if (os.path.exists(os.path.join(local_path, "model_index.json"))
            and os.path.exists(os.path.join(local_path, "text_encoder", "config.json"))
            and os.path.exists(os.path.join(local_path, "text_encoder_2", "config.json"))):
        print("✓ " + t("Local model found at: {path}", path=local_path))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=repo_id))

    enable_hf_file_progress()

    from huggingface_hub import snapshot_download

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        revision=HF_REVISION,
        local_dir=local_path,
        # False: un token caducado en la caché de Hugging Face haría fallar este repo público.
        token=get_hf_token() or False,
        max_workers=2,
    )
    print("✓ " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_pipe(device):
    # CLIP-L y T5-XXL en NF4. En CPU también funciona (bitsandbytes), mucho más lento.
    clip = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16).to(device)
    t5 = T5EncoderModel.from_pretrained(os.path.join(MODEL_ID, "text_encoder_2"), dtype=torch.bfloat16, device_map=device)
    return FluxPipeline(
        scheduler=None, vae=None, transformer=None,
        text_encoder=clip, tokenizer=AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer"),
        text_encoder_2=t5, tokenizer_2=AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2"),
    )


def encode_text(pipe, prompt, device):
    # El mismo texto va a los dos: T5 da la secuencia [1, 512, 4096] (con el relleno, el transformer
    # no recibe máscara) y CLIP-L el vector global [1, 768].
    with torch.inference_mode():
        t5, clip, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
    return {"t5": t5.to(torch.bfloat16).cpu(), "clip": clip.to(torch.bfloat16).cpu()}


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def find_samples():
    """[(nombre, ruta)]: cada imagen del dataset es una muestra."""
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    return [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f)) for f in files]


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    return read_text_file(path).strip()


def encode_latents(vae, jobs):
    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            img = fit(Image.open(src).convert("RGB"), bw, bh)

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0)
            img_tensor = ((img_tensor.float() / 127.5) - 1.0).to("cuda", dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.mode().float()
            latent = ((z - vae.config.shift_factor) * vae.config.scaling_factor).to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] {t('Image')}: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_flux1():
    if not os.path.exists(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)

    samples = find_samples()
    if not samples:
        print("[!] " + t("No images found in '{path}'. Please add images.", path=DATASET_PATH))
        sys.exit(1)
    print(f"  {t('Samples'):<22}: {len(samples)}")

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el dataset.
    expected = {f"{name}_{suffix}.pt" for name, _ in samples for suffix in ("latent", "embed")}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # ── FASE 1: TEXT ENCODERS (CLIP-L + T5-XXL NF4) ─────────────────────────
    # Los text encoders y el VAE nunca coinciden en VRAM.
    print("\n" + t("Loading {name}...", name="Text Encoders (CLIP-L + T5-XXL NF4)"))
    pipe = load_text_pipe("cuda")

    custom_prompt = PREVIEW_CUSTOM_PROMPT
    if custom_prompt:
        c_prompt = with_trigger(custom_prompt)
        print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt))
        torch.save(encode_text(pipe, c_prompt, "cuda"), os.path.join(CACHE_DIR, "_custom_embed.pt"))
        # El trainer y el servidor comparan con este texto para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
    else:
        for f in ("_custom_embed.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _) in enumerate(samples, 1):
        torch.save(encode_text(pipe, with_trigger(read_caption(name)), "cuda"), os.path.join(CACHE_DIR, f"{name}_embed.pt"))
        print(f"[{idx}/{len(samples)}] {t('Text')}: {name}")

    del pipe
    free_vram()

    # ── FASE 2: VAE (16 canales, 8x) ────────────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual o los captions
    # solo cuesta la fase de texto.
    jobs = []
    for name, src in samples:
        with Image.open(src) as im:
            bw, bh = bucket_size(*im.size)
        lat_path = os.path.join(CACHE_DIR, f"{name}_latent.pt")
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 8, bw // 8)):
            continue
        jobs.append((src, f"{name}_latent.pt", (bw, bh)))

    if len(jobs) < len(samples):
        print("\n" + t("{n} latents already cached, skipped.", n=len(samples) - len(jobs)))
    if not jobs:
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE"))
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, jobs)
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def encode_preview_prompt(cache_dir, prompt, device):
    """
    Codifica solo el prompt manual de las previews. Lo lanza el servidor al guardar un prompt nuevo.
    Sin entrenamiento en marcha va en GPU; con el entrenamiento en marcha va en CPU para no tocar
    su VRAM (~1 min), y el trainer relee el embedding antes de la siguiente preview.
    """
    where = device.upper().replace("CUDA", "GPU")
    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    pipe = load_text_pipe(device)
    embed = encode_text(pipe, prompt, device)
    del pipe
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
        preprocess_flux1()
