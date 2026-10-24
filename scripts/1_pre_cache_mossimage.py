# -*- coding: utf-8 -*-
"""
1_pre_cache_mossimage.py — Pre-caché de latentes + embeddings para MOSS-Image
Pre-cache of latents + embeddings for MOSS-Image

Lee configuración desde pre_cache_settings_mossimage.json si existe.
Reads configuration from pre_cache_settings_mossimage.json if present.

El texto se codifica como MOSSImagePipeline.encode_prompt de OpenMOSS (Apache-2.0): plantilla de
chat de Qwen con enable_thinking, última capa de Qwen3.5-9B, sin relleno. Los latentes son los del
VAE de FLUX.2, en parches 2x2 y normalizados con su batch norm, como en FLUX.3 Klein.
Base y SFT comparten text encoder y VAE: la misma caché sirve para entrenar sobre los dos.
El text encoder se elige con "text_encoder", con los mismos modos que Qwen-Image 2.1.
"""
import os
import gc
import math
import json
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel, AutoTokenizer
import logging
import sys
from i18n import t
import gpu_stats
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Las capas de atención lineal de Qwen3.5 usan la versión PyTorch de referencia si no están
# flash-linear-attention ni causal-conv1d (Triton / compilación CUDA, sin paquetes para Windows): da
# el mismo resultado, solo es más lenta, y aquí codifica unos pocos cientos de tokens por caption.
logging.getLogger("transformers.integrations.hub_kernels").addFilter(
    lambda record: "falling back to its reference PyTorch implementation" not in record.getMessage())

HF_REPO_ID = "AcademiaSD/MOSS-Image-NF4-for-LoRA-Training"

# Margen de VRAM que se deja libre al repartir el text encoder entre GPU y RAM.
OFFLOAD_RESERVE_BYTES = int(1.5 * 1024**3)
# Modo del text encoder -> (carpeta text_encoder_<precisión>, repartir entre GPU y RAM).
# Error relativo frente a BF16 (medido): INT8 ~7-14 %, NF4 ~20-30 %. VRAM: BF16 ~18 GB, INT8 ~10 GB
# (13 GB de pico al cargar), NF4 ~6.5 GB. BF16_offload es exacto y cabe en cualquier GPU: lo que no
# entra en VRAM se ejecuta desde RAM.
TEXT_ENCODER_MODES = {
    "BF16_offload": ("BF16", True),
    "BF16":         ("BF16", False),
    "INT8":         ("INT8", False),
    "NF4":          ("NF4",  False),
}
# Límite de la condición de texto en el entrenamiento original (plantilla incluida).
MAX_TEXT_TOKENS = 4096

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Models/MOSS-Image-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_mossimage",
    "target_area": 768 * 768,
    "max_side": 1536,
    "multiple": 16,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "text_encoder": "BF16_offload",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "settings/pre_cache_settings_mossimage.json"

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
TEXT_ENCODER = cfg.get("text_encoder", DEFAULTS["text_encoder"])
if TEXT_ENCODER not in TEXT_ENCODER_MODES:
    print("⚠ " + t("Invalid Text Encoder {name}. Using BF16_offload.", name=TEXT_ENCODER))
    TEXT_ENCODER = "BF16_offload"
TEXT_ENCODER_DIR = f"text_encoder_{TEXT_ENCODER_MODES[TEXT_ENCODER][0]}"

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_mossimage_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2: lados múltiplos de 16 px.
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using 16.", n=MULTIPLE))
    MULTIPLE = 16

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {'Text Encoder':<22}: {TEXT_ENCODER}")
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


def text_encoder_complete(path):
    # Están todos los ficheros de pesos del índice: un config.json suelto no basta (una descarga
    # cortada a medias, o los .json pequeños de las otras versiones).
    index = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index):
        return False
    with open(index, "r", encoding="utf-8") as f:
        shards = set(json.load(f)["weight_map"].values())
    return all(os.path.exists(os.path.join(path, s)) for s in shards)


def ensure_model_downloaded(local_path, repo_id):
    if (text_encoder_complete(os.path.join(local_path, TEXT_ENCODER_DIR))
            and os.path.exists(os.path.join(local_path, "vae", "diffusion_pytorch_model.safetensors"))):
        print("✓ " + t("Local model found at: {path}", path=local_path))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=repo_id))

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(t("huggingface_hub is required. Install it with: pip install huggingface_hub"))

    # Solo el text encoder elegido. El transformer lo descarga el entrenamiento (Base o SFT, el elegido).
    others = [f"text_encoder_{p}/*" for p, _ in TEXT_ENCODER_MODES.values() if f"text_encoder_{p}" != TEXT_ENCODER_DIR]
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
        ignore_patterns=["transformer/*", "transformer_sft/*", "LoRAs/*"] + others,
    )
    print("✓ " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_encoder(device):
    # BF16_offload: las capas que no caben en la VRAM libre se quedan en RAM y se ejecutan desde allí.
    # INT8 y NF4 llevan su quantization_config en config.json y se cargan enteros en la GPU, igual
    # que BF16. En CPU también funciona, mucho más lento.
    path = os.path.join(MODEL_ID, TEXT_ENCODER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_ID, "tokenizer"))
    if device == "cpu":
        return AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu").eval(), tokenizer
    if not TEXT_ENCODER_MODES[TEXT_ENCODER][1]:
        return AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval(), tokenizer
    free_bytes, _ = gpu_stats.mem_get_info()
    gpu_budget = max(free_bytes - OFFLOAD_RESERVE_BYTES, 0)
    print(f"  {t('GPU budget'):<22}: {gpu_budget / 1024**3:.1f} GB ({t('rest in RAM')})")
    te = AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto",
                                   max_memory={0: gpu_budget, "cpu": "512GiB"})
    return te.eval(), tokenizer


def encode_text(te, tokenizer, prompt, device):
    # [longitud, 4096]: la última capa con la plantilla de chat; cada prompt con su longitud real
    # (el trainer rellena y pasa la máscara). Un prompt vacío también lleva la plantilla.
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS).to(device)
    with torch.inference_mode():
        hidden = te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
    return hidden.to(torch.bfloat16).cpu()


def encode_and_save(te, tokenizer, prompt, name):
    torch.save(encode_text(te, tokenizer, prompt, "cuda"), os.path.join(CACHE_DIR, f"{name}_embed.pt"))


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def find_samples():
    """[(nombre, ruta)]: cada imagen del dataset es una muestra, con su nombre.txt como caption."""
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    return [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f)) for f in files]


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def patchify(z):
    # [B, 32, 2H, 2W] -> [B, 128, H, W], el mismo orden que deshace el pipeline al decodificar.
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def encode_latents(vae, jobs, device):
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, torch.float32)

    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            img = fit(Image.open(src).convert("RGB"), bw, bh)

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0)
            img_tensor = (img_tensor.float() / 127.5) - 1.0
            img_tensor = img_tensor.to(device, dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.mode().float()
            latent = ((patchify(z) - bn_mean) / bn_std).to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] {t('Image')}: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_mossimage():
    if not os.path.exists(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)

    samples = find_samples()
    if not samples:
        print("[!] " + t("No images found in '{path}'. Please add images.", path=DATASET_PATH))
        sys.exit(1)
    print(f"  {t('LoRA Type'):<22}: normal | {t('{n} samples', n=len(samples))}")

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el dataset.
    expected = {f"{name}_{suffix}.pt" for name, _ in samples for suffix in ("latent", "embed")}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # ── FASE 1: TEXT ENCODER (Qwen3.5-9B) ───────────────────────────────────
    print("\n" + t("Loading {name}...", name=f"Text Encoder (Qwen3.5-9B {TEXT_ENCODER})"))
    te, tokenizer = load_text_encoder("cuda")

    encode_and_save(te, tokenizer, "", "_neg")

    if PREVIEW_CUSTOM_PROMPT:
        c_prompt = with_trigger(PREVIEW_CUSTOM_PROMPT)
        print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt))
        encode_and_save(te, tokenizer, c_prompt, "_custom")
        # El trainer y el servidor comparan con este texto para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
    else:
        for f in ("_custom_embed.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _) in enumerate(samples, 1):
        encode_and_save(te, tokenizer, with_trigger(read_caption(name)), name)
        print(f"[{idx}/{len(samples)}] {t('Text')}: {name}")

    del te, tokenizer
    free_vram()

    # ── FASE 2: VAE (FLUX.2, 32 canales) ────────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se vuelve a
    # codificar: relanzar el pre-caché para cambiar el prompt manual o los captions solo cuesta el texto.
    jobs = []
    for name, src in samples:
        with Image.open(src) as im:
            jobs.append((src, f"{name}_latent.pt", bucket_size(*im.size)))

    pending = []
    for src, out, (bw, bh) in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 16, bw // 16)):
            continue
        pending.append((src, out, (bw, bh)))

    if len(pending) < len(jobs):
        print("\n" + t("{n} latents already cached, skipped.", n=len(jobs) - len(pending)))
    if not pending:
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (FLUX.2)"))
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, pending, "cuda")
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def encode_preview_prompt(cache_dir, prompt, device):
    """
    Codifica solo el prompt manual de las previews. Lo lanza el servidor al guardar un prompt nuevo.
    Sin entrenamiento en marcha va en GPU; con el entrenamiento en marcha va en CPU para no tocar su
    VRAM, y el trainer relee el embedding antes de la siguiente preview.
    """
    where = device.upper().replace("CUDA", "GPU")
    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    te, tokenizer = load_text_encoder(device)
    embeds = encode_text(te, tokenizer, prompt, device)
    del te, tokenizer
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_embed.pt", lambda f: torch.save(embeds, f))
    print("[Custom Prompt] " + t("Ready."), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_mossimage()
