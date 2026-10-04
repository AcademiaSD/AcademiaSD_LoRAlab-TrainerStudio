# -*- coding: utf-8 -*-
"""
1_pre_cache_sdxl.py — Pre-caché de latentes + embeddings para SDXL (Base, Pony, Illustrious, NoobAI o un checkpoint propio)
Pre-cache of latents + embeddings for SDXL (Base, Pony, Illustrious, NoobAI or your own checkpoint)

Lee configuración desde pre_cache_settings_sdxl.json si existe.
Reads configuration from pre_cache_settings_sdxl.json if present.
"""
import os
import gc
import math
import json
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODELS_DIR = "SDXL-Models"

# Modelos con los que se entrena: se descargan la primera vez. prefix y negative son para las
# previews (los prefijos de calidad de cada familia); el LoRA se entrena con los captions tal cual.
PRESETS = {
    "sdxl_base": {
        "name": "SDXL Base 1.0", "repo": "stabilityai/stable-diffusion-xl-base-1.0", "file": "sd_xl_base_1.0.safetensors",
        "prefix": "", "negative": "", "cfg": 6.0,
    },
    "pony_v6": {
        "name": "Pony Diffusion V6 XL", "repo": "LyliaEngine/Pony_Diffusion_V6_XL",
        "file": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "prefix": "score_9, score_8_up, score_7_up, ", "negative": "score_4, score_5, score_6, worst quality, low quality", "cfg": 7.0,
    },
    "illustrious_v01": {
        "name": "Illustrious XL v0.1", "repo": "OnomaAIResearch/Illustrious-xl-early-release-v0", "file": "Illustrious-XL-v0.1.safetensors",
        "prefix": "masterpiece, best quality, ",
        "negative": "worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, watermark", "cfg": 6.0,
    },
    "juggernaut_xi": {
        "name": "Juggernaut XI v11", "repo": "RunDiffusion/Juggernaut-XI-v11", "file": "Juggernaut-XI-byRunDiffusion.safetensors",
        "prefix": "", "negative": "worst quality, low quality, blurry, deformed, bad anatomy, watermark, text", "cfg": 5.0,
    },
    "realvis_v5": {
        "name": "RealVisXL V5.0", "repo": "SG161222/RealVisXL_V5.0", "file": "RealVisXL_V5.0_fp16.safetensors",
        "prefix": "", "negative": "worst quality, low quality, blurry, deformed, bad anatomy, watermark, text", "cfg": 5.0,
    },
    "noobai_v11": {
        "name": "NoobAI-XL 1.1", "repo": "Laxhar/noobai-XL-1.1", "file": "NoobAI-XL-v1.1.safetensors",
        "prefix": "masterpiece, best quality, newest, ",
        "negative": "worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, watermark", "cfg": 5.0,
    },
}

# Los captions largos se codifican por bloques de 75 tokens (hasta 3, como kohya): CLIP solo ve 77.
MAX_CHUNKS = 3

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_preset": "sdxl_base",
    "custom_checkpoint": "",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_sdxl",
    "target_area": 1024 * 1024,
    "max_side": 2048,
    "multiple": 64,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "quality_prefix": False,
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "settings/pre_cache_settings_sdxl.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_PRESET = cfg.get("model_preset", DEFAULTS["model_preset"])
CUSTOM_CHECKPOINT = cfg.get("custom_checkpoint", DEFAULTS["custom_checkpoint"]).strip().strip('"')
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()
# Prefijo de calidad del preset (score_9... en Pony) delante de los captions y del prompt de preview.
QUALITY_PREFIX = bool(cfg.get("quality_prefix", DEFAULTS["quality_prefix"]))

if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_sdxl_{PROJECT_NAME}"
else:
    CACHE_DIR = cfg.get("cache_dir", DEFAULTS["cache_dir"])

# VAE 8x y tres bajadas de resolución en la UNet: lados múltiplos de 64 px.
if MULTIPLE not in (64, 128):
    print(f"⚠ Invalid Multiple {MULTIPLE}. Defaulting to 64 / Múltiplo inválido {MULTIPLE}. Usando 64 por defecto.")
    MULTIPLE = 64

if MODEL_PRESET != "custom" and MODEL_PRESET not in PRESETS:
    print(f"⚠ Unknown model preset {MODEL_PRESET}. Using SDXL Base / Preset desconocido {MODEL_PRESET}. Se usa SDXL Base.")
    MODEL_PRESET = "sdxl_base"

print(f"  Model / Modelo              : {CUSTOM_CHECKPOINT if MODEL_PRESET == 'custom' else PRESETS[MODEL_PRESET]['name']}")
print(f"  Project Name / Proyecto     : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
print(f"  Quality Prefix / Prefijo    : {'yes / sí' if QUALITY_PREFIX else 'no'}")
print(f"  Dataset Path / Ruta Dataset : {DATASET_PATH}")
print(f"  Cache Dir / Carpeta Caché   : {CACHE_DIR}")
print(f"  Target Area / Área Objetivo : {TARGET_AREA} px²")
print(f"  Max Side / Lado Máximo      : {MAX_SIDE}")
print(f"  Multiple / Múltiplo         : {MULTIPLE}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_hf_token():
    if os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token = json.load(f).get("token", "").strip()
                if token:
                    print("✓ Using HF Token / Usando token de HF")
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


def resolve_model():
    """Ruta del checkpoint y datos del preset; descarga el del preset si no está."""
    if MODEL_PRESET == "custom":
        if not os.path.isfile(CUSTOM_CHECKPOINT):
            print(f"[!] Custom checkpoint not found / No se encuentra el checkpoint propio: {CUSTOM_CHECKPOINT}")
            sys.exit(1)
        return {"preset": "custom", "name": os.path.basename(CUSTOM_CHECKPOINT), "checkpoint": os.path.abspath(CUSTOM_CHECKPOINT),
                "prefix": "", "negative": "worst quality, low quality", "cfg": 6.0}

    p = PRESETS[MODEL_PRESET]
    path = os.path.join(MODELS_DIR, p["file"])
    if not os.path.exists(path):
        print(f"⚠ {p['name']} not found / no encontrado. Downloading from Hugging Face / Descargando desde Hugging Face: {p['repo']} (~7 GB)")
        enable_hf_file_progress()
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(p["repo"], p["file"], local_dir=MODELS_DIR, token=get_hf_token())
    else:
        print(f"✓ Local model found at / Modelo local encontrado en: {path}")
    return {"preset": MODEL_PRESET, "name": p["name"], "checkpoint": os.path.abspath(path),
            "prefix": p["prefix"] if QUALITY_PREFIX else "", "negative": p["negative"], "cfg": p["cfg"]}


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


def load_text_encoders(checkpoint, device):
    # Solo los dos CLIP y sus tokenizers: la UNet y el VAE no se cargan.
    pipe = StableDiffusionXLPipeline.from_single_file(checkpoint, unet=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    return pipe


def encode_text(pipe, prompt, device):
    """
    Como kohya: el caption se trocea en bloques de 75 tokens (hasta 3); cada bloque va con BOS/EOS y
    relleno a 77, y de cada CLIP se toma la penúltima capa (768 + 1280 = 2048 por token). El vector
    "pooled" es el del segundo CLIP en el primer bloque. Devuelve {"emb": [77*k, 2048], "pooled": [1280]}.
    """
    chunks_per_encoder = []
    for tok in (pipe.tokenizer, pipe.tokenizer_2):
        ids = tok(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        blocks = [ids[i:i + 75] for i in range(0, max(len(ids), 1), 75)][:MAX_CHUNKS]
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        chunks_per_encoder.append([[tok.bos_token_id] + b + [tok.eos_token_id] + [pad] * (75 - len(b)) for b in blocks])

    n = min(len(chunks_per_encoder[0]), len(chunks_per_encoder[1]))
    embs, pooled = [], None
    with torch.inference_mode():
        for i in range(n):
            ids1 = torch.tensor([chunks_per_encoder[0][i]], device=device)
            ids2 = torch.tensor([chunks_per_encoder[1][i]], device=device)
            h1 = pipe.text_encoder(ids1, output_hidden_states=True).hidden_states[-2]
            out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)
            embs.append(torch.cat([h1, out2.hidden_states[-2]], dim=-1)[0])
            if pooled is None:
                pooled = out2.text_embeds[0]
    return {"emb": torch.cat(embs).to(torch.bfloat16).cpu(), "pooled": pooled.to(torch.bfloat16).cpu()}


def zeros_embed():
    # SDXL usa ceros como negativo cuando no hay prompt negativo (force_zeros_for_empty_prompt).
    return {"emb": torch.zeros(77, 2048, dtype=torch.bfloat16), "pooled": torch.zeros(1280, dtype=torch.bfloat16)}


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro. Devuelve la imagen y la esquina del recorte.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh)), (top, left)


def find_samples():
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    return [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f)) for f in files]


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_latents(vae, jobs):
    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            with Image.open(src) as im:
                orig = (im.height, im.width)
                img, crop = fit(im.convert("RGB"), bw, bh)
            x = (F_vision.pil_to_tensor(img).unsqueeze(0).float() / 127.5 - 1.0).to("cuda", dtype=vae.dtype)
            lat = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
            # Las condiciones de tamaño de SDXL: tamaño original y esquina del recorte (en la imagen escalada).
            data = {"lat": lat[0].to(torch.bfloat16).cpu(), "orig": list(orig), "crop": list(crop)}
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(data, tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            print(f"[{idx}/{len(jobs)}] Image / Imagen: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_sdxl():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        sys.exit(1)

    samples = find_samples()
    if not samples:
        print(f"[!] No images found in '{DATASET_PATH}'. Please add images.")
        sys.exit(1)
    print(f"  Samples / Muestras          : {len(samples)}")

    model = resolve_model()
    # Otro modelo = otros text encoders: los textos y latentes cacheados con el anterior no valen.
    model_file = os.path.join(CACHE_DIR, "_model.json")
    previous = json.load(open(model_file, encoding="utf-8")) if os.path.exists(model_file) else {}
    if previous.get("checkpoint") not in (None, model["checkpoint"]):
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".pt"):
                os.remove(os.path.join(CACHE_DIR, f))
        print("  Model changed: cache cleared / Modelo cambiado: caché vaciada")

    expected = {f"{name}_{suffix}.pt" for name, _ in samples for suffix in ("latent", "embed")}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print(f"  Removed {len(stale)} stale cache files / Eliminados {len(stale)} ficheros antiguos de la caché")

    # ── FASE 1: TEXT ENCODERS (CLIP-L + CLIP-bigG) ──────────────────────────
    print(f"\nLoading Text Encoders (CLIP-L + CLIP-bigG) from / Cargando Text Encoders de: {model['name']}")
    pipe = load_text_encoders(model["checkpoint"], "cuda")

    torch.save(encode_text(pipe, model["negative"], "cuda") if model["negative"] else zeros_embed(),
               os.path.join(CACHE_DIR, "_neg_embed.pt"))
    # Bloque vacío para igualar la longitud de los textos de un mismo batch.
    torch.save(encode_text(pipe, "", "cuda"), os.path.join(CACHE_DIR, "_pad_embed.pt"))

    if PREVIEW_CUSTOM_PROMPT:
        c_prompt = with_trigger(PREVIEW_CUSTOM_PROMPT)
        print(f"[Custom Prompt Cache] Encoding: '{model['prefix']}{c_prompt}'")
        torch.save(encode_text(pipe, model["prefix"] + c_prompt, "cuda"), os.path.join(CACHE_DIR, "_custom_embed.pt"))
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
    else:
        for f in ("_custom_embed.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _) in enumerate(samples, 1):
        caption = model["prefix"] + with_trigger(read_caption(name))
        data = encode_text(pipe, caption, "cuda")
        torch.save(data, os.path.join(CACHE_DIR, f"{name}_embed.pt"))
        print(f"[{idx}/{len(samples)}] Text / Texto: {name} | {data['emb'].shape[0] // 77} block(s) / bloque(s)")

    del pipe
    free_vram()

    # El trainer y las previews usan el mismo checkpoint y sus ajustes de preview.
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    # ── FASE 2: VAE ─────────────────────────────────────────────────────────
    jobs = []
    for name, src in samples:
        with Image.open(src) as im:
            bw, bh = bucket_size(*im.size)
        lat_path = os.path.join(CACHE_DIR, f"{name}_latent.pt")
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True)["lat"].shape[-2:]) == (bh // 8, bw // 8)):
            continue
        jobs.append((src, f"{name}_latent.pt", (bw, bh)))

    if len(jobs) < len(samples):
        print(f"\n{len(samples) - len(jobs)} latents already cached, skipped / latentes ya cacheados, se saltan.")
    if not jobs:
        print("\n✓ Pre-caching finished! / ¡Pre-caché finalizado!")
        return

    # El VAE de SDXL da NaN en FP16; en BF16 funciona.
    print("\nLoading VAE... / Cargando VAE...")
    vae = AutoencoderKL.from_single_file(model["checkpoint"], torch_dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, jobs)
    del vae
    free_vram()
    print("\n✓ Pre-caching finished! VRAM freed / ¡Pre-caché finalizado! VRAM liberada.")


def encode_preview_prompt(cache_dir, prompt, device):
    """
    Codifica solo el prompt manual de las previews, con el mismo checkpoint que la caché. Lo lanza el
    servidor al guardar un prompt nuevo; con el entrenamiento en marcha va en CPU.
    """
    model = json.load(open(os.path.join(cache_dir, "_model.json"), encoding="utf-8"))
    where = device.upper().replace("CUDA", "GPU")
    print(f"[Custom Prompt] Prompt: encoding on {where} / Codificando en {where}: '{prompt}'", flush=True)
    pipe = load_text_encoders(model["checkpoint"], device)
    if device == "cpu":
        pipe.text_encoder.float()
        pipe.text_encoder_2.float()
    data = encode_text(pipe, model["prefix"] + prompt, device)
    del pipe
    free_vram()

    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_embed.pt", lambda f: torch.save(data, f))
    print("[Custom Prompt] Ready / Listo.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_sdxl()
