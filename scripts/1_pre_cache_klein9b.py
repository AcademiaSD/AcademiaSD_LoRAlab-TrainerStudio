# -*- coding: utf-8 -*-
"""
1_pre_cache_klein9b.py — Pre-caché de latentes + embeddings para FLUX.2 [klein] 9B
Pre-cache of latents + embeddings for FLUX.2 [klein] 9B

Lee configuración desde pre_cache_settings_klein9b.json si existe.
Reads configuration from pre_cache_settings_klein9b.json if present.
"""
import os
import gc
import math
import json
import warnings
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from transformers import AutoTokenizer, Qwen3Model
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4 en CPU (prompt manual durante el entrenamiento): bitsandbytes avisa de que no usa el kernel rápido.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

HF_REPO_ID = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "FLUX.2-Klein-9B_NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_klein9b",
    "target_area": 768 * 768,
    "max_side": 1536,
    "multiple": 16,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "preview_edit_image": "",
    "lora_type": "normal",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "settings/pre_cache_settings_klein9b.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID     = cfg.get("model_id",     DEFAULTS["model_id"])
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()
# "normal" (imagen + caption) o "edit" (pares nombre_before / nombre_after + instrucción).
LORA_TYPE = cfg.get("lora_type", DEFAULTS["lora_type"])
# Solo edición: imagen de antes (fuera del dataset) para ver en las previews cómo aplica el efecto.
PREVIEW_EDIT_IMAGE = cfg.get("preview_edit_image", "").strip() if LORA_TYPE == "edit" else ""

# Formato automático de carpeta de caché según nombre del proyecto
if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_klein9b_{PROJECT_NAME}"
else:
    CACHE_DIR = cfg.get("cache_dir", DEFAULTS["cache_dir"])

# VAE 8x + parches 2x2 en el transformer: lados múltiplos de 16 px.
if MULTIPLE not in (16, 32, 64):
    print(f"⚠ Invalid Multiple {MULTIPLE}. Defaulting to 16 / Múltiplo inválido {MULTIPLE}. Usando 16 por defecto.")
    MULTIPLE = 16

print(f"  Model ID / ID Modelo        : {MODEL_ID}")
print(f"  Project Name / Proyecto     : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
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
                token_data = json.load(f)
                token = token_data.get("token", "").strip()
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
            and os.path.exists(os.path.join(local_path, "text_encoder", "config.json"))):
        print(f"✓ Local model found at / Modelo local encontrado en: {local_path}")
        return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with pip install huggingface_hub")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
    )
    print(f"✓ Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_encoder(device):
    # Qwen3-8B en NF4 (28 capas, ~3.8 GB). En CPU también funciona (bitsandbytes), mucho más lento.
    te = Qwen3Model.from_pretrained(os.path.join(MODEL_ID, "text_encoder"), dtype=torch.bfloat16, device_map=device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    return te, tok


def encode_text(te, tok, prompt, device):
    # Klein concatena hidden_states[9, 18, 27] del prompt con plantilla de chat, rellenado a 512
    # tokens: [1, 512, 12288]. El transformer no recibe máscara, así que el relleno se guarda tal cual.
    with torch.inference_mode():
        emb = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, [prompt], dtype=torch.bfloat16, device=device)
    return emb.cpu()


def encode_and_save(te, tok, prompt, name):
    torch.save(encode_text(te, tok, prompt, "cuda"), os.path.join(CACHE_DIR, f"{name}_embed.pt"))


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def find_samples():
    """
    [(nombre, ruta_después_o_imagen, ruta_antes_o_None)]. En un LoRA de edición solo cuentan
    los pares nombre_before / nombre_after; en uno normal, cada imagen es una muestra.
    """
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    stems = {os.path.splitext(f)[0]: f for f in files}
    if LORA_TYPE != "edit":
        return [(stem, os.path.join(DATASET_PATH, f), None) for stem, f in stems.items()]
    pairs = [(s[:-len("_before")], stems[s[:-len("_before")] + "_after"], f) for s, f in stems.items()
             if s.endswith("_before") and s[:-len("_before")] + "_after" in stems]
    return [(name, os.path.join(DATASET_PATH, after), os.path.join(DATASET_PATH, before)) for name, after, before in pairs]


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def preview_edit_source(samples, image_path=None):
    # Imagen de antes para la preview de edición: la del usuario (fuera del dataset, mide si generaliza)
    # o, si no hay, el antes del primer par.
    image_path = PREVIEW_EDIT_IMAGE if image_path is None else image_path
    return image_path if image_path and os.path.exists(image_path) else samples[0][2]


def encode_latents(vae, jobs, device):
    # Como el pipeline: latente de 32 canales -> parches 2x2 (128 canales) -> batch norm del VAE.
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, torch.float32)

    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            img = fit(Image.open(src).convert("RGB"), bw, bh)

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0)
            img_tensor = (img_tensor.float() / 127.5) - 1.0
            img_tensor = img_tensor.to(device, dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.mode().float()
            latent = ((Flux2KleinPipeline._patchify_latents(z) - bn_mean) / bn_std).to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] Image / Imagen: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_klein9b():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        sys.exit(1)

    samples = find_samples()
    edit = LORA_TYPE == "edit"
    if not samples:
        if edit:
            print(f"[!] Edit LoRA but no name_before / name_after pairs in '{DATASET_PATH}' / "
                  f"LoRA de edición pero no hay pares nombre_before / nombre_after.")
        else:
            print(f"[!] No images found in '{DATASET_PATH}'. Please add images.")
        sys.exit(1)
    print(f"  LoRA type / Tipo de LoRA    : {'edit (before/after pairs) / edición (pares antes/después)' if edit else 'normal'} | {len(samples)} samples")

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el
    # dataset y, al pasar de edición a normal, los antes (_ctrl).
    suffixes = ("latent", "embed") + (("ctrl",) if edit else ())
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in suffixes}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print(f"  Removed {len(stale)} stale cache files / Eliminados {len(stale)} ficheros antiguos de la caché")

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # Bucket de cada muestra: lo fija el después (o la imagen); el antes se ajusta al mismo tamaño.
    buckets = {}
    for name, target, _ in samples:
        with Image.open(target) as im:
            buckets[name] = bucket_size(*im.size)
    preview_src = preview_edit_source(samples) if edit else None
    if preview_src:
        with Image.open(preview_src) as im:
            buckets["_custom"] = bucket_size(*im.size)

    # ── FASE 1: TEXT ENCODER (Qwen3-8B NF4) ─────────────────────────────────
    # El text encoder y el VAE nunca coinciden en VRAM. Klein solo lee el texto: en edición
    # la imagen de antes no pasa por el text encoder, va al transformer como latente.
    print("\nLoading Text Encoder (Qwen3-8B NF4)... / Cargando Text Encoder (Qwen3-8B NF4)...")
    te, tok = load_text_encoder("cuda")

    encode_and_save(te, tok, "", "_neg")

    # Prompt manual de las previews. En edición va con la imagen de antes de la preview;
    # sin prompt manual se usa la instrucción del primer par.
    custom_prompt = PREVIEW_CUSTOM_PROMPT or (read_caption(samples[0][0]) if edit else "")
    if custom_prompt:
        c_prompt = with_trigger(custom_prompt)
        print(f"[Custom Prompt Cache] Encoding: '{c_prompt}'" + (f" + {os.path.basename(preview_src)}" if edit else ""))
        encode_and_save(te, tok, c_prompt, "_custom")
        # El trainer y el servidor comparan con estos textos para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
        with open(os.path.join(CACHE_DIR, "_custom_image.txt"), "w", encoding="utf-8") as f:
            f.write(PREVIEW_EDIT_IMAGE)
    # Restos de un prompt anterior, o de la preview de edición si el dataset ya no es de pares.
    stale = () if custom_prompt else ("_custom_embed.pt", "_custom_prompt.txt", "_custom_image.txt")
    if not (custom_prompt and edit):
        stale += ("_custom_ctrl.pt",)
    for f in stale:
        if os.path.exists(os.path.join(CACHE_DIR, f)):
            os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _, _) in enumerate(samples, 1):
        encode_and_save(te, tok, with_trigger(read_caption(name)), name)
        print(f"[{idx}/{len(samples)}] Text / Texto: {name}")

    del te, tok
    free_vram()

    # ── FASE 2: VAE (FLUX.2, 32 canales) ────────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual o los captions
    # solo cuesta la fase de texto. En edición, el antes se guarda como <nombre>_ctrl.pt.
    # La imagen de la preview de edición se codifica siempre: puede ser otro fichero más antiguo.
    jobs = [(target, f"{name}_latent.pt", buckets[name]) for name, target, _ in samples]
    jobs += [(control, f"{name}_ctrl.pt", buckets[name]) for name, _, control in samples if control]
    if preview_src and custom_prompt:
        jobs.append((preview_src, "_custom_ctrl.pt", buckets["_custom"]))

    pending = []
    for src, out, (bw, bh) in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (out != "_custom_ctrl.pt" and os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 16, bw // 16)):
            continue
        pending.append((src, out, (bw, bh)))

    if len(pending) < len(jobs):
        print(f"\n{len(jobs) - len(pending)} latents already cached, skipped / latentes ya cacheados, se saltan.")
    if not pending:
        print("\n✓ Pre-caching finished! / ¡Pre-caché finalizado!")
        return

    print("\nLoading VAE (FLUX.2)... / Cargando VAE (FLUX.2)...")
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, pending, "cuda")
    del vae
    free_vram()
    print("\n✓ Pre-caching finished! VRAM freed / ¡Pre-caché finalizado! VRAM liberada.")


def encode_preview_prompt(cache_dir, prompt, image_path, device):
    """
    Codifica solo el prompt manual de las previews (y en edición su imagen de antes). Lo lanza el
    servidor al guardar un prompt o una imagen nuevos. Sin entrenamiento en marcha va en GPU; con el
    entrenamiento en marcha va en CPU (VAE en FP32) para no tocar su VRAM, y el trainer relee el
    embedding antes de la siguiente preview.
    """
    global CACHE_DIR
    CACHE_DIR = cache_dir
    where = device.upper().replace("CUDA", "GPU")
    has_image = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(cache_dir))
    if has_image:
        src = preview_edit_source(find_samples(), image_path)
        with Image.open(src) as im:
            bucket = bucket_size(*im.size)
        # Se escribe aparte y se coloca al final junto al texto.
        # El servidor lee las líneas "[Custom Prompt] Image/Prompt" para mostrar la fase en la GUI.
        print(f"[Custom Prompt] Image: encoding reference image on {where} / Codificando imagen de referencia en {where}: {src}", flush=True)
        vae = AutoencoderKLFlux2.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        encode_latents(vae, [(src, "_custom_ctrl.new", bucket)], device)
        del vae
        free_vram()

    print(f"[Custom Prompt] Prompt: encoding on {where} / Codificando en {where}: '{prompt}'", flush=True)
    te, tok = load_text_encoder(device)
    embeds = encode_text(te, tok, prompt, device)
    del te, tok
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    if has_image:
        os.replace(os.path.join(cache_dir, "_custom_ctrl.new"), os.path.join(cache_dir, "_custom_ctrl.pt"))
    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_image.txt", lambda f: open(f, "w", encoding="utf-8").write(image_path))
    replace("_custom_embed.pt", lambda f: torch.save(embeds, f))
    print("[Custom Prompt] Ready / Listo.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <imagen de antes o ""> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_klein9b()
