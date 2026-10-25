# -*- coding: utf-8 -*-
"""
1_pre_cache_mingimage.py — Pre-caché de latentes + embeddings para Ming-Image (Design)
Pre-cache of latents + embeddings for Ming-Image (Design)

El "text encoder" de Ming-Image no es un encoder de texto corriente: es un MLLM completo
(BailingMoE-V2 + ViT) al que se le pide la condición con `image_gen_only_extract_hidden_states`.
Devuelve dos tensores por prompt:
  - scale_embeds (256 tokens x 2560)  -> cap_feats   (el que ve el DiT)
  - directvlm    (N tokens  x 3840)   -> cap_feats_2 (estados del LLM, proyección directa)
Los negativos del CFG son ceros del mismo tamaño (así los construye el pipeline oficial).

Lee configuración desde pre_cache_settings_mingimage.json si existe.
Reads configuration from pre_cache_settings_mingimage.json if present.
"""
import os
import gc
import math
import json
import warnings
import sys

import torch
import torchvision.transforms.functional as F_vision
from PIL import Image

from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: las capas cuya dimensión no es múltiplo de 64 usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")
# El LLM del código oficial usa la API antigua de máscaras de transformers (modeling_attn_mask_utils).
# Migrarla a masking_utils es reescribir su máscara de atención: de momento solo se calla el aviso.
warnings.filterwarnings("ignore", message=".*attention mask API under `transformers.modeling_attn_mask_utils`.*")

# El código oficial (MIT) vive en mingimage_vendor/ porque la arquitectura no está en diffusers.
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mingimage_vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from processing_bailingmm2 import load_bailingmm2_processor
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

# El mismo paquete que usa el trainer: MLLM ya en NF4 (9,35 GB) en lugar de los 38 GB del original.
HF_REPO_ID = "AcademiaSD/Ming-Image-NF4-for-LoRA-Training"

# El pre-caché no necesita el DiT (3,4 GB en NF4); solo el MLLM, el connector, el mlp y el VAE.
# transformer/config.json sí hace falta: de ahí sale el perfil del checkpoint (zero_masked).
MODEL_FILES = ["config.json", "mllm/*", "connector/*", "mlp/*", "vae/*", "scheduler/*", "transformer/config.json"]

# NF4 = MLLM en 4 bits en VRAM (~11,6 GB). BF16 = exacto pero no cabe en 16 GB: se reparte con device_map.
# BF16 solo es posible con el checkpoint original (Ming-Image_Raw): el paquete NF4 ya viene cuantizado.
TEXT_ENCODER_MODES = ("NF4", "BF16")

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Models/Ming-Image_NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_mingimage",
    "target_area": 768 * 768,
    "max_side": 1280,
    "multiple": 32,
    "text_encoder": "NF4",
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    # "normal" (texto a imagen) o "edit" (pares nombre_before / nombre_after + instrucción).
    "lora_type": "normal",
    "preview_edit_image": "",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "settings/pre_cache_settings_mingimage.json"

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
TEXT_ENCODER = cfg.get("text_encoder", DEFAULTS["text_encoder"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()
LORA_TYPE    = "edit" if cfg.get("lora_type", DEFAULTS["lora_type"]) == "edit" else "normal"
PREVIEW_EDIT_IMAGE = cfg.get("preview_edit_image", "").strip() if LORA_TYPE == "edit" else ""

# Formato automático de carpeta de caché según nombre del proyecto
CACHE_DIR = str(resolve_cache_dir(f"./cached_data_mingimage_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2 en el transformer: múltiplos de 16 px como mínimo; se usa 32 o 64.
if MULTIPLE not in (32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using {d}.", n=MULTIPLE, d=32))
    MULTIPLE = 32

if TEXT_ENCODER not in TEXT_ENCODER_MODES:
    print("⚠ " + t("Invalid Text Encoder {name}. Using NF4.", name=TEXT_ENCODER))
    TEXT_ENCODER = "NF4"

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Target Area'):<22}: {TARGET_AREA} px²")
print(f"  {t('Max Side'):<22}: {MAX_SIDE}")
print(f"  {t('Multiple'):<22}: {MULTIPLE}")
print(f"  {'Text Encoder':<22}: {TEXT_ENCODER}")
print(f"  {t('LoRA Type'):<22}: {t('edit (before/after pairs)') if LORA_TYPE == 'edit' else 'normal'}")

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


def model_is_complete(local_path):
    return all(os.path.exists(os.path.join(local_path, p)) for p in (
        "config.json", os.path.join("mllm", "config.json"), os.path.join("connector", "config.json"),
        os.path.join("mlp", "config.json"), os.path.join("vae", "config.json"),
        os.path.join("transformer", "config.json"),
    ))


def ensure_model_downloaded(local_path, repo_id):
    if model_is_complete(local_path):
        print("✓ " + t("Local model found at: {path}", path=local_path))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=repo_id))

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(t("huggingface_hub is required. Install it with: pip install huggingface_hub"))

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=4,
        allow_patterns=MODEL_FILES,
    )
    print("✓ " + t("Model downloaded to: {path}", path=downloaded_path))
    return downloaded_path


def with_trigger(prompt):
    if not TRIGGER_WORD or TRIGGER_WORD.lower() in prompt.lower():
        return prompt
    # Caption JSON (el formato de Ming): el trigger va dentro de canvas_settings.image_style, como lo
    # pone el captioner. Delante del "{" el MLLM recibiría un JSON roto.
    try:
        data = json.loads(prompt)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("layers"), list):
        cs = data.get("canvas_settings")
        if not isinstance(cs, dict):
            cs = data["canvas_settings"] = {}
        style = str(cs.get("image_style", "") or "")
        cs["image_style"] = f"{TRIGGER_WORD}, {style}".strip(", ")
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"{TRIGGER_WORD}, {prompt}".strip(", ")


def mllm_is_prequantized():
    # El paquete NF4 guarda el MLLM ya cuantizado (quantization_config en mllm/config.json).
    try:
        with open(os.path.join(MODEL_ID, "mllm", "config.json"), "r", encoding="utf-8") as f:
            return "quantization_config" in json.load(f)
    except (OSError, ValueError):
        return False


def load_text_encoder(device="cuda"):
    """MLLM en 4 bits (o BF16 repartido) sin cargar el DiT."""
    mode = TEXT_ENCODER
    kw = dict(
        dtype=torch.bfloat16,
        attn_implementation="eager",   # el LLM de Bailing solo implementa eager / flash_attention_2
        load_image_gen=True,
        load_image_gen_diffusion=False,
        image_gen_device=device,
    )
    if mllm_is_prequantized():
        if mode != "NF4":
            print("⚠ " + t("The MLLM of {path} is already NF4; BF16 needs the original checkpoint. Using NF4.", path=MODEL_ID))
        # Ya viene cuantizado: transformers lo carga tal cual, sin recuantizar.
        kw["device_map"] = {"": "cuda:0" if device.startswith("cuda") else "cpu"}
        if not device.startswith("cuda"):
            # En CPU todo en fp32: 16 s por prompt frente a 71 s en bf16 (la CPU no tiene el autocast
            # de la GPU). No sale idéntico al de la GPU (cos ~0,94: el MoE elige otros expertos con
            # cualquier cambio de precisión), pero solo se usa para las previews en caliente.
            kw["dtype"] = torch.float32
        model = BailingMM2NativeForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
        if not device.startswith("cuda"):
            from bitsandbytes.nn import Linear4bit
            for m in model.modules():
                if isinstance(m, Linear4bit):
                    m.compute_dtype = torch.float32
        return model
    if mode == "NF4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # En GPU hay autocast bf16 (acumula en fp32). En CPU no: con bf16 el matmul
            # acumula en bf16 y el texto se desvía del de la GPU. En fp32 coincide.
            bnb_4bit_compute_dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
        )
        kw["device_map"] = {"": "cuda:0" if device.startswith("cuda") else "cpu"}
        try:
            return BailingMM2NativeForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
        except Exception as exc:
            # El 4 bits en CPU no siempre está disponible; para la preview durante el
            # entrenamiento se cae a BF16 exacto (cabe en RAM, es más lento).
            if device.startswith("cuda"):
                raise
            print("⚠ " + t("4-bit on CPU failed ({err}); falling back to BF16.", err=str(exc)[:160]))
            kw.pop("quantization_config", None)

    # En GPU el reparto automático usa el espacio libre; en CPU nunca debe tocar la GPU
    # (el prompt manual se codifica mientras el entrenamiento ocupa la VRAM).
    kw["device_map"] = "auto" if device.startswith("cuda") else {"": "cpu"}
    return BailingMM2NativeForConditionalGeneration.from_pretrained(MODEL_ID, **kw)


def encode_prompt(model, processor, prompt, image=None):
    """
    Devuelve (cap_feats, cap_feats_2) ya en CPU y en bf16. En edición `image` es el antes: va
    delante del texto en el mensaje, como en el infer.py oficial (image-edit), y el ViT lo mete
    en las dos condiciones. La tarea se declara text-to-image porque la validación de image-edit
    pide el tensor de referencia del DiT, que aquí no se usa (el latente va aparte, en _ctrl).
    """
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
    messages = [{"role": "HUMAN", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, _ = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            inputs[key] = value.to(torch.bfloat16)

    embeds, _neg, embeds2, _neg2 = model.generate(
        **inputs,
        image_gen=True,
        image_gen_task="text-to-image",
        image_gen_only_extract_hidden_states=True,
    )
    if embeds is None or embeds2 is None:
        raise RuntimeError(t("The model did not return the image conditioning embeddings."))
    return embeds.detach().to("cpu", torch.bfloat16), embeds2.detach().to("cpu", torch.bfloat16)


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def find_samples():
    """
    ([(nombre, ruta del objetivo, ruta del antes o None)], _before saltados).
    Normal: cada imagen es una muestra; en un par se ignora el _before (es la entrada de una
    edición, no el objetivo de un texto→imagen) y el caption es el de nombre_after.txt.
    Edit: solo los pares; la muestra se llama como el par y su instrucción está en nombre.txt.
    """
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    stems = {os.path.splitext(f)[0]: f for f in files}
    pairs = [s[:-len("_before")] for s in stems if s.endswith("_before") and s[:-len("_before")] + "_after" in stems]
    if LORA_TYPE == "edit":
        return [(base, os.path.join(DATASET_PATH, stems[base + "_after"]), os.path.join(DATASET_PATH, stems[base + "_before"]))
                for base in pairs], 0
    paired_before = {base + "_before" for base in pairs}
    samples = [(stem, os.path.join(DATASET_PATH, f), None) for stem, f in stems.items() if stem not in paired_before]
    return samples, len(paired_before)


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_latents(vae, jobs, device="cuda"):
    z_dim = vae.config.z_dim
    scaling = vae.config.scaling_factor
    shift = vae.config.shift_factor or 0.0

    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            # El VAE de Ming lee RGBA: las imágenes opacas llevan alfa = 1, como en el pipeline.
            img = fit(Image.open(src).convert("RGBA"), bw, bh)

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0).unsqueeze(2)
            img_tensor = (img_tensor.float() / 127.5) - 1.0
            img_tensor = img_tensor.to(device, dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.mode().float()
            latent = ((z - shift) * scaling)
            if latent.dim() == 5:
                latent = latent[:, :, 0]
            assert latent.shape[1] == z_dim, f"z_dim {latent.shape[1]} != {z_dim}"
            latent = latent.to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] {t('Image')}: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preview_edit_source(samples, image_path=None):
    # Antes de la preview de edición: la imagen elegida (fuera del dataset, mide si generaliza)
    # o, si no hay, el antes del primer par.
    image_path = PREVIEW_EDIT_IMAGE if image_path is None else image_path
    return image_path if image_path and os.path.exists(image_path) else samples[0][2]


def load_before(path, bucket):
    # El antes, al mismo bucket que el objetivo: el DiT lo concatena como un segundo fotograma.
    with Image.open(path) as im:
        return fit(im.convert("RGBA"), *bucket)


def preprocess_mingimage():
    if not os.path.exists(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)

    samples, skipped = find_samples()
    edit = LORA_TYPE == "edit"
    if not samples:
        if edit:
            print("[!] " + t("Edit LoRA but there are no name_before / name_after pairs in '{path}'.", path=DATASET_PATH))
        else:
            print("[!] " + t("No images found in '{path}'. Please add images.", path=DATASET_PATH))
        sys.exit(1)
    print(f"  {t('Samples'):<22}: {len(samples)}" + (f" ({t('before/after pairs')})" if edit else ""))
    if skipped:
        print(f"  {'Pairs':<22}: {skipped} _before skipped (input of an edit, not the target)")
        print("⚠ " + t("The dataset has name_before / name_after pairs: set LoRA Type to Edit to train editing instead of text-to-image."))
    if edit and TARGET_AREA < 1024 * 1024:
        print("⚠ " + t("The official image-edit task works at 1024: smaller buckets train faster but are out of its distribution."))

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el dataset.
    suffixes = ("latent", "embed", "embed2") + (("ctrl",) if edit else ())
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in suffixes}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    buckets = {}
    for name, path, _before in samples:
        with Image.open(path) as im:
            buckets[name] = bucket_size(*im.size)

    # ── FASE 1: MLLM (BailingMoE-V2 + ViT + connector) ──────────────────────
    # El MLLM y el VAE nunca coinciden en VRAM.
    print("\n" + t("Loading {name}...", name=f"Text Encoder (Ming-Image MLLM {TEXT_ENCODER})"))
    processor = load_bailingmm2_processor(os.path.join(MODEL_ID, "mllm"))
    model = load_text_encoder("cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ {t('Ready.')} {n_params / 1e9:.1f} B params | VRAM {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    embeds, embeds2 = encode_prompt(model, processor, "")
    # El negativo del pipeline oficial son ceros exactos (no el embedding del prompt vacío):
    # se guarda con la forma real para que el servidor y el trainer sepan que la caché está completa.
    torch.save(torch.zeros_like(embeds), os.path.join(CACHE_DIR, "_neg_embed.pt"))
    torch.save(torch.zeros_like(embeds2), os.path.join(CACHE_DIR, "_neg_embed2.pt"))
    print(f"[neg] cap_feats {tuple(embeds.shape)} | directvlm {tuple(embeds2.shape)} (zeros)")

    # Prompt manual de las previews (en edición, con su imagen de antes).
    custom_prompt = PREVIEW_CUSTOM_PROMPT
    custom_src = preview_edit_source(samples) if edit and custom_prompt else None
    for f in ("_custom_ctrl.pt", "_custom_image.txt"):
        if not custom_src and os.path.exists(os.path.join(CACHE_DIR, f)):
            os.remove(os.path.join(CACHE_DIR, f))
    if custom_prompt:
        c_prompt = with_trigger(custom_prompt)
        print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt)
              + (f" + {os.path.basename(custom_src)}" if custom_src else ""))
        if custom_src:
            with Image.open(custom_src) as im:
                buckets["_custom"] = bucket_size(*im.size)
        c_embeds, c_embeds2 = encode_prompt(model, processor, c_prompt,
                                            load_before(custom_src, buckets["_custom"]) if custom_src else None)
        torch.save(c_embeds, os.path.join(CACHE_DIR, "_custom_embed.pt"))
        torch.save(c_embeds2, os.path.join(CACHE_DIR, "_custom_embed2.pt"))
        # El trainer y el servidor comparan con estos textos para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
        if custom_src:
            with open(os.path.join(CACHE_DIR, "_custom_image.txt"), "w", encoding="utf-8") as f:
                f.write(PREVIEW_EDIT_IMAGE)
        print(f"[custom] cap_feats {tuple(c_embeds.shape)} | directvlm {tuple(c_embeds2.shape)}")
    else:
        for f in ("_custom_embed.pt", "_custom_embed2.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    missing = []
    with torch.inference_mode():
        for idx, (name, _path, before) in enumerate(samples, 1):
            caption = read_caption(name)
            if edit and not caption:
                missing.append(name)
            caption = with_trigger(caption)
            e, e2 = encode_prompt(model, processor, caption, load_before(before, buckets[name]) if before else None)
            torch.save(e, os.path.join(CACHE_DIR, f"{name}_embed.pt"))
            torch.save(e2, os.path.join(CACHE_DIR, f"{name}_embed2.pt"))
            print(f"[{idx}/{len(samples)}] {t('Text')}: {name} | {e.shape[1]} + {e2.shape[1]} tokens")
    if missing:
        print("⚠ " + t("{n} pairs without an instruction in name.txt (only the trigger is used): {names}",
                      n=len(missing), names=", ".join(missing[:10])))

    del model, processor
    free_vram()

    # ── FASE 2: VAE (Qwen-Image, RGBA) ──────────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual solo cuesta el texto.
    jobs = []
    for name, path, before in samples:
        jobs.append((path, f"{name}_latent.pt", buckets[name]))
        if before:
            jobs.append((before, f"{name}_ctrl.pt", buckets[name]))
    if custom_src:
        jobs.append((custom_src, "_custom_ctrl.pt", buckets["_custom"]))

    pending = []
    for src, out, (bw, bh) in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 8, bw // 8)):
            continue
        pending.append((src, out, (bw, bh)))

    if len(pending) < len(jobs):
        print("\n" + t("{n} latents already cached, skipped.", n=len(jobs) - len(pending)))
    if not pending:
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (Ming-Image, RGBA)"))
    vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, pending)
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def encode_preview_prompt(cache_dir, prompt, image_path, device):
    """
    Codifica solo el prompt manual de las previews (y, en edición, su imagen de antes). Lo lanza
    el servidor al guardar uno nuevo. Sin entrenamiento va en GPU; con el entrenamiento en marcha
    va en CPU para no tocar su VRAM, y el trainer relee el embedding antes de la siguiente preview.
    """
    global CACHE_DIR, LORA_TYPE
    CACHE_DIR = cache_dir
    where = device.upper().replace("CUDA", "GPU")

    # La caché dice si el LoRA es de edición (los _ctrl.pt de las muestras), no los ajustes.
    edit = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(cache_dir))
    image = None
    if edit:
        LORA_TYPE = "edit"
        src = preview_edit_source(find_samples()[0], image_path)
        with Image.open(src) as im:
            bucket = bucket_size(*im.size)
        image = load_before(src, bucket)
        # Se escribe aparte y se coloca al final junto al texto: el antes nuevo no encaja con el embedding viejo.
        # El servidor lee las líneas "[Custom Prompt] Image/Prompt" para mostrar la fase en la GUI.
        print("[Custom Prompt] Image: " + t("Encoding the reference image on {device}: {path}", device=where, path=src), flush=True)
        vae = AutoencoderKLQwenImage.from_pretrained(
            MODEL_ID, subfolder="vae", dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        encode_latents(vae, [(src, "_custom_ctrl.new", bucket)], device)
        del vae
        free_vram()

    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    processor = load_bailingmm2_processor(os.path.join(MODEL_ID, "mllm"))
    model = load_text_encoder(device)
    with torch.inference_mode():
        embeds, embeds2 = encode_prompt(model, processor, prompt, image)
    del model, processor
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    def write_text(text):
        def write(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return write

    if image is not None:
        os.replace(os.path.join(cache_dir, "_custom_ctrl.new"), os.path.join(cache_dir, "_custom_ctrl.pt"))
        replace("_custom_image.txt", write_text(image_path))
    replace("_custom_prompt.txt", write_text(prompt))
    replace("_custom_embed2.pt", lambda f: torch.save(embeds2, f))
    replace("_custom_embed.pt", lambda f: torch.save(embeds, f))
    print("[Custom Prompt] " + t("Ready."), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <imagen de antes o ""> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    elif len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # Forma anterior, sin imagen (la de un servidor arrancado antes de la edición).
        encode_preview_prompt(sys.argv[2], sys.argv[3], "", sys.argv[4])
    else:
        preprocess_mingimage()
