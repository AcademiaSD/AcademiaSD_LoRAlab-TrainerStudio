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

LoRA de edición (lora_type "edit"): pares nombre_before / nombre_after con la instrucción en
nombre.txt. El después es el objetivo; el antes se guarda como referencia (<nombre>_ctrl.pt). Cómo
depende de la base de edición (edit_base), igual que el edit_format del nodo Text Encode MOSS-Image
Edit de ComfyUI:
  - "armV4" (akhaliq/MOSS-Image-Edit-HQ, formato ai-toolkit): la referencia hasta 0,5 MP y el text
    encoder también la ve, reducida a 384x384 y tras "Picture N:".
  - "chinoll" (chinoll/Moss-Image-Edit): el lado largo de la referencia limitado a reference_max_edge
    (512) y la instrucción codificada como un prompt normal.
La base queda escrita en la caché (_edit_base.txt): el trainer la lee de ahí. Con armV4, cada
muestra guarda además su prompt negativo vacío con la referencia (<nombre>_uncond.pt), para la rama
negativa de las previews.
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
# Bases de edición. armV4: referencias hasta control_image_max_pixels de su ai-toolkit (0,5 MP) y,
# para el text encoder, reducidas a 384x384 como hace ComfyUI con el edit_format ai-toolkit.
EDIT_BASES = ("armV4", "chinoll")
REFERENCE_MAX_PIXELS = 524288
VISION_MAX_PIXELS = 384 * 384
EDIT_BASE_FILE = "_edit_base.txt"

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
    "lora_type": "normal",
    "preview_edit_image": "",
    "preview_edit_images": [],
    "reference_max_edge": 512,
    "edit_base": "armV4",
}

# Imágenes de referencia de la preview de edición (el nodo de ComfyUI admite hasta 4).
PREVIEW_MAX = 4


def as_image_list(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        value = []
    return [str(p).strip() for p in value if str(p).strip()]


def config_preview_images(cfg):
    images = as_image_list(cfg.get("preview_edit_images"))
    return (images or as_image_list(cfg.get("preview_edit_image")))[:PREVIEW_MAX]

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
# "normal" (imagen + caption) o "edit" (pares nombre_before / nombre_after + instrucción).
LORA_TYPE = cfg.get("lora_type", DEFAULTS["lora_type"])
# Solo edición: imágenes de referencia (fuera del dataset) para ver en las previews cómo aplica el efecto.
PREVIEW_EDIT_IMAGES = config_preview_images(cfg) if LORA_TYPE == "edit" else []
REFERENCE_MAX_EDGE = int(cfg.get("reference_max_edge", DEFAULTS["reference_max_edge"]))
EDIT_BASE = cfg.get("edit_base", DEFAULTS["edit_base"])
if EDIT_BASE not in EDIT_BASES:
    print("⚠ " + t("Invalid Edit Base {name}. Using armV4.", name=EDIT_BASE))
    EDIT_BASE = "armV4"

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_mossimage_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2: lados múltiplos de 16 px.
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using 16.", n=MULTIPLE))
    MULTIPLE = 16
REFERENCE_MAX_EDGE = max(16, REFERENCE_MAX_EDGE // 16 * 16)

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {'Text Encoder':<22}: {TEXT_ENCODER}")
print(f"  {t('Target Area'):<22}: {TARGET_AREA} px²")
print(f"  {t('Max Side'):<22}: {MAX_SIDE}")
print(f"  {t('Multiple'):<22}: {MULTIPLE}")
if LORA_TYPE == "edit":
    print(f"  {t('Edit Base'):<22}: {EDIT_BASE}")
    if EDIT_BASE == "chinoll":
        print(f"  {t('Reference Max Edge'):<22}: {REFERENCE_MAX_EDGE}")

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


def reference_size(w: int, h: int, base=None):
    # Solo se reduce y los lados se redondean a múltiplos de 16. armV4: hasta 0,5 MP, como su
    # entrenamiento en ai-toolkit; chinoll: hasta que el lado largo quepa en REFERENCE_MAX_EDGE, como
    # Apply MOSS Edit.
    if (base or EDIT_BASE) == "armV4":
        s = min(1.0, math.sqrt(REFERENCE_MAX_PIXELS / (w * h)))
    else:
        s = min(1.0, REFERENCE_MAX_EDGE / max(w, h))
    return max(16, round(w * s / 16) * 16), max(16, round(h * s / 16) * 16)


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


def vision_patches(img):
    """
    Una imagen PIL como la ve Qwen3.5 en el nodo Text Encode MOSS-Image Edit de ComfyUI: reducida
    (interpolación "area") hasta 384x384 px de área, llevada con bilineal a múltiplos de 32,
    normalizada a [-1, 1] y en parches de 16 px repetidos en los 2 fotogramas del parche temporal.
    Devuelve (parches [n, 1536], rejilla [1, 3]).
    """
    x = F_vision.pil_to_tensor(img.convert("RGB")).float().unsqueeze(0) / 255.0
    h, w = x.shape[-2:]
    s = min(1.0, math.sqrt(VISION_MAX_PIXELS / (h * w)))
    if s < 1.0:
        x = torch.nn.functional.interpolate(x, size=(round(h * s), round(w * s)), mode="area")
    h, w = x.shape[-2:]
    patch, merge = 16, 2
    factor = patch * merge
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    x = torch.nn.functional.interpolate(x, size=(hb, wb), mode="bilinear", align_corners=False)[0]
    x = (x - 0.5) / 0.5
    gh, gw = hb // patch, wb // patch
    x = x.unsqueeze(0).repeat(2, 1, 1, 1).reshape(1, 2, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * 2 * patch * patch)
    return x, torch.tensor([[1, gh, gw]])


def encode_text(te, tokenizer, prompt, device, images=()):
    # [longitud, 4096]: la última capa con la plantilla de chat; cada prompt con su longitud real
    # (el trainer rellena y pasa la máscara). Un prompt vacío también lleva la plantilla.
    if not images:
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS).to(device)
        with torch.inference_mode():
            hidden = te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
        return hidden.to(torch.bfloat16).cpu()

    # Con imágenes (armV4), como el edit_format ai-toolkit de ComfyUI: "Picture N:" y el bloque de
    # visión de cada una delante de la instrucción, con la misma plantilla de chat.
    # La torre de visión en FP32, como ComfyUI: en BF16 algunos tokens de imagen se alejan mucho
    # (medido: similitud mínima 0,42 frente a 0,96). No está cuantizada en INT8 ni en NF4.
    if next(te.visual.parameters()).dtype != torch.float32:
        te.visual.float()
    vision = [vision_patches(img) for img in images]
    text = "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(len(vision))) + prompt
    formatted = f"<|im_start|>user\n{text.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ids = tokenizer(formatted, return_tensors="pt").input_ids[0]
    # Cada <|image_pad|> pasa a ser un token por cada 2x2 parches de su imagen.
    pad = te.config.image_token_id
    pieces, start, counts = [], 0, iter(int(grid.prod()) // 4 for _, grid in vision)
    for pos in (ids == pad).nonzero().flatten().tolist():
        pieces += [ids[start:pos], torch.full((next(counts),), pad, dtype=ids.dtype)]
        start = pos + 1
    ids = torch.cat(pieces + [ids[start:]])[:MAX_TEXT_TOKENS].unsqueeze(0).to(device)
    with torch.inference_mode():
        hidden = te(input_ids=ids, attention_mask=torch.ones_like(ids),
                    mm_token_type_ids=(ids == pad).int(),
                    pixel_values=torch.cat([p for p, _ in vision]).to(device),
                    image_grid_thw=torch.cat([g for _, g in vision]).to(device),
                    use_cache=False, return_dict=True).last_hidden_state[0]
    return hidden.to(torch.bfloat16).cpu()


def encode_and_save(te, tokenizer, prompt, name, images=(), suffix="embed"):
    torch.save(encode_text(te, tokenizer, prompt, "cuda", images), os.path.join(CACHE_DIR, f"{name}_{suffix}.pt"))


def reference_image(path, size):
    # La referencia tal como se cachea (_ctrl): recortada a la proporción y reducida a su tamaño.
    with Image.open(path) as im:
        return fit(im.convert("RGB"), *size)


def fit(img, bw, bh):
    # Escala hasta cubrir el bucket y recorta al centro.
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def dataset_pairs():
    """(pares nombre_before/nombre_after, imágenes que no forman parte de ningún par)."""
    stems = {os.path.splitext(f)[0] for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}
    pairs = {s[:-len("_before")] for s in stems if s.endswith("_before") and s[:-len("_before")] + "_after" in stems}
    loose = stems - {p + suffix for p in pairs for suffix in ("_before", "_after")}
    return pairs, loose


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


def preview_edit_sources(samples, image_paths=None):
    paths = PREVIEW_EDIT_IMAGES if image_paths is None else as_image_list(image_paths)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("⚠ " + t("Preview reference image not found (it is skipped; later image numbers shift): {path}", path=", ".join(missing)))
    paths = [p for p in paths if os.path.exists(p)][:PREVIEW_MAX]
    if paths:
        return paths
    return [samples[0][2]] if samples and samples[0][2] else []


def custom_ctrl_name(index):
    return "_custom_ctrl.pt" if index == 0 else f"_custom_ctrl_{index + 1}.pt"


def custom_ctrl_names(count):
    return {custom_ctrl_name(i) for i in range(count)}


def remove_stale_custom_ctrls(expected):
    for f in os.listdir(CACHE_DIR):
        if f.startswith("_custom_ctrl") and f.endswith(".pt") and f not in expected:
            os.remove(os.path.join(CACHE_DIR, f))


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

    # Pares nombre_before / nombre_after con el tipo en "normal": se entrenarían como imágenes sueltas,
    # sin la referencia y sin su caption (nombre.txt). Si solo hay pares, es un LoRA de edición.
    global LORA_TYPE, PREVIEW_EDIT_IMAGES
    pairs, loose = dataset_pairs()
    if LORA_TYPE != "edit" and pairs and not loose:
        print("⚠ " + t("The dataset only has name_before / name_after pairs: training an EDIT LoRA. Set LoRA Type to Edit to use the preview reference image."))
        LORA_TYPE = "edit"
        PREVIEW_EDIT_IMAGES = config_preview_images(cfg)
    elif LORA_TYPE != "edit" and pairs:
        print("⚠ " + t("The dataset mixes {n} before/after pairs with normal images: with LoRA Type normal the pairs train as loose images without their caption.", n=len(pairs)))

    samples = find_samples()
    edit = LORA_TYPE == "edit"
    if not samples:
        if edit:
            print("[!] " + t("Edit LoRA but there are no name_before / name_after pairs in '{path}'.", path=DATASET_PATH))
        else:
            print("[!] " + t("No images found in '{path}'. Please add images.", path=DATASET_PATH))
        sys.exit(1)
    print(f"  {t('LoRA Type'):<22}: {t('edit (before/after pairs)') if edit else 'normal'} | {t('{n} samples', n=len(samples))}")

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el
    # dataset y, al pasar de edición a normal, los antes (_ctrl).
    vision = edit and EDIT_BASE == "armV4"
    suffixes = ("latent", "embed") + (("ctrl",) if edit else ()) + (("uncond",) if vision else ())
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in suffixes}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # Bucket de cada muestra: lo fija el después (o la imagen). El antes es la referencia: se
    # recorta a la proporción del después, para que los píxeles casen, y se reduce a reference_size.
    buckets = {}
    for name, target, _ in samples:
        with Image.open(target) as im:
            buckets[name] = bucket_size(*im.size)
    preview_sources = preview_edit_sources(samples) if edit else []
    preview_sizes = []
    for src in preview_sources:
        with Image.open(src) as im:
            preview_sizes.append(reference_size(*im.size))

    # ── FASE 1: TEXT ENCODER (Qwen3.5-9B) ───────────────────────────────────
    print("\n" + t("Loading {name}...", name=f"Text Encoder (Qwen3.5-9B {TEXT_ENCODER})"))
    te, tokenizer = load_text_encoder("cuda")

    encode_and_save(te, tokenizer, "", "_neg")
    # El trainer lee de aquí la base con la que se codificó la caché de edición.
    base_file = os.path.join(CACHE_DIR, EDIT_BASE_FILE)
    if edit:
        with open(base_file, "w", encoding="utf-8") as f:
            f.write(EDIT_BASE)
    elif os.path.exists(base_file):
        os.remove(base_file)

    # Prompt manual de las previews. En edición va con las imágenes de referencia de la preview;
    # sin prompt manual se usa la instrucción del primer par.
    custom_prompt = PREVIEW_CUSTOM_PROMPT or (read_caption(samples[0][0]) if edit else "")
    if os.path.exists(os.path.join(CACHE_DIR, "_custom_uncond.pt")) and not (custom_prompt and vision):
        os.remove(os.path.join(CACHE_DIR, "_custom_uncond.pt"))
    if custom_prompt:
        c_prompt = with_trigger(custom_prompt)
        print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt) + (f" + {', '.join(os.path.basename(p) for p in preview_sources)}" if edit else ""))
        images = [reference_image(src, size) for src, size in zip(preview_sources, preview_sizes)] if vision else ()
        encode_and_save(te, tokenizer, c_prompt, "_custom", images)
        if vision:
            encode_and_save(te, tokenizer, "", "_custom", images, "uncond")
        # El trainer y el servidor comparan con estos textos para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
        with open(os.path.join(CACHE_DIR, "_custom_image.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(PREVIEW_EDIT_IMAGES))
    else:
        for f in ("_custom_embed.pt", "_custom_prompt.txt", "_custom_image.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))
    expected_custom_ctrls = custom_ctrl_names(len(preview_sources)) if custom_prompt and edit else set()

    for idx, (name, _, control) in enumerate(samples, 1):
        # armV4: con su referencia, y también el negativo vacío con ella (rama negativa de las previews).
        images = [reference_image(control, reference_size(*buckets[name]))] if vision else ()
        encode_and_save(te, tokenizer, with_trigger(read_caption(name)), name, images)
        if vision:
            encode_and_save(te, tokenizer, "", name, images, "uncond")
        print(f"[{idx}/{len(samples)}] {t('Text')}: {name}")

    del te, tokenizer
    free_vram()

    # ── FASE 2: VAE (FLUX.2, 32 canales) ────────────────────────────────────
    # Un latente ya cacheado con el tamaño actual y más nuevo que su imagen no se vuelve a
    # codificar: relanzar el pre-caché para cambiar el prompt manual o los captions solo cuesta el
    # texto. En edición, el antes se guarda como <nombre>_ctrl.pt. Las imágenes de la preview de
    # edición se codifican siempre: pueden ser otros ficheros más antiguos.
    jobs = [(target, f"{name}_latent.pt", buckets[name]) for name, target, _ in samples]
    jobs += [(control, f"{name}_ctrl.pt", reference_size(*buckets[name])) for name, _, control in samples if control]
    if custom_prompt and edit:
        jobs += [(src, custom_ctrl_name(i), size) for i, (src, size) in enumerate(zip(preview_sources, preview_sizes))]

    pending = []
    for src, out, (bw, bh) in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (not out.startswith("_custom_ctrl") and os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 16, bw // 16)):
            continue
        pending.append((src, out, (bw, bh)))

    if len(pending) < len(jobs):
        print("\n" + t("{n} latents already cached, skipped.", n=len(jobs) - len(pending)))
    if not pending:
        remove_stale_custom_ctrls(expected_custom_ctrls)
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (FLUX.2)"))
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, pending, "cuda")
    remove_stale_custom_ctrls(expected_custom_ctrls)
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def parse_preview_images_arg(value):
    value = (value or "").strip()
    if value.startswith("["):
        try:
            return as_image_list(json.loads(value))
        except (ValueError, TypeError):
            return []
    return as_image_list(value)


def encode_preview_prompt(cache_dir, prompt, images_arg, device):
    """
    Codifica solo el prompt manual de las previews y, en edición, sus imágenes de referencia. Lo
    lanza el servidor al guardar un prompt nuevo. Sin entrenamiento en marcha va en GPU; con el
    entrenamiento en marcha va en CPU para no tocar su VRAM, y el trainer relee el embedding antes
    de la siguiente preview.
    """
    global CACHE_DIR, EDIT_BASE
    CACHE_DIR = cache_dir
    where = device.upper().replace("CUDA", "GPU")
    requested = parse_preview_images_arg(images_arg)
    has_image = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(cache_dir))
    sources = preview_edit_sources(find_samples(), requested) if has_image else []
    # La base con la que se codificó la caché (una caché anterior a armV4 es de chinoll).
    base_file = os.path.join(cache_dir, EDIT_BASE_FILE)
    EDIT_BASE = open(base_file, "r", encoding="utf-8").read().strip() if os.path.exists(base_file) else "chinoll"
    vision = bool(sources) and EDIT_BASE == "armV4"

    jobs = []
    for i, src in enumerate(sources):
        with Image.open(src) as im:
            jobs.append((src, custom_ctrl_name(i) + ".new", reference_size(*im.size)))
    if sources:
        # Se escriben aparte y se colocan al final junto al texto.
        # El servidor lee las líneas "[Custom Prompt] Image/Prompt" para mostrar la fase en la GUI.
        print("[Custom Prompt] Image: " + t("Encoding the reference image on {device}: {path}", device=where, path=", ".join(os.path.basename(p) for p in sources)), flush=True)
        vae = AutoencoderKLFlux2.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        encode_latents(vae, jobs, device)
        del vae
        free_vram()

    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    te, tokenizer = load_text_encoder(device)
    images = [reference_image(src, size) for src, _, size in jobs] if vision else ()
    embeds = encode_text(te, tokenizer, prompt, device, images)
    uncond = encode_text(te, tokenizer, "", device, images) if vision else None
    del te, tokenizer
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    for i in range(len(sources)):
        name = custom_ctrl_name(i)
        os.replace(os.path.join(cache_dir, name + ".new"), os.path.join(cache_dir, name))
    remove_stale_custom_ctrls(custom_ctrl_names(len(sources)))
    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_image.txt", lambda f: open(f, "w", encoding="utf-8").write("\n".join(requested)))
    if uncond is not None:
        replace("_custom_uncond.pt", lambda f: torch.save(uncond, f))
    elif os.path.exists(os.path.join(cache_dir, "_custom_uncond.pt")):
        os.remove(os.path.join(cache_dir, "_custom_uncond.pt"))
    replace("_custom_embed.pt", lambda f: torch.save(embeds, f))
    print("[Custom Prompt] " + t("Ready."), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <rutas JSON o una ruta> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_mossimage()
