# -*- coding: utf-8 -*-
"""
1_pre_cache_fibo2.py — Pre-caché de latentes + embeddings para Fibo 2.0 (BRIA)
Pre-cache of latents + embeddings for Fibo 2.0 (BRIA)

Fase 1: el text encoder (Qwen3-VL-4B, cuantizado a NF4) codifica los captions y,
en modo edición, la imagen de antes que acompaña a la instrucción.
Fase 2: el VAE (AutoencoderKLFlux2) codifica el objetivo y la imagen de antes.

Al terminar, el entrenamiento (2_train_lora_fibo2.py) no necesita ni el text encoder
ni el VAE: solo latentes y embeddings.

Lee configuración desde pre_cache_settings_fibo2.json si existe.
Reads configuration from pre_cache_settings_fibo2.json if present.
"""
import os
import gc
import json
import warnings
import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKLFlux2
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import (
    BriaFibo2Pipeline,
    DEFAULT_NEGATIVE_PROMPT,
)
from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
import logging
import sys
from i18n import t
import gpu_stats
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import fibo2_caption

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# LLM.int8 avisa en cada matmul de que pasa las entradas de bf16 a fp16.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
# NF4: las capas de visión no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

# El caption negativo de text-to-image que recomienda BRIA (el pipeline oficial lo usa por defecto).
# En edición el negativo es el caption vacío: es con el que se entrenó.
NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, '
    'no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, '
    'no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast",'
    '"overexposed highlights","lens flare","slight blur","motion blur","haze"]},'
    '"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame",'
    '"focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'
)

# Modo del text encoder -> (carpeta text_encoder_<precisión>, repartir entre GPU y RAM).
# BF16_offload es exacto y cabe en cualquier GPU: lo que no entra en VRAM se ejecuta desde RAM.
TEXT_ENCODER_MODES = {
    "BF16_offload": ("BF16", True),
    "BF16":         ("BF16", False),
    "INT8":         ("INT8", False),
    "NF4":          ("NF4",  False),
}
# VRAM que se deja libre al repartir el text encoder entre GPU y RAM (activaciones, contexto CUDA).
OFFLOAD_RESERVE_BYTES = int(1.5 * 1024**3)

# Fibo 2.0 se entrenó con captions JSON de hasta 1000+ palabras (VGL) y con descripciones cortas.
MAX_SEQUENCE_LENGTH = 4096

# El modelo NF4 publicado en Hugging Face (lo mismo que el conversor genera en local).
HF_REPO_ID = "AcademiaSD/Fibo-2.0-NF4-for-LoRA-Training"

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Models/Fibo2_Dev-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_fibo2",
    "target_area": 1024 * 1024,
    "max_side": 1536,
    "multiple": 16,
    "text_encoder": "NF4",
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "preview_edit_image": "",
    "lora_type": "normal",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "settings/pre_cache_settings_fibo2.json"

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
# "normal" (imagen + caption) o "edit" (pares nombre_before / nombre_after + instrucción).
LORA_TYPE = cfg.get("lora_type", DEFAULTS["lora_type"])
# Solo edición: imagen de antes (fuera del dataset) para ver en las previews cómo aplica el efecto.
PREVIEW_EDIT_IMAGE = cfg.get("preview_edit_image", "").strip() if LORA_TYPE == "edit" else ""

# Formato automático de carpeta de caché según nombre del proyecto
CACHE_DIR = str(resolve_cache_dir(f"./cached_data_fibo2_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# Fibo 2.0 trabaja en la rejilla de 16 px del VAE (8 px del VAE x el parche 2x2 del transformer).
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using {d}.", n=MULTIPLE, d=16))
    MULTIPLE = 16

if TEXT_ENCODER not in TEXT_ENCODER_MODES:
    print("⚠ " + t("Invalid Text Encoder {name}. Using BF16_offload.", name=TEXT_ENCODER))
    TEXT_ENCODER = "BF16_offload"

TEXT_ENCODER_DIR = f"text_encoder_{TEXT_ENCODER_MODES[TEXT_ENCODER][0]}"

print(f"  {t('Model ID'):<22}: {MODEL_ID}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Trigger Word'):<22}: {TRIGGER_WORD}")
print(f"  {t('Dataset Path'):<22}: {DATASET_PATH}")
print(f"  {t('Cache Dir'):<22}: {CACHE_DIR}")
print(f"  {t('Target Area'):<22}: {TARGET_AREA} px²")
print(f"  {t('Max Side'):<22}: {MAX_SIDE}")
print(f"  {t('Multiple'):<22}: {MULTIPLE}")
print(f"  {'Text Encoder':<22}: {TEXT_ENCODER}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


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


def ensure_model_downloaded(local_path, text_encoder_dir):
    if (os.path.exists(os.path.join(local_path, "model_index.json"))
            and os.path.exists(os.path.join(local_path, text_encoder_dir, "config.json"))):
        print("✓ " + t("Local model found at: {path}", path=f"{local_path} ({text_encoder_dir})"))
        return local_path

    print("⚠ " + t("Local model not found at: {path}", path=local_path))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=HF_REPO_ID))

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(t("huggingface_hub is required. Install it with: pip install huggingface_hub"))

    hf_token = get_hf_token()

    # Solo la variante del text encoder elegida: las tres juntas son ~16 GB.
    other_text_encoders = [f"text_encoder_{p}/*" for p, _ in TEXT_ENCODER_MODES.values()
                           if f"text_encoder_{p}" != text_encoder_dir]

    downloaded_path = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=local_path,
        token=hf_token,
        max_workers=2,
        ignore_patterns=other_text_encoders,
    )
    print("✓ " + t("Model downloaded to: {path}", path=downloaded_path))

    if not os.path.exists(os.path.join(local_path, text_encoder_dir, "config.json")):
        # El repo puede publicar solo una variante del text encoder.
        print("[!] " + t("The model on Hugging Face does not have the text encoder {name}.",
                         name=text_encoder_dir))
        print("  " + t("Choose another one in the pre-cache, or build it locally with:"))
        print("      tools/fibo2/5_conversor_Fibo2_NF4.py")
        sys.exit(3)

    return downloaded_path


def image_size(aspect_ratio, area, multiple=16):
    """
    El (ancho, alto) con el que Fibo 2.0 se entrenó para una relación de aspecto: de los tamaños
    múltiplos de `multiple`, con al menos 4 múltiplos por lado y dentro del 10% de `area`, el que
    recorta menos de una imagen con esa proporción y luego el más cercano a `area`.

    Es el mismo criterio que BriaFibo2ImageProcessor.get_image_size, con la rejilla ampliable
    a 32 o 64 px (más de un tamaño por cubo = menos cubos = lotes más homogeneous).
    """
    # El área puede llegar como float (al escalar para el tope de lado) y range() exige enteros.
    # The area can arrive as a float (when scaled for the side cap) and range() needs integers.
    area = int(area)
    min_side = 4 * multiple
    sizes = [
        (width, height)
        for width in range(min_side, 2 * area // min_side, multiple)
        for height in range(min_side, 2 * area // width, multiple)
        if 0.9 * area <= width * height <= 1.1 * area
    ]

    def crop(size):
        ratio = size[0] / size[1]
        return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

    if not sizes:
        # Área inalcanzable con esa rejilla: se usa el múltiplo más cercano.
        side = max(min_side, int((area ** 0.5 / multiple) + 0.5) * multiple)
        return side, side

    return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))


def image_size_capped(aspect_ratio, area, max_side, multiple=16):
    """Como image_size, pero ninguna lado pasa de max_side."""
    width, height = image_size(aspect_ratio, area, multiple)
    longest = max(width, height)
    if max_side and longest > max_side:
        # El resultado se escala al tope, manteniendo la proporción y el múltiplo. Escalar
        # el área y volver a buscar en la rejilla puede devolver un tamaño que aún se pasa.
        # The result is scaled to the cap, keeping the proportion and the multiple. Scaling
        # the area and searching the grid again can return a size that is still over.
        scale = max_side / longest
        min_side = 4 * multiple
        width = max(min_side, int(width * scale / multiple) * multiple)
        height = max(min_side, int(height * scale / multiple) * multiple)
    return width, height


def compact(caption):
    """El caption en la forma en que se entrenó: JSON sin espacios."""
    return fibo2_caption.compact(caption)


def with_trigger(prompt):
    """Añade el trigger word al caption sin romperlo (ver fibo2_caption.with_trigger)."""
    return fibo2_caption.with_trigger(prompt, TRIGGER_WORD, is_edit=(LORA_TYPE == "edit"))


def load_text_encoder():
    from transformers import Qwen3VLForConditionalGeneration

    offload = TEXT_ENCODER_MODES[TEXT_ENCODER][1]
    path = os.path.join(MODEL_ID, TEXT_ENCODER_DIR)

    if not offload:
        return Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")

    # Las capas que no caben en la VRAM libre se quedan en RAM y se ejecutan desde allí.
    free_bytes, _ = gpu_stats.mem_get_info()
    gpu_budget = max(free_bytes - OFFLOAD_RESERVE_BYTES, 0)
    print(f"  {t('GPU budget'):<22}: {gpu_budget / 1024**3:.1f} GB ({t('rest in RAM')})")

    return Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: gpu_budget, "cpu": "512GiB"},
    )


def load_fonts():
    """Las fuentes con las que se dibujaron los textos del caption en el entrenamiento."""
    from diffusers.pipelines.bria_fibo2.text_images import load_font

    paths = []
    for name in ("Aileron-Regular.otf", "DejaVuSans.ttf"):
        path = os.path.join(MODEL_ID, "fonts", name)
        if os.path.exists(path):
            paths.append(path)
    if not paths:
        print("⚠ " + t("Fonts not found in {path}: the texts of the caption will be drawn with the default font.",
                       path=os.path.join(MODEL_ID, "fonts")))
        return []
    return [load_font(p) for p in paths]


def caption_text_images(caption, fonts):
    """Las imágenes del texto del caption (text_render), dibujadas como en el entrenamiento."""
    from diffusers.pipelines.bria_fibo2.text_images import draw_text_image

    entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
    images = [
        draw_text_image(entry["text"], fonts)
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]
    ]
    return images or None


def save_tensor(name, tensor):
    """Escritura atómica: el trainer puede estar leyendo la caché."""
    tmp = os.path.join(CACHE_DIR, name + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, os.path.join(CACHE_DIR, name))


def save_prompt(name, embeds, mask, base_embeds=None, base_mask=None):
    """Guarda los embeddings de una muestra: (6, tokens, 7680) y su máscara (tokens,)."""
    torch.save(embeds[0].to(torch.bfloat16).cpu().contiguous(), os.path.join(CACHE_DIR, f"{name}_embed.pt"))
    torch.save(mask[0].bool().cpu().contiguous(), os.path.join(CACHE_DIR, f"{name}_mask.pt"))
    if base_embeds is not None:
        torch.save(base_embeds[0].to(torch.bfloat16).cpu().contiguous(),
                   os.path.join(CACHE_DIR, f"{name}_base_embed.pt"))
        torch.save(base_mask[0].bool().cpu().contiguous(), os.path.join(CACHE_DIR, f"{name}_base_mask.pt"))
    return embeds.shape[2]


def dataset_pairs():
    """(pares nombre_before/nombre_after, imágenes que no forman parte de ningún par)."""
    stems = {os.path.splitext(f)[0] for f in os.listdir(DATASET_PATH) if f.lower().endswith(IMAGE_SUFFIXES)}
    pairs = {s[:-len("_before")] for s in stems if s.endswith("_before") and s[:-len("_before")] + "_after" in stems}
    loose = stems - {p + suffix for p in pairs for suffix in ("_before", "_after")}
    return pairs, loose


def find_samples():
    """
    [(nombre, ruta_después_o_imagen, ruta_antes_o_None)]. En un LoRA de edición solo cuentan
    los pares nombre_before / nombre_after; en uno normal, cada imagen es una muestra.
    """
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith(IMAGE_SUFFIXES))
    stems = {os.path.splitext(f)[0]: f for f in files}
    if LORA_TYPE != "edit":
        return [(stem, os.path.join(DATASET_PATH, f), None) for stem, f in stems.items()]
    pairs = [(s[:-len("_before")], stems[s[:-len("_before")] + "_after"], f) for s, f in stems.items()
             if s.endswith("_before") and s[:-len("_before")] + "_after" in stems]
    return [(name, os.path.join(DATASET_PATH, after), os.path.join(DATASET_PATH, before)) for name, after, before in pairs]


def read_caption(name):
    """
    El caption de una muestra: <nombre>.json (caption estructurado, gana) o <nombre>.txt (frase).
    """
    json_path = fibo2_caption.caption_path(os.path.join(DATASET_PATH, name))
    if json_path.endswith(fibo2_caption.JSON_EXT) and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return compact({"short_description": str(data)})
            return compact(data)
        except (OSError, ValueError) as exc:
            print(f"⚠ {name}.json no se pudo leer ({exc}); se usa el .txt")
    txt_path = fibo2_caption.caption_path(os.path.join(DATASET_PATH, name), prefer_json=False)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def preview_edit_source(samples, image_path=None):
    # Imagen de antes para la preview de edición: la del usuario (fuera del dataset, mide si generaliza)
    # o, si no hay, el antes del primer par.
    image_path = PREVIEW_EDIT_IMAGE if image_path is None else image_path
    return image_path if image_path and os.path.exists(image_path) else samples[0][2]


def load_image(path):
    """La imagen tal y como la ve el modelo: RGB, con la orientación EXIF corregida."""
    from PIL import ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def prepare_latents(vae, images, device, dtype=torch.float32):
    """
    Los latentes del VAE normalizados con sus estadísticas de BatchNorm, como los normaliza
    el pipeline: se parchean (2x2), se normalizan en ese formato y se deshacen.
    """
    out = []
    for img in images:
        img = img.contiguous().to(device=device, dtype=vae.dtype)
        latents = BriaFibo2Pipeline._patchify_latents(vae.encode(img).latent_dist.mode().float())
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        scale = (1 / torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps)).view(1, -1, 1, 1)
        scale = scale.to(latents.device, latents.dtype)
        latents = (latents - mean) * scale
        latents = BriaFibo2Pipeline._unpatchify_latents(latents)
        out.append(latents.to(dtype))
    return out


def preprocess_images(proc, images, width, height, area):
    """Los recortes que ve el VAE y las copias que ve el text encoder."""
    vae_images, text_encoder_images = proc.preprocess(
        images,
        height=height,
        width=width,
        area=area,
        text_encoder_image_size=768,
    )
    return vae_images, text_encoder_images


def encode_prompt(pipe, prompt, device, images=None, fonts=None, clean=True):
    """
    (embeddings, máscara, embeddings sin los textos del caption).

    Fibo 2.0 lee seis haces de features del text encoder (el del perceptor y uno por bloque
    de inyección). Si el caption tiene textos que dibujar, el pipeline usa el caption sin
    ellos en los pasos de más ruido: por eso se cachean los dos.
    """
    caption, _ = pipe._parse_prompt(prompt)
    texts = caption_text_images(caption, fonts or [])

    # El pipeline oficial limpia el caption (quita valores vacíos y las notas de
    # puntuación) y lo serializa compacto ANTES de codificarlo. En edición no lo
    # limpia: el pipeline construye él el JSON de la edición con la instrucción.
    if clean and images is None and not isinstance(caption, str):
        clean_prompt = compact(caption or {})
    else:
        clean_prompt = prompt

    embeds, mask = pipe.encode_prompt(clean_prompt, images, device, text_images=texts)
    if mask is None:
        # (lote, tokens): los seis haces de features no son tokens, son canales.
        mask = torch.ones((embeds.shape[0], embeds.shape[2]), dtype=torch.bool, device=embeds.device)

    base_embeds = base_mask = None
    if texts and images is None:
        # Sólo en text-to-image: en edición los textos se ven en todos los pasos.
        try:
            marked = json.loads(clean_prompt)
        except (json.JSONDecodeError, TypeError):
            marked = None
        if isinstance(marked, dict):
            base_embeds, base_mask = pipe.encode_prompt(
                compact(pipe._mark_text_images(marked, "")), None, device)
            if base_mask is None:
                base_mask = torch.ones((base_embeds.shape[0], base_embeds.shape[2]),
                                       dtype=torch.bool, device=base_embeds.device)
    return embeds, mask, base_embeds, base_mask


def preprocess_fibo2():
    if not os.path.exists(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)

    # Pares nombre_before / nombre_after con el tipo en "normal": se entrenarían como imágenes sueltas,
    # sin la referencia y sin su caption (nombre.txt). Si solo hay pares, es un LoRA de edición.
    global LORA_TYPE, PREVIEW_EDIT_IMAGE
    pairs, loose = dataset_pairs()
    if LORA_TYPE != "edit" and pairs and not loose:
        print("⚠ " + t("The dataset only has name_before / name_after pairs: training an EDIT LoRA. Set LoRA Type to Edit to use the preview reference image."))
        LORA_TYPE = "edit"
        PREVIEW_EDIT_IMAGE = cfg.get("preview_edit_image", "").strip()
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
    # dataset y, al pasar de edición a normal, los antes (_ref).
    suffixes = ("latent", "embed", "mask") + (("ref",) if edit else ())
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in suffixes}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(local_path=MODEL_ID, text_encoder_dir=TEXT_ENCODER_DIR)

    # Los ajustes del proyecto en la carpeta de la cache: el trainer los lee para poner la
    # resolucion en los metadatos del LoRA.
    if PROJECT_NAME:
        with open(os.path.join(CACHE_DIR, f"pre_cache_settings_{PROJECT_NAME}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── TAMAÑOS ──────────────────────────────────────────────────────────────
    # En edición manda la imagen de antes (como en el pipeline); en text-to-image, la imagen.
    # El tamaño se fija una vez y lo usan las dos fases.
    sizes = {}
    for name, target, control in samples:
        source = control if control else target
        sizes[name] = image_size_capped(_aspect(source), TARGET_AREA, MAX_SIDE, MULTIPLE)

    preview_src = preview_edit_source(samples) if edit else None
    if preview_src:
        sizes["_custom"] = image_size_capped(_aspect(preview_src), TARGET_AREA, MAX_SIDE, MULTIPLE)

    # ── FASE 1: TEXT ENCODER (Qwen3-VL-4B) ──────────────────────────────────
    # El text encoder y el VAE nunca coinciden en VRAM.
    print("\n" + t("Loading {name}...", name=f"Text Encoder (Qwen3-VL-4B {TEXT_ENCODER})"))
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        vae=None,
        text_encoder=load_text_encoder(),
        dtype=torch.bfloat16,
    )
    pipe.tokenizer_max_length = MAX_SEQUENCE_LENGTH
    proc = pipe.image_processor
    fonts = load_fonts()

    with torch.inference_mode():
        # El caption vacío: es el negativo con el que se entrenó y el que sustituye al texto
        # en el text dropout del entrenamiento.
        embeds, mask, _, _ = encode_prompt(pipe, DEFAULT_NEGATIVE_PROMPT, "cuda", None, fonts,
                                           clean=False)
        save_tensor("_neg_embed.pt", embeds[0])
        save_tensor("_neg_mask.pt", mask[0])

        # El negativo de la guía: en text-to-image BRIA usa el caption largo; en edición, el vacío.
        negative = NEGATIVE_T2I if not edit else DEFAULT_NEGATIVE_PROMPT
        neg_embeds, neg_mask, _, _ = encode_prompt(pipe, negative, "cuda", None, fonts, clean=False)
        save_tensor("_negcfg_embed.pt", neg_embeds[0])
        save_tensor("_negcfg_mask.pt", neg_mask[0])

        # Prompt manual de las previews. En edición va con la imagen de antes de la preview;
        # sin prompt manual se usa la instrucción del primer par.
        custom_prompt = PREVIEW_CUSTOM_PROMPT or (read_caption(samples[0][0]) if edit else "")
        if custom_prompt:
            c_prompt = with_trigger(custom_prompt)
            print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt) + (f" + {os.path.basename(preview_src)}" if edit else ""))
            images = None
            if edit and preview_src:
                width, height = sizes["_custom"]
                _, images = preprocess_images(proc, [load_image(preview_src)], width, height, TARGET_AREA)
            embeds, mask, base_embeds, base_mask = encode_prompt(pipe, c_prompt, "cuda", images, fonts)
            save_prompt("_custom", embeds, mask, base_embeds, base_mask)
            # El trainer y el servidor comparan con estos textos para saber si la caché está al día.
            with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(c_prompt)
            with open(os.path.join(CACHE_DIR, "_custom_image.txt"), "w", encoding="utf-8") as f:
                f.write(PREVIEW_EDIT_IMAGE)
        # Restos de un prompt anterior, o de la preview de edición si el dataset ya no es de pares.
        stale = () if custom_prompt else ("_custom_embed.pt", "_custom_mask.pt", "_custom_prompt.txt", "_custom_image.txt")
        if not (custom_prompt and edit):
            stale += ("_custom_base_embed.pt", "_custom_base_mask.pt", "_custom_ref.pt")
        for f in stale:
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

        for idx, (name, target, control) in enumerate(samples, 1):
            width, height = sizes[name]
            images = None
            if control:
                _, images = preprocess_images(proc, [load_image(control)], width, height, TARGET_AREA)
            caption = read_caption(name)
            if not caption:
                print(f"  ⚠ {t('Sample {name} has no caption: it is trained with an empty one.', name=name)}")
            tokens = save_prompt(name, *encode_prompt(pipe, with_trigger(caption), "cuda", images, fonts))
            print(f"[{idx}/{len(samples)}] {t('Text')}: {name} | {tokens} tokens")

    del pipe
    free_vram()

    # ── FASE 2: VAE (AutoencoderKLFlux2, 32 canales) ───────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual o los captions
    # solo cuesta la fase de texto.
    jobs = []
    for name, target, control in samples:
        jobs.append((target, f"{name}_latent.pt", sizes[name], name))
        if control:
            jobs.append((control, f"{name}_ref.pt", sizes[name], name))
    if preview_src and custom_prompt and edit:
        jobs.append((preview_src, "_custom_ref.pt", sizes["_custom"], "_custom"))

    pending = []
    for src, out, (width, height), name in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (out != "_custom_ref.pt" and os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (height // 8, width // 8)):
            continue
        pending.append((src, out, (width, height), name))

    if len(pending) < len(jobs):
        print("\n" + t("{n} latents already cached, skipped.", n=len(jobs) - len(pending)))
    if not pending:
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (FLUX.2)"))
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, proc, pending, "cuda")
    del vae
    free_vram()
    print("\n✓ " + t("Pre-caching finished! VRAM freed."))


def encode_latents(vae, proc, jobs, device):
    with torch.inference_mode():
        for idx, (src, out, (width, height), name) in enumerate(jobs, 1):
            image = load_image(src)
            vae_images, _ = preprocess_images(proc, [image], width, height, TARGET_AREA)
            latent = prepare_latents(vae, vae_images, device, torch.float32)[0]
            save_tensor(out, latent.cpu().contiguous())
            print(f"[{idx}/{len(jobs)}] {t('Image')}: {os.path.basename(src)} -> {out} | {width}x{height}", flush=True)


def encode_preview_prompt(cache_dir, prompt, image_path, device):
    """
    Codifica solo el prompt manual de las previews (y en edición su imagen de antes). Lo lanza el
    servidor al guardar un prompt o una imagen nuevos. Sin entrenamiento en marcha va en GPU, con el
    text encoder elegido para el Pre-Cache; con el entrenamiento en marcha va en CPU (text encoder
    BF16 exacto) para no tocar su VRAM, y el trainer relee el embedding antes de la siguiente preview.
    """
    global CACHE_DIR
    CACHE_DIR = cache_dir
    where = device.upper().replace("CUDA", "GPU")
    image = None
    proc = proc_or_default()
    samples = find_samples()
    if any(f.endswith("_ref.pt") and not f.startswith("_") for f in os.listdir(cache_dir)):
        src = preview_edit_source(samples, image_path)
        width, height = image_size_capped(_aspect(src), TARGET_AREA, MAX_SIDE, MULTIPLE)
        vae_images, images = preprocess_images(proc, [load_image(src)], width, height, TARGET_AREA)
        image = images
        print("[Custom Prompt] Image: " + t("Encoding the reference image on {device}: {path}", device=where, path=src), flush=True)
        # Mismo dtype que las referencias del dataset: en float32 el latente_preview sería
        # ligeramente distinto del que se entrenó.
        vae = AutoencoderKLFlux2.from_pretrained(
            MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to(device)
        latents = prepare_latents(vae, vae_images, device, torch.float32)
        save_tensor("_custom_ref.pt", latents[0].cpu().contiguous())
        del vae
        free_vram()

    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    if device == "cuda":
        te = load_text_encoder()
    else:
        from transformers import Qwen3VLForConditionalGeneration
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(MODEL_ID, "text_encoder_BF16"), dtype=torch.bfloat16, device_map="cpu")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, text_encoder=te, dtype=torch.bfloat16)
    pipe.tokenizer_max_length = MAX_SEQUENCE_LENGTH
    with torch.inference_mode():
        embeds, mask, base_embeds, base_mask = encode_prompt(pipe, prompt, device, image, load_fonts())
    del pipe, te
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_mask.pt", lambda f: torch.save(mask[0].bool().cpu().contiguous(), f))
    if base_embeds is not None:
        replace("_custom_base_mask.pt", lambda f: torch.save(base_mask[0].bool().cpu().contiguous(), f))
    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_image.txt", lambda f: open(f, "w", encoding="utf-8").write(image_path))
    replace("_custom_embed.pt", lambda f: torch.save(embeds[0].to(torch.bfloat16).cpu().contiguous(), f))
    if base_embeds is not None:
        replace("_custom_base_embed.pt", lambda f: torch.save(base_embeds[0].to(torch.bfloat16).cpu().contiguous(), f))
    else:
        # El prompt anterior sí tenía textos: sus embeddings sin ellos ya no valen.
        for stale in ("_custom_base_embed.pt", "_custom_base_mask.pt"):
            if os.path.exists(os.path.join(cache_dir, stale)):
                os.remove(os.path.join(cache_dir, stale))
    print("[Custom Prompt] " + t("Ready."), flush=True)


def _aspect(path):
    """La relación de aspecto de la imagen YA enderezada por el EXIF, como la ve el modelo."""
    from PIL import ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.width / im.height


def proc_or_default():
    return BriaFibo2ImageProcessor(vae_scale_factor=16)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <imagen de antes o ""> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_fibo2()
