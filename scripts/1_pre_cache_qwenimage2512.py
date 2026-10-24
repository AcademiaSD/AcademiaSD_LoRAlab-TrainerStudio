# -*- coding: utf-8 -*-
"""
1_pre_cache_qwenimage2512.py — Pre-caché de latentes + embeddings para Qwen-Image-2512 / Qwen-Image-Edit-2511
Pre-cache of latents + embeddings for Qwen-Image-2512 / Qwen-Image-Edit-2511

Lee configuración desde settings/pre_cache_settings_qwenimage2512.json si existe.
Reads configuration from settings/pre_cache_settings_qwenimage2512.json if present.

El texto se codifica igual que QwenImagePipeline (imagen) y QwenImageEditPlusPipeline (edición):
plantilla de chat, última capa oculta de Qwen2.5-VL-7B y sin los tokens de la plantilla.
"""
import os
import gc
import math
import json
import logging
import sys
import warnings

import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import AutoencoderKLQwenImage
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from i18n import t
import gpu_stats
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import qwen_image2512_model as qm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# LLM.int8 avisa en cada matmul de que pasa las entradas de bf16 a fp16.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

HF_REPO_ID = "AcademiaSD/Qwen-Image-2512-NF4-for-LoRA-Training"

# Modo del text encoder -> (carpeta text_encoder_<precisión>, repartir entre GPU y RAM).
# VRAM: BF16 ~16.6 GB, INT8 ~9 GB, NF4 ~5.5 GB. BF16_offload es exacto y cabe en cualquier GPU:
# lo que no entra en VRAM se ejecuta desde RAM.
TEXT_ENCODER_MODES = {
    "BF16_offload": ("BF16", True),
    "BF16":         ("BF16", False),
    "INT8":         ("INT8", False),
    "NF4":          ("NF4",  False),
}
# VRAM que se deja libre al repartir el text encoder entre GPU y RAM (activaciones, contexto CUDA).
# Negativo del CFG de las previews de edición (el de los ejemplos de QwenImageEditPlusPipeline).
EDIT_NEGATIVE = " "
OFFLOAD_RESERVE_BYTES = int(1.5 * 1024**3)

DEFAULTS = {
    "model_id": "Models/Qwen-Image-2512-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_qwenimage2512",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 32,
    "text_encoder": "BF16_offload",
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "preview_edit_image": "",
    "preview_edit_images": [],
    "lora_type": "normal",
}

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


CONFIG_PATH = "settings/pre_cache_settings_qwenimage2512.json"

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
# Solo edición: imágenes de referencia (fuera del dataset) para ver en las previews cómo aplica el efecto.
PREVIEW_EDIT_IMAGES = config_preview_images(cfg) if LORA_TYPE == "edit" else []

CACHE_DIR = str(resolve_cache_dir(f"./cached_data_qwenimage2512_{PROJECT_NAME}" if PROJECT_NAME else cfg.get("cache_dir"), DEFAULTS["cache_dir"]))

# VAE 8x + parches 2x2 en el transformer: lados múltiplos de 16 px como mínimo.
if MULTIPLE not in (16, 32, 64):
    print("⚠ " + t("Invalid Multiple {n}. Using {d}.", n=MULTIPLE, d=32))
    MULTIPLE = 32

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


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_hf_token():
    if os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token = json.load(f).get("token", "").strip()
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


def bucket_size(w: int, h: int, area=None):
    area = TARGET_AREA if area is None else area
    ar = w / h
    bh = math.sqrt(area / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        s = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * s) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * s) // MULTIPLE * MULTIPLE)
    return bw, bh


def ensure_model_downloaded(edit):
    needed = [TEXT_ENCODER_DIR, "vae", "tokenizer"] + (["processor"] if edit else [])
    if all(os.path.exists(os.path.join(MODEL_ID, d)) for d in needed):
        print("✓ " + t("Local model found at: {path}", path=f"{MODEL_ID} ({TEXT_ENCODER_DIR})"))
        return

    print("⚠ " + t("Local model not found at: {path}", path=f"{MODEL_ID} ({TEXT_ENCODER_DIR})"))
    print("  " + t("Downloading from Hugging Face: {repo}", repo=HF_REPO_ID))
    enable_hf_file_progress()
    from huggingface_hub import snapshot_download
    # Solo la variante del text encoder elegida y nada de los transformers (eso lo baja el trainer).
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MODEL_ID,
        token=get_hf_token(),
        max_workers=2,
        allow_patterns=[f"{d}/*" for d in needed] + ["model_index.json", "LICENSE", "README.md"],
    )
    print("✓ " + t("Model downloaded to: {path}", path=MODEL_ID))


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_encoder(folder, offload, device):
    path = os.path.join(MODEL_ID, folder)
    if device == "cpu":
        return AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu").eval()
    if not offload:
        return AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

    # Las capas que no caben en la VRAM libre se quedan en RAM y se ejecutan desde allí.
    free_bytes, _ = gpu_stats.mem_get_info()
    gpu_budget = max(free_bytes - OFFLOAD_RESERVE_BYTES, 0)
    print(f"  {t('GPU budget'):<22}: {gpu_budget / 1024**3:.1f} GB ({t('rest in RAM')})")
    return AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto",
                                     max_memory={0: gpu_budget, "cpu": "512GiB"}).eval()


class TextEncoder:
    """Qwen2.5-VL-7B con el tokenizer (imagen) y el processor (edición, ve la imagen de antes)."""

    def __init__(self, folder, offload, device, edit):
        self.device = device
        self.model = load_text_encoder(folder, offload, device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_ID, "tokenizer"))
        self.processor = AutoProcessor.from_pretrained(os.path.join(MODEL_ID, "processor")) if edit else None

    def encode(self, prompt, images=None):
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            drop = qm.DROP_T2I
            inputs = self.tokenizer([qm.PROMPT_T2I.format(prompt)], max_length=qm.MAX_TEXT_TOKENS + drop,
                                    truncation=True, return_tensors="pt")
        else:
            # Las referencias a ~384² para el text encoder, como QwenImageEditPlusPipeline.
            prepared = []
            for image in images:
                cw, ch = qm.calculate_dimensions(qm.CONDITION_IMAGE_AREA, image.width / image.height)
                prepared.append(image.convert("RGB").resize((cw, ch), Image.LANCZOS))
            drop = qm.DROP_EDIT
            inputs = self.processor(text=[qm.PROMPT_EDIT.format(qm.edit_image_prompt(len(prepared)) + prompt)],
                                    images=prepared,
                                    padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        # Todas las salidas del processor, también mm_token_type_ids: sin ella transformers 5 no calcula
        # la MRoPE 3D de la imagen y cae a posiciones 1D (es lo que hace hoy QwenImageEditPlusPipeline, que
        # solo pasa cuatro claves; el resultado cambia un 62%). ComfyUI sí usa las posiciones 3D.
        hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1][0]
        embeds = hidden[inputs["attention_mask"][0].bool()][drop:].unsqueeze(0).to(torch.bfloat16)
        return embeds, torch.ones(embeds.shape[:2], dtype=torch.bool)


def encode_and_save(te, prompt, name, images=None):
    embeds, mask = te.encode(prompt, images)
    torch.save(embeds.cpu(), os.path.join(CACHE_DIR, f"{name}_embed.pt"))
    torch.save(mask, os.path.join(CACHE_DIR, f"{name}_mask.pt"))
    return embeds.shape[1]


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


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def preview_edit_sources(samples, image_paths=None):
    paths = PREVIEW_EDIT_IMAGES if image_paths is None else as_image_list(image_paths)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("⚠ " + t("Preview reference image not found (it is skipped; later <imageN> numbers shift): {path}", path=", ".join(missing)))
    paths = [p for p in paths if os.path.exists(p)][:PREVIEW_MAX]
    if paths:
        return paths
    return [samples[0][2]] if samples and samples[0][2] else []


def preview_image_area(count):
    return max(MULTIPLE * MULTIPLE, int(TARGET_AREA * 2 / max(1, count + 1)))


def custom_ctrl_name(index):
    return "_custom_ctrl.pt" if index == 0 else f"_custom_ctrl_{index + 1}.pt"


def custom_ctrl_names(count):
    return {custom_ctrl_name(i) for i in range(count)}


def remove_stale_custom_ctrls(expected):
    for f in os.listdir(CACHE_DIR):
        if f.startswith("_custom_ctrl") and f.endswith(".pt") and f not in expected:
            os.remove(os.path.join(CACHE_DIR, f))


def encode_latents(vae, jobs, device):
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(vae.config.latents_std,  device=device, dtype=torch.float32).view(1, z_dim, 1, 1, 1)

    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            img = fit(Image.open(src).convert("RGB"), bw, bh)
            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0).unsqueeze(2)
            img_tensor = ((img_tensor.float() / 127.5) - 1.0).to(device, dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.mode().float()
            latent = ((z - latents_mean) / latents_std)[:, :, 0].to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] {t('Image')}: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_qwenimage2512():
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
    suffixes = ("latent", "embed", "mask") + (("ctrl",) if edit else ())
    expected = {f"{name}_{suffix}.pt" for name, _, _ in samples for suffix in suffixes}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    ensure_model_downloaded(edit)

    # Bucket de cada muestra: lo fija el después (o la imagen); el antes se ajusta al mismo tamaño.
    buckets = {}
    for name, target, _ in samples:
        with Image.open(target) as im:
            buckets[name] = bucket_size(*im.size)
    preview_sources = preview_edit_sources(samples) if edit else []
    preview_buckets = []
    preview_area = preview_image_area(len(preview_sources))
    for src in preview_sources:
        with Image.open(src) as im:
            preview_buckets.append(bucket_size(*im.size, area=preview_area))
    if preview_buckets:
        buckets["_custom"] = preview_buckets[0]

    # ── FASE 1: TEXT ENCODER (Qwen2.5-VL-7B) ────────────────────────────────
    # El text encoder y el VAE nunca coinciden en VRAM.
    print("\n" + t("Loading {name}...", name=f"Text Encoder (Qwen2.5-VL-7B {TEXT_ENCODER})"))
    te = TextEncoder(TEXT_ENCODER_DIR, TEXT_ENCODER_MODES[TEXT_ENCODER][1], "cuda", edit)

    with torch.inference_mode():
        encode_and_save(te, "", "_neg")

        # Prompt manual de las previews. En edición va con la imagen de antes de la preview;
        # sin prompt manual se usa la instrucción del primer par.
        custom_prompt = PREVIEW_CUSTOM_PROMPT or (read_caption(samples[0][0]) if edit else "")
        if custom_prompt:
            c_prompt = with_trigger(custom_prompt)
            print("[Custom Prompt Cache] " + t("Encoding: '{prompt}'", prompt=c_prompt) + (f" + {', '.join(os.path.basename(p) for p in preview_sources)}" if edit else ""))
            preview_images = []
            for src, bucket in zip(preview_sources, preview_buckets):
                with Image.open(src) as im:
                    preview_images.append(fit(im.convert("RGB"), *bucket))
            encode_and_save(te, c_prompt, "_custom", images=preview_images)
            if edit:
                # El negativo del CFG de edición también ve las referencias, como en QwenImageEditPlusPipeline.
                encode_and_save(te, EDIT_NEGATIVE, "_custom_neg", images=preview_images)
            # El trainer y el servidor comparan con estos textos para saber si la caché está al día.
            with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(c_prompt)
            with open(os.path.join(CACHE_DIR, "_custom_image.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(PREVIEW_EDIT_IMAGES))
        # Restos de un prompt anterior, o de la preview de edición si el dataset ya no es de pares.
        stale = () if custom_prompt else ("_custom_embed.pt", "_custom_mask.pt", "_custom_prompt.txt", "_custom_image.txt")
        if not (custom_prompt and edit):
            stale += ("_custom_ctrl.pt", "_custom_neg_embed.pt", "_custom_neg_mask.pt")
        for f in stale:
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))
        expected_custom_ctrls = custom_ctrl_names(len(preview_sources)) if custom_prompt and edit else set()

        for idx, (name, target, control) in enumerate(samples, 1):
            image = fit(Image.open(control).convert("RGB"), *buckets[name]) if control else None
            tokens = encode_and_save(te, with_trigger(read_caption(name)), name, images=[image] if image else None)
            print(f"[{idx}/{len(samples)}] {t('Text')}: {name} | {tokens} tokens")

    del te
    free_vram()

    # ── FASE 2: VAE (Qwen-Image, 16 canales) ─────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual o los captions
    # solo cuesta la fase de texto. En edición, el antes se guarda como <nombre>_ctrl.pt.
    # La imagen de la preview de edición se codifica siempre: puede ser otro fichero más antiguo.
    jobs = [(target, f"{name}_latent.pt", buckets[name]) for name, target, _ in samples]
    jobs += [(control, f"{name}_ctrl.pt", buckets[name]) for name, _, control in samples if control]
    if custom_prompt and edit:
        jobs += [(src, custom_ctrl_name(i), bucket)
                 for i, (src, bucket) in enumerate(zip(preview_sources, preview_buckets))]

    pending = []
    for src, out, (bw, bh) in jobs:
        lat_path = os.path.join(CACHE_DIR, out)
        if (not out.startswith("_custom_ctrl") and os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 8, bw // 8)):
            continue
        pending.append((src, out, (bw, bh)))

    if len(pending) < len(jobs):
        print("\n" + t("{n} latents already cached, skipped.", n=len(jobs) - len(pending)))
    if not pending:
        remove_stale_custom_ctrls(expected_custom_ctrls)
        print("\n✓ " + t("Pre-caching finished!"))
        return

    print("\n" + t("Loading {name}...", name="VAE (Qwen-Image)"))
    vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16).to("cuda")
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
    global CACHE_DIR
    CACHE_DIR = cache_dir
    where = device.upper().replace("CUDA", "GPU")
    requested = parse_preview_images_arg(images_arg)
    edit = any(f.endswith("_ctrl.pt") and not f.startswith("_") for f in os.listdir(cache_dir))
    sources = preview_edit_sources(find_samples(), requested) if edit else []
    buckets = []
    area = preview_image_area(len(sources))
    for src in sources:
        with Image.open(src) as im:
            buckets.append(bucket_size(*im.size, area=area))
    images = []
    for src, bucket in zip(sources, buckets):
        with Image.open(src) as im:
            images.append(fit(im.convert("RGB"), *bucket))

    if sources:
        # El servidor lee las líneas "[Custom Prompt] Image/Prompt" para mostrar la fase en la GUI.
        print("[Custom Prompt] Image: " + t("Encoding the reference image on {device}: {path}", device=where, path=", ".join(os.path.basename(p) for p in sources)), flush=True)
        vae = AutoencoderKLQwenImage.from_pretrained(
            MODEL_ID, subfolder="vae", dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        encode_latents(vae, [(src, custom_ctrl_name(i) + ".new", bucket)
                             for i, (src, bucket) in enumerate(zip(sources, buckets))], device)
        del vae
        free_vram()

    print("[Custom Prompt] Prompt: " + t("Encoding the prompt on {device}: '{prompt}'", device=where, prompt=prompt), flush=True)
    if device == "cuda":
        te = TextEncoder(TEXT_ENCODER_DIR, TEXT_ENCODER_MODES[TEXT_ENCODER][1], "cuda", edit)
    else:
        te = TextEncoder("text_encoder_BF16", False, "cpu", edit)
    with torch.inference_mode():
        embeds, mask = te.encode(prompt, images)
        neg = te.encode(EDIT_NEGATIVE, images) if images else None
    del te
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_mask.pt", lambda f: torch.save(mask, f))
    if images:
        for i in range(len(images)):
            name = custom_ctrl_name(i)
            os.replace(os.path.join(cache_dir, name + ".new"), os.path.join(cache_dir, name))
        replace("_custom_neg_mask.pt", lambda f: torch.save(neg[1], f))
        replace("_custom_neg_embed.pt", lambda f: torch.save(neg[0].cpu(), f))
    else:
        for name in ("_custom_neg_mask.pt", "_custom_neg_embed.pt"):
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                os.remove(path)
    remove_stale_custom_ctrls(custom_ctrl_names(len(images)))
    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_image.txt", lambda f: open(f, "w", encoding="utf-8").write("\n".join(requested)))
    replace("_custom_embed.pt", lambda f: torch.save(embeds.cpu(), f))
    print("[Custom Prompt] " + t("Ready."), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <rutas JSON o una ruta> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_qwenimage2512()
