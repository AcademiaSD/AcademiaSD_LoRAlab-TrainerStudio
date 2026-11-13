# -*- coding: utf-8 -*-
"""
0_forge_dataset.py — Dataset Forge: analiza las referencias o genera el dataset (ver dataset_forge.py).
Dataset Forge: analyzes the references or generates the dataset.

    python 0_forge_dataset.py analyze    captions que faltan + ficha del personaje (Qwen3-VL-8B NF4)
    python 0_forge_dataset.py generate   imágenes nuevas con el modelo elegido (4 pasos con su LoRA turbo):
                                         Qwen-Image 2.1 NF4 + Viggle, o Qwen-Image-Edit-2511 NF4 + Lightning

Lee settings/dataset_forge_settings.json (lo escribe la página) y va dejando el progreso en
settings/dataset_forge_status.json. VRAM: el text encoder (BF16) se reparte entre GPU y RAM; el
transformer de 2.1 (7B) cabe en la GPU y el de 2511 (20B) usa BlockSwap, como sus entrenadores.
"""
import gc
import json
import math
import os
import re
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from i18n import t
from text_io import read_text_file
from project_layout import resolve_model_dir
import dataset_forge as forge
import gpu_stats
import qwen_image2512_model as qm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

CAPTIONER_DIR = str(resolve_model_dir("Models/Captioner-Qwen3-VL-8B", "Models/Captioner-Qwen3-VL-8B"))
CAPTIONER_REPO = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"
MODEL_ID = str(resolve_model_dir("Models/Qwen-Image-2512-NF4", "Models/Qwen-Image-2512-NF4"))
MODEL_REPO = "AcademiaSD/Qwen-Image-2512-NF4-for-LoRA-Training"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
OFFLOAD_RESERVE_BYTES = int(1.5 * 1024**3)

CAPTION_PROMPT = (
    "Describe this image for training a text-to-image model, in at most 150 words. "
    "One single paragraph of plain prose, no lists and no preamble. Cover the main subject and what they "
    "are doing, their appearance, hair and clothing, the pose, the framing and camera angle, the setting and "
    "background, the lighting and the colors. Do not invent a name and do not mention that this is an image or a photo."
)
IDENTITY_PROMPT = (
    "These photos show the same adult person. Reply with JSON only, no other text: "
    '{"subject": "woman" if the person is female, "man" if male, "person" only if you cannot tell, '
    '"age": "about N years old", '
    '"hair": "<length> <texture> <color> hair", '
    '"eyes": "<color> eyes", '
    '"marks": "visible facial hair, glasses, freckles, moles or tattoos, or an empty string"}. '
    "Never say what is absent, and do not mention clothing, pose, background or lighting."
)

cfg = {**forge.DEFAULTS, **forge.read_json(forge.SETTINGS_FILE)}
SRC = cfg["source_dir"]
REFS = [os.path.join(SRC, r) for r in cfg["references"]]


def status(**kw):
    st = forge.read_json(forge.STATUS_FILE)
    st.update(kw)
    forge.write_json(forge.STATUS_FILE, st)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def hf_token():
    return forge.read_json("settings/HF_token.json").get("token", "").strip() or None


def open_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def read_caption(path):
    txt = os.path.splitext(path)[0] + ".txt"
    try:
        return read_text_file(txt).strip()
    except OSError:
        return ""


# =============================================================================
# ANALIZAR / ANALYZE
# =============================================================================

def load_captioner():
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    path = os.path.join(CAPTIONER_DIR, "text_encoder_NF4")
    if not os.path.exists(os.path.join(path, "config.json")):
        print(t("Downloading the captioner from Hugging Face: {repo} (~5 GB)", repo=CAPTIONER_REPO), flush=True)
        snapshot_download(repo_id=CAPTIONER_REPO, local_dir=CAPTIONER_DIR, token=hf_token(), max_workers=2,
                          allow_patterns=["text_encoder_NF4/*", "processor/*"])
    print(t("Loading {name}...", name="Qwen3-VL-8B (NF4)"), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, AutoProcessor.from_pretrained(os.path.join(CAPTIONER_DIR, "processor"))


def ask(model, processor, images, prompt, max_tokens=300):
    images = [im if max(im.size) <= 768 else im.resize((round(im.width * 768 / max(im.size)), round(im.height * 768 / max(im.size))), Image.LANCZOS)
              for im in images]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.3)
    return " ".join(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).split())


def parse_identity(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    subject = str(data.get("subject", "person")).strip().lower()
    # Solo adultos: un captioner que diga girl / boy de un adulto joven se queda en woman / man.
    subject = {"girl": "woman", "boy": "man"}.get(subject, subject)
    parts = []
    for key in ("age", "hair", "eyes", "marks", "identity"):
        value = data.get(key, "")
        value = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        value = value.strip().rstrip(".")
        if value and "<" not in value and not re.match(r"^(none|no |n/a)", value, re.I):
            parts.append(value)
    identity = ", ".join(parts)
    if subject not in ("woman", "man"):
        # Si el sujeto no queda claro, a veces lo dice la ficha («young woman with ...»).
        found = re.search(r"\b(woman|man|girl|boy|female|male)\b", identity, re.I)
        subject = {"girl": "woman", "female": "woman", "boy": "man", "male": "man"}.get(
            found.group(1).lower(), found.group(1).lower()) if found else "person"
    # Sin la palabra del sujeto, que ya va en la instrucción: «young woman with blonde hair» -> «young, blonde hair».
    identity = re.sub(r"\b(?:an? )?(young |adult |middle-aged )?(woman|man|girl|boy|person)( with)?\b", lambda m: m.group(1).strip() + "," if m.group(1) else "", identity, flags=re.I)
    identity = re.sub(r"\s*,\s*,|\s{2,}", ", ", identity).strip(" ,")
    return subject, identity


def analyze():
    status(phase="analyze", message=t("Loading the captioner..."))
    model, processor = load_captioner()
    images = [open_image(p) for p in REFS]
    captions = {}
    for i, (path, im) in enumerate(zip(REFS, images), 1):
        caption = read_caption(path)
        if not caption:
            caption = ask(model, processor, [im], CAPTION_PROMPT)
            print(f"[{i}/{len(REFS)}] {os.path.basename(path)}: {caption}", flush=True)
        captions[os.path.basename(path)] = caption
    print(t("Describing the character..."), flush=True)
    subject, identity = parse_identity(ask(model, processor, images[:4], IDENTITY_PROMPT, 160))
    print(f"  {subject}: {identity}", flush=True)
    status(phase="done", result={"subject": subject, "identity": identity, "captions": captions},
           message=t("Analysis ready."))


# =============================================================================
# GENERAR / GENERATE
# =============================================================================

def fit(img, bw, bh):
    ratio = max(bw / img.width, bh / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
    left, top = (img.width - bw) // 2, (img.height - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def download(repo, local, patterns, label):
    from huggingface_hub import snapshot_download
    print(t("Downloading from Hugging Face: {repo}", repo=repo), flush=True)
    status(message=t("Downloading {name} (first time only)...", name=label))
    snapshot_download(repo_id=repo, local_dir=local, token=hf_token(), max_workers=2, allow_patterns=patterns)


def turbo_hooks(transformer, layers, label):
    """Suma un LoRA destilado (pocos pasos) a la salida de cada capa: layers = {capa: (down, up, escala)}."""
    def make_hook(a, b, scale):
        def hook(module, args, output):
            return output + F.linear(F.linear(args[0].to(a.dtype), a), b).to(output.dtype) * scale
        return hook
    hooks = [transformer.get_submodule(name).register_forward_hook(make_hook(a.to("cuda"), b.to("cuda"), s))
             for name, (a, b, s) in layers.items()]
    print(f"  🚀 {label}: " + t("{n} layers (strength {s})", n=len(hooks), s=1.0), flush=True)
    return hooks


def vae_latent(vae, img):
    """Imagen PIL ya ajustada a su tamaño -> latente normalizado [1, C, h, w]."""
    z = vae.config.z_dim
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None, :, None].float() / 127.5 - 1
    lat = vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mode().float()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, z, 1, 1, 1)
    return ((lat - mean) / std)[:, :, 0].to(torch.bfloat16)


def vae_image(vae, lat):
    """Latente [1, C, h, w] -> imagen PIL RGB (el VAE de 2.1 devuelve RGBA: se queda el color)."""
    lat = lat.to(vae.dtype).unsqueeze(2)
    mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda", dtype=lat.dtype).view(1, -1, 1, 1, 1)
    try:
        img = vae.decode(lat * std + mean, return_dict=False)[0][:, :3, 0]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        vae.enable_tiling()
        img = vae.decode(lat * std + mean, return_dict=False)[0][:, :3, 0]
    img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(img)


def te_budget():
    free_b, _ = gpu_stats.mem_get_info()
    return {0: max(free_b - OFFLOAD_RESERVE_BYTES, 0), "cpu": "512GiB"}


class Qwen21:
    """
    Qwen-Image 2.1 (Qwen Research License, no comercial): DiT de 7B en NF4, entero en la GPU. Se usa sin
    CFG, como el pipeline oficial. Calidad «best»: 20 pasos del modelo base (mejor anatomía y parecido:
    68% de media frente a 64% en la prueba); «fast»: LoRA turbo de Viggle, 4 pasos. Las referencias entran
    como <image1>, <image2>... al text encoder (Qwen3-VL-8B) y como latentes LIMPIOS delante del objetivo.
    """
    LABEL = "Qwen-Image 2.1"
    DIR = str(resolve_model_dir("Models/Qwen-Image21-NF4", "Models/Qwen-Image21-NF4"))
    REPO = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"
    TURBO = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
    TAGS = True   # la instrucción nombra las referencias como <image1>, <image2>...

    def __init__(self, quality="best"):
        self.turbo = quality == "fast"
        self.steps = 4 if self.turbo else 20
        self.hooks = []

    def ensure(self):
        needed = ["transformer", "text_encoder_BF16", "vae", "processor", "scheduler"]
        if not (all(os.path.exists(os.path.join(self.DIR, d)) for d in needed)
                and os.path.exists(os.path.join(self.DIR, "LoRAs", self.TURBO))):
            download(self.REPO, self.DIR, [f"{d}/*" for d in needed] + ["model_index.json", "LoRAs/" + self.TURBO], self.LABEL)

    def ref_area(self, area, n):
        # Como el pipeline oficial, a la resolución de salida; con 3-4 referencias, menos (VRAM).
        return area if n <= 2 else int(area * 2 / (n + 1))

    def encode(self, texts, refs):
        from diffusers import DiffusionPipeline
        from transformers import Qwen3VLForConditionalGeneration
        print(t("Loading {name}...", name="Qwen3-VL-8B (text encoder)"), flush=True)
        te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(self.DIR, "text_encoder_BF16"), dtype=torch.bfloat16,
                                                             device_map="auto", max_memory=te_budget())
        pipe = DiffusionPipeline.from_pretrained(self.DIR, transformer=None, vae=None, text_encoder=te, dtype=torch.bfloat16)
        rgba = [im.convert("RGBA") for im in refs]
        out = []
        with torch.inference_mode():
            for i, text in enumerate(texts, 1):
                emb, mask, imgmask = pipe.encode_prompt(prompt=text, image=rgba, device="cuda")
                if mask is None:
                    mask = torch.ones(emb.shape[:2], dtype=torch.bool)
                out.append((emb.cpu(), mask.bool().cpu(), imgmask.bool().cpu()))
                print(f"  {t('Encoding prompts')} {i}/{len(texts)}", flush=True)
        del pipe, te
        free_vram()
        return out

    def load(self, refs):
        from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
        from safetensors import safe_open
        self.vae = AutoencoderKLQwenImage21.from_pretrained(self.DIR, subfolder="vae", dtype=torch.bfloat16).to("cuda")
        with torch.inference_mode():
            self.ctrls = [vae_latent(self.vae, im.convert("RGBA")) for im in refs]
        self.vae.to("cpu")
        free_vram()
        print(t("Loading {name}...", name=self.LABEL + " (NF4)"), flush=True)
        self.transformer = load_qwen21_transformer(self.DIR).to("cuda").eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.DIR, subfolder="scheduler")
        print(f"  {self.steps} " + t("steps") + (" (turbo)" if self.turbo else ""), flush=True)
        if not self.turbo:
            return
        with safe_open(os.path.join(self.DIR, "LoRAs", self.TURBO), framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("lora_adapter_metadata", "{}"))
            scale = meta.get("transformer.lora_alpha", 1) / meta.get("transformer.r", 1)
            layers = {k[len("transformer."):-len(".lora_A.weight")]: (
                f.get_tensor(k).to(torch.bfloat16), f.get_tensor(k.replace(".lora_A.", ".lora_B.")).to(torch.bfloat16), scale)
                for k in f.keys() if k.endswith(".lora_A.weight")}
        self.hooks = turbo_hooks(self.transformer, layers, "Turbo LoRA (Viggle)")
        # El turbo se entrenó sin shift_terminal (con el 0.02 de la base el último paso sale mal).
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config, shift_terminal=None)

    def sample(self, cond, w, h, seed):
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        emb, mask, imgmask = (x.to("cuda") for x in cond)
        H, W = h // 16, w // 16   # VAE 16x y sin parches: cada latente es un token
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 64, H, W), generator=g, device="cuda", dtype=torch.bfloat16).view(1, 64, H * W).transpose(1, 2)
        sc = self.scheduler.config
        mu = qm.calculate_shift(H * W, sc.base_image_seq_len, sc.max_image_seq_len, sc.base_shift, sc.max_shift)
        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / self.steps, self.steps), device="cuda", mu=mu)
        self.scheduler.set_begin_index(0)
        # Huecos de las referencias en el texto + uno por cada grupo 2x2 del objetivo.
        text = torch.zeros(emb.shape[1], dtype=torch.bool, device="cuda")
        text[:imgmask.shape[1]] = imgmask[0]
        img_mask = torch.cat([text, torch.ones(H * W // 4, dtype=torch.bool, device="cuda")]).unsqueeze(0)
        refs = [c.view(1, c.shape[1], -1).transpose(1, 2) for c in self.ctrls]
        shapes = [[(1, c.shape[2], c.shape[3]) for c in self.ctrls] + [(1, H, W)]]
        cache = QwenImage21KVCache(len(self.transformer.transformer_blocks))
        for i, t_ in enumerate(self.scheduler.timesteps):
            out = self.transformer(hidden_states=torch.cat(refs + [latents], dim=1), encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=None if mask.all() else mask,
                                   timestep=t_.expand(1).to(torch.bfloat16) / 1000, img_shapes=shapes, img_mask=img_mask,
                                   kv_cache=cache, kv_cache_mode="extract" if i == 0 else "cached", return_dict=False)[0]
            latents = self.scheduler.step(out[:, -H * W:], t_, latents, return_dict=False)[0]
        self.vae.to("cuda")
        img = vae_image(self.vae, latents.transpose(1, 2).reshape(1, 64, H, W))
        self.vae.to("cpu")
        return img


def load_qwen21_transformer(model_dir):
    """El transformer de 2.1 desde su caché NF4 (como 2_train_lora_qwen_image21.py)."""
    from diffusers import QwenImage21Transformer2DModel
    return load_nf4_cache(model_dir, QwenImage21Transformer2DModel)


def load_nf4_cache(model_dir, cls):
    """
    Caché NF4 de los entrenadores de Qwen-Image 2.1, FLUX.2/3 Klein y Fibo 2.0 (mismo formato): transformer/
    index.json + weights/ (una capa por fichero, NF4 o BF16) + others.safetensors (normas, embedders).
    """
    from accelerate import init_empty_weights
    from bitsandbytes.functional import QuantState
    from bitsandbytes.nn import Linear4bit, Params4bit
    from safetensors import safe_open
    from safetensors.torch import load_file
    cache_dir = os.path.join(model_dir, "transformer")
    with open(os.path.join(cache_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    with init_empty_weights():
        tr = cls.from_config(cls.load_config(cache_dir))

    def put(name, module):
        parent, _, child = name.rpartition(".")
        setattr(tr.get_submodule(parent) if parent else tr, child, module)

    weights = os.path.join(cache_dir, "weights")
    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights, info["file"]), framework="pt", device="cpu") as f:
            data = f.get_tensor("weight")
            qs = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}
            bias = f.get_tensor("bias") if "bias" in f.keys() else None
        weight = Params4bit(data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs, device="cpu")
        weight.bnb_quantized = True
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info.get("bias", False),
                               quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        if bias is not None:   # Fibo 2.0 guarda capas con bias
            layer.bias = torch.nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)
        put(name, layer)
    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=info.get("bias", False), dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights, info["file"])), assign=True)
        put(name, layer)
    tr.load_state_dict(load_file(os.path.join(cache_dir, index["others"])), strict=False, assign=True)
    tr.requires_grad_(False)
    return tr


class Edit2511:
    """
    Qwen-Image-Edit-2511 (Apache 2.0) NF4 + Lightning (4 pasos), con BlockSwap como el entrenador de
    Qwen-Image 2512. Referencias al text encoder (Qwen2.5-VL-7B) a ~384² y como latentes LIMPIOS
    detrás del objetivo (QwenImageEditPlusPipeline).
    """
    LABEL = "Qwen-Image-Edit-2511"
    DIR = MODEL_ID
    TAGS = False
    steps = 4   # Lightning

    def ensure(self):
        needed = ["transformer_edit", "text_encoder_BF16", "vae", "tokenizer", "processor", "scheduler"]
        if not (all(os.path.exists(os.path.join(self.DIR, d)) for d in needed)
                and os.path.exists(os.path.join(self.DIR, "LoRAs", LIGHTNING))):
            download(MODEL_REPO, self.DIR, [f"{d}/*" for d in needed] + ["model_index.json", "LoRAs/" + LIGHTNING], self.LABEL)

    def ref_area(self, area, n):
        return max(512 * 512, int(min(area, qm.VAE_IMAGE_AREA) * 2 / (n + 1)))

    def encode(self, texts, refs):
        from transformers import AutoModel, AutoProcessor
        print(t("Loading {name}...", name="Qwen2.5-VL-7B (text encoder)"), flush=True)
        te = AutoModel.from_pretrained(os.path.join(self.DIR, "text_encoder_BF16"), dtype=torch.bfloat16,
                                       device_map="auto", max_memory=te_budget()).eval()
        processor = AutoProcessor.from_pretrained(os.path.join(self.DIR, "processor"))
        small = []
        for im in refs:
            cw, ch = qm.calculate_dimensions(qm.CONDITION_IMAGE_AREA, im.width / im.height)
            small.append(im.resize((cw, ch), Image.LANCZOS))
        out = []
        with torch.inference_mode():
            for i, text in enumerate(texts, 1):
                inputs = processor(text=[qm.PROMPT_EDIT.format(qm.edit_image_prompt(len(small)) + text)], images=small,
                                   padding=True, return_tensors="pt").to(te.device)
                hidden = te(**inputs, output_hidden_states=True).hidden_states[-1][0]
                out.append(hidden[inputs["attention_mask"][0].bool()][qm.DROP_EDIT:].unsqueeze(0).to(torch.bfloat16).cpu())
                print(f"  {t('Encoding prompts')} {i}/{len(texts)}", flush=True)
        del te
        free_vram()
        return out

    def load(self, refs):
        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
        from safetensors import safe_open
        self.vae = AutoencoderKLQwenImage.from_pretrained(self.DIR, subfolder="vae", dtype=torch.bfloat16).to("cuda")
        self.vae.enable_tiling()
        with torch.inference_mode():
            self.ctrls = [vae_latent(self.vae, im) for im in refs]
        self.vae.to("cpu")
        free_vram()
        print(t("Loading {name}...", name=self.LABEL + " (NF4)"), flush=True)
        self.transformer = qm.load_nf4_transformer(os.path.join(self.DIR, qm.TRANSFORMER_FOLDERS["edit"]))
        free_gb = gpu_stats.mem_get_info()[0] / 1e9
        latent_hw = sum(c.shape[2] * c.shape[3] for c in self.ctrls) + max(c.shape[2] * c.shape[3] for c in self.ctrls)
        resident = qm.auto_resident_blocks(self.transformer, free_gb, qm.training_reserve_gb(latent_hw, False, 0))
        print(t("Blocks on GPU") + f": {resident}/{len(self.transformer.transformer_blocks)}", flush=True)
        qm.BlockSwap(self.transformer, resident)
        self.transformer.eval()
        layers = {}
        with safe_open(os.path.join(self.DIR, "LoRAs", LIGHTNING), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".lora_down.weight"):
                    name = k[:-len(".lora_down.weight")]
                    down = f.get_tensor(k).to(torch.bfloat16)
                    layers[name] = (down, f.get_tensor(name + ".lora_up.weight").to(torch.bfloat16),
                                    f.get_tensor(name + ".alpha").item() / down.shape[0])
        self.hooks = turbo_hooks(self.transformer, layers, "Lightning")
        base = FlowMatchEulerDiscreteScheduler.from_pretrained(self.DIR, subfolder="scheduler")
        # Lightning se destiló con shift fijo 3 y sin shift_terminal.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base.config, base_shift=float(np.log(3)),
                                                                     max_shift=float(np.log(3)), shift_terminal=None)

    def sample(self, emb, w, h, seed):
        H, W = h // 8, w // 8   # en latente (px / 8); el transformer junta parches 2x2
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = qm.pack_latents(torch.randn((1, 16, H, W), generator=g, device="cuda", dtype=torch.bfloat16))
        sc = self.scheduler.config
        mu = qm.calculate_shift(latents.shape[1], sc.base_image_seq_len, sc.max_image_seq_len, sc.base_shift, sc.max_shift)
        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / self.steps, self.steps), device="cuda", mu=mu)
        self.scheduler.set_begin_index(0)
        shapes = [[(1, H // 2, W // 2)] + [(1, c.shape[2] // 2, c.shape[3] // 2) for c in self.ctrls]]
        emb = emb.to("cuda")
        for t_ in self.scheduler.timesteps:
            hidden = torch.cat([latents] + [qm.pack_latents(c) for c in self.ctrls], dim=1)
            out = self.transformer(hidden_states=hidden, encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                                   timestep=t_.expand(1).to(torch.bfloat16) / 1000, img_shapes=shapes, return_dict=False)[0]
            latents = self.scheduler.step(out[:, :latents.shape[1]], t_, latents, return_dict=False)[0]
        self.vae.to("cuda")
        img = vae_image(self.vae, qm.unpack_latents(latents, H, W))
        self.vae.to("cpu")
        return img


class Klein9B:
    """
    FLUX.2 [klein] Base 9B (FLUX Non-Commercial License) NF4 + el LoRA turbo de kalle07 (4 pasos, sin
    CFG), como las previews de su entrenador. El text encoder (Qwen3-8B NF4) solo lee texto: las
    referencias entran como latentes LIMPIOS detrás del objetivo, cada una con sus ids (t = 10, 20...).
    Mejor anatomía que los Qwen con turbo.
    """
    LABEL = "FLUX.2 Klein 9B"
    DIR = str(resolve_model_dir("Models/FLUX.2-Klein-9B_NF4", "Models/FLUX.2-Klein-9B_NF4"))
    REPO = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"
    TURBO_REPO = "kalle07/FLUX.2-klein-9B-turbo-lora-set"
    TURBO = "Flux_Klein_9b_Turbo_lora_rank_64_bf16_standard.safetensors"
    TAGS = "reference"
    steps = 4   # turbo de kalle07
    # Formato BFL del turbo -> módulos de diffusers (como 2_train_lora_klein9b.py). Las qkv fusionadas se
    # reparten en q/k/v: misma matriz down, filas consecutivas de la up.
    BLOCKS = {
        "double_blocks": ("transformer_blocks", {
            "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
            "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
            "img_attn.proj": ("attn.to_out.0",), "txt_attn.proj": ("attn.to_add_out",),
            "img_mlp.0": ("ff.linear_in",), "img_mlp.2": ("ff.linear_out",),
            "txt_mlp.0": ("ff_context.linear_in",), "txt_mlp.2": ("ff_context.linear_out",),
            "img_attn.norm.query_norm": ("attn.norm_q",), "img_attn.norm.key_norm": ("attn.norm_k",),
            "txt_attn.norm.query_norm": ("attn.norm_added_q",), "txt_attn.norm.key_norm": ("attn.norm_added_k",),
        }),
        "single_blocks": ("single_transformer_blocks", {
            "linear1": ("attn.to_qkv_mlp_proj",), "linear2": ("attn.to_out",),
            "norm.query_norm": ("attn.norm_q",), "norm.key_norm": ("attn.norm_k",),
        }),
    }
    TOP = {
        "img_in": "x_embedder", "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out", "final_layer.adaLN_modulation.1": "norm_out.linear",
    }

    def ensure(self):
        needed = ["transformer", "text_encoder", "tokenizer", "vae", "scheduler"]
        if not all(os.path.exists(os.path.join(self.DIR, d)) for d in needed):
            download(self.REPO, self.DIR, [f"{d}/*" for d in needed] + ["model_index.json", "LICENSE.md"], self.LABEL)
        if not os.path.exists(os.path.join(self.DIR, "LoRAs", self.TURBO)):
            from huggingface_hub import hf_hub_download
            hf_hub_download(self.TURBO_REPO, self.TURBO, local_dir=os.path.join(self.DIR, "LoRAs"))

    def ref_area(self, area, n):
        return area if n <= 2 else int(area * 2 / (n + 1))

    def encode(self, texts, refs):
        from diffusers import Flux2KleinPipeline
        from transformers import AutoTokenizer, Qwen3Model
        print(t("Loading {name}...", name="Qwen3-8B NF4 (text encoder)"), flush=True)
        te = Qwen3Model.from_pretrained(os.path.join(self.DIR, "text_encoder"), dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(self.DIR, subfolder="tokenizer")
        out = []
        with torch.inference_mode():
            for i, text in enumerate(texts, 1):
                out.append(Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, [text], dtype=torch.bfloat16, device="cuda").cpu())
                print(f"  {t('Encoding prompts')} {i}/{len(texts)}", flush=True)
        del te
        free_vram()
        return out

    def _targets(self, name):
        parts = name.split(".")
        if parts[0] in self.BLOCKS:
            prefix, layers = self.BLOCKS[parts[0]]
            return [f"{prefix}.{parts[1]}.{x}" for x in layers[".".join(parts[2:])]]
        return [self.TOP[name]]

    def load(self, refs):
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
        from safetensors import safe_open
        self.P = Flux2KleinPipeline
        self.vae = AutoencoderKLFlux2.from_pretrained(self.DIR, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
        self.bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).float().to("cuda")
        self.bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).float() + self.vae.config.batch_norm_eps).to("cuda")
        self.ctrls = []
        with torch.inference_mode():
            for im in refs:
                x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float() / 127.5 - 1
                z = self.vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mode().float()
                self.ctrls.append(((self.P._patchify_latents(z) - self.bn_mean) / self.bn_std).to(torch.bfloat16))
        self.vae.to("cpu")
        free_vram()
        print(t("Loading {name}...", name=self.LABEL + " (NF4)"), flush=True)
        self.transformer = load_flux2_transformer(self.DIR).to("cuda").eval()
        # Turbo: capas como hooks (lora_down / lora_up, escala 1) y diferencias de las normas q/k.
        layers = {}
        with safe_open(os.path.join(self.DIR, "LoRAs", self.TURBO), framework="pt", device="cpu") as f:
            for k in f.keys():
                name = k[len("diffusion_model."):]
                if name.endswith(".diff"):
                    w = self.transformer.get_submodule(self._targets(name[:-len(".diff")])[0]).weight
                    w.data += f.get_tensor(k).to(w.device, w.dtype)
                    continue
                if not name.endswith(".lora_down.weight"):
                    continue
                name = name[:-len(".lora_down.weight")]
                down = f.get_tensor(k).to(torch.bfloat16)
                up = f.get_tensor(k.replace(".lora_down.", ".lora_up.")).to(torch.bfloat16)
                targets = self._targets(name)
                if targets == ["norm_out.linear"]:
                    shift, scale = up.chunk(2)   # BFL guarda (shift, scale); diffusers espera (scale, shift)
                    up = torch.cat([scale, shift])
                for target, rows in zip(targets, up.chunk(len(targets))):
                    layers[target] = (down, rows.contiguous(), 1.0)
        self.hooks = turbo_hooks(self.transformer, layers, "Turbo LoRA (kalle07)")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.DIR, subfolder="scheduler")

    def sample(self, emb, w, h, seed):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        H, W = h // 16, w // 16   # VAE 8x + parches 2x2 ya en el latente (128 canales)
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 128, H, W), generator=g, device="cuda", dtype=torch.bfloat16).view(1, 128, H * W).transpose(1, 2)
        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / self.steps, self.steps), device="cuda", mu=compute_empirical_mu(H * W, self.steps))
        self.scheduler.set_begin_index(0)
        emb = emb.to("cuda")
        img_ids = self.P._prepare_latent_ids(latents.new_empty(1, 1, H, W)).to("cuda")
        ctrl_ids = self.P._prepare_image_ids([c[:1] for c in self.ctrls]).to("cuda")
        ids = torch.cat([img_ids, ctrl_ids], dim=1)
        txt_ids = self.P._prepare_text_ids(emb).to("cuda")
        refs = [c.view(1, c.shape[1], -1).transpose(1, 2) for c in self.ctrls]
        for t_ in self.scheduler.timesteps:
            out = self.transformer(hidden_states=torch.cat([latents] + refs, dim=1), encoder_hidden_states=emb,
                                   timestep=(t_ / 1000).expand(1).to(torch.bfloat16), img_ids=ids, txt_ids=txt_ids,
                                   return_dict=False)[0]
            latents = self.scheduler.step(out[:, :H * W], t_, latents, return_dict=False)[0]
        self.vae.to("cuda")
        lat = latents.transpose(1, 2).reshape(1, 128, H, W).float()
        lat = self.P._unpatchify_latents(lat * self.bn_std + self.bn_mean).to(self.vae.dtype)
        try:
            img = self.vae.decode(lat, return_dict=False)[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.vae.enable_tiling()
            img = self.vae.decode(lat, return_dict=False)[0]
        self.vae.to("cpu")
        img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(img)


def load_flux2_transformer(model_dir):
    """El transformer de FLUX.2 Klein 9B desde su caché NF4 (como 2_train_lora_klein9b.py)."""
    from diffusers import Flux2Transformer2DModel
    return load_nf4_cache(model_dir, Flux2Transformer2DModel)


class Flux3Klein:
    """
    FLUX.3 Image Klein Base 5.5B (FLUX Kommunity License) NF4. Calidad «best»: como las previews de su
    entrenador, shift constante 6.93, 28 pasos y CFG 3.5 con negativo vacío. «fast»: LoRA turbo PDD de
    NoiseToNova (misma licencia), 8 pasos sin CFG (lo lleva destilado): ~6x más rápido y en la prueba más
    parecido (64% frente a 58%). El text encoder (Qwen3-VL-4B) solo lee texto; las referencias van LIMPIAS
    por su propio flujo (image_cond), con l = -1, -2...
    """
    LABEL = "FLUX.3 Klein"
    DIR = str(resolve_model_dir("Models/FLUX.3-Image-Klein_NF4", "Models/FLUX.3-Image-Klein_NF4"))
    REPO = "AcademiaSD/FLUX.3-Image-Klein-NF4-for-LoRA-Training"
    TURBO_REPO = "NoiseToNova/Flux-3-Klein-turbo-pdd-4_8-steps"
    TURBO = "Flux-3-Klein-turbo-pdd-v0.1-8_4step-lora-r128.safetensors"
    TAGS = "reference"
    CFG = 3.5
    SHIFT = 6.93
    ROPE = 0.5

    def __init__(self, quality="fast", steps=0):
        self.turbo = quality == "fast"
        # PDD solo admite 4 u 8 (saltos de 32 o 16 en su rejilla de 128); sin turbo, cualquiera (0 = 28).
        if self.turbo:
            self.steps = steps if steps in (4, 8) else 8
        else:
            self.steps = steps if 4 <= steps <= 100 else 28
        self.hooks = []

    def ensure(self):
        if not (os.path.exists(os.path.join(self.DIR, "transformer", "index.json"))
                and os.path.exists(os.path.join(self.DIR, "text_encoder", "config.json"))):
            download(self.REPO, self.DIR, None, self.LABEL)
        if self.turbo and not os.path.exists(os.path.join(self.DIR, "LoRAs", self.TURBO)):
            download(self.TURBO_REPO, os.path.join(self.DIR, "LoRAs"), [self.TURBO, "pdd_config.json"], "Turbo LoRA (PDD)")

    def ref_area(self, area, n):
        return area if n <= 2 else int(area * 2 / (n + 1))

    def encode(self, texts, refs):
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
        from transformers import AutoProcessor, Qwen3VLModel
        print(t("Loading {name}...", name="Qwen3-VL-4B (text encoder)"), flush=True)
        path = os.path.join(self.DIR, "text_encoder")
        processor = AutoProcessor.from_pretrained(path)
        te = Qwen3VLModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", max_memory=te_budget())
        enc = lambda s: Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(te, processor, s, device="cuda").to(torch.bfloat16).cpu()
        out = []
        with torch.inference_mode():
            self.neg = enc("")
            for i, text in enumerate(texts, 1):
                out.append(enc(text))
                print(f"  {t('Encoding prompts')} {i}/{len(texts)}", flush=True)
        del te, processor
        free_vram()
        return out

    def _ids(self, H, W, l=0):
        ids = torch.cartesian_prod(torch.arange(1), torch.arange(H), torch.arange(W), torch.tensor([l])).float()
        ids[:, 1:3] *= self.ROPE
        return ids.to("cuda")

    @staticmethod
    def _txt_ids(n):
        return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n)).float().to("cuda")

    def load(self, refs):
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
        from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep
        self.vae = AutoencoderKLFlux2.from_pretrained(self.DIR, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
        self.bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).float().to("cuda")
        self.bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).float() + self.vae.config.batch_norm_eps).to("cuda")
        ctrls = []
        with torch.inference_mode():
            for im in refs:
                x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float() / 127.5 - 1
                z = self.vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mode().float()
                ctrls.append(((Flux3VaeEncoderStep._patchify_latents(z) - self.bn_mean) / self.bn_std).to(torch.bfloat16))
        self.vae.to("cpu")
        free_vram()
        self.cond = torch.cat([c.flatten(2).transpose(1, 2) for c in ctrls], dim=1)
        self.cond_ids = torch.cat([self._ids(c.shape[2], c.shape[3], -(1 + i)) for i, c in enumerate(ctrls)])
        print(t("Loading {name}...", name=self.LABEL + " (NF4)"), flush=True)
        self.transformer = load_nf4_cache(self.DIR, Flux3Transformer2DModel).to("cuda").eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=self.SHIFT, use_dynamic_shifting=False)
        self.sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
        print(f"  {self.steps} " + t("steps") + (" (turbo)" if self.turbo else f" · CFG {self.CFG}"), flush=True)
        if self.turbo:
            self._load_turbo()

    def _load_turbo(self):
        """
        LoRA PDD (formato PEFT, escala alpha/rank) como hooks + proj_out.weight: un banco de 128 cabezas de
        salida, una por paso de su rejilla de sigmas (ya desplazada). Cada salto de 16 pasos usa la media de
        sus cabezas pesada por dt, como el wrapper de su ficha; los sigmas son los del inicio de cada salto.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler
        from safetensors.torch import load_file
        with open(os.path.join(self.DIR, "LoRAs", "pdd_config.json"), encoding="utf-8") as f:
            pc = json.load(f)
        sd = load_file(os.path.join(self.DIR, "LoRAs", self.TURBO))
        heads = sd.pop("proj_out.weight").float()
        scale = pc["lora_alpha"] / pc["lora_rank"]
        layers = {k.split(".lora_A.")[0]: (v.to(torch.bfloat16), sd[k.replace(".lora_A.", ".lora_B.")].to(torch.bfloat16), scale)
                  for k, v in sd.items() if ".lora_A." in k}
        self.hooks = turbo_hooks(self.transformer, layers, "Turbo LoRA (PDD)")
        sig = torch.tensor(pc["sigmas"])
        bounds = list(range(0, pc["trajectory_steps"] + 1, pc["trajectory_steps"] // self.steps))
        dt = sig.diff()
        self.heads = [torch.einsum("t,toi->oi", dt[a:b] / dt[a:b].sum(), heads[a:b]).to("cuda", torch.bfloat16)
                      for a, b in zip(bounds[:-1], bounds[1:])]
        self.block = 0
        self.hooks.append(self.transformer.proj_out.register_forward_hook(
            lambda m, args, out: F.linear(args[0].to(torch.bfloat16), self.heads[self.block], m.bias).to(out.dtype)))
        self.sigmas = sig[bounds[:-1]].numpy()
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)

    def sample(self, emb, w, h, seed):
        H, W = h // 16, w // 16   # VAE 8x + parches 2x2 (128 canales)
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 128, H, W), generator=g, device="cuda", dtype=torch.bfloat16).view(1, 128, H * W).transpose(1, 2)
        self.scheduler.set_timesteps(sigmas=self.sigmas, device="cuda")
        self.scheduler.set_begin_index(0)
        img_ids = self._ids(H, W)
        # El turbo lleva el CFG destilado: solo la pasada positiva.
        conds = [(e.to("cuda"), self._txt_ids(e.shape[1])) for e in ((emb,) if self.turbo else (emb, self.neg))]
        for i, t_ in enumerate(self.scheduler.timesteps):
            self.block = i
            timestep = (t_ / 1000).expand(1).to(torch.bfloat16)
            v = [self.transformer(hidden_states=latents, encoder_hidden_states=e, txt_ids=ti, img_ids=img_ids,
                                  image_cond_hidden_states=self.cond, image_cond_ids=self.cond_ids,
                                  timestep=timestep, return_dict=False)[0] for e, ti in conds]
            v = v[0] if self.turbo else v[1] + self.CFG * (v[0] - v[1])
            latents = self.scheduler.step(v, t_, latents, return_dict=False)[0]
        self.vae.to("cuda")
        lat = latents.transpose(1, 2).reshape(1, 128, H, W).float() * self.bn_std + self.bn_mean
        lat = lat.view(1, 32, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, H * 2, W * 2).to(self.vae.dtype)
        try:
            img = self.vae.decode(lat, return_dict=False)[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.vae.enable_tiling()
            img = self.vae.decode(lat, return_dict=False)[0]
        self.vae.to("cpu")
        img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(img)


class Fibo2:
    """
    Fibo 2.0 Dev (BRIA, CC BY-NC 4.0) NF4: genera y edita con varias referencias. Como el pipeline oficial
    de edición: 30 pasos, CFG 5 con el caption vacío como negativo, la instrucción envuelta en el JSON de
    edición ({"image_1": marca, ..., "edit_instruction": ...}) y las referencias como context_latents.
    """
    LABEL = "Fibo 2.0"
    DIR = str(resolve_model_dir("Models/Fibo2_Dev-NF4", "Models/Fibo2_Dev-NF4"))
    REPO = "AcademiaSD/Fibo-2.0-NF4-for-LoRA-Training"
    TAGS = "reference"
    steps = 30
    CFG = 5.0
    hooks = []

    def ensure(self):
        needed = ["transformer", "text_encoder_BF16", "vae", "processor", "scheduler"]
        if not all(os.path.exists(os.path.join(self.DIR, d)) for d in needed):
            download(self.REPO, self.DIR, [f"{d}/*" for d in needed] + ["model_index.json", "fonts/*"], self.LABEL)

    def ref_area(self, area, n):
        return area if n <= 2 else int(area * 2 / (n + 1))

    def encode(self, texts, refs):
        from diffusers import DiffusionPipeline
        from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import DEFAULT_NEGATIVE_PROMPT
        from transformers import Qwen3VLForConditionalGeneration
        print(t("Loading {name}...", name="Qwen3-VL-4B (text encoder)"), flush=True)
        te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(self.DIR, "text_encoder_BF16"), dtype=torch.bfloat16,
                                                             device_map="auto", max_memory=te_budget())
        pipe = DiffusionPipeline.from_pretrained(self.DIR, transformer=None, vae=None, text_encoder=te, dtype=torch.bfloat16)
        pipe.tokenizer_max_length = 4096
        # Las referencias ya vienen recortadas a su tamaño: se pide ese mismo tamaño (con una sola, el
        # procesador la recortaría al de la salida).
        self.vae_refs, te_refs = pipe.image_processor.preprocess(
            refs, width=refs[0].width, height=refs[0].height, area=refs[0].width * refs[0].height,
            text_encoder_image_size=pipe.text_encoder_image_size)
        mask_of = lambda e, m: m if m is not None else torch.ones((e.shape[0], e.shape[2]), dtype=torch.bool, device=e.device)
        out = []
        with torch.inference_mode():
            e, m = pipe.encode_prompt(DEFAULT_NEGATIVE_PROMPT, None, "cuda")
            self.neg = (e.cpu(), mask_of(e, m).cpu())
            for i, text in enumerate(texts, 1):
                e, m = pipe.encode_prompt(text, te_refs, "cuda")
                out.append((e.cpu(), mask_of(e, m).cpu()))
                print(f"  {t('Encoding prompts')} {i}/{len(texts)}", flush=True)
        del pipe, te
        free_vram()
        return out

    def load(self, refs):
        from diffusers import AutoencoderKLFlux2, BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler, BriaFibo2Transformer2DModel
        self.P = BriaFibo2Pipeline
        self.vae = AutoencoderKLFlux2.from_pretrained(self.DIR, subfolder="vae", dtype=torch.bfloat16).to("cuda")
        self.bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).float().to("cuda")
        self.bn_scale = (1 / torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps)).view(1, -1, 1, 1).float().to("cuda")
        self.ctrls = []
        with torch.inference_mode():
            for img in self.vae_refs:
                lat = self.P._patchify_latents(self.vae.encode(img.contiguous().to("cuda", self.vae.dtype)).latent_dist.mode().float())
                self.ctrls.append(self.P._unpatchify_latents((lat - self.bn_mean) * self.bn_scale).to(torch.bfloat16))
        self.vae.to("cpu")
        free_vram()
        print(t("Loading {name}...", name=self.LABEL + " (NF4)"), flush=True)
        self.transformer = load_nf4_cache(self.DIR, BriaFibo2Transformer2DModel).to("cuda").eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.DIR, subfolder="scheduler")
        print(f"  {self.steps} " + t("steps") + f" · CFG {self.CFG}", flush=True)

    def sample(self, cond, w, h, seed):
        H, W = h // 8, w // 8   # latente de 32 canales; el transformer junta parches 2x2
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 32, H, W), generator=g, device="cuda", dtype=torch.float32)
        sc = self.scheduler.config
        mu = qm.calculate_shift((H // 2) * (W // 2), sc.base_image_seq_len, sc.max_image_seq_len, sc.base_shift, sc.max_shift)
        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / self.steps, self.steps), device="cuda", mu=mu)
        self.scheduler.set_begin_index(0)
        pos, neg = [(e.to("cuda"), None if m.all() else m.to("cuda")) for e, m in (cond, self.neg)]
        for t_ in self.scheduler.timesteps:
            x = latents.to(torch.bfloat16)
            timestep = t_.expand(1).to(torch.float32) / 1000
            p, n = (self.transformer(hidden_states=x, timestep=timestep, encoder_hidden_states=e, encoder_attention_mask=m,
                                     context_latents=self.ctrls, return_dict=False)[0].float() for e, m in (pos, neg))
            latents = self.scheduler.step(n + self.CFG * (p - n), t_, latents, return_dict=False)[0]
        self.vae.to("cuda")
        lat = self.P._patchify_latents(latents.float()) / self.bn_scale + self.bn_mean
        lat = self.P._unpatchify_latents(lat).to(self.vae.dtype)
        try:
            img = self.vae.decode(lat, return_dict=False)[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.vae.enable_tiling()
            img = self.vae.decode(lat, return_dict=False)[0]
        self.vae.to("cpu")
        img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(img)


BACKENDS = {"qwen21": Qwen21, "flux3klein": Flux3Klein, "fibo2": Fibo2, "klein9b": Klein9B, "edit2511": Edit2511}


def make_scorer():
    try:
        import face_match
        if not face_match.installed():
            raise RuntimeError
    except Exception:
        print("[!] " + t("Face likeness is not installed: the images are kept without checking the likeness."), flush=True)
        return None
    scorer = face_match.Scorer(REFS)
    if not scorer.faces:
        print("[!] " + t("No face found in the reference images: the images are kept without checking the likeness."), flush=True)
        return None
    print(t("Likeness reference: {n} faces.", n=scorer.faces), flush=True)
    return scorer


def generate():
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    rejected_dir = os.path.join(out_dir, "_rejected")
    plan, subject, identity = cfg["plan"], cfg["subject"] or "person", cfg["identity"].strip()
    threshold, tries = float(cfg["threshold"]), max(1, int(cfg["max_tries"]))
    area = int(cfg["resolution"]) ** 2
    seed0 = int(cfg["seed"]) or int(time.time()) % 1000000
    trigger = cfg["trigger_word"]
    cls = BACKENDS.get(cfg.get("model"), Flux3Klein)
    if cls is Flux3Klein:
        model = cls(cfg.get("quality", "fast"), int(cfg.get("steps") or 0))
    else:
        model = cls(cfg.get("quality", "fast")) if cls is Qwen21 else cls()
    items = []
    status(phase="generate", total=len(plan), items=items, output_dir=out_dir, message=t("Preparing..."))
    print(f"{t('Model')}: {model.LABEL}", flush=True)

    # Referencias en la carpeta nueva, con su caption (el de origen, el del análisis o ninguno).
    if cfg["include_references"]:
        ref_caps = cfg.get("ref_captions") or {}
        for path in REFS:
            name = os.path.basename(path)
            shutil.copy2(path, os.path.join(out_dir, name))
            caption = read_caption(path) or ref_caps.get(name, "")
            if caption:
                with open(os.path.join(out_dir, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
                    f.write(forge.training_caption(trigger, caption) if trigger and not caption.lower().startswith(trigger.lower()) else caption)

    scorer = make_scorer() if threshold > 0 else None
    model.ensure()
    # Las mismas referencias (ya recortadas a su tamaño) para el text encoder y el VAE.
    refs = []
    for p in REFS[:4]:
        im = open_image(p)
        refs.append(fit(im, *qm.calculate_dimensions(model.ref_area(area, min(len(REFS), 4)), im.width / im.height)))
    status(message=t("Encoding the prompts..."))
    conds = model.encode([forge.instruction(p["text"], subject, identity, len(refs), model.TAGS) for p in plan], refs)
    status(message=t("Loading {name}...", name=model.LABEL))
    model.load(refs)

    kept = 0
    for i, (item, cond) in enumerate(zip(plan, conds), 1):
        w, h = qm.calculate_dimensions(area, item["ratio"])
        best, t0 = None, time.time()
        for k in range(tries):
            seed = seed0 + i * 1000 + k
            status(message=t("Image {i}/{n}, try {k}/{m}", i=i, n=len(plan), k=k + 1, m=tries))
            with torch.inference_mode():
                img = model.sample(cond, w, h, seed)
            score = scorer.score(img) if scorer else None
            if best is None or (score or 0) > (best[1] or 0):
                best = (img, score, seed)
            if scorer is None or (score is not None and score >= threshold):
                break
        img, score, seed = best
        ok = scorer is None or (score is not None and score >= threshold)
        name = f"forge_{i:03d}"
        target = out_dir if ok else rejected_dir
        os.makedirs(target, exist_ok=True)
        img.save(os.path.join(target, name + ".png"))
        with open(os.path.join(target, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(forge.training_caption(trigger, item["text"]))
        kept += ok
        items.append({"file": name + ".png", "ok": ok, "score": score, "seed": seed, "text": item["text"]})
        shown = "–" if score is None else f"{score:.0f}%"
        print(f"[{i}/{len(plan)}] {name}.png {('✓' if ok else '✗')} {shown} · {time.time() - t0:.0f} s · {item['text']}", flush=True)
        status(items=items, kept=kept)

    for hk in model.hooks:
        hk.remove()
    status(phase="done", kept=kept, message=t("Done: {kept} of {n} images kept.", kept=kept, n=len(plan)))
    print("[OK] " + t("Done: {kept} of {n} images kept.", kept=kept, n=len(plan)), flush=True)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    import signal
    # Stop manda CTRL_BREAK en Windows (SIGINT en Linux): sin esto Python moriría sin marcar «stopped».
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if mode == "analyze":
            analyze()
        elif mode == "generate":
            generate()
        else:
            raise SystemExit("usage: 0_forge_dataset.py analyze|generate")
    except KeyboardInterrupt:
        status(phase="stopped", message=t("Stopped."))
        print("\n[!] " + t("Stopped."), flush=True)
    except Exception as exc:
        status(phase="error", message=str(exc))
        raise
