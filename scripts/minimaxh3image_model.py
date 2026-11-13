# -*- coding: utf-8 -*-
"""
minimaxh3image_model.py

Shared pieces of the MiniMax-H3-Image trainer that are NOT the training engine:

* the packed-sequence layout `[text | reference images | generated image]`,
* the 6-step schedule the checkpoint was distilled on, one output head per step,
* a low-VRAM NF4 loader for inference (previews, tests): every quantized linear
  stays packed in system RAM and is dequantized on the GPU only while it runs,
  so the 33B DiT samples inside ~3 GB of VRAM at any card size,
* the image VAE (encode for the pre-cache, decode for previews).

The layout functions follow the released diffusers pipeline
(`modular_pipelines/minimax_h3_image`, Apache-2.0) so the coordinates match the
ones the checkpoint was trained with.

Piezas compartidas del entrenador de MiniMax-H3-Image que no son el motor de
entrenamiento: layout de la secuencia, calendario de 6 pasos con una cabeza por
paso, cargador NF4 de poca VRAM para inferencia y el VAE de imagen.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_TAG = 0
TEXT_TAG = 1
ROPE_SPATIAL_SCALE = 32
VAE_RATIO = 16
PATCH = (1, 2, 2)
CANVAS_MULTIPLE = 32            # VAE_RATIO * patch
LATENT_CHANNELS = 24
TEXT_ENCODER_LAYER = 50
REFERENCE_T = 0.999             # reference rows are held at t = 0.999 on every step
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# The sampling grid the per-step heads were distilled on. Head i runs at sigma i.
# Calendario sobre el que se destilaron las cabezas. La cabeza i corre en sigma i.
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)
NUM_PDD_HEADS = 6


def pdd_timestep(step_index):
    """float32 `t = 1 - sigma` of a sampling step, rounded as the released sampler does."""
    return float(torch.tensor(1.0 - PDD_SIGMAS[step_index], dtype=torch.float32))


def pdd_sigma(step_index):
    """The sigma actually stepped on: `1 - t` after the float32 round trip (0 for the terminal one)."""
    if step_index >= NUM_PDD_HEADS:
        return 0.0
    return float(1.0 - torch.tensor(pdd_timestep(step_index), dtype=torch.float32))


# =============================================================================
# PACKED SEQUENCE / SECUENCIA EMPAQUETADA
# =============================================================================

def patchify(latents):
    """(B, C, H, W) or (B, C, 1, H, W) -> (B * rows, C * 4), row-major over the 2x2 patch grid."""
    if latents.dim() == 4:
        latents = latents[:, :, None]
    pt, ph, pw = PATCH
    b, c, f, h, w = latents.shape
    if h % ph or w % pw:
        raise ValueError("Latent {}x{} is not divisible by the 2x2 patch.".format(h, w))
    latents = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, c * pt * ph * pw).contiguous()


def unpatchify(rows, latent_h, latent_w):
    """(rows, C * 4) -> (1, C, H, W). Inverse of `patchify` for one image."""
    _, ph, pw = PATCH
    c = rows.shape[-1] // (ph * pw)
    rows = rows.reshape(-1, latent_h // ph, latent_w // pw, c, ph, pw).permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(-1, c, latent_h, latent_w).contiguous()


def _spatial_axis(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # numpy on purpose: np.linspace(endpoint=False) is not what torch.linspace computes.
    return torch.from_numpy(np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE).to(torch.float64)


def _frame_grid(latent_h, latent_w):
    sqrt_area = np.sqrt(latent_h * latent_w)
    grids = torch.meshgrid(_spatial_axis(latent_h, PATCH[1], sqrt_area), _spatial_axis(latent_w, PATCH[2], sqrt_area), indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def build_packed_sequence(text_token_tags, reference_grids, latent_h, latent_w):
    """
    Layout `[text | reference images | generated image]`.

    text_token_tags: (num_text,) long - 1 for text, 0 for the rows of a reference's
        vision block inside the text stream. reference_grids: [(latent_h, latent_w)].
    Returns position_ids (S, 3) float64, token_tags (S,), image_indices, text_indices,
    num_reference_rows.
    """
    _, ph, pw = PATCH
    n_text = int(text_token_tags.shape[0])
    n_ref = sum((h // ph) * (w // pw) for h, w in reference_grids)
    n_tgt = (latent_h // ph) * (latent_w // pw)
    seq = n_text + n_ref + n_tgt

    position_ids = torch.zeros(seq, 3, dtype=torch.float64)
    position_ids[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    # Every image is one frame on the rotary clock, which starts where the text ends.
    cursor, clock = n_text, float(n_text)
    for h, w in reference_grids:
        rows = slice(cursor, cursor + (h // ph) * (w // pw))
        cursor = rows.stop
        position_ids[rows, 0] = clock
        position_ids[rows, 1:] = _frame_grid(h, w)
        clock += 1.0
    position_ids[cursor:, 0] = clock
    position_ids[cursor:, 1:] = _frame_grid(latent_h, latent_w)

    text_indices = torch.arange(n_text)
    image_indices = torch.arange(n_text, seq)
    token_tags = torch.empty(seq, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = IMAGE_TAG
    return position_ids, token_tags, image_indices, text_indices, n_ref


def row_timesteps(t_target, seq_len, image_indices, num_reference_rows):
    """(unique timesteps, per-row index): target/text rows at `t_target`, reference rows at 0.999."""
    rows = torch.full((seq_len,), float(t_target), dtype=torch.float32)
    if num_reference_rows:
        rows[image_indices[:num_reference_rows].cpu()] = REFERENCE_T
    return torch.unique(rows, sorted=True, return_inverse=True)


# -----------------------------------------------------------------------------
# MiniMax-H3 (video) DiT: the model the LoRA is TRAINED on
# -----------------------------------------------------------------------------
# MiniMax-H3-Image is distilled to 6 steps with one output head per step: plain flow matching
# on it turns the high-noise heads into the dataset's average (blotches). It was fine-tuned from
# MiniMax-H3, which is NOT distilled, shares the latent space and every LoRA module name, and a
# LoRA trained on it carries over to MiniMax-H3-Image. So the LoRA trains on MiniMax-H3 and
# MiniMax-H3-Image only renders the previews. The layout below is the one the MiniMax-H3 trainer
# packs a still image with: [text | 1 silent audio row | video], one timestep for every row.
# H3-Image esta destilado: se entrena sobre MiniMax-H3 (sin destilar) y H3-Image solo hace previews.

AUDIO_CHANNELS = 32
AUDIO_TAG = 2


def build_video_packed_sequence(n_text, latent_h, latent_w, text_token_tags=None):
    """
    `[text | audio | video]` for ONE still image on the MiniMax-H3 DiT, with a single silent audio
    row. Returns position_ids (S, 3) float64, token_tags, video_indices, audio_indices, text_indices.
    """
    _, ph, pw = PATCH
    n_video = (latent_h // ph) * (latent_w // pw)
    n_audio = 1
    seq = n_text + n_audio + n_video
    text_indices = torch.arange(n_text)
    audio_indices = torch.arange(n_text, n_text + n_audio)
    video_indices = torch.arange(n_text + n_audio, seq)
    token_tags = torch.empty(seq, dtype=torch.long)
    token_tags[text_indices] = (text_token_tags.long() if text_token_tags is not None else TEXT_TAG)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = IMAGE_TAG
    position_ids = torch.zeros(seq, 3, dtype=torch.float64)
    position_ids[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    # The single frame and the audio row both start where the text ends; the audio row sits on
    # the first column of the width axis, as the MiniMax-H3 trainer places it.
    width_axis = _spatial_axis(latent_w, pw, np.sqrt(latent_h * latent_w))
    position_ids[audio_indices, 0] = float(n_text)
    position_ids[audio_indices, 2] = float(width_axis[0])
    position_ids[video_indices, 0] = float(n_text)
    position_ids[video_indices, 1:] = _frame_grid(latent_h, latent_w)
    return position_ids, token_tags, video_indices, audio_indices, text_indices


def sample_training_sigma(image_tokens, generator, mu=0.0, std=1.0, resolution_shift=True):
    """
    Logit-normal noise level with the resolution-dependent shift the MiniMax-H3 trainer uses:
    shift = exp(0.5 + (tokens - 256) * (1.15 - 0.5) / (6400 - 256)). NOT the video sampler's 12.0.
    Nivel de ruido logit-normal con el shift por resolucion del entrenador de MiniMax-H3.
    """
    base = float(torch.sigmoid(mu + std * torch.randn(1, generator=generator)))
    if not resolution_shift:
        return base
    shift = math.exp(0.5 + (float(image_tokens) - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0))
    return shift * base / (1.0 + (shift - 1.0) * base)


def attach_lora(target_model, source_model):
    """
    Shares the LoRA of `source_model` (the MiniMax-H3 DiT being trained) with `target_model`
    (MiniMax-H3-Image for a preview): same Parameters, no copy, module by module by name.
    Comparte el LoRA del DiT en entrenamiento con H3-Image para una preview, sin copiarlo.
    """
    source = {n: m for n, m in source_model.named_modules() if isinstance(m, LazyNF4Linear) and m.lora_A is not None}
    shared = 0
    for name, mod in target_model.named_modules():
        src = source.get(name)
        if isinstance(mod, LazyNF4Linear) and src is not None:
            mod.lora_A, mod.lora_B, mod.lora_scale, mod.lora_enabled = src.lora_A, src.lora_B, src.lora_scale, True
            shared += 1
    return shared


def canvas_size(width, height, multiple=CANVAS_MULTIPLE):
    return max(multiple, round(width / multiple) * multiple), max(multiple, round(height / multiple) * multiple)


# =============================================================================
# NF4 LOADER FOR INFERENCE / CARGADOR NF4 PARA INFERENCIA
# =============================================================================

class LazyNF4Linear(nn.Module):
    """
    A frozen NF4 linear that lives packed in system RAM. `forward` uploads the
    packed bytes, dequantizes, runs and frees, so VRAM only ever holds one
    weight. Inference only: there is no backward through the weight.
    """

    def __init__(self, tensors, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self._packed = tensors["weight"]
        self._stats = {k[len("quant_state."):]: v for k, v in tensors.items() if k.startswith("quant_state.")}
        self._bias = tensors.get("bias")
        # Zero-size bf16 parameter: diffusers reads the module dtype/device from its parameters.
        self.dtype_probe = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)

        self.lora_A = None
        self.lora_B = None
        self.lora_scale = 1.0
        self.lora_enabled = True
        # Frozen LoRAs added on top (an adapter during training): [(A, B, scale)], never trained.
        # LoRAs congelados encima (un adaptador durante el entrenamiento); nunca se entrenan.
        self.frozen = []
        self.frozen_enabled = True

    def make_resident(self, device):
        """Keep the PACKED bytes on the GPU (0.33 GB per block): no upload per forward."""
        self._packed = self._packed.to(device)
        self._stats = {k: v.to(device) for k, v in self._stats.items()}
        if self._bias is not None:
            self._bias = self._bias.to(device)

    def make_offloaded(self):
        """Back to system RAM: undoes make_resident when the GPU turns out to be too full."""
        self._packed = self._packed.to("cpu")
        self._stats = {k: v.to("cpu") for k, v in self._stats.items()}
        if self._bias is not None:
            self._bias = self._bias.to("cpu")

    def add_lora(self, rank, alpha, rslora=False, device="cuda"):
        """fp32 LoRA on top of the frozen NF4 weight. B starts at zero."""
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scale = float(alpha) / (math.sqrt(rank) if rslora else float(rank))

    def forward(self, x):
        import bitsandbytes.functional as bnbF
        from bitsandbytes.nn import Params4bit
        with torch.no_grad():
            p = Params4bit.from_prequantized(data=self._packed, quantized_stats=dict(self._stats), device=x.device)
            w = bnbF.dequantize_4bit(p.data, p.quant_state).to(x.dtype)
            b = self._bias.to(x.device, x.dtype) if self._bias is not None else None
        out = F.linear(x, w, b)
        del p, w
        if self.lora_A is not None and self.lora_enabled:
            delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            out = out + (delta * self.lora_scale).to(out.dtype)
        if self.frozen and self.frozen_enabled:
            for a, b, scale in self.frozen:
                out = out + (F.linear(F.linear(x.to(a.dtype), a), b) * scale).to(out.dtype)
        return out


def set_frozen_enabled(model, enabled):
    for mod in model.modules():
        if isinstance(mod, LazyNF4Linear):
            mod.frozen_enabled = bool(enabled)


def lora_file_to_modules(path):
    """
    A LoRA file in the exported (original checkpoint) layout -> {diffusers module name: (A, B)}:
    qkv_proj split into to_q/to_k/to_v, out_proj -> to_out.0, mlp.fc1 ->
    ff.net.0.proj with the SwiGLU halves back to [value; gate], mlp.fc2 -> ff.net.2. Scaling is
    already baked into B by the exporter.
    Un LoRA exportado -> modulos de diffusers (inverso del exportador).
    """
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as fh:
        meta = fh.metadata() or {}
    sd = load_file(path)
    # qkv_proj is [q; k; v] (ComfyUI's layout). Only the files this trainer exported before it
    # wrote `qkv_layout` interleave it per attention head (128 rows each) - the published
    # de-distillation adapter is one of them.
    # Los LoRAs de imagen anteriores a `qkv_layout` van intercalados por cabeza.
    interleaved = ("qkv_layout" not in meta and "MiniMax-H3-Image" in meta.get("trained_with", ""))
    out_map = {}
    for key in sd:
        if not key.endswith(".lora_A.weight"):
            continue
        base = key[: -len(".lora_A.weight")]
        base = base[base.index("blocks."):] if "blocks." in base else base
        if key.split("blocks.")[0].endswith("token_refiner."):
            dbase = "token_refiner.refiner_" + base
        else:
            dbase = base.replace("blocks.", "transformer_blocks.", 1)
        A = sd[key].float()
        B = sd[key.replace("lora_A", "lora_B")].float()
        if dbase.endswith("attn.qkv_proj"):
            stem = dbase[: -len("qkv_proj")]
            out = B.shape[0] // 3
            if interleaved:
                rows = B.reshape(out // 128, 3, 128, -1)
                parts = [rows[:, i].reshape(out, -1) for i in range(3)]
            else:
                parts = list(B.chunk(3, dim=0))
            for n, part in zip(("to_q", "to_k", "to_v"), parts):
                out_map[stem + n] = (A, part.contiguous())
        elif dbase.endswith("attn.out_proj"):
            out_map[dbase.replace("out_proj", "to_out.0")] = (A, B)
        elif dbase.endswith("mlp.fc1"):
            gate, value = B.chunk(2, dim=0)
            out_map[dbase.replace("mlp.fc1", "ff.net.0.proj")] = (A, torch.cat([value, gate], 0))
        elif dbase.endswith("mlp.fc2"):
            out_map[dbase.replace("mlp.fc2", "ff.net.2")] = (A, B)
    return out_map


def add_frozen_lora(model, path, strength=1.0, device="cuda", dtype=torch.bfloat16):
    """Adds an exported LoRA as a frozen extra. Returns how many modules took it."""
    mods = dict(model.named_modules())
    n = 0
    for name, (A, B) in lora_file_to_modules(path).items():
        mod = mods.get(name)
        if isinstance(mod, LazyNF4Linear):
            mod.frozen.append((A.to(device, dtype), B.to(device, dtype), float(strength)))
            n += 1
    return n


def set_lora_enabled(model, enabled):
    """Turns every LazyNF4Linear LoRA on or off: off gives the base model's prediction."""
    for mod in model.modules():
        if isinstance(mod, LazyNF4Linear):
            mod.lora_enabled = bool(enabled)


def _set_submodule(root, name, module):
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def find_transformer_dir(nf4_dir):
    for cand in (os.path.join(nf4_dir, "transformers"), os.path.join(nf4_dir, "transformer"), nf4_dir):
        if os.path.isfile(os.path.join(cand, "index.json")):
            return cand
    raise FileNotFoundError("No NF4 transformer cache (index.json) under {}".format(nf4_dir))


def load_dit_lowvram(nf4_dir, device="cuda"):
    """
    An NF4 DiT: quantized linears lazy in RAM, everything else on `device`. The class comes from the
    cache's config.json: MiniMaxH3ImageTransformer2DModel (MiniMax-H3-Image) or
    MiniMaxH3Transformer3DModel (MiniMax-H3, the video model). Both share every module name the
    LoRA touches, so a LoRA trained on one attaches to the other unchanged.
    La clase sale del config.json: imagen o video. Comparten los nombres de los modulos del LoRA.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file

    tdir = find_transformer_dir(nf4_dir)
    wdir = os.path.join(tdir, "weights")
    with open(os.path.join(tdir, "config.json"), encoding="utf-8") as fh:
        raw_config = json.load(fh)
    config = {k: v for k, v in raw_config.items() if not k.startswith("_")}
    with open(os.path.join(tdir, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    if raw_config.get("_class_name") == "MiniMaxH3Transformer3DModel":
        from diffusers import MiniMaxH3Transformer3DModel as model_cls
    else:
        from minimaxh3image_vendor import MiniMaxH3ImageTransformer2DModel as model_cls
    with init_empty_weights():
        model = model_cls(**config)

    critical = index.get("precision_critical", {})
    critical_layers = {name.rsplit(".", 1)[0] for name in critical if name.endswith(".weight")}
    for layer, rec in index["quantized"].items():
        if layer in critical_layers:
            continue  # loaded in float below
        _set_submodule(model, layer, LazyNF4Linear(load_file(os.path.join(wdir, rec["file"])), rec["in_features"], rec["out_features"]))
    for section in ("other", "precision_critical"):
        for name, rec in index.get(section, {}).items():
            tensor = load_file(os.path.join(wdir, rec["file"]))[name]
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=tensor.dtype)
    model.rope.to(device)
    for mod in model.modules():
        if isinstance(mod, LazyNF4Linear):
            mod.dtype_probe.data = mod.dtype_probe.data.to(device)
    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError("Parameters still on meta after the NF4 load: {}".format(left[:8]))
    return model.eval()


# =============================================================================
# VAE
# =============================================================================

def load_vae(nf4_dir, device="cuda", dtype=torch.float32):
    from minimaxh3image_vendor import AutoencoderKLMiniMaxH3Image
    return AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(nf4_dir, "vae"), torch_dtype=dtype).to(device).eval()


def _vae_stats(vae, device):
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_image(vae, image, generator=None, sample=True):
    """PIL RGB (sides multiple of 32) -> normalized latent (1, 24, h, w) float32 on CPU."""
    device = next(vae.parameters()).device
    pixels = torch.from_numpy(np.array(image.convert("RGB"))).to(device).permute(2, 0, 1)[None]
    pixels = pixels[..., : pixels.shape[-2] // VAE_RATIO * VAE_RATIO, : pixels.shape[-1] // VAE_RATIO * VAE_RATIO]
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
    pixels = ((pixels.to(torch.float32).div(255.0) - mean) / std).to(next(vae.parameters()).dtype)
    posterior = vae.encode(pixels, return_dict=False)[0]
    latents = (posterior.sample(generator=generator) if sample else posterior.mode()).float()
    latents = latents[..., : latents.shape[-2] // 2 * 2, : latents.shape[-1] // 2 * 2]
    lmean, lstd = _vae_stats(vae, device)
    return ((latents - lmean) / lstd).cpu()


@torch.no_grad()
def decode_latent(vae, latents):
    """Normalized latent (1, 24, h, w) -> PIL image."""
    from PIL import Image
    device = next(vae.parameters()).device
    lmean, lstd = _vae_stats(vae, device)
    latents = latents.to(device, torch.float32) * lstd + lmean
    image = vae.decode(latents.to(next(vae.parameters()).dtype), return_dict=False)[0].float()
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    array = (image[0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array)


# =============================================================================
# SAMPLER
# =============================================================================

@torch.no_grad()
def sample(model, prompt_embeds, width, height, seed=0, text_token_tags=None, reference_latents=None,
           device="cuda", progress=None):
    """
    The released 6-step sampler. `prompt_embeds` is (1, T, 5120) - layer 50 of the
    Qwen3-VL conditioner. `reference_latents` is a list of normalized (1, 24, h, w)
    latents for an edit. Returns the normalized target latent (1, 24, h, w).
    """
    latent_h, latent_w = height // VAE_RATIO, width // VAE_RATIO
    n_text = prompt_embeds.shape[1]
    if text_token_tags is None:
        text_token_tags = torch.full((n_text,), TEXT_TAG, dtype=torch.long)
    refs = list(reference_latents or [])
    position_ids, token_tags, image_indices, text_indices, n_ref = build_packed_sequence(
        text_token_tags, [tuple(r.shape[-2:]) for r in refs], latent_h, latent_w)
    seq = position_ids.shape[0]

    gen = torch.Generator("cpu").manual_seed(int(seed))
    target = patchify(torch.randn((1, LATENT_CHANNELS, 1, latent_h, latent_w), generator=gen, dtype=torch.float32))
    rows = []
    for ref in refs:
        noise = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
        sigma = torch.tensor(1.0, dtype=torch.float32) - torch.tensor(REFERENCE_T, dtype=torch.float32)
        rows.append(patchify((1.0 - sigma) * ref.float() + sigma * noise))
    latents = torch.cat(rows + [target]).to(device)

    kwargs = dict(
        encoder_hidden_states=prompt_embeds.to(device),
        token_tags=token_tags.to(device),
        position_ids=position_ids.to(device),
        image_indices=image_indices.to(device),
        text_indices=text_indices.to(device),
    )
    for i in range(NUM_PDD_HEADS):
        unique_t, t_idx = row_timesteps(pdd_timestep(i), seq, image_indices, n_ref)
        velocity = model(hidden_states=latents[None], timestep=unique_t.to(device), timestep_indices=t_idx.to(device),
                         pdd_step_index=i, return_dict=False, **kwargs)[0]
        # Data-ward velocity, Euler in the "velocity" form the heads were distilled with.
        latents[n_ref:] = latents[n_ref:] + (pdd_sigma(i) - pdd_sigma(i + 1)) * velocity[0, n_ref:].float()
        if progress:
            progress(i + 1, NUM_PDD_HEADS)
    return unpatchify(latents[n_ref:].cpu(), latent_h, latent_w)
