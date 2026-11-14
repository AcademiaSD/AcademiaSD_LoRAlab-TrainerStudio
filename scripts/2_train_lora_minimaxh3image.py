# -*- coding: utf-8 -*-
"""
2_train_lora_minimaxh3image.py

LoRA trainer for MiniMax-H3-Image (33B single-stream DiT, NF4).

THE LoRA IS TRAINED ON MiniMax-H3, NOT ON MiniMax-H3-Image.

MiniMax-H3-Image is distilled to 6 sampling steps with one output head per step and no CFG.
Fine-tuning it directly was measured to fail both ways: plain flow matching turns its
high-noise heads into the dataset's average within ~100 steps (previews become blotches, even
at lr 1e-4), and keeping those heads anchored to the base model leaves it sharp but unable to
learn a face. MiniMax-H3-Image was fine-tuned from MiniMax-H3, which is NOT distilled, shares
its latent space and every module name a LoRA touches; a LoRA trained on MiniMax-H3 was shown
to carry an identity over to MiniMax-H3-Image cleanly. So:

* training: the MiniMax-H3 (video) DiT, NF4 (Models/MiniMax-H3-NF4/transformers), standard
  flow matching exactly as the MiniMax-H3 trainer packs a still image: [text | 1 silent audio
  row | video], logit-normal sigma with the resolution shift, data-ward velocity x0 - noise;
* latents: the MiniMax-H3-Image VAE (the pre-cache of the MiniMax-H3 Image card);
* previews: MiniMax-H3-Image itself, 6 steps, CFG 1, with the LoRA being trained attached;
* export: the original checkpoint layout, which loads on both models.

Memory: every frozen linear stays NF4-packed. `resident_blocks` blocks keep their packed bytes
on the GPU; the rest are uploaded just in time. Each block runs under gradient checkpointing,
so only ONE block ever holds its dequantized weights during backward.

El LoRA se entrena sobre MiniMax-H3 (sin destilar), con latentes del VAE de H3-Image, y las
previews salen de MiniMax-H3-Image. Entrenar directamente el modelo destilado o lo emborrona o
no aprende (medido).
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import gc
import json
import math
import random
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir, resolve_output_dir
import lora_options
import training_metrics
import minimaxh3image_model as M

STEP_WORD = t("Step")  # the GUI parses "<Step> N/Total" from the console

DEFAULTS = {
    "project_name": "",
    "trigger_word": "",
    "nf4_cache_dir": "Models/MiniMax-H3-Image-NF4",
    "nf4_repo": "AcademiaSD/MiniMax-H3-Image-NF4-for-LoRA-Training",
    # The DiT the LoRA trains on: MiniMax-H3 (video, not distilled). Only its transformers/ folder.
    # El DiT sobre el que se entrena: MiniMax-H3 (video, sin destilar). Solo su carpeta transformers/.
    "train_nf4_dir": "Models/MiniMax-H3-NF4",
    "train_nf4_repo": "AcademiaSD/MiniMax-H3-NF4",
    # "image" (default): the LoRA trains on MiniMax-H3-Image itself, one distilled step (and its
    # head) at a time, with `adapter_lora` frozen on top: a de-distillation adapter that absorbs what
    # plain flow matching does to the distilled heads, so the LoRA only learns the subject. Previews
    # and use are WITHOUT the adapter. Measured on a face dataset it gives clearly more likeness on
    # H3-Image than "video", where the LoRA trains on MiniMax-H3 (not distilled) and is transferred.
    # "image" (por defecto): entrena sobre H3-Image con un adaptador de des-destilacion congelado,
    # que no se usa al generar. "video": entrena sobre MiniMax-H3 y se transfiere (menos parecido).
    "train_dit": "image",
    # A path, or a file name inside <nf4_cache_dir>/adapter/ (downloaded from nf4_repo if missing).
    # "" or "none" trains without adapter. / Ruta, o nombre dentro de <nf4>/adapter/. "none" = sin adaptador.
    "adapter_lora": "MiniMaxH3Image_dedistill_adapter.safetensors",
    "adapter_strength": 1.5,
    "preview_with_adapter": False,
    "pdd_step_weights": [1, 1, 1, 1, 1, 1],
    "logit_normal_mu": 0.0,
    "logit_normal_std": 1.0,
    "sigma_resolution_shift": True,
    "cache_dir": "./Cache_Data/cached_data_minimaxh3image",
    "output_dir": "./Outputs/minimaxh3image_lora_output",
    "total_steps": 1500,
    "grad_accum_steps": 1,
    "lr": 4e-4,
    "lr_scheduler": "constant_with_warmup",
    "lr_cycles": 3,
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "lora_rank": 16,
    "lora_alpha": 32,
    "use_rslora": False,
    "use_loraplus": False,
    "loraplus_ratio": 16.0,
    # One step in N trains with the empty caption the pre-cache encodes (_neg_*). The model has no
    # CFG, so this is not about guidance: it keeps the concept from hanging on caption wording alone.
    # Un paso de cada N entrena con el caption vacio del pre-cache. El modelo no tiene CFG.
    "caption_dropout": 0.05,
    # token_refiner blocks are trained too, as in the MiniMax-H3 reference LoRAs.
    "lora_exclude_refiner": False,
    "lora_key_prefix": "diffusion_model.",
    "lora_save_raw_copy": False,
    # Relative weight of each of the 6 distilled steps when sampling the noise level.
    # Peso relativo de cada uno de los 6 pasos destilados al muestrear el nivel de ruido.
    # 0 = automatic from the card and the longest cached sequence. / 0 = automatico.
    "resident_blocks": 0,
    "vram_budget_gb": 0,
    "save_every": 200,
    "seed": 42,
    "preview_every": 0,
    "preview_caption_mode": "first",   # first | rotate | random
    "preview_seed": 42,
    # preview_width x preview_height if both are set; else preview_size as an AREA side
    # (1024 = about 1024 x 1024 px at the sample's own aspect); else the cached sample's size.
    # Ancho x alto si estan los dos; si no, preview_size como lado de AREA; si no, el de la muestra.
    "preview_size": 0,
    "preview_width": 0,
    "preview_height": 0,
    "preview_vae_device": "cuda",
    # Strength of the LoRA in the previews, like the LoRA strength in ComfyUI. Only the preview:
    # training and the saved LoRAs do not change. Can be changed while training.
    # Fuerza del LoRA en las previews, como en ComfyUI. No cambia el entrenamiento ni los LoRAs.
    "preview_lora_strength": 1.0,
    "debug_training": False,
}

CONFIG_PATH = os.environ.get("LORALAB_TRAIN_CONFIG") or "settings/train_settings_minimaxh3image.json"
CFG = {}
if os.path.isfile(CONFIG_PATH):
    with open(CONFIG_PATH, encoding="utf-8") as _fh:
        CFG = json.load(_fh)


def cfg(key):
    value = CFG.get(key, DEFAULTS[key])
    return DEFAULTS[key] if value is None or value == "" and not isinstance(DEFAULTS[key], str) else value


def cfg_bool(key):
    value = cfg(key)
    return value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)


NF4_DIR = str(resolve_model_dir(cfg("nf4_cache_dir"), DEFAULTS["nf4_cache_dir"]))
TRAIN_NF4_DIR = str(resolve_model_dir(cfg("train_nf4_dir"), DEFAULTS["train_nf4_dir"]))
# Derived from the project name, exactly as the pre-cache writes it: a cache_dir saved by an older
# version of the GUI pointed at the MiniMax-H3 (video) cache, made with the video VAE.
# Se deriva del nombre del proyecto, igual que el pre-cache: un cache_dir guardado por una version
# anterior apuntaba a la cache de MiniMax-H3 (video), hecha con el VAE de video.
_PROJECT = str(CFG.get("project_name") or "").strip()
CACHE_DIR = str(resolve_cache_dir("./cached_data_minimaxh3image_" + _PROJECT if _PROJECT else cfg("cache_dir"), DEFAULTS["cache_dir"]))
OUTPUT_DIR = str(resolve_output_dir(cfg("output_dir"), DEFAULTS["output_dir"]))
TOTAL_STEPS = int(cfg("total_steps"))
GRAD_ACCUM = max(1, int(cfg("grad_accum_steps")))
LR = float(cfg("lr"))
RANK = int(cfg("lora_rank"))
ALPHA = float(cfg("lora_alpha"))
USE_RSLORA = cfg_bool("use_rslora")
SEED = int(cfg("seed"))
SAVE_EVERY = int(cfg("save_every"))
PREVIEW_EVERY = int(cfg("preview_every"))
DEBUG = cfg_bool("debug_training")
TRAIN_ON_IMAGE = str(cfg("train_dit")).strip().lower() == "image"
DEVICE = "cuda"

STATE_DIR = os.path.join(OUTPUT_DIR, "resume")
STOP_REQUESTED = False
# True while the LoRA or the optimizer state is being written: a Stop then waits for it to finish
# instead of interrupting it half-way. Mientras se escribe el LoRA, Stop espera a que termine.
CRITICAL = False


@contextlib.contextmanager
def critical():
    global CRITICAL
    CRITICAL = True
    try:
        yield
    finally:
        CRITICAL = False

LORA_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


_PROGRESS_OPEN = [False]


def log(msg):
    # The step line is rewritten in place with "\r": anything else starts on a fresh line.
    # La linea de paso se reescribe con "\r": cualquier otro mensaje empieza en linea nueva.
    if _PROGRESS_OPEN[0]:
        print()
        _PROGRESS_OPEN[0] = False
    print(msg, flush=True)


def progress(line):
    print("\r" + line, end="", flush=True)
    _PROGRESS_OPEN[0] = True


_live_mtime = [None]


def reload_live_settings():
    """
    Live settings: when the GUI rewrites the JSON with Save JSON while training runs, the keys
    below apply on the next step. One getmtime per step. Rank, alpha, folders and the VRAM plan
    are fixed at start and still need Stop -> Resume.
    Ajustes en caliente: al guardar el JSON con el entrenamiento en marcha, estas claves se
    aplican en el paso siguiente.
    """
    global TOTAL_STEPS, LR, SAVE_EVERY, PREVIEW_EVERY
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return
    if _live_mtime[0] is None or mtime == _live_mtime[0]:
        _live_mtime[0] = _live_mtime[0] or mtime
        return
    _live_mtime[0] = mtime
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            fresh = json.load(fh)
    except (OSError, ValueError):
        return
    live = ("total_steps", "lr", "save_every", "max_grad_norm", "weight_decay", "caption_dropout", "preview_every",
            "preview_caption_mode", "preview_seed", "preview_size", "preview_width", "preview_height",
            "preview_vae_device", "preview_lora_strength")
    changed = [k for k in live if k in fresh and fresh[k] != CFG.get(k)]
    for key in changed:
        CFG[key] = fresh[key]
    if changed:
        TOTAL_STEPS, LR = int(cfg("total_steps")), float(cfg("lr"))
        SAVE_EVERY, PREVIEW_EVERY = int(cfg("save_every")), int(cfg("preview_every"))
        log("[LIVE] " + ", ".join("{}={}".format(k, CFG[k]) for k in changed))


def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def step_seed(step, salt=0):
    """splitmix64 of (seed, step): step N draws the same noise whether or not the run was resumed."""
    x = (SEED * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9 + salt * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0x7FFFFFFFFFFFFFFF


# =============================================================================
# DATA
# =============================================================================

def check_cache_vae(cache_dir):
    """The latents must come from the MiniMax-H3-Image VAE. A MiniMax-H3 (video) cache is refused."""
    vae = None
    try:
        with open(os.path.join(cache_dir, "cache_info.json"), encoding="utf-8") as fh:
            vae = json.load(fh).get("vae")
    except (OSError, ValueError):
        pass
    if not os.path.isfile(os.path.join(cache_dir, "cache_info.json")):
        raise RuntimeError(t("No MiniMax-H3-Image cache in {0}. Run Pre-Cache first.", cache_dir))
    if vae != "minimaxh3image":
        raise RuntimeError(t("This cache was not made with the MiniMax-H3-Image VAE ({0}). Run Pre-Cache from the "
                             "MiniMax-H3 Image card: it writes its own cache folder.", vae or "no cache_info.json"))


def load_entries(cache_dir):
    check_cache_vae(cache_dir)
    entries = []
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith("_video_latent.pt") or filename.startswith("_"):
            continue
        base = filename[: -len("_video_latent.pt")]
        embeds_path = os.path.join(cache_dir, base + "_prompt_root_prompt_embeds.pt")
        if not os.path.isfile(embeds_path):
            continue
        info = {}
        try:
            with open(os.path.join(cache_dir, base + "_info.json"), encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception:
            pass
        if str(info.get("kind", "image")) not in ("image", "edit"):
            continue  # clips and audio takes belong to the MiniMax-H3 trainer
        latent = torch.load(os.path.join(cache_dir, filename), map_location="cpu", weights_only=True).float()
        if latent.dim() == 5:
            if latent.shape[2] != 1:
                continue
            latent = latent[:, :, 0]
        latent = latent[..., : latent.shape[-2] // 2 * 2, : latent.shape[-1] // 2 * 2].contiguous()
        embeds = torch.load(embeds_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
        refs = []
        while os.path.isfile(os.path.join(cache_dir, "{}_ref{}_latent.pt".format(base, len(refs)))):
            ref = torch.load(os.path.join(cache_dir, "{}_ref{}_latent.pt".format(base, len(refs))),
                             map_location="cpu", weights_only=True).float()
            refs.append(ref.reshape(1, M.LATENT_CHANNELS, ref.shape[-2], ref.shape[-1]))
        tags_path = os.path.join(cache_dir, base + "_text_token_tags.pt")
        tags = (torch.load(tags_path, map_location="cpu", weights_only=True).long() if os.path.isfile(tags_path)
                else torch.full((embeds.shape[1],), M.TEXT_TAG, dtype=torch.long))
        if refs:
            log("[DATASET][!] " + t("{0}: edit pairs are not supported yet; trained as a plain image.", base))
            refs = []
        if TRAIN_ON_IMAGE:
            layout = M.build_packed_sequence(tags, [], latent.shape[-2], latent.shape[-1])
        else:
            layout = M.build_video_packed_sequence(embeds.shape[1], latent.shape[-2], latent.shape[-1], tags)
        entries.append({"name": base, "latent": latent, "embeds": embeds, "refs": refs, "tags": tags,
                        "layout": layout, "prompt": info.get("prompt", ""), "tokens": int(layout[0].shape[0])})
    if not entries:
        raise RuntimeError(t("No valid entries found in the cache.") + " " + cache_dir)
    return entries


def load_empty_caption(cache_dir):
    """Embedding of the empty caption (written by the pre-cache as _neg_*), or None."""
    path = os.path.join(cache_dir, "_neg_root_prompt_embeds.pt")
    if not os.path.isfile(path):
        return None
    embeds = torch.load(path, map_location="cpu", weights_only=True).to(torch.bfloat16)
    tags = torch.full((embeds.shape[1],), M.TEXT_TAG, dtype=torch.long)
    return {"embeds": embeds, "tags": tags}


def video_tokens(entry):
    return (entry["latent"].shape[-2] // M.PATCH[1]) * (entry["latent"].shape[-1] // M.PATCH[2])


# =============================================================================
# MODEL
# =============================================================================

def block_checkpoint(fn, *args):
    # Non-reentrant: the forward keeps no graph, and backward recomputes one block
    # at a time, so only that block's dequantized weights are alive.
    return torch_checkpoint(fn, *args, use_reentrant=False)


VRAM_SLACK_GB = 2.0
# The LoRA in training: fp32 weights + grads + the two AdamW moments, 16 bytes per parameter, and
# ~5.4 M parameters per unit of rank. The other constants were measured at rank 16.
# El LoRA en entrenamiento cuesta 16 bytes por parametro; el resto se midio con rank 16.
VRAM_LORA_GB_PER_RANK = 0.08
# When a step's peak plus what other programs use goes past the card, Windows moves memory to
# "shared GPU memory" (system RAM over PCIe) instead of failing, and a step gets 2-3 times slower.
# This is the margin the guard keeps below the card's size.
# Si el pico de un paso mas lo de otros programas pasa de la VRAM, Windows usa la memoria compartida.
VRAM_GUARD_GB = 0.1
VRAM_BASE_GB = 4.2
VRAM_PER_TOKEN_GB = 0.0006
VRAM_BLOCK_GB = 0.32


def plan_resident_blocks(num_blocks, max_tokens):
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    budget = min(float(cfg("vram_budget_gb") or 0) or total, total) - VRAM_SLACK_GB
    # Measured on an RTX 5080 (allocated peak, GiB, rank 16): 1,104 tokens -> 7.07 with 8 resident
    # blocks and 12.03 with 24; 344 tokens -> 12.01 with 25. A block costs 0.31, the rest is the
    # non-block weights, the dequantized block of the backward and the activations. SLACK covers
    # what the allocator peak does not see: the CUDA context, the desktop, the browser running the
    # GUI and the allocator's fragmentation. 1.2 GB was too little on a 16 GB card under Windows
    # (30 blocks at 1,145 tokens spilled into shared memory and ran 3 times slower); 2.0 matches the
    # runs measured there. vram_guard() corrects whatever is still off once training runs.
    # Medido en una RTX 5080. La GUI (perfiles de VRAM) usa las mismas constantes.
    floor = VRAM_BASE_GB + VRAM_PER_TOKEN_GB * max_tokens + VRAM_LORA_GB_PER_RANK * (RANK - 16)
    fit = int(max(0, min(num_blocks, (budget - floor) // VRAM_BLOCK_GB)))
    # A manual value wins, even above the plan: whoever types one is measuring.
    # Un valor manual manda, incluso por encima del plan: quien lo escribe esta midiendo.
    manual = int(cfg("resident_blocks") or 0)
    if manual > 0:
        if manual > fit:
            log("[VRAM][!] " + t("resident_blocks={0} is above the {1} the VRAM plan fits; it may run out of VRAM.", manual, fit))
        return min(num_blocks, manual)
    return fit


def vram_guard(model):
    """
    After a step, checks whether the step's peak plus other programs' VRAM got within
    VRAM_GUARD_GB of the card's size. If so, moves resident blocks back to RAM (last ones
    first), two at a time: a block streamed from RAM costs
    a fraction of a second per step, spilling into Windows' shared GPU memory costs 2-3 times the
    step. Works on the plan's value and on a hand-typed one alike. Returns True when it moved any.
    Si la tarjeta se queda sin VRAM libre, devuelve bloques residentes a la RAM antes de que
    Windows empiece a usar la memoria compartida.
    """
    # mem_get_info()'s free VRAM leaves out what PyTorch keeps cached between steps, so right
    # after a step the card always looks full: that number is useless here. What matters is the
    # step's PEAK (max_memory_reserved) plus what other programs hold, against the card's size.
    # La VRAM "libre" no cuenta la cache de PyTorch: se compara el pico del paso con la tarjeta.
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    others = max(0, total - free - reserved)
    peak = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    over = (peak + others - total) / 2**30
    model._vram_seen = max(getattr(model, "_vram_seen", -1e9), over)
    resident = getattr(model, "_resident_blocks", 0)
    # Only an overflow seen on two checks in a row counts: a single spike (the first steps'
    # one-off allocations, a bigger bucket) must not cost resident blocks for the whole run.
    # Solo cuenta si se repite en dos comprobaciones seguidas: un pico suelto no.
    strikes = getattr(model, "_vram_strikes", 0) + 1 if over >= -VRAM_GUARD_GB else 0
    model._vram_strikes = strikes
    if strikes < 2 or resident <= 0:
        return False
    model._vram_strikes = 0
    move = min(resident, 2)
    blocks = list(model.transformer_blocks)
    for block in blocks[resident - move:resident]:
        for mod in block.modules():
            if isinstance(mod, M.LazyNF4Linear):
                mod.make_offloaded()
    model._resident_blocks = resident - move
    gc_cuda()
    log("[VRAM][!] " + t("The step peaked at {0} GB and other programs use {1} GB, on a {2} GB card: {3} resident blocks moved back to RAM, {4} remain. Set Resident Blocks to {4} or less next time.",
                        "{:.1f}".format(peak / 2**30), "{:.1f}".format(others / 2**30), "{:.1f}".format(total / 2**30),
                        move, model._resident_blocks))
    return True


# Margin kept free when suggesting MORE resident blocks: steps differ in size (buckets, captions)
# and a preview adds its own peak. / Margen al recomendar mas bloques residentes.
VRAM_ADVICE_MARGIN_GB = 1.0


def vram_advice(model):
    """
    Once the first steps have run: if their highest peak left room on the card, says how many
    more resident blocks would fit. Each resident block saves its upload from RAM on every step.
    Tras los primeros pasos: si sobra VRAM, dice cuantos bloques residentes mas caben.
    """
    seen = getattr(model, "_vram_seen", None)
    resident = getattr(model, "_resident_blocks", 0)
    total_blocks = len(model.transformer_blocks)
    if seen is None or resident >= total_blocks:
        return
    room = -seen - VRAM_ADVICE_MARGIN_GB
    more = min(total_blocks - resident, int(room // VRAM_BLOCK_GB))
    if more >= 2:
        log("[VRAM] " + t("The steps leave {0} GB of VRAM unused: about {1} more resident blocks fit. Set Resident Blocks to {2} next time for faster steps.",
                          "{:.1f}".format(-seen), more, resident + more))


def ensure_nf4():
    """
    Downloads AcademiaSD/MiniMax-H3-Image-NF4-for-LoRA-Training if the transformer or the VAE is
    missing. The pre-cache normally fetched it already; this covers a cache made elsewhere.
    Descarga el NF4 si falta el transformer o el VAE (normalmente ya lo bajo el pre-cache).
    """
    if all(os.path.isfile(os.path.join(NF4_DIR, sub, "config.json")) for sub in ("transformers", "vae")) \
            and os.path.isfile(os.path.join(NF4_DIR, "transformers", "index.json")):
        return
    repo = str(cfg("nf4_repo"))
    log("[DOWNLOAD] MiniMax-H3-Image NF4 ({}, ~19 GB) -> {}".format(repo, NF4_DIR))
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars
    enable_progress_bars()
    os.makedirs(NF4_DIR, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=NF4_DIR, token=_hf_token())
    log("[DOWNLOAD] OK")


def _hf_token():
    try:
        with open(os.path.join("settings", "HF_token.json"), encoding="utf-8") as fh:
            return (json.load(fh).get("token") or "").strip() or None
    except Exception:
        return None


def ensure_train_dit():
    """
    The MiniMax-H3 (video) DiT the LoRA trains on: ONLY transformers/ of AcademiaSD/MiniMax-H3-NF4
    (~17 GB). No video VAE, no audio VAE; the text encoder was fetched by the pre-cache.
    Solo transformers/ del repo NF4 de MiniMax-H3 (~17 GB).
    """
    if os.path.isfile(os.path.join(TRAIN_NF4_DIR, "transformers", "index.json")):
        return
    repo = str(cfg("train_nf4_repo"))
    log("[DOWNLOAD] " + t("MiniMax-H3 DiT the LoRA trains on ({0}/transformers, ~17 GB) -> {1}", repo, TRAIN_NF4_DIR))
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars
    enable_progress_bars()
    os.makedirs(TRAIN_NF4_DIR, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=TRAIN_NF4_DIR, allow_patterns=["transformers/*"], token=_hf_token())
    log("[DOWNLOAD] OK")


def ensure_adapter():
    """
    The de-distillation adapter for train_dit "image": an absolute/relative path, or a file name in
    <nf4_cache_dir>/adapter/, downloaded from nf4_repo (adapter/<name>) when it is missing.
    El adaptador: ruta o nombre dentro de <nf4>/adapter/; si falta se descarga de nf4_repo.
    """
    name = str(cfg("adapter_lora") or "").strip()
    if not name or name.lower() == "none":
        return ""
    if os.path.isfile(name):
        return name
    local = os.path.join(NF4_DIR, "adapter", os.path.basename(name))
    if os.path.isfile(local):
        return local
    repo = str(cfg("nf4_repo"))
    log("[DOWNLOAD] " + t("De-distillation adapter ({0}/adapter/{1}) -> {2}", repo, os.path.basename(name), local))
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id=repo, filename="adapter/" + os.path.basename(name), local_dir=NF4_DIR, token=_hf_token())
    except Exception as exc:
        raise RuntimeError(t("De-distillation adapter not found: {0} ({1})", local, exc))
    log("[DOWNLOAD] OK")
    return local


def build_model(max_tokens):
    ensure_nf4()
    if not TRAIN_ON_IMAGE:
        ensure_train_dit()
    t0 = time.time()
    model = M.load_dit_lowvram(NF4_DIR if TRAIN_ON_IMAGE else TRAIN_NF4_DIR, DEVICE)
    model.requires_grad_(False)
    exclude_refiner = cfg_bool("lora_exclude_refiner")
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, M.LazyNF4Linear) or not name.endswith(LORA_SUFFIXES):
            continue
        if not (name.startswith("transformer_blocks.") or (name.startswith("token_refiner.") and not exclude_refiner)):
            continue
        mod.add_lora(RANK, ALPHA, USE_RSLORA, DEVICE)
        targets.append(name)
    blocks = list(model.transformer_blocks)
    resident = plan_resident_blocks(len(blocks), max_tokens)
    for block in blocks[:resident]:
        for mod in block.modules():
            if isinstance(mod, M.LazyNF4Linear):
                mod.make_resident(DEVICE)
    model._resident_blocks = resident
    adapter = ensure_adapter() if TRAIN_ON_IMAGE else ""
    if adapter:
        n = M.add_frozen_lora(model, adapter, float(cfg("adapter_strength")), DEVICE)
        log("[ADAPTER] {} frozen on {} modules (strength {})".format(os.path.basename(adapter), n, cfg("adapter_strength")))
    model.gradient_checkpointing = True
    model._gradient_checkpointing_func = block_checkpoint
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("[MODEL] " + ("MiniMax-H3-Image DiT NF4 (+ de-distillation adapter)" if TRAIN_ON_IMAGE else "MiniMax-H3 (video) DiT NF4, trained for MiniMax-H3-Image") + " | {} LoRA modules, {:,} trainable params | {} of {} blocks resident | {:.0f}s | VRAM {:.2f} GB".format(
        len(targets), trainable, resident, len(blocks), time.time() - t0, torch.cuda.memory_allocated() / 2**30))
    return model


def lora_state(model):
    """{diffusers module name + .lora_A/.lora_B.weight: tensor}, raw (no scaling baked)."""
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, M.LazyNF4Linear) and mod.lora_A is not None:
            state[name + ".lora_A.weight"] = mod.lora_A.detach().float().cpu()
            state[name + ".lora_B.weight"] = mod.lora_B.detach().float().cpu()
    return state


def load_lora_state(model, state):
    mods = dict(model.named_modules())
    loaded = 0
    for key, tensor in state.items():
        name, part = key.rsplit(".lora_", 1)
        mod = mods.get(name)
        if mod is None or getattr(mod, "lora_A", None) is None:
            continue
        param = mod.lora_A if part.startswith("A") else mod.lora_B
        param.data.copy_(tensor.to(param.device, param.dtype))
        loaded += 1
    return loaded


# =============================================================================
# EXPORT (original checkpoint layout, the one ComfyUI loads)
# =============================================================================

def _fuse_qkv(q_A, q_B, k_A, k_B, v_A, v_B):
    """Three independent LoRAs (to_q/to_k/to_v) -> one exact LoRA for the fused qkv_proj, order [q; k; v]."""
    ranks = (q_A.shape[0], k_A.shape[0], v_A.shape[0])
    outs = (q_B.shape[0], k_B.shape[0], v_B.shape[0])
    B = torch.zeros((sum(outs), sum(ranks)), dtype=torch.float32)
    row = col = 0
    for part, out, rank in zip((q_B, k_B, v_B), outs, ranks):
        B[row:row + out, col:col + rank] = part.float()
        row, col = row + out, col + rank
    # [q; k; v] in three blocks: the layout of the checkpoints ComfyUI loads (its MiniMax-H3
    # attention does qkv_proj(x).split(heads * head_dim)). Verified on the ComfyUI H3-Image
    # checkpoint: its qkv_proj row norms match [to_q; to_k; to_v] (correlation 1.0) and not a
    # per-head interleave (0.21). LoRAs exported before `qkv_layout` was written used that
    # interleave; minimaxh3image_model.lora_file_to_modules still reads them.
    # [q; k; v] en tres bloques, como el checkpoint que carga ComfyUI.
    return torch.cat([q_A.float(), k_A.float(), v_A.float()], dim=0), B


def export_lora(model, path):
    """
    diffusers layout -> original checkpoint layout: transformer_blocks/refiner_blocks -> blocks,
    to_q/to_k/to_v fused into attn.qkv_proj, to_out.0 -> out_proj, ff.net.0.proj -> mlp.fc1 with
    the SwiGLU halves swapped back ([value; gate] -> [gate; value]), ff.net.2 -> mlp.fc2. The
    alpha/rank scaling is baked into lora_B.
    """
    prefix = str(cfg("lora_key_prefix") or "")
    scale = ALPHA / (math.sqrt(RANK) if USE_RSLORA else float(max(1, RANK)))
    raw = {k: (v * scale if ".lora_B." in k else v) for k, v in lora_state(model).items()}
    state = {}
    for base in sorted({k[: -len("attn.to_q.lora_A.weight")] for k in raw if k.endswith("attn.to_q.lora_A.weight")}):
        new = prefix + base.replace("transformer_blocks", "blocks").replace("refiner_blocks", "blocks")
        A, B = _fuse_qkv(*[raw[base + "attn.to_{}.lora_{}.weight".format(m, ab)] for m in "qkv" for ab in "AB"])
        state[new + "attn.qkv_proj.lora_A.weight"] = A
        state[new + "attn.qkv_proj.lora_B.weight"] = B
        state[new + "attn.out_proj.lora_A.weight"] = raw[base + "attn.to_out.0.lora_A.weight"]
        state[new + "attn.out_proj.lora_B.weight"] = raw[base + "attn.to_out.0.lora_B.weight"]
        state[new + "mlp.fc1.lora_A.weight"] = raw[base + "ff.net.0.proj.lora_A.weight"]
        value, gate = raw[base + "ff.net.0.proj.lora_B.weight"].chunk(2, dim=0)
        state[new + "mlp.fc1.lora_B.weight"] = torch.cat([gate, value], dim=0)
        state[new + "mlp.fc2.lora_A.weight"] = raw[base + "ff.net.2.lora_A.weight"]
        state[new + "mlp.fc2.lora_B.weight"] = raw[base + "ff.net.2.lora_B.weight"]
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}
    trigger = str(cfg("trigger_word") or "").strip()
    project = str(cfg("project_name") or "").strip()
    meta = {
        "format": "minimaxh3_lora", "lora_key_prefix": prefix, "baked_scaling": "{:.6f}".format(scale),
        "qkv_fused": "true", "qkv_layout": "q_k_v", "swiglu_fc1_halves_swapped": "true",
        "trained_with": "AcademiaSD LoRAlab MiniMax-H3-Image (trained on the {} DiT{})".format(
            "MiniMax-H3-Image" if TRAIN_ON_IMAGE else "MiniMax-H3",
            ", adapter " + os.path.basename(str(cfg("adapter_lora"))) if TRAIN_ON_IMAGE and str(cfg("adapter_lora") or "").strip().lower() not in ("", "none") else ""),
        "ss_sd_model_name": "MiniMax-H3-Image", "ss_base_model_version": "MiniMax-H3-Image",
        "ss_network_dim": str(RANK), "ss_network_alpha": str(lora_options.export_alpha(ALPHA, RANK, USE_RSLORA)),
        "ss_learning_rate": str(LR), "ss_max_train_steps": str(TOTAL_STEPS), "ss_seed": str(SEED),
    }
    if trigger:
        meta.update({"trigger_word": trigger, "ss_tag_frequency": json.dumps({"dataset": {trigger: 1}})})
    if project or trigger:
        meta["ss_output_name"] = project or trigger
    save_file(state, path, metadata=meta)
    if cfg_bool("lora_save_raw_copy"):
        save_file({prefix + k: v.to(torch.bfloat16) for k, v in raw.items()}, path.replace(".safetensors", ".rawkeys.safetensors"))
    return len(state)


def save_resume(model, optimizer, step, elapsed):
    os.makedirs(STATE_DIR, exist_ok=True)
    # weights -> optimizer -> step file; the step file only if both succeeded.
    tmp = os.path.join(STATE_DIR, "lora.safetensors.tmp")
    save_file(lora_state(model), tmp)
    os.replace(tmp, os.path.join(STATE_DIR, "lora.safetensors"))
    tmp = os.path.join(STATE_DIR, "optimizer.pt.tmp")
    torch.save(optimizer.state_dict(), tmp)
    os.replace(tmp, os.path.join(STATE_DIR, "optimizer.pt"))
    tmp = os.path.join(STATE_DIR, "step.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"step": step, "elapsed": elapsed, "rank": RANK, "alpha": ALPHA}, fh)
    os.replace(tmp, os.path.join(STATE_DIR, "step.json"))


def try_resume(model, optimizer):
    path = os.path.join(STATE_DIR, "step.json")
    if not os.path.isfile(path):
        return 0, 0.0
    with open(path, encoding="utf-8") as fh:
        info = json.load(fh)
    if int(info.get("rank", RANK)) != RANK:
        log("[RESUME] " + t("Saved rank {0} differs from the configured rank {1}: starting from zero.", info.get("rank"), RANK))
        return 0, 0.0
    load_lora_state(model, load_file(os.path.join(STATE_DIR, "lora.safetensors")))
    try:
        optimizer.load_state_dict(torch.load(os.path.join(STATE_DIR, "optimizer.pt"), map_location=DEVICE))
    except Exception as exc:
        log("[RESUME][WARN] optimizer: {}".format(exc))
    return int(info.get("step", 0)), float(info.get("elapsed", 0.0))


# =============================================================================
# STEP
# =============================================================================

def forward_velocity(model, entry, noised_video, noised_audio, timestep, empty=None):
    """One forward of the MiniMax-H3 DiT on a still image. Returns the velocity of the video rows."""
    embeds = entry["embeds"]
    position_ids, token_tags, video_indices, audio_indices, text_indices = entry["layout"]
    if empty is not None:
        embeds = empty["embeds"]
        position_ids, token_tags, video_indices, audio_indices, text_indices = M.build_video_packed_sequence(
            embeds.shape[1], entry["latent"].shape[-2], entry["latent"].shape[-1], empty["tags"])
    seq = position_ids.shape[0]
    out = model(
        hidden_states=noised_video[None].to(DEVICE), audio_hidden_states=noised_audio[None].to(DEVICE),
        encoder_hidden_states=embeds.to(DEVICE),
        timestep=torch.tensor([timestep], dtype=torch.float32, device=DEVICE),
        timestep_indices=torch.zeros(seq, dtype=torch.long, device=DEVICE),
        token_tags=token_tags.to(DEVICE), position_ids=position_ids.to(DEVICE),
        video_indices=video_indices.to(DEVICE), audio_indices=audio_indices.to(DEVICE),
        text_indices=text_indices.to(DEVICE), return_dict=False,
    )[0]
    return out[0].float()


def forward_velocity_image(model, entry, noised, step_index, empty=None):
    """One forward of MiniMax-H3-Image at distilled step `step_index` (its own output head)."""
    embeds, layout = entry["embeds"], entry["layout"]
    if empty is not None:
        embeds = empty["embeds"]
        layout = M.build_packed_sequence(empty["tags"], [], entry["latent"].shape[-2], entry["latent"].shape[-1])
    position_ids, token_tags, image_indices, text_indices, n_ref = layout
    unique_t, t_idx = M.row_timesteps(M.pdd_timestep(step_index), position_ids.shape[0], image_indices, n_ref)
    out = model(
        hidden_states=noised[None].to(DEVICE), encoder_hidden_states=embeds.to(DEVICE),
        timestep=unique_t.to(DEVICE), timestep_indices=t_idx.to(DEVICE), token_tags=token_tags.to(DEVICE),
        position_ids=position_ids.to(DEVICE), image_indices=image_indices.to(DEVICE),
        text_indices=text_indices.to(DEVICE), pdd_step_index=step_index, return_dict=False,
    )[0]
    return out[0].float()


def train_step_image(model, entry, step, empty_caption=None):
    """
    Flow matching on MiniMax-H3-Image at one of its 6 distilled noise levels, through that step's
    head: noised = (1 - sigma) x0 + sigma noise, target = x0 - noise.
    """
    generator = torch.Generator("cpu").manual_seed(step_seed(step))
    drop = (empty_caption is not None
            and float(torch.rand(1, generator=generator)) < float(cfg("caption_dropout")))
    weights = torch.tensor([float(w) for w in cfg("pdd_step_weights")][: M.NUM_PDD_HEADS])
    step_index = int(torch.multinomial(weights, 1, generator=generator))
    sigma = M.pdd_sigma(step_index)
    x0 = M.patchify(entry["latent"])
    noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32)
    noised = (1.0 - sigma) * x0 + sigma * noise
    target = (x0 - noise).to(DEVICE)
    pred = forward_velocity_image(model, entry, noised, step_index, empty_caption if drop else None)
    return F.mse_loss(pred, target), sigma


def train_step(model, entry, step, empty_caption=None):
    if TRAIN_ON_IMAGE:
        return train_step_image(model, entry, step, empty_caption)
    """
    Flow matching on MiniMax-H3, as its own trainer does for a still: noised = (1 - sigma) x0 +
    sigma noise, t = 1 - sigma (t = 1 is clean), target = x0 - noise. The silent audio row is
    noised with the same sigma and gets no loss.
    """
    generator = torch.Generator("cpu").manual_seed(step_seed(step))
    drop = (empty_caption is not None
            and float(torch.rand(1, generator=generator)) < float(cfg("caption_dropout")))
    sigma = M.sample_training_sigma(video_tokens(entry), generator, float(cfg("logit_normal_mu")),
                                    float(cfg("logit_normal_std")), cfg_bool("sigma_resolution_shift"))
    x0 = M.patchify(entry["latent"])
    noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32)
    noised = (1.0 - sigma) * x0 + sigma * noise
    audio = sigma * torch.randn((1, M.AUDIO_CHANNELS), generator=generator, dtype=torch.float32)
    target = (x0 - noise).to(DEVICE)          # data-ward velocity, the H3 convention
    pred = forward_velocity(model, entry, noised, audio, 1.0 - sigma, empty_caption if drop else None)
    return F.mse_loss(pred, target), sigma


@torch.no_grad()
def run_preview(model, entries, step):
    mode = str(cfg("preview_caption_mode")).lower()
    count = step // max(1, PREVIEW_EVERY)
    entry = entries[0] if mode == "first" else (entries[count % len(entries)] if mode.startswith("rotate")
                                                 else random.Random(step_seed(step, 7)).choice(entries))
    height, width = entry["latent"].shape[-2] * M.VAE_RATIO, entry["latent"].shape[-1] * M.VAE_RATIO
    if int(cfg("preview_width") or 0) > 0 and int(cfg("preview_height") or 0) > 0 and not entry["refs"]:
        width, height = int(cfg("preview_width")), int(cfg("preview_height"))
    elif int(cfg("preview_size") or 0) > 0:
        scale = int(cfg("preview_size")) / math.sqrt(width * height)
        width, height = width * scale, height * scale
    width, height = M.canvas_size(width, height)
    t0 = time.time()
    strength = float(cfg("preview_lora_strength") if cfg("preview_lora_strength") not in (None, "") else 1.0)
    # The preview is rendered by MiniMax-H3-Image, the model the LoRA is for, with the LoRA being
    # trained attached (shared tensors, no copy). Its NF4 weights stay packed in RAM and are
    # uploaded layer by layer: it costs ~3 GB of VRAM on top of training, and is freed afterwards.
    # La preview la hace MiniMax-H3-Image con el LoRA en entrenamiento enganchado, sin copiarlo.
    if TRAIN_ON_IMAGE:
        # The model in training IS MiniMax-H3-Image: the preview shows it as it will be used,
        # without the adapter (unless preview_with_adapter).
        M.set_frozen_enabled(model, cfg_bool("preview_with_adapter"))
        try:
            with lora_strength(model, strength):
                    latent = M.sample(model, entry["embeds"], width, height,
                                  seed=(lambda v: v if v >= 0 else step_seed(step, 11) % 2**31)(int(cfg("preview_seed"))),
                                  text_token_tags=entry["tags"], device=DEVICE)
        finally:
            M.set_frozen_enabled(model, True)
            gc_cuda()
    else:
        image_model = M.load_dit_lowvram(NF4_DIR, DEVICE)
        try:
            shared = M.attach_lora(image_model, model)
            if DEBUG:
                log("[PREVIEW] LoRA modules attached to MiniMax-H3-Image: {}".format(shared))
            for mod in image_model.modules():
                if isinstance(mod, M.LazyNF4Linear) and mod.lora_A is not None:
                    mod.lora_scale = mod.lora_scale * strength
            latent = M.sample(image_model, entry["embeds"], width, height,
                              seed=(lambda v: v if v >= 0 else step_seed(step, 11) % 2**31)(int(cfg("preview_seed"))),
                              text_token_tags=entry["tags"], device=DEVICE)
        finally:
            del image_model
            gc_cuda()
    vae_device = str(cfg("preview_vae_device") or "cuda").lower()
    vae = M.load_vae(NF4_DIR, vae_device)
    try:
        image = M.decode_latent(vae, latent)
    finally:
        del vae
        gc_cuda()
    path = os.path.join(OUTPUT_DIR, "preview_step_{}.png".format(step))
    image.save(path)
    log("[PREVIEW] {} | {}x{} | {} {} | CFG 1 | LoRA x{:g} | {:.0f}s | {}".format(
        os.path.basename(path), width, height, M.NUM_PDD_HEADS, t("steps"), strength, time.time() - t0, entry["prompt"][:80]))


@contextlib.contextmanager
def lora_strength(model, strength):
    """Multiplies the LoRA scale of every module by `strength` and puts it back afterwards."""
    mods = [m for m in model.modules() if isinstance(m, M.LazyNF4Linear) and m.lora_A is not None]
    saved = [m.lora_scale for m in mods]
    for m in mods:
        m.lora_scale = m.lora_scale * strength
    try:
        yield
    finally:
        for m, v in zip(mods, saved):
            m.lora_scale = v


def fmt_time(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return "{} Hours {} Minutes / {} Horas {} Minutos".format(hours, minutes, hours, minutes)


def main():
    global STOP_REQUESTED
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("=" * 78)
    if TRAIN_ON_IMAGE:
        log("  MINIMAX-H3-IMAGE LoRA TRAINER - trains on MiniMax-H3-Image + de-distillation adapter")
        log("  Train DiT : {} (MiniMax-H3-Image) | adapter {} x{}".format(NF4_DIR, cfg("adapter_lora") or "none", cfg("adapter_strength")))
    else:
        log("  MINIMAX-H3-IMAGE LoRA TRAINER - trains on the MiniMax-H3 DiT, previews on MiniMax-H3-Image")
        log("  Train DiT : {} (MiniMax-H3)".format(TRAIN_NF4_DIR))
    log("  Image NF4 : {} (VAE, previews)".format(NF4_DIR))
    log("  Cache dir : {}".format(CACHE_DIR))
    log("  Output dir: {}".format(OUTPUT_DIR))
    log("  Steps {} | LR {} | rank {} alpha {} | accum {}".format(TOTAL_STEPS, LR, RANK, ALPHA, GRAD_ACCUM))
    log("=" * 78)

    # Stop interrupts whatever is running right away: a step (~5 s) or a preview (a 2048 px one
    # takes over 2 minutes and used to run to the end first). Only the writing of the LoRA and the
    # optimizer state is never cut: the stop then happens right after it.
    # Stop interrumpe al momento el paso o la preview en curso; solo espera si se esta guardando.
    def handle_signal(signum, frame):
        global STOP_REQUESTED
        first = not STOP_REQUESTED
        STOP_REQUESTED = True
        if first:
            log("[STOP] Stop signal received, stopping now. / Senal de parada recibida, deteniendo ya.")
        if not CRITICAL:
            raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, handle_signal)
            except Exception:
                pass

    entries = load_entries(CACHE_DIR)
    max_tokens = max(e["tokens"] for e in entries)
    edits = sum(1 for e in entries if e["refs"])
    empty_caption = load_empty_caption(CACHE_DIR)
    log("[DATASET] {} samples ({} edit pairs) | longest sequence {} tokens | caption dropout {}".format(
        len(entries), edits, max_tokens, cfg("caption_dropout") if empty_caption else "off (no empty caption in the cache)"))

    model = build_model(max_tokens)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg_bool("use_loraplus"):
        named = [(n + (".lora_B" if n.endswith("lora_B") else ""), p) for n, p in model.named_parameters()]
        groups = lora_options.loraplus_groups(named, float(cfg("loraplus_ratio")))
    else:
        groups = [{"params": params, "lr_mult": 1.0}]
    optimizer = torch.optim.AdamW(groups, lr=LR, weight_decay=float(cfg("weight_decay")), betas=(0.9, 0.999))
    start_step, elapsed_before = try_resume(model, optimizer)
    if start_step:
        log("[RESUME] " + t("Resuming from step {0}", start_step))
    metrics = training_metrics.MetricsLog(OUTPUT_DIR, start_step)
    warmup = int(cfg("warmup_steps"))
    name = "MiniMaxH3Image_LoRA"

    run_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    # The last step whose optimizer update is applied: what a Stop saves and Resume restarts from.
    # With grad accumulation the micro-steps after it only hold gradients, which a stop discards.
    last_update = start_step
    avg_time, running_loss, gnorm = 0.0, 0.0, 0.0
    log("\n" + t("STARTING TRAINING! {n} {kind} in {b} buckets.", n=len(entries), kind=t("images"),
                 b=len({tuple(e["latent"].shape[-2:]) for e in entries})))
    try:
        while step < TOTAL_STEPS:
            reload_live_settings()
            if STOP_REQUESTED or step >= TOTAL_STEPS:
                break
            epoch, pos = divmod(step, len(entries))
            order = list(range(len(entries)))
            random.Random(step_seed(epoch, 3)).shuffle(order)
            entry = entries[order[pos]]
            t0 = time.time()
            factor = lora_options.lr_factor(str(cfg("lr_scheduler")), step // GRAD_ACCUM, max(1, TOTAL_STEPS // GRAD_ACCUM),
                                            warmup, float(cfg("min_lr_ratio")), int(cfg("lr_cycles")))
            for group in optimizer.param_groups:
                group["lr"] = LR * factor * group.get("lr_mult", 1.0)
                group["weight_decay"] = float(cfg("weight_decay"))
            loss, sigma = train_step(model, entry, step, empty_caption)
            if not torch.isfinite(loss):
                raise RuntimeError(t("Non-finite loss at step {0}", step + 1))
            (loss / GRAD_ACCUM).backward()
            step += 1
            # The norm only exists when the optimizer steps: in between, the last one is shown.
            if step % GRAD_ACCUM == 0:
                with critical():
                    gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg("max_grad_norm"))))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    last_update = step
            dt = time.time() - t0
            if 2 < step - start_step <= 20 or step % max(1, PREVIEW_EVERY) == 1:
                vram_guard(model)
                if step - start_step == 20:
                    vram_advice(model)
            step_loss = float(loss)
            running_loss += step_loss
            avg_time = dt if avg_time == 0 else 0.1 * dt + 0.9 * avg_time
            eta_s = max(0, TOTAL_STEPS - step) * avg_time
            pct = min(1.0, step / max(1, TOTAL_STEPS))
            bar = "\u2588" * int(pct * 20) + "\u2591" * (20 - int(pct * 20))
            progress("{} {:4d}/{} [{}] {:5.1f}% | Loss {:.4f} | gnorm {:.3f} | lr {:.2e} | {:.2f}s/it | ETA {:02d}:{:02d}:{:02d}".format(
                STEP_WORD, step, TOTAL_STEPS, bar, pct * 100, running_loss / max(1, step - start_step), gnorm, LR * factor,
                avg_time, int(eta_s // 3600), int(eta_s % 3600 // 60), int(eta_s % 60)))
            metrics.log(step, step_loss, LR * factor, gnorm, dt, TOTAL_STEPS)
            if DEBUG:
                log("[DEBUG] step {} | sigma {:.4f} | loss {:.4f} | {:.2f}s | VRAM peak {:.2f} GB".format(
                    step, sigma, step_loss, dt, torch.cuda.max_memory_allocated() / 2**30))

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0 and step < TOTAL_STEPS and step == last_update:
                log(t("Saving checkpoint state at step {n}...", n=step))
                with critical():
                    keys = export_lora(model, os.path.join(OUTPUT_DIR, "{}_step_{}.safetensors".format(name, step)))
                    save_resume(model, optimizer, step, elapsed_before + time.time() - run_start)
                log("[SAVE] {}_step_{}.safetensors ({} keys)".format(name, step, keys))
            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0 and not STOP_REQUESTED:
                try:
                    run_preview(model, entries, step)
                except Exception as exc:
                    log("[PREVIEW][WARN] {}".format(exc))
                    gc_cuda()
                # The preview's own peak is not a training step's: the guard must not see it.
                torch.cuda.reset_peak_memory_stats()

    except KeyboardInterrupt:
        pass

    total = elapsed_before + time.time() - run_start
    finished = step >= TOTAL_STEPS and not STOP_REQUESTED
    # A second Stop while this is written must not leave a half-written LoRA behind.
    # Un segundo Stop mientras se guarda no debe dejar un LoRA a medias.
    with critical():
        save_resume(model, optimizer, step if finished else last_update, total)
        if finished:
            keys = export_lora(model, os.path.join(OUTPUT_DIR, "MiniMaxH3Image_FINAL_LoRA.safetensors"))
            log("[SAVE] MiniMaxH3Image_FINAL_LoRA.safetensors ({} keys)".format(keys))
            log("\n" + t("Training completed!"))
        elif last_update > start_step:
            # Stopped: the LoRA of the last completed step, unless that step was just saved.
            path = os.path.join(OUTPUT_DIR, "{}_step_{}.safetensors".format(name, last_update))
            if not os.path.isfile(path):
                keys = export_lora(model, path)
                log("[SAVE] {} ({} keys)".format(os.path.basename(path), keys))
            log("[STOP] Resume continues from step {0}. / Reanudar sigue desde el paso {0}.".format(last_update))
    log("Total training time / Tiempo total de entrenamiento: " + fmt_time(total))


if __name__ == "__main__":
    main()
