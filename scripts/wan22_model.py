# -*- coding: utf-8 -*-
"""
wan22_model.py — Piezas comunes del trainer de Wan 2.2 T2V-A14B
Shared pieces of the Wan 2.2 T2V-A14B trainer

Wan 2.2 A14B son dos Transformers de 14B (MoE por pasos): "transformer" es el experto de
ruido alto (composición y movimiento, sigma >= 0.875) y "transformer_2" el de ruido bajo
(detalle y parecido). Cada uno se entrena con su propio LoRA.

Modelo: no hay un NF4 publicado, así que se convierte aquí la primera vez. Cada experto se
descarga shard a shard del repo oficial (FP32, ~57 GB por experto), se cuantiza en GPU y el
shard original se borra en cuanto queda convertido: el disco nunca guarda más de un shard en
FP32. Si se corta, al relanzar sigue por el shard siguiente.

    Models/Wan22-T2V-A14B-NF4/
        model_index.json, scheduler/, tokenizer/, vae/, text_encoder/   del repo oficial
        transformer/config.json, transformer_2/config.json
        transformer_nf4/   index.json + part_XX.safetensors (~7.6 GB)   experto de ruido alto
        transformer_2_nf4/ index.json + part_XX.safetensors (~7.6 GB)   experto de ruido bajo

Las Linear de los 40 bloques van en NF4 (QuantState packed, el formato del resto de
trainers); las de entrada/salida, timestep y texto, en BF16. Las normas, la rope y las
tablas de modulación, en FP32, como _keep_in_fp32_modules de diffusers.
"""
import json
import logging
import math
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from bitsandbytes.functional import QuantState, quantize_4bit
from bitsandbytes.nn import Linear4bit, Params4bit
from diffusers import WanTransformer3DModel
from safetensors import safe_open
from safetensors.torch import save_file
from i18n import t

HF_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
# Los mismos ficheros con los expertos ya en NF4 (~7.2 GB cada uno). Si el repo no existe o le falta
# un experto, ese experto se convierte en local desde HF_REPO_ID.
NF4_REPO_ID = "AcademiaSD/Wan22-T2V-A14B-NF4-for-LoRA-Training"
MODEL_DEFAULT = "Models/Wan22-T2V-A14B-NF4"

# Nombres de diffusers: transformer = ruido alto, transformer_2 = ruido bajo.
EXPERTS = {"high": "transformer", "low": "transformer_2"}
# model_index.json del repo: el experto de ruido alto actúa con t >= 0.875 * 1000.
BOUNDARY = 0.875

# Ritmo de los vídeos de Wan 2.2 A14B: 16 fps y 4k+1 fotogramas (el VAE comprime 4x en el tiempo).
FPS = 16
# Wan 2.1 VAE: 16 canales, 8x en espacio; el Transformer usa parches 2x2.
LATENT_CHANNELS = 16
SPATIAL = 8
MULTIPLE = 16
TEXT_LEN = 512
TEXT_DIM = 4096

# Negativo por defecto de Wan (sample_neg_prompt del repo oficial).
NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
                   "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                   "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

KEEP_FP32 = ("rope", "time_embedder", "scale_shift_table", "norm1", "norm2", "norm3")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi")


# =============================================================================
# HUGGING FACE
# =============================================================================

def get_hf_token():
    try:
        with open("settings/HF_token.json", "r", encoding="utf-8") as f:
            token = json.load(f).get("token", "").strip()
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "").strip() or None


def enable_hf_file_progress():
    # Lanzado desde la GUI la salida no es una terminal y tqdm se desactiva: una descarga de
    # varios GB parecería colgada. Se fuerzan las barras de progreso por fichero (en bytes).
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


_nf4_files = None


def nf4_repo_files():
    """Ficheros del repo NF4 (vacío si no existe, es privado sin token o no hay conexión)."""
    global _nf4_files
    if _nf4_files is None:
        from huggingface_hub import HfApi
        try:
            _nf4_files = set(HfApi().list_repo_files(NF4_REPO_ID, token=get_hf_token()))
        except Exception:
            _nf4_files = set()
    return _nf4_files


def ensure_base(model_dir, text_encoder=False):
    """Todo menos los Transformers: VAE, scheduler, tokenizer y (para el pre-caché) UMT5-XXL (~11 GB)."""
    need = [os.path.join(model_dir, "vae", "config.json"),
            os.path.join(model_dir, "tokenizer", "tokenizer_config.json"),
            os.path.join(model_dir, "transformer_2", "config.json")]
    if text_encoder:
        need.append(os.path.join(model_dir, "text_encoder", "config.json"))
    if all(os.path.exists(p) for p in need):
        return

    from huggingface_hub import snapshot_download
    enable_hf_file_progress()
    patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "vae/*",
                "transformer/config.json", "transformer_2/config.json"]
    if text_encoder:
        patterns.append("text_encoder/*")
    files = nf4_repo_files()
    repo = NF4_REPO_ID if all(f in files for f in ("vae/config.json", "tokenizer/tokenizer_config.json",
                                                   "transformer_2/config.json"))         and (not text_encoder or "text_encoder/config.json" in files) else HF_REPO_ID
    print(t("Downloading from Hugging Face: {repo}", repo=repo) + (" (~12 GB)" if text_encoder else " (~0.5 GB)"), flush=True)
    snapshot_download(repo_id=repo, local_dir=model_dir, token=get_hf_token(), max_workers=4,
                      allow_patterns=patterns)


def nf4_dir(model_dir, expert):
    return os.path.join(model_dir, EXPERTS[expert] + "_nf4")


def expert_ready(model_dir, expert):
    return os.path.exists(os.path.join(nf4_dir(model_dir, expert), "index.json"))


def quantized_names(config_dir):
    """Linear de los bloques: son las que van en NF4 (el 99% de los pesos). También todas las Linear."""
    with init_empty_weights():
        model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(config_dir))
    return {name: (m.in_features, m.out_features) for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and name.startswith("blocks.")}, \
           {name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}


def convert_shard(raw, out_dir, part, quantized, linears, index):
    """Un shard FP32 del repo -> out_dir/part: Linear de los bloques en NF4, el resto en BF16 o FP32."""
    tensors = {}
    with safe_open(raw, framework="pt", device="cpu") as f:
        for key in f.keys():
            module, _, kind = key.rpartition(".")
            tensor = f.get_tensor(key)
            if module in quantized and kind == "weight":
                # Mismo procedimiento que Params4bit.cuda(): bloques de 64, estadísticas comprimidas.
                w4, qs = quantize_4bit(tensor.to("cuda", torch.float32), blocksize=64, compress_statistics=True,
                                       quant_type="nf4", quant_storage=torch.uint8)
                tensors[key] = w4.cpu()
                for k, v in qs.as_dict(packed=True).items():
                    tensors[f"{key}.qs.{k}"] = v.cpu()
                index["quantized"][module] = {"file": part, "in": quantized[module][0], "out": quantized[module][1]}
                del w4, qs
            elif any(p in key for p in KEEP_FP32) and module not in linears:
                tensors[key] = tensor.float().contiguous()
            else:
                tensors[key] = tensor.to(torch.bfloat16).contiguous()
            del tensor
    torch.cuda.empty_cache()

    # Escritura atómica: un part a medias no se da por bueno al relanzar.
    tmp = os.path.join(out_dir, part + ".tmp")
    save_file(tensors, tmp)
    os.replace(tmp, os.path.join(out_dir, part))
    index["files"].append(part)


def download_expert(model_dir, expert):
    """Baja el experto ya convertido de NF4_REPO_ID. False si el repo no lo tiene."""
    sub = EXPERTS[expert] + "_nf4"
    if f"{sub}/index.json" not in nf4_repo_files():
        return False

    from huggingface_hub import hf_hub_download, snapshot_download
    enable_hf_file_progress()
    label = "high noise" if expert == "high" else "low noise"
    print(t("Downloading the {label} expert in NF4 from Hugging Face: {repo} (~7.6 GB)",
            label=label, repo=NF4_REPO_ID), flush=True)
    # index.json al final: es lo que marca el experto como listo, y una descarga cortada no debe parecerlo.
    snapshot_download(repo_id=NF4_REPO_ID, local_dir=model_dir, token=get_hf_token(), max_workers=4,
                      allow_patterns=[f"{sub}/*"], ignore_patterns=[f"{sub}/index.json"])
    hf_hub_download(NF4_REPO_ID, f"{sub}/index.json", local_dir=model_dir, token=get_hf_token())
    print("[OK] " + t("{label} expert ready: {path}", label=label, path=nf4_dir(model_dir, expert)), flush=True)
    return True


def ensure_expert(model_dir, expert):
    """
    Deja un experto en NF4: lo baja de NF4_REPO_ID si está allí y, si no, lo convierte en local
    (descarga, cuantiza y borra cada shard por turno).
    """
    if expert_ready(model_dir, expert):
        return
    ensure_base(model_dir)
    if download_expert(model_dir, expert):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(t("CUDA is not available. NF4 quantization needs CUDA."))

    from huggingface_hub import hf_hub_download
    enable_hf_file_progress()

    sub = EXPERTS[expert]
    out_dir = nf4_dir(model_dir, expert)
    raw_dir = os.path.join(model_dir, "_raw_download")
    os.makedirs(out_dir, exist_ok=True)

    index_file = hf_hub_download(HF_REPO_ID, f"{sub}/diffusion_pytorch_model.safetensors.index.json",
                                 local_dir=raw_dir, token=get_hf_token())
    with open(index_file, "r", encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))

    quantized, linears = quantized_names(os.path.join(model_dir, sub))
    label = "high noise" if expert == "high" else "low noise"
    print("\n" + "=" * 70)
    print(t("Converting the {label} expert to NF4: {n} shards (~57 GB in FP32, downloaded and deleted one at a time).",
            label=label, n=len(shards)), flush=True)
    print("=" * 70, flush=True)

    progress_path = os.path.join(out_dir, "index_partial.json")
    index = {"quantized": {}, "files": []}
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            index = json.load(f)

    for i, shard in enumerate(shards, 1):
        part = f"part_{i:02d}.safetensors"
        if part in index["files"] and os.path.exists(os.path.join(out_dir, part)):
            continue
        print(f"\n[{i}/{len(shards)}] " + t("Downloading {name}...", name=shard), flush=True)
        raw = hf_hub_download(HF_REPO_ID, f"{sub}/{shard}", local_dir=raw_dir, token=get_hf_token())

        print(f"[{i}/{len(shards)}] " + t("Quantizing to NF4..."), flush=True)
        convert_shard(raw, out_dir, part, quantized, linears, index)
        os.remove(raw)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

    missing = set(quantized) - set(index["quantized"])
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=sorted(missing)[:5]))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({**index, "format": "Wan22-NF4", "version": 1, "model_id": HF_REPO_ID, "subfolder": sub,
                   "quant_type": "nf4", "quant_state_serialization": "as_dict(packed=True)"}, f, indent=2)
    os.remove(progress_path)
    shutil.copy2(os.path.join(model_dir, sub, "config.json"), out_dir)
    shutil.rmtree(raw_dir, ignore_errors=True)
    print("[OK] " + t("{label} expert ready: {path}", label=label, path=out_dir), flush=True)


def load_expert(model_dir, expert):
    """
    Construye el experto vacío desde su config.json y lo rellena con los part_XX en CPU.
    No hace falta el modelo en BF16: cada Linear de los bloques se crea ya como Linear4bit.
    """
    folder = nf4_dir(model_dir, expert)
    with open(os.path.join(folder, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(folder))

    for name, info in index["quantized"].items():
        parent_name, _, child = name.rpartition(".")
        # En meta: nn.Linear.__init__ reservaría y rellenaría con kaiming una matriz que se tira enseguida.
        with torch.device("meta"):
            layer = Linear4bit(info["in"], info["out"], bias=True, quant_type="nf4", compute_dtype=torch.bfloat16)
        setattr(model.get_submodule(parent_name), child, layer)

    others = {}
    for part in index["files"]:
        with safe_open(os.path.join(folder, part), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if ".qs." in key:
                    continue
                module = key.rpartition(".")[0]
                if module in index["quantized"] and key.endswith(".weight"):
                    qs = {k.split(".qs.", 1)[1]: f.get_tensor(k) for k in keys if k.startswith(key + ".qs.")}
                    weight = Params4bit(f.get_tensor(key), requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
                    weight.quant_state = QuantState.from_dict(qs, device="cpu")
                    weight.bnb_quantized = True
                    model.get_submodule(module).weight = weight
                else:
                    others[key] = f.get_tensor(key)

    model.load_state_dict(others, strict=False, assign=True)
    model.requires_grad_(False)
    # diffusers crea la rope en float64 (en CUDA): cada producto con q y k sale en float64, el doble de
    # temporales que en float32, que es lo que ya usa con MPS. Con 33 fotogramas eran ~1 GB por bloque.
    model.rope.freqs_cos = model.rope.freqs_cos.float()
    model.rope.freqs_sin = model.rope.freqs_sin.float()

    missing = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=missing[:5]))
    return model


def _attn_half(block, hidden_states, temb, rotary_emb):
    # Primera mitad de WanTransformerBlock.forward (diffusers, temb [B, 6, D] de los 14B): autoatención.
    shift_msa, scale_msa, gate_msa, _, _, _ = (block.scale_shift_table + temb.float()).chunk(6, dim=1)
    norm_hidden_states = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = block.attn1(norm_hidden_states, None, None, rotary_emb)
    return (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)


def _ffn_half(block, hidden_states, encoder_hidden_states, temb):
    # Segunda mitad: atención al texto y FFN.
    _, _, _, c_shift_msa, c_scale_msa, c_gate_msa = (block.scale_shift_table + temb.float()).chunk(6, dim=1)
    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    hidden_states = hidden_states + block.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    norm_hidden_states = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


def split_block_checkpoint(block, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """
    Gradient checkpointing de un bloque en dos mitades en vez de una. Al recalcular en el backward solo
    está viva media bloque: los residuales de diffusers van en FP32 y con vídeo un bloque entero son
    ~45 veces el tamaño de las activaciones (6.5 GB con 33 fotogramas a 480p). Reentrante: lo que
    guarda (la entrada de cada mitad) pasa por save_for_backward, que es lo que save_on_cpu manda a la RAM.
    """
    from torch.utils.checkpoint import checkpoint
    hidden_states = checkpoint(_attn_half, block, hidden_states, temb, rotary_emb, use_reentrant=True)
    return checkpoint(_ffn_half, block, hidden_states, encoder_hidden_states, temb, use_reentrant=True)


def is_lora_target(name, module, targets):
    if not isinstance(module, (torch.nn.Linear, Linear4bit)):
        return False
    if targets == "all":
        return True
    # "blocks": autoatención, atención al texto y FFN de los 40 bloques.
    return name.startswith("blocks.")


# =============================================================================
# EXPORTACIÓN: nombres del repo original de Wan (los que lee ComfyUI)
# =============================================================================

_BLOCK_RENAME = {
    "attn1.to_q": "self_attn.q", "attn1.to_k": "self_attn.k", "attn1.to_v": "self_attn.v", "attn1.to_out.0": "self_attn.o",
    "attn2.to_q": "cross_attn.q", "attn2.to_k": "cross_attn.k", "attn2.to_v": "cross_attn.v", "attn2.to_out.0": "cross_attn.o",
    "ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2",
}
_TOP_RENAME = {
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "proj_out": "head.head",
}


def comfy_key(name):
    """blocks.3.attn1.to_q -> diffusion_model.blocks.3.self_attn.q (ComfyUI, Wan2GP y diffusers lo cargan)."""
    if name.startswith("blocks."):
        _, n, rest = name.split(".", 2)
        return f"diffusion_model.blocks.{n}.{_BLOCK_RENAME.get(rest, rest)}"
    return "diffusion_model." + _TOP_RENAME.get(name, name)


# Normas que traen los LoRAs de ComfyUI (pesos "diff"). En el original la norma de la atención al texto
# es norm3; en diffusers, norm2.
_NORM_RENAME = {"self_attn.norm_q": "attn1.norm_q", "self_attn.norm_k": "attn1.norm_k",
                "cross_attn.norm_q": "attn2.norm_q", "cross_attn.norm_k": "attn2.norm_k", "norm3": "norm2"}
_BLOCK_FROM_COMFY = {**{v: k for k, v in _BLOCK_RENAME.items()}, **_NORM_RENAME}
_TOP_FROM_COMFY = {v: k for k, v in _TOP_RENAME.items()}


def diffusers_name(key):
    """diffusion_model.blocks.3.self_attn.q -> blocks.3.attn1.to_q (lo contrario de comfy_key)."""
    key = key.removeprefix("diffusion_model.")
    if key.startswith("blocks."):
        _, n, rest = key.split(".", 2)
        return f"blocks.{n}.{_BLOCK_FROM_COMFY.get(rest, rest)}"
    return _TOP_FROM_COMFY.get(key, key)


# =============================================================================
# PREVIEWS TURBO (lightx2v)
# =============================================================================

# LoRA de destilación de pasos y CFG de Wan 2.1 T2V 14B, el que se usa con Wan 2.2 A14B en ComfyUI.
TURBO_REPO = "Kijai/WanVideo_comfy"
TURBO_FILE = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"


def ensure_turbo(model_dir):
    folder = os.path.join(model_dir, "turbo")
    path = os.path.join(folder, TURBO_FILE)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        enable_hf_file_progress()
        print(t("Downloading the turbo LoRA for the previews: {repo} (~317 MB)", repo=TURBO_REPO), flush=True)
        hf_hub_download(TURBO_REPO, TURBO_FILE, local_dir=folder, token=get_hf_token())
    return path


class TurboLoRA:
    """
    El LoRA lightx2v (formato de ComfyUI: lora_down/lora_up sin alpha, o sea escala 1; diff_b para los bias
    y diff para los pesos de las normas), puesto en un experto solo mientras dura una preview. Las capas
    LoRA van como hooks que suman su salida; bias y normas se cambian en su sitio y remove() los deja
    como estaban. Así no toca el LoRA que se entrena ni el optimizador.
    """

    def __init__(self, path):
        self.parts = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                for suffix, part in ((".lora_down.weight", "down"), (".lora_up.weight", "up"),
                                     (".diff_b", "diff_b"), (".diff", "diff")):
                    if key.endswith(suffix):
                        name = diffusers_name(key[:-len(suffix)])
                        self.parts.setdefault(name, {})[part] = f.get_tensor(key).to(torch.bfloat16)
                        break
        self._gpu = {}

    def apply(self, model, strength):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        modules = dict(base.named_modules())
        handles, saved = [], []
        for name, p in self.parts.items():
            module = modules.get(name)
            if module is None:
                continue
            if "down" in p:
                if name not in self._gpu:
                    self._gpu[name] = (p["down"].to("cuda"), p["up"].to("cuda"))
                down, up = self._gpu[name]

                def hook(_, args, out, down=down, up=up):
                    return out + (strength * F.linear(F.linear(args[0].to(down.dtype), down), up)).to(out.dtype)

                handles.append(module.register_forward_hook(hook))
            # Con LoRA de PEFT la capa original está en base_layer.
            target = getattr(module, "base_layer", module)
            for part, attr in (("diff_b", "bias"), ("diff", "weight")):
                weight = getattr(target, attr, None)
                if part in p and weight is not None and weight.shape == p[part].shape:
                    saved.append((weight, weight.data.clone()))
                    weight.data += strength * p[part].to(weight.device, weight.dtype)
        return handles, saved

    def remove(self, state):
        handles, saved = state
        for h in handles:
            h.remove()
        for weight, original in saved:
            weight.data = original.to(weight.device)

    def release(self):
        """Fuera de la GPU al acabar la preview (~0.3 GB)."""
        self._gpu.clear()


def turbo_sigmas(steps, shift=8.0, a=0.6, b=0.6):
    """
    Los de ComfyUI con el scheduler beta: su tabla de 1000 sigmas de Wan (shift 8 por defecto) y de ella los
    pasos que reparte una Beta(0.6, 0.6), más juntos al principio y al final. La inversa de la Beta se
    integra aquí (x = sin² θ deja la densidad sin infinitos) para no depender de scipy.
    """
    t_ = np.arange(1, 1001) / 1000
    table = shift * t_ / (1 + (shift - 1) * t_)
    theta = np.linspace(0, np.pi / 2, 200001)
    pdf = 2 * np.sin(theta) ** (2 * a - 1) * np.cos(theta) ** (2 * b - 1)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(theta))])
    cdf /= cdf[-1]
    ppf = np.interp(1 - np.linspace(0, 1, steps, endpoint=False), cdf, np.sin(theta) ** 2)
    sigmas, last = [], -1
    for i in np.rint(ppf * (len(table) - 1)).astype(int):
        if i != last:
            sigmas.append(float(table[i]))
        last = i
    return sigmas + [0.0]


# =============================================================================
# VÍDEO (ffmpeg)
# =============================================================================

def ffbin(name):
    return shutil.which(name) or shutil.which(name + ".exe")


def video_info(path):
    """(ancho, alto, fps, duración en s) del primer stream de vídeo, o None si ffprobe no lo lee."""
    probe = ffbin("ffprobe")
    if not probe:
        raise RuntimeError(t("ffmpeg/ffprobe not found in PATH: they are needed to read videos."))
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height,r_frame_rate:format=duration",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out)
        s = data["streams"][0]
        num, _, den = s["r_frame_rate"].partition("/")
        return int(s["width"]), int(s["height"]), float(num) / float(den or 1), float(data["format"]["duration"])
    except Exception:
        return None


def valid_frames(n):
    """El mayor 4k+1 que no pasa de n (1, 5, 9, ..., 81)."""
    return max(1, (int(n) - 1) // 4 * 4 + 1)


def clip_frames(duration, max_frames):
    """Fotogramas que se entrenan de un clip: los que caben a 16 fps, en la rejilla 4k+1 y sin pasar de max_frames."""
    return valid_frames(min(int(max_frames), max(1, math.floor(duration * FPS + 1e-6))))


def read_video(path, frames, width, height):
    """Los `frames` primeros fotogramas a 16 fps, escalados hasta cubrir width x height y recortados al centro: [F, H, W, 3] uint8."""
    ffmpeg = ffbin("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(t("ffmpeg/ffprobe not found in PATH: they are needed to read videos."))
    vf = (f"fps={FPS},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height}")
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", str(frames),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=600, check=True).stdout
    got = len(raw) // (width * height * 3)
    if got < frames:
        raise RuntimeError(t("Only {got} of {n} frames could be read", got=got, n=frames))
    return np.frombuffer(raw[:frames * width * height * 3], dtype=np.uint8).reshape(frames, height, width, 3)


def write_video(frames, path, fps=FPS):
    """[F, H, W, 3] uint8 -> mp4 H.264 (lo reproduce cualquier navegador). Sin ffmpeg, un .webp animado."""
    ffmpeg = ffbin("ffmpeg")
    if ffmpeg:
        h, w = frames.shape[1:3]
        tmp = path + ".tmp.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-movflags", "+faststart", tmp], input=frames.tobytes(), check=True, timeout=600)
        os.replace(tmp, path)
        return path
    from PIL import Image
    path = os.path.splitext(path)[0] + ".webp"
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, quality=90)
    return path


# =============================================================================
# VAE (Wan 2.1: 16 canales normalizados con su media y desviación por canal)
# =============================================================================

def latent_stats(vae, device, dtype=torch.float32):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std
