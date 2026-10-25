# -*- coding: utf-8 -*-
"""
mingimage_model.py — DiT de Ming-Image (arquitectura custom) y carga NF4
Ming-Image DiT (custom architecture) and NF4 loading

El DiT de Ming-Image no está en diffusers: es el `DiffusionTransformer` del repo oficial
(MIT), vendorizado en `mingimage_vendor/diffusion/`. Es la misma arquitectura que Z-Image
con dos campos extra en el config (`alignment_padding_mode`, `multi_frame_output`) y una
segunda condición (`cap_feats_2`, N tokens x 3840).

Aquí viven:
  - `DiffusionTransformer` reexportado (para el entrenador y el conversor NF4).
  - `load_nf4_transformer(model_dir)`: construye el modelo vacío y lo rellena desde la
    caché NF4 (`transformer/{config.json, index.json, others.safetensors, weights/}`),
    con el mismo formato que `Z-Image_NF4` y que consume `2_train_lora_mingimage.py`.
"""
import os
import json
import sys

import torch
from accelerate import init_empty_weights
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from safetensors.torch import load_file

from i18n import t

# El código oficial (MIT) vive en mingimage_vendor/ porque no está en diffusers.
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mingimage_vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from diffusion.transformer import DiffusionTransformer  # noqa: E402


def load_nf4_transformer(model_dir):
    """
    Construye el Transformer vacío desde transformer/config.json y lo rellena con la
    caché NF4 (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    """
    cache_dir = os.path.join(model_dir, "transformer")
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(t("index.json not found in the NF4 cache: {path}", path=cache_dir))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = DiffusionTransformer.from_config(DiffusionTransformer.load_config(cache_dir))

    weights_dir = os.path.join(cache_dir, "weights")

    def set_module(name, module):
        parent_name, _, child_name = name.rpartition(".")
        setattr(transformer.get_submodule(parent_name) if parent_name else transformer, child_name, module)

    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias = f.get_tensor("bias") if info["bias"] else None
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device="cpu")
        weight.bnb_quantized = True

        # En meta: nn.Linear.__init__ reserva y rellena con kaiming una matriz FP32 que se tira enseguida.
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"], quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=info["bias"], dtype=torch.bfloat16)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    transformer.load_state_dict(load_file(os.path.join(cache_dir, index["others"])), strict=False, assign=True)

    missing = [n for n, t_ in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t_.is_meta]
    if missing:
        raise RuntimeError(t("NF4 cache incomplete: {names}", names=missing[:5]))

    print("[OK] " + t("NF4 layers: {a} | BF16 layers: {b}", a=len(index["quantized"]), b=len(index["unquantized"])))
    return transformer
