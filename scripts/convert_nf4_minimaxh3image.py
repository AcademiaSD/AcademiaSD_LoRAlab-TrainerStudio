# -*- coding: utf-8 -*-
"""
convert_nf4_minimaxh3image.py

Converts the MiniMax-H3-Image DiT (diffusers layout, 66 GB bf16) into the same
per-layer NF4 cache the MiniMax-H3 trainer already reads ("MiniMax-H3-NF4" v1):

    <out>/transformers/config.json, config_nf4.json, index.json, metadata.json
    <out>/transformers/weights/<layer>.safetensors        Linear4bit, NF4 + double quant
    <out>/transformers/weights/other-<param>.safetensors  non-linear params, raw
    <out>/transformers/weights/pc-<param>.safetensors     precision-critical, unquantized
    <out>/vae, <out>/scheduler, pdd_config.json, LICENSE  copied as they are

It never instantiates the model: every tensor is read straight from the
safetensors shards, one at a time, so it needs ~1 GB of RAM and ~1 GB of VRAM.
It is resumable: a layer whose file already exists with the recorded size is
skipped.

The text encoder is NOT converted: MiniMax-H3-Image conditions on the unchanged
Qwen3-VL of MiniMax-H3, so Models/MiniMax-H3-NF4/text_encoder is reused.

Convierte el DiT de MiniMax-H3-Image al mismo formato NF4 por capas que ya lee el
entrenador de MiniMax-H3. No instancia el modelo y se puede reanudar. El text
encoder no se convierte: es el mismo Qwen3-VL de MiniMax-H3.

Usage / Uso:
    python convert_nf4_minimaxh3image.py [--src Models/MiniMax-H3-Image-Raw]
                                         [--out Models/MiniMax-H3-Image-NF4]
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch
from bitsandbytes.nn import Linear4bit
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "MiniMax-H3-NF4"
VERSION = 1

# Kept in float next to their NF4 copy. The model declares proj_in, time_embedder
# and the per-step heads as fp32 (`_keep_in_fp32_modules`); the output AdaLN is
# added because NF4 error there breaks the cancellation of the final modulation.
# Se guardan en float ademas de en NF4: el modelo los declara fp32 y el AdaLN de
# salida no tolera el error de NF4.
PRECISION_CRITICAL_PREFIXES = ("proj_in.", "time_embedder.", "pdd_proj_out.", "norm_out.")


NOTICE = """MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, Copyright \u00a9 2026 MiniMax. All Rights Reserved.

Powered by MiniMax-H3-Image.

The full text of the MiniMax-H3-Image Community License Agreement is in the LICENSE file of this repository. Use, reproduction and distribution are only authorized within the Applicable Territory defined in that Agreement, which excludes the European Union, the United Kingdom, the Republic of Korea and the United States, and are subject to its Acceptable Use Policy (Exhibit A).

MODIFIED FILES
--------------
This repository is a modified redistribution of MiniMaxAI/MiniMax-H3-Image, prepared by AcademiaSD for LoRA training with AcademiaSD LoRAlab Trainer Studio.

  transformers/weights/*.safetensors
      MODIFIED by AcademiaSD. Converted from the original bfloat16 transformer
      (transformer_diffusers/) by 4-bit NormalFloat (NF4) quantization with
      bitsandbytes, double quantization, compute dtype bfloat16, stored one
      layer per file. Files prefixed "other-" hold the original non-linear
      parameters unchanged; files prefixed "pc-" hold unquantized copies of the
      precision-critical modules (proj_in, time_embedder, norm_out and the six
      per-step output heads pdd_proj_out).

  transformers/index.json, transformers/metadata.json,
  transformers/config_nf4.json
      CREATED by AcademiaSD to describe the quantized layout.

  transformers/config.json
      Copied from transformer_diffusers/config.json, unchanged.

  vae/, scheduler/, pdd_config.json, modular_model_index.json, LICENSE
      Copied from MiniMaxAI/MiniMax-H3-Image, unchanged.

The text encoder is not included: MiniMax-H3-Image conditions on the unchanged Qwen3-VL text encoder of MiniMax-H3, which is distributed separately.
"""


def fname(name):
    return name.replace(".", "__")


def atomic_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def scan_shards(src_dir):
    """{tensor name: (shard path, dtype, shape)} over every shard of the DiT."""
    shards = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError("No .safetensors shards in {}".format(src_dir))
    where = {}
    for path in shards:
        with safe_open(path, "pt") as st:
            for key in st.keys():
                sl = st.get_slice(key)
                where[key] = (path, sl.get_dtype(), list(sl.get_shape()))
    return where


def read_tensor(where, name):
    with safe_open(where[name][0], "pt") as st:
        return st.get_tensor(name)


def quantize_linear(weight, bias, device):
    """One nn.Linear -> the tensors of a bitsandbytes Linear4bit (NF4, double quant)."""
    out_features, in_features = weight.shape
    lin = Linear4bit(
        in_features, out_features, bias=bias is not None,
        compute_dtype=torch.bfloat16, quant_type="nf4", compress_statistics=True,
    )
    # bitsandbytes quantizes when the float weight moves to CUDA.
    lin.weight.data = weight.to(torch.float32)
    if bias is not None:
        lin.bias.data = bias.to(torch.bfloat16)
    lin = lin.to(device)
    tensors = {"weight": lin.weight.data.detach().cpu().contiguous()}
    if bias is not None:
        tensors["bias"] = lin.bias.data.detach().to(torch.bfloat16).cpu().contiguous()
    for key, value in lin.weight.quant_state.as_dict(packed=True).items():
        tensors["quant_state." + key] = value.detach().cpu().contiguous()
    del lin
    return tensors


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(root, "Models", "MiniMax-H3-Image-Raw"))
    ap.add_argument("--out", default=os.path.join(root, "Models", "MiniMax-H3-Image-NF4"))
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new layers (testing).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[ERROR] bitsandbytes NF4 quantization needs a CUDA GPU / hace falta una GPU CUDA.")
    device = torch.device("cuda")

    src_dit = os.path.join(args.src, "transformer_diffusers")
    out_dit = os.path.join(args.out, "transformers")
    out_w = os.path.join(out_dit, "weights")
    os.makedirs(out_w, exist_ok=True)

    with open(os.path.join(src_dit, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    atomic_json(config, os.path.join(out_dit, "config.json"))
    atomic_json(dict(config, _name_or_path="./MiniMax-H3-Image"), os.path.join(out_dit, "config_nf4.json"))

    where = scan_shards(src_dit)
    names = sorted(where)
    linears = sorted(n[: -len(".weight")] for n in names if n.endswith(".weight") and len(where[n][2]) == 2)
    linear_params = set()
    for layer in linears:
        linear_params.add(layer + ".weight")
        if layer + ".bias" in where:
            linear_params.add(layer + ".bias")
    others = [n for n in names if n not in linear_params]
    critical = [n for n in names if n.startswith(PRECISION_CRITICAL_PREFIXES)]
    print("[NF4] {} tensors: {} linear layers, {} other, {} precision-critical".format(
        len(names), len(linears), len(others), len(critical)), flush=True)

    index_path = os.path.join(out_dit, "index.json")
    index = {"format": FORMAT, "version": VERSION, "quantized": {}, "unquantized": {}, "other": {}, "precision_critical": {}}
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as fh:
                index.update(json.load(fh))
        except Exception:
            pass

    def done(section, key):
        rec = index[section].get(key)
        if not rec:
            return False
        path = os.path.join(out_w, rec["file"])
        return os.path.isfile(path) and os.path.getsize(path) == rec["file_size"]

    # 1. Raw copies: non-linear params and the precision-critical ones.
    for section, prefix, group, status in (
        ("other", "other-", others, "saved_raw"),
        ("precision_critical", "pc-", critical, "saved_unquantized"),
    ):
        for name in group:
            if done(section, name):
                continue
            tensor = read_tensor(where, name).contiguous()
            file = prefix + fname(name) + ".safetensors"
            save_file({name: tensor}, os.path.join(out_w, file))
            index[section][name] = {
                "file": file, "file_size": os.path.getsize(os.path.join(out_w, file)),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""), "status": status,
            }
        atomic_json(index, index_path)

    # 2. Linear layers -> NF4.
    started = time.time()
    converted = 0
    for i, layer in enumerate(linears, 1):
        if done("quantized", layer):
            continue
        weight = read_tensor(where, layer + ".weight")
        bias = read_tensor(where, layer + ".bias") if layer + ".bias" in where else None
        tensors = quantize_linear(weight, bias, device)
        out_features, in_features = weight.shape
        file = fname(layer) + ".safetensors"
        meta = {
            "format": FORMAT, "version": str(VERSION), "layer_name": layer, "type": "Linear4bit",
            "in_features": str(in_features), "out_features": str(out_features), "bias": str(bias is not None),
            "quant_type": "nf4", "compute_dtype": "bfloat16", "bnb_quantized": "true",
        }
        tmp = os.path.join(out_w, file + ".tmp")
        save_file(tensors, tmp, metadata=meta)
        os.replace(tmp, os.path.join(out_w, file))
        index["quantized"][layer] = {
            "file": file, "file_size": os.path.getsize(os.path.join(out_w, file)),
            "in_features": in_features, "out_features": out_features, "bias": bias is not None,
            "quant_type": "nf4", "compute_dtype": "bfloat16", "status": "saved",
        }
        converted += 1
        del weight, bias, tensors
        if converted % 10 == 0 or i == len(linears):
            atomic_json(index, index_path)
            torch.cuda.empty_cache()
            print("[NF4] {}/{} {}  ({:.0f}s)".format(i, len(linears), layer, time.time() - started), flush=True)
        if args.limit and converted >= args.limit:
            break
    atomic_json(index, index_path)

    complete = len(index["quantized"]) == len(linears)
    atomic_json({
        "format": FORMAT, "version": VERSION, "component_name": "transformer", "component_type": "transformer",
        "model": "MiniMax-H3-Image", "quant_type": "nf4", "compute_dtype": "bfloat16",
        "total_linear": len(linears), "quantized": len(index["quantized"]), "unquantized": 0,
        "incremental": True, "other": len(index["other"]), "complete": complete,
    }, os.path.join(out_dit, "metadata.json"))

    # 3. Everything else the trainer needs, copied as it is. The VAE stays float.
    for sub, dst in (("vae_diffusers", "vae"), ("scheduler", "scheduler")):
        s, d = os.path.join(args.src, sub), os.path.join(args.out, dst)
        if os.path.isdir(s) and not os.path.isdir(d):
            shutil.copytree(s, d)
    for file in ("pdd_config.json", "modular_model_index.json", "LICENSE"):
        s = os.path.join(args.src, file)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, file))

    # Section III.4 of the MiniMax-H3-Image Community License: every distribution carries a NOTICE,
    # and III.2: modified files carry a notice saying so. / Seccion III.4 y III.2 de la licencia.
    with open(os.path.join(args.out, "NOTICE"), "w", encoding="utf-8") as fh:
        fh.write(NOTICE)
    print("[NF4] {} -> {} ({} of {} linear layers)".format(
        "DONE" if complete else "PARTIAL", args.out, len(index["quantized"]), len(linears)), flush=True)


if __name__ == "__main__":
    main()
