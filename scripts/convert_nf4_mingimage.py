# -*- coding: utf-8 -*-
"""
convert_nf4_mingimage.py — Conversor de Ming-Image (Design) al formato NF4 del entrenador
Ming-Image (Design) to the trainer NF4 format converter

Genera un paquete autocontenido para `AcademiaSD/Ming-Image-NF4-for-LoRA-Training`:

  Ming-Image_NF4/
    mllm/          MLLM cuantizado NF4 (prequantizado con bitsandbytes) + tokenizer
    connector/     proyector Qwen2 en bf16 (el loader lo carga en bf16 igualmente)
    mlp/           query tokens + proj_in/proj_out (fp32, tal cual)
    transformer/   DiT NF4: config.json, index.json, metadata.json,
                   others.safetensors y weights/*.safetensors
    vae/           AutoencoderKLQwenImage (RGBA)
    scheduler/     FlowMatchEulerDiscreteScheduler
    config.json    config raíz del paquete

El formato de `transformer/` es el mismo que `Z-Image_NF4` y lo consume
`mingimage_model.load_nf4_transformer()`:

  - `quantized`:   nn.Linear 4-bit (peso empaquetado + quant_state serializado)
  - `unquantized`: nn.Linear en bf16 (embedders, final layer, adaLN, t_embedder, cap_embedder)
  - `others`:      normas, biases, tokens aprendidos…

Uso:
    python scripts/convert_nf4_mingimage.py --src Ming-Image_Raw --dst Ming-Image_NF4
    python scripts/convert_nf4_mingimage.py --only transformer      # solo el DiT
    python scripts/convert_nf4_mingimage.py --only mllm             # solo el text encoder
    python scripts/convert_nf4_mingimage.py --only extras           # connector/vae/scheduler/mlp

Es reanudable: los ficheros ya escritos se saltan salvo `--overwrite`.
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mingimage_vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

HF_REPO_ID = "inclusionAI/Ming-Image-0.1-Design"

# Capas que no se cuantizan: son pequeñas y/o sensibles a la precisión.
# (Misma lista que Z-Image_NF4; el DiT es la misma arquitectura.)
SKIP_QUANT = ("all_x_embedder", "all_final_layer", "t_embedder", "cap_embedder")
SKIP_QUANT_ANY = ("adaLN_modulation",)
BLOCKSIZE = 64


def log(msg=""):
    print(msg, flush=True)


def file_name(module_name):
    """Nombre de fichero del formato Z-Image: puntos -> __ (los guiones se quedan)."""
    return module_name.replace(".", "__") + ".safetensors"


def scan_source(src_dir):
    """Índice de pesos + formas, clasificado en cuantizados / bf16 / otros."""
    index_path = os.path.join(src_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("No se encuentra " + index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    by_file = {}
    for key, fname in weight_map.items():
        by_file.setdefault(fname, []).append(key)

    shapes = {}
    for fname, keys in by_file.items():
        with safe_open(os.path.join(src_dir, fname), framework="pt", device="cpu") as f:
            for key in keys:
                shapes[key] = tuple(f.get_slice(key).get_shape())

    quantized, unquantized = {}, {}
    used = set()
    for key, shape in shapes.items():
        if not key.endswith(".weight") or len(shape) != 2:
            continue
        name = key[: -len(".weight")]
        if any(name.startswith(p) for p in SKIP_QUANT) or any(p in name for p in SKIP_QUANT_ANY):
            info = {
                "file": file_name(name),
                "in_features": shape[1],
                "out_features": shape[0],
                "bias": (name + ".bias") in shapes,
            }
            unquantized[name] = info
            used.add(key)
            used.add(name + ".bias")
        else:
            info = {
                "file": file_name(name),
                "in_features": shape[1],
                "out_features": shape[0],
                "bias": (name + ".bias") in shapes,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
            }
            quantized[name] = info
            used.add(key)
            used.add(name + ".bias")

    others = [k for k in shapes if k not in used]
    return weight_map, shapes, quantized, unquantized, others


def convert_transformer(src, dst, device, overwrite):
    src_dir = os.path.join(src, "transformer")
    dst_dir = os.path.join(dst, "transformer")
    weights_dir = os.path.join(dst_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    weight_map, shapes, quantized, unquantized, others = scan_source(src_dir)
    log("  DiT: {q} capas NF4, {u} capas bf16, {o} tensores sueltos".format(
        q=len(quantized), u=len(unquantized), o=len(others)))

    t0 = time.time()

    def target(name, info):
        return os.path.join(weights_dir, info["file"])

    # ── 1. Capas 4-bit ──────────────────────────────────────────────────────
    done = 0
    for name, info in quantized.items():
        path = target(name, info)
        if not overwrite and os.path.exists(path):
            done += 1
            continue
        with safe_open(os.path.join(src_dir, weight_map[name + ".weight"]), framework="pt", device="cpu") as f:
            weight = f.get_tensor(name + ".weight").to(device=device, dtype=torch.bfloat16).contiguous()
            bias = f.get_tensor(name + ".bias") if info["bias"] else None

        from bitsandbytes.functional import quantize_4bit

        packed, qs = quantize_4bit(weight, blocksize=BLOCKSIZE, quant_type="nf4", compress_statistics=True)
        sd = {"weight": packed.cpu()}
        for k, v in qs.as_dict(packed=True).items():
            sd["quant_state." + k] = v.cpu()
        if bias is not None:
            sd["bias"] = bias.cpu()
        save_file(sd, path, metadata={"format": "pt"})

        del weight, packed, qs, sd
        done += 1
        if done % 20 == 0 or done == len(quantized):
            log("    NF4 {}/{}".format(done, len(quantized)))

    # ── 2. Capas en bf16 ────────────────────────────────────────────────────
    for name, info in unquantized.items():
        path = target(name, info)
        if not overwrite and os.path.exists(path):
            continue
        with safe_open(os.path.join(src_dir, weight_map[name + ".weight"]), framework="pt", device="cpu") as f:
            sd = {"weight": f.get_tensor(name + ".weight")}
            if info["bias"]:
                sd["bias"] = f.get_tensor(name + ".bias")
        save_file(sd, path, metadata={"format": "pt"})

    # ── 3. Resto (normas, biases, tokens aprendidos) ────────────────────────
    others_path = os.path.join(dst_dir, "others.safetensors")
    if overwrite or not os.path.exists(others_path):
        # Agrupado por shard: abrir/cerrar el mismo safetensors 200 veces seguidas
        # provoca un access violation en Windows (0xC0000005).
        by_file = {}
        for key in others:
            by_file.setdefault(weight_map[key], []).append(key)

        sd = {}
        for fname, keys in by_file.items():
            with safe_open(os.path.join(src_dir, fname), framework="pt", device="cpu") as f:
                for key in keys:
                    sd[key] = f.get_tensor(key)
            log("    others: {} ({} tensores)".format(fname, len(keys)))
        save_file(sd, others_path, metadata={"format": "pt"})

    # ── 4. config.json / index.json / metadata.json ─────────────────────────
    shutil.copy2(os.path.join(src_dir, "config.json"), os.path.join(dst_dir, "config.json"))

    index = {"quantized": quantized, "unquantized": unquantized, "others": "others.safetensors"}
    with open(os.path.join(dst_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    metadata = {
        "format": "MingImage-NF4",
        "version": 1,
        "model_id": HF_REPO_ID,
        "dtype": "bfloat16",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "quantized_layers": len(quantized),
        "unquantized_layers": len(unquantized),
        "other_tensors": len(others),
        "skip_quant": list(SKIP_QUANT),
        "skip_quant_any": list(SKIP_QUANT_ANY),
        "bitsandbytes_prequantized": True,
        "quant_state_serialization": "as_dict(packed=True)",
        "loader": "mingimage_model.load_nf4_transformer",
    }
    with open(os.path.join(dst_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log("  DiT listo en {:.1f} s".format(time.time() - t0))


def convert_mllm(src, dst, device):
    """Cuantiza el MLLM (BailingMoE-V2 + ViT) y lo guarda prequantizado."""
    from transformers import BitsAndBytesConfig
    from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

    src_dir = os.path.join(src, "mllm")
    dst_dir = os.path.join(dst, "mllm")
    os.makedirs(dst_dir, exist_ok=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log("  Cargando MLLM en NF4 (esto tarda ~1 min)...")
    t0 = time.time()
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        src_dir,
        load_image_gen=False,          # solo el MLLM: los demás módulos van aparte
        dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quantization_config,
        device_map={"": device},
    )
    log("  MLLM cargado en {:.1f} s | VRAM {:.1f} GB".format(time.time() - t0, torch.cuda.memory_allocated() / 1e9))

    # transformers 5 espera _tied_weights_keys como dict; el LM de Ming lo declara
    # como lista (formato 4.x) y save_pretrained revienta en _get_tied_weight_keys.
    fixed = 0
    for module in model.modules():
        twk = getattr(module, "_tied_weights_keys", None)
        if isinstance(twk, (list, tuple)):
            module._tied_weights_keys = {k: k for k in twk}
            fixed += 1
    if fixed:
        log("  _tied_weights_keys normalizado en {} módulo(s)".format(fixed))

    log("  Guardando MLLM prequantizado...")
    model.save_pretrained(dst_dir, safe_serialization=True, max_shard_size="5GB")

    # El tokenizer/processor no lo escribe save_pretrained.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "preprocessor_config.json"):
        source = os.path.join(src_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(dst_dir, name))

    del model
    torch.cuda.empty_cache()
    log("  MLLM listo.")


def convert_extras(src, dst):
    """connector (a bf16) + mlp + vae + scheduler + config raíz."""
    # connector: en disco está fp32; el loader lo carga siempre en bf16.
    src_dir = os.path.join(src, "connector")
    dst_dir = os.path.join(dst, "connector")
    os.makedirs(dst_dir, exist_ok=True)

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    for shard in shards:
        target = os.path.join(dst_dir, shard)
        source = os.path.join(src_dir, shard)
        if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(source) // 2:
            continue
        sd = load_file(source)
        save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, target, metadata={"format": "pt"})
        log("    connector: {} -> bf16".format(shard))

    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        source = os.path.join(src_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(dst_dir, name))

    # config del connector en bf16 (coherente con los pesos guardados)
    cfg_path = os.path.join(dst_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["torch_dtype"] = "bfloat16"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # módulos que se copian tal cual
    for name in ("mlp", "vae", "scheduler"):
        source = os.path.join(src, name)
        target = os.path.join(dst, name)
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)
        log("    {}: copiado".format(name))

    root_cfg = os.path.join(src, "config.json")
    if os.path.exists(root_cfg):
        shutil.copy2(root_cfg, os.path.join(dst, "config.json"))


def main():
    parser = argparse.ArgumentParser(description="Convierte Ming-Image a NF4 para el entrenador.")
    parser.add_argument("--src", default="Ming-Image_Raw", help="checkpoint original (BF16)")
    parser.add_argument("--dst", default="Ming-Image_NF4", help="destino del paquete NF4")
    parser.add_argument("--only", choices=("all", "transformer", "mllm", "extras"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="rehace lo que ya exista")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit("No existe el checkpoint de origen: " + args.src)
    os.makedirs(args.dst, exist_ok=True)

    log("Origen : " + os.path.abspath(args.src))
    log("Destino: " + os.path.abspath(args.dst))

    if args.only in ("all", "transformer"):
        log("\n[1/3] DiT -> NF4")
        convert_transformer(args.src, args.dst, args.device, args.overwrite)

    if args.only in ("all", "mllm"):
        log("\n[2/3] MLLM -> NF4 prequantizado")
        convert_mllm(args.src, args.dst, args.device)

    if args.only in ("all", "extras"):
        log("\n[3/3] connector / mlp / vae / scheduler")
        convert_extras(args.src, args.dst)

    log("\n[OK] Paquete NF4 en " + os.path.abspath(args.dst))


if __name__ == "__main__":
    main()
