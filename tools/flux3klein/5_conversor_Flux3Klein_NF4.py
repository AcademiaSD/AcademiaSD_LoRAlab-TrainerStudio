# -*- coding: utf-8 -*-
"""
5_conversor_Flux3Klein_NF4.py — FLUX 3 Image Klein Base -> caché NF4 para entrenar LoRAs.
FLUX 3 Image Klein Base -> NF4 cache for LoRA training.

Entrada: la carpeta del modelo en formato diffusers (black-forest-labs/flux-3-image-klein-base):
    transformer/   diffusion_pytorch_model-*.safetensors + index (BF16, ~11 GB)
    text_encoder/  Qwen3-VL-4B-Instruct (BF16): se copia tal cual, los scripts lo cuantizan al cargarlo
    vae/           autoencoder

Salida (OUTPUT_DIR), el mismo formato que Klein 9B, Krea 2 o Z-Image:
    transformer/config.json
    transformer/index.json        qué capas son NF4 y dónde está cada una
    transformer/weights/          una capa NF4 por fichero (peso uint8 + QuantState packed=True)
    transformer/others.safetensors  el resto en BF16 (embedders, modulaciones, normas, salida)
    transformer/metadata.json
    text_encoder/, vae/, configuraciones, LICENSE.md y NOTICE.md copiados

A diferencia de los otros conversores, este lee el transformer tensor a tensor desde los
.safetensors: no lo carga entero, así que funciona con poca RAM y también sin GPU (bitsandbytes
cuantiza en CPU, más despacio). Con GPU cuantiza en ella, como el resto.
"""
import json
import os
import shutil
import sys
import time

import torch
from bitsandbytes.functional import quantize_4bit
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "./Models/flux-3-image-klein-base"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./Models/FLUX.3-Image-Klein_NF4"

# Capas lineales de los bloques: 5 dobles (pesos por flujo: imagen, imagen de referencia y texto)
# y 28 simples. Embedders, modulaciones y la capa de salida se quedan en BF16.
QUANT_BLOCKS = ("mode_blocks.", "txt_mode_blocks.", "single_transformer_blocks.")
# Lo mismo que Params4bit por defecto, que es lo que reconstruye el cargador de los entrenadores.
BLOCKSIZE = 64
COMPRESS_STATISTICS = True
COPY_IGNORE = ("transformer", "variants", "model.safetensors", "ae.safetensors", ".cache", "*.lock")


def json_safe(value):
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def quantizable(name, tensor):
    return name.endswith(".weight") and tensor.dim() == 2 and name.startswith(QUANT_BLOCKS)


def convert_transformer(src, dst, device):
    with open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json"), "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    by_file = {}
    for name, file in weight_map.items():
        by_file.setdefault(file, []).append(name)

    weights_dir = os.path.join(dst, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    index = {"quantized": {}, "unquantized": {}, "others": "others.safetensors"}
    others, done = {}, 0
    # Los bias de las capas NF4 van en su fichero: se recogen primero.
    quant_layers = set()
    for file, names in by_file.items():
        with safe_open(os.path.join(src, file), framework="pt", device="cpu") as f:
            for name in names:
                if name.endswith(".weight") and name.startswith(QUANT_BLOCKS) and len(f.get_slice(name).get_shape()) == 2:
                    quant_layers.add(name[:-len(".weight")])

    for file, names in sorted(by_file.items()):
        with safe_open(os.path.join(src, file), framework="pt", device="cpu") as f:
            for name in sorted(names):
                layer = name.rsplit(".", 1)[0]
                if layer in quant_layers and name.endswith(".bias"):
                    continue  # se guarda con su peso
                tensor = f.get_tensor(name)
                if not quantizable(name, tensor):
                    others[name] = tensor.to(torch.bfloat16).contiguous()
                    continue
                # Copia FP32 temporal, el mismo procedimiento que los demás conversores.
                packed, qs = quantize_4bit(tensor.to(device=device, dtype=torch.float32).contiguous(),
                                           blocksize=BLOCKSIZE, compress_statistics=COMPRESS_STATISTICS,
                                           quant_type="nf4", quant_storage=torch.uint8)
                qs_dict = qs.as_dict(packed=True)
                tensors = {"weight": packed.cpu().contiguous()}
                bias_name = layer + ".bias"
                has_bias = bias_name in weight_map
                if has_bias:
                    with safe_open(os.path.join(src, weight_map[bias_name]), framework="pt", device="cpu") as fb:
                        tensors["bias"] = fb.get_tensor(bias_name).to(torch.bfloat16).contiguous()
                out_f, in_f = tensor.shape
                metadata = {"layer_name": layer, "type": "Linear4bit", "quant_type": "nf4", "bnb_quantized": "true",
                            "in_features": str(in_f), "out_features": str(out_f), "bias": str(has_bias)}
                for key, value in qs_dict.items():
                    if torch.is_tensor(value):
                        tensors["quant_state." + key] = value.detach().cpu().contiguous()
                    else:
                        metadata["qs_" + key] = json.dumps(json_safe(value))
                filename = layer.replace(".", "__") + ".safetensors"
                save_file(tensors, os.path.join(weights_dir, filename), metadata=metadata)
                index["quantized"][layer] = {"file": filename, "in_features": in_f, "out_features": out_f,
                                             "bias": has_bias, "quant_type": "nf4", "compute_dtype": "bfloat16",
                                             "quant_state_keys": list(qs_dict.keys())}
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(quant_layers)} capas NF4...", flush=True)
                del tensor, packed, qs

    save_file(others, os.path.join(dst, "others.safetensors"))
    with open(os.path.join(dst, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    shutil.copy2(os.path.join(src, "config.json"), dst)
    with open(os.path.join(dst, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "Flux3Klein-NF4", "version": 1, "source": "black-forest-labs/flux-3-image-klein-base",
                   "dtype": "bfloat16", "quant_type": "nf4", "compute_dtype": "bfloat16",
                   "blocksize": BLOCKSIZE, "compress_statistics": COMPRESS_STATISTICS,
                   "quantized_layers": len(index["quantized"]), "unquantized_layers": 0, "other_tensors": len(others),
                   "quant_blocks": list(QUANT_BLOCKS), "bitsandbytes_prequantized": True,
                   "quant_state_serialization": "as_dict(packed=True)", "loader": "Params4bit + QuantState.from_dict"},
                  f, indent=2)
    return len(index["quantized"]), len(others)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" FLUX 3 IMAGE KLEIN BASE -> NF4")
    print("=" * 70)
    print("Modelo:", os.path.abspath(MODEL_ID))
    print("Salida:", os.path.abspath(OUTPUT_DIR))
    print("Cuantiza en:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (más lento)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Todo menos el transformer BF16 y las variantes FP8/NVFP4 se copia tal cual.
    shutil.copytree(MODEL_ID, OUTPUT_DIR, ignore=shutil.ignore_patterns(*COPY_IGNORE), dirs_exist_ok=True)

    src = os.path.join(MODEL_ID, "transformer")
    if os.path.isdir(src):
        dst = os.path.join(OUTPUT_DIR, "transformer")
        shutil.rmtree(dst, ignore_errors=True)
        q, o = convert_transformer(src, dst, device)
        print(f"Capas NF4: {q} | tensores BF16 en others.safetensors: {o}")
    else:
        print("[AVISO] No hay transformer/ en el modelo: solo se copiaron los demás componentes.")
    print(f"Hecho en {(time.time() - t0) / 60:.1f} min -> {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
