# -*- coding: utf-8 -*-
"""
5_conversor_QwenImage2512_NF4.py — Qwen-Image-2512 y Qwen-Image-Edit-2511 -> caché NF4 para entrenar LoRAs.
Qwen-Image-2512 and Qwen-Image-Edit-2511 -> NF4 cache for LoRA training.

Uso / usage:
    python 5_conversor_QwenImage2512_NF4.py <Qwen-Image-2512> <Qwen-Image-Edit-2511 o ""> <salida>

Entrada: las carpetas diffusers de Qwen/Qwen-Image-2512 y Qwen/Qwen-Image-Edit-2511. Los dos repos
comparten text encoder y VAE (mismos ficheros, mismo hash); de Edit-2511 basta con transformer/ y processor/.

Salida (mismo formato que MOSS-Image y FLUX.3 Klein):
    transformer/       2512 en NF4: index.json, weights/ (una capa por fichero), others.safetensors
    transformer_edit/  Edit-2511 en NF4, igual
    text_encoder_BF16/ Qwen2.5-VL-7B en BF16, sin cambios (~16.6 GB)
    text_encoder_INT8/ el mismo en INT8 (bitsandbytes LLM.int8, ~9 GB)
    text_encoder_NF4/  el mismo en NF4 (bitsandbytes, ~5.5 GB)
                       En las dos cuantizadas la torre de visión queda en BF16: la edición la usa.
    processor/         de Edit-2511 (el pre-caché de edición pasa la imagen al text encoder)
    tokenizer/, vae/, scheduler/, LICENSE y model_index.json copiados de 2512

Lee el transformer tensor a tensor (41 GB en BF16): funciona con poca RAM y también sin GPU (más despacio).
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
from transformers import AutoModel, BitsAndBytesConfig

T2I_DIR = sys.argv[1] if len(sys.argv) > 1 else "./Models/Qwen-Image-2512"
EDIT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./Models/Qwen-Image-Edit-2511"
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "./Models/Qwen-Image-2512-NF4"
# Solo el transformer (p. ej. para añadir Edit-2511 a una salida que ya tiene el text encoder).
SKIP_TEXT_ENCODER = os.environ.get("SKIP_TEXT_ENCODER") == "1"

# Las capas lineales de los 60 bloques, modulaciones incluidas (img_mod / txt_mod: 3072 -> 18432,
# casi 7 GB en BF16). Los embedders (img_in, txt_in, time_text_embed) y la salida quedan en BF16.
QUANT_BLOCKS = ("transformer_blocks.",)
# La modulación del primer bloque se queda en BF16 (113 MB): en NF4 deja grano por toda la imagen con el
# muestreo normal (20-50 pasos); con ella en BF16 la imagen sale como la del modelo original. Es la misma
# capa que la documentación de diffusers excluye al cargar Qwen-Image en 4 bits (llm_int8_skip_modules).
SKIP_LAYERS = ("transformer_blocks.0.img_mod.1",)
BLOCKSIZE = 64
COMPRESS_STATISTICS = True
COPY_IGNORE = ("transformer", "text_encoder", ".cache", "*.lock", "README.md")


def json_safe(value):
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def convert_transformer(src, dst, device, source_repo):
    with open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json"), "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    by_file = {}
    for name, file in weight_map.items():
        by_file.setdefault(file, []).append(name)

    weights_dir = os.path.join(dst, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    index = {"quantized": {}, "unquantized": {}, "others": "others.safetensors"}
    others = {}
    total = sum(1 for n in weight_map if n.startswith(QUANT_BLOCKS) and n.endswith(".weight") and "norm" not in n
                and n[:-len(".weight")] not in SKIP_LAYERS)

    for file, names in sorted(by_file.items()):
        with safe_open(os.path.join(src, file), framework="pt", device="cpu") as f:
            for name in sorted(names):
                tensor = f.get_tensor(name)
                if not (name.endswith(".weight") and tensor.dim() == 2 and name.startswith(QUANT_BLOCKS))                         or name[:-len(".weight")] in SKIP_LAYERS:
                    # Normas, bias (también los de las capas NF4: el loader los asigna a su Linear4bit) y embedders.
                    others[name] = tensor.to(torch.bfloat16).contiguous()
                    continue
                layer = name[:-len(".weight")]
                packed, qs = quantize_4bit(tensor.to(device=device, dtype=torch.float32).contiguous(),
                                           blocksize=BLOCKSIZE, compress_statistics=COMPRESS_STATISTICS,
                                           quant_type="nf4", quant_storage=torch.uint8)
                qs_dict = qs.as_dict(packed=True)
                tensors = {"weight": packed.cpu().contiguous()}
                out_f, in_f = tensor.shape
                has_bias = layer + ".bias" in weight_map
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
                if len(index["quantized"]) % 50 == 0:
                    print(f"  {len(index['quantized'])}/{total} capas NF4...", flush=True)
                del tensor, packed, qs

    save_file(others, os.path.join(dst, "others.safetensors"))
    with open(os.path.join(dst, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    shutil.copy2(os.path.join(src, "config.json"), dst)
    with open(os.path.join(dst, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "QwenImage2512-NF4", "version": 1, "source": source_repo,
                   "dtype": "bfloat16", "quant_type": "nf4", "compute_dtype": "bfloat16",
                   "blocksize": BLOCKSIZE, "compress_statistics": COMPRESS_STATISTICS,
                   "quantized_layers": len(index["quantized"]), "unquantized_layers": 0, "other_tensors": len(others),
                   "quant_blocks": list(QUANT_BLOCKS), "skip_layers": list(SKIP_LAYERS), "bitsandbytes_prequantized": True,
                   "quant_state_serialization": "as_dict(packed=True)", "loader": "Params4bit + QuantState.from_dict"},
                  f, indent=2)
    return len(index["quantized"]), len(others)


TEXT_ENCODER_QUANT = {
    "text_encoder_INT8": BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"]),
    "text_encoder_NF4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"]),
}


def convert_text_encoder(src, dst, config):
    # Se cuantiza al cargar (capa a capa en la GPU) y se guarda ya cuantizado: el pre-caché lo carga
    # con su quantization_config, sin leer los 16.6 GB en BF16. AutoModel = Qwen2_5_VLModel, sin lm_head:
    # el pre-caché solo lee hidden states.
    model = AutoModel.from_pretrained(src, dtype=torch.bfloat16, quantization_config=config, device_map={"": 0})
    shutil.rmtree(dst, ignore_errors=True)
    model.save_pretrained(dst, max_shard_size="5GB")
    del model
    torch.cuda.empty_cache()


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" QWEN-IMAGE-2512 / EDIT-2511 -> NF4")
    print("=" * 70)
    print("2512     :", os.path.abspath(T2I_DIR))
    print("Edit-2511:", os.path.abspath(EDIT_DIR) if EDIT_DIR else "-")
    print("Salida   :", os.path.abspath(OUTPUT_DIR))
    print("Cuantiza en:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (más lento)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isdir(os.path.join(T2I_DIR, "vae")):
        shutil.copytree(T2I_DIR, OUTPUT_DIR, ignore=shutil.ignore_patterns(*COPY_IGNORE), dirs_exist_ok=True)
    if EDIT_DIR and os.path.isdir(os.path.join(EDIT_DIR, "processor")):
        shutil.copytree(os.path.join(EDIT_DIR, "processor"), os.path.join(OUTPUT_DIR, "processor"), dirs_exist_ok=True)

    for src, folder, repo in ((T2I_DIR, "transformer", "Qwen/Qwen-Image-2512"),
                              (EDIT_DIR, "transformer_edit", "Qwen/Qwen-Image-Edit-2511")):
        if not src or not os.path.isdir(os.path.join(src, "transformer")):
            print(f"[AVISO] Sin {folder}: no se encontró {src}/transformer")
            continue
        dst = os.path.join(OUTPUT_DIR, folder)
        shutil.rmtree(dst, ignore_errors=True)
        print(f"\n{repo} -> {folder}/")
        q, o = convert_transformer(os.path.join(src, "transformer"), dst, device, repo)
        print(f"Capas NF4: {q} | tensores BF16 en others.safetensors: {o}")

    te_src = os.path.join(T2I_DIR, "text_encoder")
    if not SKIP_TEXT_ENCODER and os.path.isdir(te_src):
        shutil.copytree(te_src, os.path.join(OUTPUT_DIR, "text_encoder_BF16"),
                        ignore=shutil.ignore_patterns(".cache"), dirs_exist_ok=True)
        for folder, config in TEXT_ENCODER_QUANT.items():
            print(f"\nText encoder Qwen2.5-VL-7B -> {folder}/")
            convert_text_encoder(te_src, os.path.join(OUTPUT_DIR, folder), config)
    print(f"\nHecho en {(time.time() - t0) / 60:.1f} min -> {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
