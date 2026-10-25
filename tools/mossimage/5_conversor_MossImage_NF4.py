# -*- coding: utf-8 -*-
"""
5_conversor_MossImage_NF4.py — MOSS-Image Base y SFT -> caché NF4 para entrenar LoRAs.
MOSS-Image Base and SFT -> NF4 cache for LoRA training.

Uso / usage:
    python 5_conversor_MossImage_NF4.py <MOSS-Image-Base> <MOSS-Image-SFT o ""> <salida>

Entrada: las carpetas diffusers de OpenMOSS-Team/MOSS-Image-Base y MOSS-Image-SFT. Los dos repos
comparten text encoder, tokenizer y VAE (mismos ficheros); solo cambia el transformer, así que de
SFT basta con transformer/.

Salida (mismo formato que FLUX.3 Klein):
    transformer/       Base en NF4: index.json, weights/ (una capa por fichero), others.safetensors
    transformer_sft/   SFT en NF4, igual
    text_encoder_BF16/ Qwen3.5-9B en BF16, sin cambios (~17 GB): exacto; el pre-caché lo reparte entre GPU y RAM
    text_encoder_INT8/ el mismo en INT8 (bitsandbytes LLM.int8, ~10 GB)
    text_encoder_NF4/  el mismo en NF4 (bitsandbytes, ~6 GB)
                       En las dos versiones cuantizadas la torre de visión, que el texto no usa, queda en BF16.
    tokenizer/, vae/, scheduler/, LICENSE y model_index.json copiados de Base

Lee el transformer tensor a tensor: funciona con poca RAM y también sin GPU (más despacio).
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

BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "./Models/MOSS-Image-Base"
SFT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./Models/MOSS-Image-SFT"
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "./Models/MOSS-Image-NF4"

# Las capas lineales de los 42 bloques. El refinador de texto (text_embedder), los embedders,
# las modulaciones y la salida se quedan en BF16.
QUANT_BLOCKS = ("layers.",)
BLOCKSIZE = 64
COMPRESS_STATISTICS = True
COPY_IGNORE = ("transformer", "text_encoder", "assets", ".cache", "*.lock", "*.py", "README.md")


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
    others, total = {}, sum(1 for n in weight_map if n.startswith(QUANT_BLOCKS) and n.endswith(".weight") and "norm" not in n)

    for file, names in sorted(by_file.items()):
        with safe_open(os.path.join(src, file), framework="pt", device="cpu") as f:
            for name in sorted(names):
                tensor = f.get_tensor(name)
                if not (name.endswith(".weight") and tensor.dim() == 2 and name.startswith(QUANT_BLOCKS)):
                    others[name] = tensor.to(torch.float32 if name == "rope.bands" else torch.bfloat16).contiguous()
                    continue
                layer = name[:-len(".weight")]
                packed, qs = quantize_4bit(tensor.to(device=device, dtype=torch.float32).contiguous(),
                                           blocksize=BLOCKSIZE, compress_statistics=COMPRESS_STATISTICS,
                                           quant_type="nf4", quant_storage=torch.uint8)
                qs_dict = qs.as_dict(packed=True)
                tensors = {"weight": packed.cpu().contiguous()}
                out_f, in_f = tensor.shape
                metadata = {"layer_name": layer, "type": "Linear4bit", "quant_type": "nf4", "bnb_quantized": "true",
                            "in_features": str(in_f), "out_features": str(out_f), "bias": "False"}
                for key, value in qs_dict.items():
                    if torch.is_tensor(value):
                        tensors["quant_state." + key] = value.detach().cpu().contiguous()
                    else:
                        metadata["qs_" + key] = json.dumps(json_safe(value))
                filename = layer.replace(".", "__") + ".safetensors"
                save_file(tensors, os.path.join(weights_dir, filename), metadata=metadata)
                index["quantized"][layer] = {"file": filename, "in_features": in_f, "out_features": out_f,
                                             "bias": False, "quant_type": "nf4", "compute_dtype": "bfloat16",
                                             "quant_state_keys": list(qs_dict.keys())}
                if len(index["quantized"]) % 25 == 0:
                    print(f"  {len(index['quantized'])}/{total} capas NF4...", flush=True)
                del tensor, packed, qs

    save_file(others, os.path.join(dst, "others.safetensors"))
    with open(os.path.join(dst, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    shutil.copy2(os.path.join(src, "config.json"), dst)
    with open(os.path.join(dst, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "MossImage-NF4", "version": 1, "source": source_repo,
                   "dtype": "bfloat16", "quant_type": "nf4", "compute_dtype": "bfloat16",
                   "blocksize": BLOCKSIZE, "compress_statistics": COMPRESS_STATISTICS,
                   "quantized_layers": len(index["quantized"]), "unquantized_layers": 0, "other_tensors": len(others),
                   "quant_blocks": list(QUANT_BLOCKS), "bitsandbytes_prequantized": True,
                   "quant_state_serialization": "as_dict(packed=True)", "loader": "Params4bit + QuantState.from_dict"},
                  f, indent=2)
    return len(index["quantized"]), len(others)


TEXT_ENCODER_QUANT = {
    "text_encoder_INT8": BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"]),
    "text_encoder_NF4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual"]),
}


def convert_text_encoder(src, dst, config):
    # Se cuantiza al cargar (capa a capa en la GPU; INT8 llega a ~13 GB de pico) y se guarda ya
    # cuantizado: los scripts lo cargan con su quantization_config, sin leer los 17 GB en BF16.
    model = AutoModel.from_pretrained(src, dtype=torch.bfloat16, quantization_config=config, device_map={"": 0})
    shutil.rmtree(dst, ignore_errors=True)
    model.save_pretrained(dst, max_shard_size="5GB")
    for name in ("LICENSE", "README.md"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), dst)
    del model
    torch.cuda.empty_cache()


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" MOSS-IMAGE BASE / SFT -> NF4")
    print("=" * 70)
    print("Base:", os.path.abspath(BASE_DIR))
    print("SFT :", os.path.abspath(SFT_DIR) if SFT_DIR else "-")
    print("Salida:", os.path.abspath(OUTPUT_DIR))
    print("Cuantiza en:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (más lento)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shutil.copytree(BASE_DIR, OUTPUT_DIR, ignore=shutil.ignore_patterns(*COPY_IGNORE), dirs_exist_ok=True)
    shutil.copytree(os.path.join(BASE_DIR, "text_encoder"), os.path.join(OUTPUT_DIR, "text_encoder_BF16"),
                    ignore=shutil.ignore_patterns(".cache"), dirs_exist_ok=True)

    for src, folder, repo in ((BASE_DIR, "transformer", "OpenMOSS-Team/MOSS-Image-Base"),
                              (SFT_DIR, "transformer_sft", "OpenMOSS-Team/MOSS-Image-SFT")):
        if not src or not os.path.isdir(os.path.join(src, "transformer")):
            print(f"[AVISO] Sin {folder}: no se encontró {src}/transformer")
            continue
        dst = os.path.join(OUTPUT_DIR, folder)
        shutil.rmtree(dst, ignore_errors=True)
        print(f"\n{repo} -> {folder}/")
        q, o = convert_transformer(os.path.join(src, "transformer"), dst, device, repo)
        print(f"Capas NF4: {q} | tensores BF16 en others.safetensors: {o}")
    for folder, config in TEXT_ENCODER_QUANT.items():
        print(f"\nText encoder Qwen3.5-9B -> {folder}/")
        convert_text_encoder(os.path.join(BASE_DIR, "text_encoder"), os.path.join(OUTPUT_DIR, folder), config)
    print(f"\nHecho en {(time.time() - t0) / 60:.1f} min -> {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
