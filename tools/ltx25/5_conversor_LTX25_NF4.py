# -*- coding: utf-8 -*-
"""
5_conversor_LTX25_NF4.py — LTX-2.5 (Diffusers) -> NF4 for LoRA training / NF4 para entrenar LoRAs.

Lee ./Models/LTX25-Raw (lo que baja 4_descarga_LTX25.py) y escribe ./Models/LTX25-NF4 con el MISMO formato que
Models/LTX23-NF4, para que el loader de LTX 2.3 sirva:

    config.json              config del transformer (transformer_full = DiT completo / SFT, el que se entrena)
    index.json               {"quantized": {...}, "unquantized": {...}}  un fichero por Linear en weights/
    weights/*.safetensors    Linear de los 48 bloques en NF4 (weight + quant_state.* empaquetado, double quant)
                             Linear fuera de los bloques (proj_in/out, AdaLN, timestep, caption) en BF16
    others.safetensors       normas, tablas de modulación, rope, registros... (dtype original)
    text_encoder_NF4/        Gemma 4 12B de LTX en NF4 (bitsandbytes, igual que el Gemma 3 de 2.3)
    connectors/ vae/ audio_vae/ vocoder/ scheduler/ tokenizer/ duration_head/ model_index.json   copiados

Diferencias con el conversor de 2.3, ambas por la memoria:
  * Streaming: nunca carga el transformer entero (38 GB). Construye el modelo vacío para saber qué es Linear y
    lee cada tensor de su shard con safe_open (mmap). RAM ~ un tensor; VRAM ~ una capa.
  * Solo se cuantizan los Linear de transformer_blocks.* Las capas de entrada/salida y de timestep quedan en BF16
    (~0.85 GB): en MiniMax-H3 cuantizarlas rompía la cancelación del AdaLN final, y en un perfil de 8 GB caben.

Reanudable: cada capa se guarda y se apunta en index.json en cuanto está hecha.

Uso / Usage:
    python tools/ltx25/5_conversor_LTX25_NF4.py                 # todo
    python tools/ltx25/5_conversor_LTX25_NF4.py --only transformer|text_encoder|copy
    python tools/ltx25/5_conversor_LTX25_NF4.py --validate-te    # compara Gemma NF4 contra BF16 (CPU, lento)
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RAW = "./Models/LTX25-Raw"
OUT = "./Models/LTX25-NF4"
FORMAT = "LTX25-NF4"
VERSION = 1
COMPUTE_DTYPE = torch.bfloat16
QUANT_TYPE = "nf4"

COPY_DIRS = ["connectors", "vae", "audio_vae", "vocoder", "scheduler", "tokenizer", "duration_head"]


# ----------------------------------------------------------------------------------------------------------------
def atomic_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_filename(name):
    return name.replace(".", "__")


def file_ok(path):
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return "weight" in f.keys()
    except Exception:
        return False


def fmt_gb(n):
    return "%.2f GB" % (n / 1e9)


class ShardReader:
    """Tensores de un modelo diffusers repartido en shards, leídos uno a uno (mmap)."""

    def __init__(self, folder):
        idx = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(idx):
            self.weight_map = load_json(idx, {})["weight_map"]
        else:
            single = "diffusion_pytorch_model.safetensors"
            with safe_open(os.path.join(folder, single), framework="pt", device="cpu") as f:
                self.weight_map = {k: single for k in f.keys()}
        self.folder = folder
        self.handles = {}

    def keys(self):
        return self.weight_map.keys()

    def get(self, key):
        shard = self.weight_map[key]
        h = self.handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.folder, shard), framework="pt", device="cpu")
            self.handles[shard] = h
        return h.get_tensor(key)


# ----------------------------------------------------------------------------------------------------------------
def quantize_linear(name, weight, bias, path):
    from bitsandbytes.nn import Linear4bit, Params4bit

    out_f, in_f = weight.shape
    layer = Linear4bit(in_f, out_f, bias=bias is not None, compute_dtype=COMPUTE_DTYPE,
                       compress_statistics=True, quant_type=QUANT_TYPE, device="meta")
    layer.weight = Params4bit(weight.float().contiguous(), requires_grad=False, quant_type=QUANT_TYPE,
                              compress_statistics=True)
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias.to(COMPUTE_DTYPE).contiguous(), requires_grad=False)
    layer = layer.to("cuda")
    qw = layer.weight
    if not getattr(qw, "bnb_quantized", False) or getattr(qw, "quant_state", None) is None:
        raise RuntimeError(f"{name}: NF4 not materialized / NF4 no materializado")
    tensors = {"weight": qw.detach().cpu().contiguous()}
    if layer.bias is not None:
        tensors["bias"] = layer.bias.detach().cpu().contiguous()
    meta = {"format": FORMAT, "version": str(VERSION), "layer_name": name, "type": "Linear4bit",
            "quant_type": QUANT_TYPE, "compute_dtype": "bfloat16", "in_features": str(in_f),
            "out_features": str(out_f), "bias": str(bias is not None)}
    for k, v in qw.quant_state.as_dict(packed=True).items():
        if torch.is_tensor(v):
            tensors["quant_state." + k] = v.detach().cpu().contiguous()
        else:
            meta["quant_state_" + k] = json.dumps(v)
    save_file(tensors, path, metadata=meta)
    del layer, qw, tensors
    return in_f, out_f


def save_bf16_linear(name, weight, bias, path):
    out_f, in_f = weight.shape
    tensors = {"weight": weight.to(COMPUTE_DTYPE).contiguous()}
    if bias is not None:
        tensors["bias"] = bias.to(COMPUTE_DTYPE).contiguous()
    save_file(tensors, path, metadata={"format": FORMAT, "layer_name": name, "type": "Linear",
                                       "in_features": str(in_f), "out_features": str(out_f),
                                       "bias": str(bias is not None)})
    return in_f, out_f


def convert_transformer():
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel

    src = os.path.join(RAW, "transformer_full")
    if not os.path.isfile(os.path.join(src, "config.json")):
        sys.exit(f"[!] Missing / falta {src}. Run 4_descarga_LTX25.py first / primero.")
    if not torch.cuda.is_available():
        sys.exit("[!] CUDA not available / CUDA no disponible.")

    weights_dir = os.path.join(OUT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    shutil.copy2(os.path.join(src, "config.json"), os.path.join(OUT, "config.json"))

    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(src))
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    reader = ShardReader(src)

    index_path = os.path.join(OUT, "index.json")
    index = load_json(index_path, {"format": FORMAT, "version": VERSION, "quantized": {}, "unquantized": {}})
    linear_keys = set()
    t0 = time.time()
    total = len(linears)
    print(f"Linear layers / capas Linear: {total}")

    for i, (name, mod) in enumerate(linears.items(), 1):
        wkey, bkey = name + ".weight", name + ".bias"
        linear_keys.update((wkey, bkey))
        in_blocks = name.startswith("transformer_blocks.")
        section = "quantized" if in_blocks else "unquantized"
        fname = safe_filename(name) + ".safetensors"
        path = os.path.join(weights_dir, fname)
        if name in index[section] and file_ok(path):
            continue
        weight = reader.get(wkey)
        bias = reader.get(bkey) if bkey in reader.weight_map else None
        if tuple(weight.shape) != (mod.out_features, mod.in_features):
            raise RuntimeError(f"{name}: shape {tuple(weight.shape)} != config")
        if in_blocks:
            in_f, out_f = quantize_linear(name, weight, bias, path)
        else:
            in_f, out_f = save_bf16_linear(name, weight, bias, path)
        if not file_ok(path):
            raise RuntimeError(f"{name}: invalid file / fichero inválido")
        index[section][name] = {"file": fname, "file_size": os.path.getsize(path), "in_features": in_f,
                                "out_features": out_f, "bias": bias is not None,
                                "quant_type": QUANT_TYPE if in_blocks else "bf16",
                                "compute_dtype": "bfloat16", "status": "saved"}
        atomic_json(index, index_path)
        del weight, bias
        if i % 25 == 0 or i == total:
            torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"  [{i}/{total}] {name}  ({el:.0f} s)", flush=True)

    # Todo lo que no es Linear: normas, tablas de modulación, registros, embeddings sueltos.
    others = {}
    for k in reader.keys():
        if k not in linear_keys:
            others[k] = reader.get(k).contiguous()
    expected = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    missing = sorted(expected - linear_keys - set(others))
    if missing:
        print("[!] Not in checkpoint (buffers computed at load?) / no están en el checkpoint:", missing[:8])
    save_file(others, os.path.join(OUT, "others.safetensors"))
    print(f"others.safetensors: {len(others)} tensors, {fmt_gb(sum(t.numel() * t.element_size() for t in others.values()))}")

    q = sum(v["file_size"] for v in index["quantized"].values())
    u = sum(v["file_size"] for v in index["unquantized"].values())
    atomic_json({"format": FORMAT, "version": VERSION, "source": "Lightricks/LTX-2.5-Diffusers/transformer_full",
                 "quant_type": QUANT_TYPE, "compute_dtype": "bfloat16", "double_quant": True,
                 "quantized": len(index["quantized"]), "unquantized_bf16": len(index["unquantized"]),
                 "nf4_bytes": q, "bf16_linear_bytes": u,
                 "loader_note": "QuantState serialized with quant_state.as_dict(packed=True)"},
                os.path.join(OUT, "metadata.json"))
    print(f"[OK] transformer: NF4 {fmt_gb(q)} + BF16 {fmt_gb(u)} in {time.time() - t0:.0f} s")
    reader.handles.clear()
    del model
    gc.collect()


# ----------------------------------------------------------------------------------------------------------------
def te_class():
    import transformers
    for name in ("Gemma4UnifiedForConditionalGeneration", "Gemma4ForConditionalGeneration"):
        if hasattr(transformers, name):
            return getattr(transformers, name)
    raise RuntimeError("transformers without Gemma 4 / sin Gemma 4")


def convert_text_encoder():
    from transformers import BitsAndBytesConfig

    src = os.path.join(RAW, "text_encoder")
    dst = os.path.join(OUT, "text_encoder_NF4")
    if os.path.isfile(os.path.join(dst, "config.json")):
        print("[skip] text_encoder_NF4 exists / ya existe")
        return
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    print("Gemma 4 -> NF4 ...")
    model = te_class().from_pretrained(src, quantization_config=cfg, torch_dtype=torch.bfloat16,
                                       device_map={"": 0}, low_cpu_mem_usage=True)
    tmp = dst + ".tmp"
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="20GB")
    for extra in ("generation_config.json",):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copy2(os.path.join(src, extra), os.path.join(tmp, extra))
    os.replace(tmp, dst)
    print(f"[OK] text_encoder_NF4 in {time.time() - t0:.0f} s, VRAM peak {fmt_gb(torch.cuda.max_memory_allocated())}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def validate_text_encoder():
    """Coseno por token de los 49 estados apilados (lo que ven los connectors): NF4 en GPU contra BF16 en CPU."""
    from transformers import AutoTokenizer

    prompts = [
        "A woman with short red hair walks through a rainy neon street at night, the camera tracking beside her, "
        "puddles reflecting pink and blue light, the hiss of rain and distant traffic.",
        "Close-up of an old fisherman mending a net on a wooden pier at sunrise, gulls calling overhead.",
        "TOK, a watercolor illustration of a fox in a snowy forest.",
    ]
    tok = AutoTokenizer.from_pretrained(os.path.join(RAW, "tokenizer"))
    cls = te_class()

    def run(model, device):
        outs = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(device)
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            outs.append(torch.stack(hs, dim=-1).float().cpu())
        return outs

    q = cls.from_pretrained(os.path.join(OUT, "text_encoder_NF4"), device_map={"": 0}, torch_dtype=torch.bfloat16)
    a = run(q, "cuda")
    del q
    torch.cuda.empty_cache()
    print("BF16 on CPU (slow / lento)...")
    r = cls.from_pretrained(os.path.join(RAW, "text_encoder"), torch_dtype=torch.bfloat16, device_map={"": "cpu"})
    b = run(r, "cpu")
    del r
    for p, x, y in zip(prompts, a, b):
        cos = torch.nn.functional.cosine_similarity(x, y, dim=2)  # [1, tokens, layers]
        per_layer = cos.mean(dim=(0, 1))
        print(f"  {p[:40]!r}: mean cos {cos.mean():.4f}, worst layer {per_layer.min():.4f} "
              f"(layer {int(per_layer.argmin())}), last {per_layer[-1]:.4f}")


# ----------------------------------------------------------------------------------------------------------------
def copy_components():
    for d in COPY_DIRS:
        s, t = os.path.join(RAW, d), os.path.join(OUT, d)
        if not os.path.isdir(s):
            print(f"[!] missing / falta {s}")
            continue
        os.makedirs(t, exist_ok=True)
        for fn in os.listdir(s):
            if os.path.isfile(os.path.join(s, fn)) and not os.path.exists(os.path.join(t, fn)):
                shutil.copy2(os.path.join(s, fn), os.path.join(t, fn))
    # El scheduler/ del repo oficial es el del modelo DESTILADO (sin dynamic shifting). Se entrena y se previsualiza
    # el completo (transformer_full), que según la model card necesita use_dynamic_shifting=True y shift_terminal=0.1.
    sched = os.path.join(OUT, "scheduler", "scheduler_config.json")
    sc = load_json(sched, None)
    if sc is not None:
        sc.update(use_dynamic_shifting=True, shift_terminal=0.1)
        atomic_json(sc, sched)
    mi = load_json(os.path.join(RAW, "model_index.json"), None)
    if mi is not None:
        for k in ("prompt_enhancer", "processor", "latent_upsampler", "temporal_latent_upsampler", "diffusion_decoder"):
            if k in mi:
                mi[k] = [None, None]
        atomic_json(mi, os.path.join(OUT, "model_index.json"))
    # Ficha del repo NF4 y copia completa de la licencia (obligatoria, seccion 3.2 de la LTX-2.x Community License).
    here = os.path.dirname(os.path.abspath(__file__))
    for src, dst in (("README_NF4.md", "README.md"), ("LICENSE-2_x.txt", "LICENSE")):
        if os.path.exists(os.path.join(here, src)):
            shutil.copy2(os.path.join(here, src), os.path.join(OUT, dst))
    print("[OK] components copied / componentes copiados")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["transformer", "text_encoder", "copy"])
    ap.add_argument("--validate-te", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.validate_te:
        validate_text_encoder()
        return
    if args.only in (None, "copy"):
        copy_components()
    if args.only in (None, "transformer"):
        convert_transformer()
    if args.only in (None, "text_encoder"):
        convert_text_encoder()


if __name__ == "__main__":
    main()
