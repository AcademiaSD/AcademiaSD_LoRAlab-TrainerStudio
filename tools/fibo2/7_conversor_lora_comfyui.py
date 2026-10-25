# -*- coding: utf-8 -*-
"""
7_conversor_lora_comfyui.py — Convierte un LoRA de Fibo 2.0 al formato que carga ComfyUI.

ComfyUI no carga el transformer de diffusers: carga su propio repack, donde la atencion
(qkv), el MLP (w13) y la atencion cruzada (to_kv) estan fundidas en una sola matriz. Un
LoRA entrenado con el trainer (claves "transformer.*") no se carga tal cual.

Este script reescribe las claves sin reentrenar: las capas que se funden se rearman en un
LoRA de rango multiple (down = [A1|A2|...], up = diag(B1,B2,...)), de modo que el cambio
que aplica ComfyUI es EXACTAMENTE el mismo que el del LoRA original.

    python tools/fibo2/7_conversor_lora_comfyui.py <lora.safetensors> [-o salida.safetensors]

Si no se dice nada, la salida es <lora>_comfyui.safetensors en la misma carpeta.
"""
import argparse
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

STUDIO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(STUDIO, "scripts"))
import fibo2_lora


def read_lora(path):
    """Las parejas A/B de cada capa y el alpha de cada una."""
    pairs, alphas, meta = {}, {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            if key.endswith(".alpha"):
                module = key[: -len(".alpha")]
                module = _strip_prefix(module)
                alphas[module] = float(f.get_tensor(key).item())
                continue
            if ".lora_A.weight" in key:
                module = _strip_prefix(key.split(".lora_A.weight")[0])
                pairs.setdefault(module, {})["A"] = f.get_tensor(key)
            elif ".lora_B.weight" in key:
                module = _strip_prefix(key.split(".lora_B.weight")[0])
                pairs.setdefault(module, {})["B"] = f.get_tensor(key)
            else:
                print("  [!] Clave que no se sabe convertir, se copia tal cual:", key)
    return pairs, alphas, meta


def _strip_prefix(module):
    """Quita el prefijo del trainer ("transformer.") o el de ComfyUI ("diffusion_model.")."""
    for prefix in ("transformer.", "diffusion_model.", "unet."):
        if module.startswith(prefix):
            return module[len(prefix):]
    return module


def main():
    parser = argparse.ArgumentParser(description="Convierte un LoRA de Fibo 2.0 al formato de ComfyUI.")
    parser.add_argument("lora", help="El LoRA que hay que convertir.")
    parser.add_argument("-o", "--output", default=None, help="El LoRA de salida.")
    parser.add_argument("--force", action="store_true", help="Sobrescribe la salida si existe.")
    args = parser.parse_args()

    src = os.path.abspath(args.lora)
    if not os.path.exists(src):
        print(f"[ERROR] No existe: {src}")
        return 1
    out = os.path.abspath(args.output) if args.output else src[:-len(".safetensors")] + "_comfyui.safetensors"

    if os.path.exists(out) and not args.force:
        print(f"[ERROR] Ya existe {out}. Usa --force para sobrescribirlo.")
        return 1

    print(f"Origen: {src}")
    pairs, alphas, meta = read_lora(src)
    print(f"  capas: {len(pairs)}")

    if not pairs:
        print("[ERROR] El fichero no tiene LoRA (lora_A / lora_B).")
        return 1

    distintos = set(round(v, 6) for v in alphas.values())
    alpha = distintos.pop() if len(distintos) == 1 else None
    if alpha is None:
        print("  [!] Los alphas no son iguales: se usa el de cada capa.")
        alpha = alphas

    clean, skipped = fibo2_lora.build_lora(pairs, alpha)
    if skipped:
        print(f"  [!] {len(skipped)} capas no se pudieron mapear: {skipped[:5]}")
    print(f"  claves de salida: {len(clean)}")

    save_file(clean, out, metadata=meta or None)
    print(f"Listo: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("Copia este fichero en models/loras de ComfyUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
