# -*- coding: utf-8 -*-
"""
moss_edit_lora.py — Fusiona un LoRA de edición de MOSS-Image con el LoRA chinoll/Moss-Image-Edit.
Merges a MOSS-Image edit LoRA with the chinoll/Moss-Image-Edit LoRA.

Un LoRA de edición se entrena encima de Moss-Image-Edit, así que en ComfyUI hacen falta los dos.
Este módulo los junta en un único parche con el formato "moss-edit-lora" que lee el nodo
Load Model Patch de ComfyUI-Moss-Image-Edit (carpeta models/model_patches), para usarlo solo con
Apply MOSS Edit.

La suma de dos LoRAs es exacta si se concatenan sus matrices: A = [A_edit; A_lora] y
B = [B_edit·escala_edit, B_lora·escala_lora], con rank la suma de los dos y alpha = rank (escala 1).
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

EDIT_FILE = os.path.join("LoRAs", "Moss-Image-Edit", "adapter_model.safetensors")


class FuseError(ValueError):
    pass


def is_edit_lora(path):
    """True si el .safetensors es un LoRA de edición entrenado por LoRAlab (lleva edit_base_lora)."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return bool((f.metadata() or {}).get("edit_base_lora"))
    except Exception:
        return False


def fuse(lora_path, edit_path, out_path):
    """Escribe en out_path el parche Moss-Image-Edit con el LoRA de lora_path ya sumado."""
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    if not meta.get("edit_base_lora"):
        raise FuseError("This LoRA was not trained as an edit LoRA on top of Moss-Image-Edit.")
    if meta.get("ss_network_module") == "lycoris.kohya":
        raise FuseError("A LoKr cannot be merged into the Moss-Image-Edit patch: export it separately.")
    if not os.path.exists(edit_path):
        raise FuseError(f"Moss-Image-Edit LoRA not found: {edit_path}")

    with safe_open(str(edit_path), framework="pt", device="cpu") as f:
        config = json.loads((f.metadata() or {})["adapter_config"])
    edit = load_file(str(edit_path))
    lora = load_file(str(lora_path))
    strength = float(meta.get("edit_base_lora_strength", 1.0))

    layers = {k[len("diffusion_model."):-len(".lora_A.weight")] for k in lora if k.endswith(".lora_A.weight")}
    unknown = layers - set(config["modules"])
    if unknown:
        raise FuseError(f"The LoRA has layers the Moss-Image-Edit patch does not: {sorted(unknown)[:3]}")

    out = {}
    for name, spec in config["modules"].items():
        a = [edit[name + ".lora_A"].float()]
        b = [edit[name + ".lora_B"].float() * (spec["alpha"] / spec["rank"] * strength)]
        if name in layers:
            key = "diffusion_model." + name
            down, up = lora[key + ".lora_A.weight"].float(), lora[key + ".lora_B.weight"].float()
            alpha = float(lora[key + ".alpha"]) if key + ".alpha" in lora else down.shape[0]
            a.append(down)
            b.append(up * (alpha / down.shape[0]))
        out[name + ".lora_A"] = torch.cat(a, 0).to(torch.bfloat16).contiguous()
        out[name + ".lora_B"] = torch.cat(b, 1).to(torch.bfloat16).contiguous()
        rank = out[name + ".lora_A"].shape[0]
        spec.update(rank=rank, alpha=float(rank))

    # weights_sha256 era el hash de los pesos originales: ya no lo es.
    config.pop("weights_sha256", None)
    config["metadata"]["merged_lora"] = {k: meta[k] for k in ("ss_output_name", "trigger_word", "ss_steps",
                                                               "edit_base_lora", "edit_base_lora_revision")
                                         if k in meta}
    header = {"adapter_config": json.dumps(config, indent=2)}
    for k in ("trigger_word", "project_name", "ss_output_name"):
        if k in meta:
            header[k] = meta[k]
    header["trained_with"] = meta.get("trained_with", "AcademiaSD LoRAlab MOSS-Image")
    tmp = str(out_path) + ".tmp"
    save_file(out, tmp, metadata=header)
    os.replace(tmp, str(out_path))
