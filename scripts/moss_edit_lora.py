# -*- coding: utf-8 -*-
"""
moss_edit_lora.py — LoRAs de edición base de MOSS-Image y fusión con ellos.
Base edit LoRAs for MOSS-Image and merging with them.

Un LoRA de edición se entrena encima de un LoRA de edición base (armV4 o chinoll), así que en
ComfyUI hacen falta los dos. fuse() los junta en un único LoRA estándar (claves diffusion_model.*,
las del cargador de LoRA de ComfyUI), que se usa solo con Text Encode MOSS-Image Edit y el
edit_format de su base.

La suma de dos LoRAs es exacta si se concatenan sus matrices: A = [A_base; A_lora] y
B = [B_base·escala_base, B_lora·escala_lora], con rank la suma de los dos y alpha = rank (escala 1).
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# LoRAs de edición base; se fija la revisión probada. layout es la posición de las referencias en el
# modelo (MOSSImageTransformer.reference_layout) y edit_format, la opción del nodo de ComfyUI.
#   armV4: akhaliq/MOSS-Image-Edit-HQ (Apache-2.0), formato ai-toolkit: rank 192 en q/k/v y 64 en el
#          resto, sin alpha (escala 1).
#   chinoll: chinoll/Moss-Image-Edit, rank 64, formato "moss-edit-lora" (alpha en adapter_config).
EDIT_BASES = {
    "armV4": {"repo": "akhaliq/MOSS-Image-Edit-HQ", "revision": "e67d778c32f20756dd9bd1fc2c1d0f06c3c34b42",
              "files": ("weights/armV4/armV4.safetensors",), "folder": os.path.join("LoRAs", "MOSS-Image-Edit-HQ"),
              "mb": 495, "layout": "ai-toolkit", "edit_format": "ai-toolkit"},
    "chinoll": {"repo": "chinoll/Moss-Image-Edit", "revision": "3be9c40037e1b89b17aa85ff6f8ee805872aa558",
                "files": ("adapter_config.json", "adapter_model.safetensors"), "folder": os.path.join("LoRAs", "Moss-Image-Edit"),
                "mb": 300, "layout": "moss-image-edit", "edit_format": "Moss-Image-Edit"},
}


class FuseError(ValueError):
    pass


def base_file(model_dir, name):
    """Ruta local del LoRA de edición base name dentro del modelo."""
    base = EDIT_BASES[name]
    return os.path.join(str(model_dir), base["folder"], base["files"][-1])


def base_layers(path, device="cpu"):
    """{capa: (lora_A, lora_B, escala)} de un LoRA de edición base, en cualquiera de los dos formatos."""
    weights = load_file(str(path), device=device)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    if "adapter_config" in meta:
        # moss-edit-lora (chinoll): <capa>.lora_A / .lora_B, alpha y rank en adapter_config.
        config = json.loads(meta["adapter_config"])
        return {name: (weights[name + ".lora_A"], weights[name + ".lora_B"], spec["alpha"] / spec["rank"])
                for name, spec in config["modules"].items()}
    # ai-toolkit (armV4): diffusion_model.<capa>.lora_A.weight / .lora_B.weight; sin .alpha, escala 1.
    layers = {}
    for k in weights:
        if k.endswith(".lora_A.weight"):
            name = k[len("diffusion_model."):-len(".lora_A.weight")]
            down = weights[k]
            alpha = weights.get(f"diffusion_model.{name}.alpha")
            scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
            layers[name] = (down, weights[k.replace(".lora_A.", ".lora_B.")], scale)
    return layers


def edit_base_of(path):
    """La base de edición sobre la que se entrenó el LoRA, o None si no es un LoRA de edición de LoRAlab."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
    except Exception:
        return None
    if not meta.get("edit_base_lora"):
        return None
    # Los LoRAs anteriores a armV4 no guardan edit_base: son de chinoll.
    return meta.get("edit_base", "chinoll")


def is_edit_lora(path):
    """True si el .safetensors es un LoRA de edición entrenado por LoRAlab (lleva edit_base_lora)."""
    return edit_base_of(path) is not None


def fuse(lora_path, model_dir, out_path):
    """Escribe en out_path un LoRA estándar con la base de edición y el LoRA de lora_path sumados."""
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    name = edit_base_of(lora_path)
    if name is None:
        raise FuseError("This LoRA was not trained as an edit LoRA.")
    if name not in EDIT_BASES:
        raise FuseError(f"Unknown edit base: {name}")
    if meta.get("ss_network_module") == "lycoris.kohya":
        raise FuseError("A LoKr cannot be merged with the edit base LoRA: export it separately.")
    edit_path = base_file(model_dir, name)
    if not os.path.exists(edit_path):
        raise FuseError(f"Edit base LoRA not found (it is downloaded when training): {edit_path}")

    base = base_layers(edit_path)
    lora = load_file(str(lora_path))
    strength = float(meta.get("edit_base_lora_strength", 1.0))

    layers = {k[len("diffusion_model."):-len(".lora_A.weight")] for k in lora if k.endswith(".lora_A.weight")}
    unknown = layers - set(base)
    if unknown:
        raise FuseError(f"The LoRA has layers the edit base LoRA does not: {sorted(unknown)[:3]}")

    out = {}
    for layer, (base_a, base_b, scale) in base.items():
        a = [base_a.float()]
        b = [base_b.float() * (scale * strength)]
        key = "diffusion_model." + layer
        if layer in layers:
            down, up = lora[key + ".lora_A.weight"].float(), lora[key + ".lora_B.weight"].float()
            alpha = float(lora[key + ".alpha"]) if key + ".alpha" in lora else down.shape[0]
            a.append(down)
            b.append(up * (alpha / down.shape[0]))
        out[key + ".lora_A.weight"] = torch.cat(a, 0).to(torch.bfloat16).contiguous()
        out[key + ".lora_B.weight"] = torch.cat(b, 1).to(torch.bfloat16).contiguous()
        out[key + ".alpha"] = torch.tensor(float(out[key + ".lora_A.weight"].shape[0]))

    # Sin edit_base_lora: el resultado ya no es un LoRA para fusionar.
    header = {k: meta[k] for k in ("trigger_word", "project_name", "ss_output_name", "ss_steps") if k in meta}
    header["fused_edit_base"] = name
    header["fused_edit_base_lora"] = f"{EDIT_BASES[name]['repo']}@{EDIT_BASES[name]['revision']}"
    header["edit_format"] = EDIT_BASES[name]["edit_format"]
    header["usage"] = (f"Edit LoRA merged with {EDIT_BASES[name]['repo']}: load it alone with a LoRA loader and encode "
                       f"with Text Encode MOSS-Image Edit, edit_format {EDIT_BASES[name]['edit_format']}. Recommended strength 0.5-1.0.")
    header["trained_with"] = meta.get("trained_with", "AcademiaSD LoRAlab MOSS-Image")
    tmp = str(out_path) + ".tmp"
    save_file(out, tmp, metadata=header)
    os.replace(tmp, str(out_path))
