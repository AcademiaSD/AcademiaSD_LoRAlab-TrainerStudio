# -*- coding: utf-8 -*-
"""
lora_options.py — Opciones de entrenamiento comunes: rsLoRA, LoRA+ y LoKr.
Shared training options: rsLoRA, LoRA+ and LoKr.

rsLoRA y LoRA+ son parámetros de PEFT y del optimizador; aquí solo está la conversión del alpha al
exportar. LoKr (LyCORIS) no usa PEFT: su LoKr no admite las capas NF4 (Linear4bit), así que cada
capa se envuelve con LoKrLinear, que además nunca construye la matriz completa.

El método LoKr (factorización de Kronecker, inicialización y claves lokr_*) es el de LyCORIS
(https://github.com/KohakuBlueleaf/LyCORIS, KohakuBlueleaf, Apache-2.0), reimplementado aquí para
capas NF4. rsLoRA (Kalajdzievski, 2023) y LoRA+ (Hayou et al., 2024) siguen sus artículos.
"""
import json
import math
import os

import torch
from safetensors.torch import save_file

LOKR_WEIGHTS = "lokr_model.safetensors"
LOKR_CONFIG = "lokr_config.json"


def export_alpha(alpha, rank, rslora):
    """Alpha que se escribe en el LoRA exportado. ComfyUI y el resto aplican alpha/rank; con rsLoRA
    el LoRA se entrenó con alpha/√rank, así que se guarda alpha·√rank para que actúe igual."""
    return alpha * math.sqrt(rank) if rslora else alpha


def lokr_export(model, key_of, alpha):
    """Claves LyCORIS que lee ComfyUI: <capa>.lokr_w1 / lokr_w2_a / lokr_w2_b / alpha (escala alpha/rank).
    key_of convierte el nombre de la capa en el prefijo que espera el cargador del modelo."""
    out = {}
    for name, layer in model.lokr_layers().items():
        key = key_of(name)
        for k, v in layer.export_tensors().items():
            out[f"{key}.{k}"] = v
        out[f"{key}.alpha"] = torch.tensor(float(alpha))
    return out


def lokr_factorization(dim, factor):
    """Parte dim en (m, n), m <= n, como LyCORIS y ComfyUI: con factor exacto si divide a dim;
    si no, el divisor más cercano a √dim."""
    if 0 < factor < dim and dim % factor == 0:
        m, n = factor, dim // factor
        return (m, n) if m <= n else (n, m)
    m, n = 1, dim
    for i in range(1, int(dim ** 0.5) + 1):
        if dim % i == 0:
            m, n = i, dim // i
    return m, n


class LoKrLinear(torch.nn.Module):
    """
    LoKr sobre una capa lineal (también NF4): salida = base(x) + alpha/rank · x·(w1 ⊗ w2)ᵀ, con
    w2 = w2_a·w2_b. La matriz grande no se construye nunca: (A⊗B)·vec(X) = vec(A·X·Bᵀ), así que
    cuesta unos pocos MB de VRAM por capa en vez de una matriz del tamaño del peso.
    """

    def __init__(self, base, rank, alpha, factor):
        super().__init__()
        self.base = base
        (self.a, self.b), (self.c, self.d) = (lokr_factorization(base.out_features, factor),
                                              lokr_factorization(base.in_features, factor))
        self.scale = alpha / rank
        # Inicialización de LyCORIS: w1 y w2_a aleatorias, w2_b a cero (el LoKr empieza sin efecto).
        self.lokr_w1 = torch.nn.Parameter(torch.empty(self.a, self.c))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.b, rank))
        self.lokr_w2_b = torch.nn.Parameter(torch.zeros(rank, self.d))
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        X = x.to(self.lokr_w1.dtype).reshape(*x.shape[:-1], self.c, self.d)
        y = torch.einsum("ac,...cr->...ar", self.lokr_w1, X @ self.lokr_w2_b.T) @ self.lokr_w2_a.T
        return out + (self.scale * y.reshape(*x.shape[:-1], self.a * self.b)).to(out.dtype)

    def export_tensors(self):
        return {k: getattr(self, k).detach().to(torch.bfloat16).cpu().contiguous()
                for k in ("lokr_w1", "lokr_w2_a", "lokr_w2_b")}


class LoKrModel(torch.nn.Module):
    """Envuelve el transformer con la misma forma que el modelo de PEFT (base_model.model), para que
    las previews y el Turbo LoRA lo usen igual. Guarda y carga sus pesos sin PEFT."""

    def __init__(self, transformer, target_modules, rank, alpha, factor):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = transformer
        for p in transformer.parameters():
            p.requires_grad_(False)
        for name in target_modules:
            parent_name, _, child = name.rpartition(".")
            parent = transformer.get_submodule(parent_name) if parent_name else transformer
            setattr(parent, child, LoKrLinear(getattr(parent, child), rank, alpha, factor).to(transformer.device))
        self.config = {"r": rank, "lora_alpha": alpha, "lokr_factor": factor}

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)

    def lokr_layers(self):
        return {n: m for n, m in self.base_model.model.named_modules() if isinstance(m, LoKrLinear)}

    def lokr_state_dict(self):
        return {f"{n}.{k}": getattr(m, k).detach().float().cpu().contiguous()
                for n, m in self.lokr_layers().items() for k in ("lokr_w1", "lokr_w2_a", "lokr_w2_b")}

    def load_lokr_state_dict(self, sd):
        layers = self.lokr_layers()
        for key, v in sd.items():
            name, _, attr = key.rpartition(".")
            getattr(layers[name], attr).data.copy_(v)

    def save_pretrained(self, directory):
        save_file(self.lokr_state_dict(), os.path.join(directory, LOKR_WEIGHTS))
        with open(os.path.join(directory, LOKR_CONFIG), "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    def print_trainable_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"trainable params: {n:,} (LoKr)")


