# -*- coding: utf-8 -*-
"""
fibo2_lora.py — Exportación del LoRA de Fibo 2.0 en el formato que carga ComfyUI.

ComfyUI no entrena ni carga el transformer de diffusers: usa su propio "repack", donde

  - la atención de cada bloque tiene UNA matriz qkv (q|k|v) en vez de tres,
  - el MLP tiene UNA matriz w13 (w1|w3) con el ancho relleno a un múltiplo de 256,
  - las atenciones cruzadas (y las del perceptor) tienen una to_kv (k|v) y la to_q suelta.

Un LoRA entrenado sobre el modelo de diffusers no se puede aplicar tal cual: hay que
rearmarlo a esas matrices. Para un grupo de n partes que se funden en una sola matriz de
ComfyUI se arma un LoRA de rango n·r:

    down = [A1 | A2 | ... ]            (n·r filas)
    up   = diag(B1, B2, ...)            (n·r columnas; cada bloque en las filas de su parte)

    up @ down = [B1 A1 ; B2 A2 ; ...] = la suma exacta de los LoRA sueltos

El alpha se multiplica por n para que la escala que aplica ComfyUI (alpha / rango) siga
siendo la misma que en el LoRA de diffusers.

Las claves del fichero son las que ComfyUI busca:
    diffusion_model.<capa>.lora_A.weight   (rango, entrada)
    diffusion_model.<capa>.lora_B.weight   (salida, rango)
    diffusion_model.<capa>.alpha
"""
import math

import torch

# Ancho al que ComfyUI rellena el MLP (múltiplo de 256, para el swiglu int8).
FFN_ALIGN = 256


# ── reglas de mapeo ─────────────────────────────────────────────────────────
# (sufijo de diffusers, sufijo de ComfyUI, grupo de fusión)
RULES = (
    # La atencion del bloque es la de JointAttention (ComfyUI: "out"); las demas, la de
    # Fibo 2.0 (ComfyUI: "to_out").
    (".attention.to_out.0", ".attention.out", None),
    (".cross_attention.to_out.0", ".cross_attention.to_out", None),
    (".self_attention.to_out.0", ".self_attention.to_out", None),
    (".cross_attention.to_q", ".cross_attention.to_q", None),
    (".self_attention.to_q", ".self_attention.to_q", None),
    (".feed_forward.w2", ".feed_forward.w2", None),
    (".adaLN_modulation.0", ".adaLN_modulation.0", None),
    (".adaLN_modulation.1", ".adaLN_modulation.1", None),
    (".mlp.0", ".mlp.0", None),
    (".mlp.2", ".mlp.2", None),
    (".attention.to_q", ".attention.qkv", "qkv"),
    (".attention.to_k", ".attention.qkv", "qkv"),
    (".attention.to_v", ".attention.qkv", "qkv"),
    (".cross_attention.to_k", ".cross_attention.to_kv", "kv"),
    (".cross_attention.to_v", ".cross_attention.to_kv", "kv"),
    (".self_attention.to_k", ".self_attention.to_kv", "kv"),
    (".self_attention.to_v", ".self_attention.to_kv", "kv"),
    (".feed_forward.w1", ".feed_forward.w13", "w13"),
    (".feed_forward.w3", ".feed_forward.w13", "w13"),
    # El MLP del perceptor va en "ffn", no en "feed_forward".
    (".ffn.w2", ".ffn.w2", None),
    (".ffn.w1", ".ffn.w13", "w13"),
    (".ffn.w3", ".ffn.w13", "w13"),
)

# Orden de las partes dentro de cada grupo fusionado.
GROUP_ORDER = {
    "qkv": ("to_q", "to_k", "to_v"),
    "kv": ("to_k", "to_v"),
    "w13": ("w1", "w3"),
}


def classify(module):
    """
    (sufijo de diffusers, sufijo de ComfyUI, grupo) de una capa de diffusers.
    grupo None = la capa se mapea 1:1.
    """
    for old, new, group in RULES:
        if module.endswith(old):
            return old, new, group
    return None, None, None


def is_fused(module):
    """True si la capa de diffusers se funde con otras en una sola matriz de ComfyUI."""
    return classify(module)[2] is not None


def ffn_padding(hidden):
    """Las columnas de relleno que ComfyUI añade al MLP (a un múltiplo de 256)."""
    return int(math.ceil(hidden / FFN_ALIGN) * FFN_ALIGN) - hidden


def build_lora(pairs, alpha):
    """
    Arma el LoRA de ComfyUI.

    pairs: {ruta de diffusers: {"A": (r, entrada), "B": (salida, r)}}
    alpha: el alpha del LoRA (ya convertido con rsLoRA si toca). Puede ser un numero o un
           diccionario {ruta: alpha} para los LoRA con un alpha por capa.

    Devuelve {clave del fichero: tensor} y la lista de capas que no se pudieron mapear
    (las fusionadas, para el LoKr, que no sabe expresarlas).
    """
    out = {}
    skipped = []

    def alpha_of(module):
        if isinstance(alpha, dict):
            return float(alpha.get(module, 0.0))
        return float(alpha)

    # 1. Las capas 1:1.
    for module, ab in pairs.items():
        old, new, group = classify(module)
        if group is not None:
            continue
        comfy = (module[: -len(old)] + new) if old else module
        a, b = ab["A"], ab["B"]
        if comfy.endswith(".w2"):
            # ComfyUI ensancha la entrada del MLP: se rellenan las columnas de mas.
            pad = ffn_padding(a.shape[1])
            if pad:
                a = torch.nn.functional.pad(a, (0, pad))
        out[f"diffusion_model.{comfy}.lora_A.weight"] = a.to(torch.bfloat16).cpu().contiguous()
        out[f"diffusion_model.{comfy}.lora_B.weight"] = b.to(torch.bfloat16).cpu().contiguous()
        out[f"diffusion_model.{comfy}.alpha"] = torch.tensor(alpha_of(module))

    # 2. Los grupos fusionados.
    groups = {}
    for module, ab in pairs.items():
        old, new, group = classify(module)
        if group is None:
            continue
        comfy = module[: -len(old)] + new
        groups.setdefault(comfy, {"group": group, "parts": {}})["parts"][module] = ab

    for comfy, info in groups.items():
        group = info["group"]
        parts = info["parts"]
        ordered = []
        for name in GROUP_ORDER[group]:
            for module, ab in parts.items():
                if module.endswith("." + name):
                    ordered.append((module, ab))
                    break
        if len(ordered) != len(GROUP_ORDER[group]):
            skipped.append(comfy)
            continue

        rank = ordered[0][1]["A"].shape[0]
        in_features = ordered[0][1]["A"].shape[1]
        if any(ab["A"].shape != (rank, in_features) for _, ab in ordered):
            skipped.append(comfy)
            continue

        # Las filas de cada parte dentro de la matriz fusionada.
        offsets = []
        row = 0
        for _, ab in ordered:
            out_rows = ab["B"].shape[0]
            if group == "w13":
                # ComfyUI separa w1 y w3 con el ancho relleno.
                out_rows = out_rows + ffn_padding(out_rows)
            offsets.append((row, out_rows))
            row += out_rows

        down = torch.cat([ab["A"] for _, ab in ordered], dim=0)
        up = torch.zeros((row, len(ordered) * rank), dtype=torch.float32)
        for i, (_, ab) in enumerate(ordered):
            off, out_rows = offsets[i]
            up[off: off + ab["B"].shape[0], i * rank: (i + 1) * rank] = ab["B"].float()

        out[f"diffusion_model.{comfy}.lora_A.weight"] = down.to(torch.bfloat16).cpu().contiguous()
        out[f"diffusion_model.{comfy}.lora_B.weight"] = up.to(torch.bfloat16).cpu().contiguous()
        # El rango de ComfyUI es el del down (n·r): el alpha se multiplica para dejar la escala igual.
        out[f"diffusion_model.{comfy}.alpha"] = torch.tensor(alpha_of(ordered[0][0]) * len(ordered))

    return out, skipped


def build_lokr(lokr_layers, key_of_layer, alpha):
    """
    El LoKr solo se puede exportar para las capas que se mapean 1:1: las fusionadas no se
    pueden expresar con la factorizacion de Kronecker.
    """
    out = {}
    skipped = []
    for name, layer in lokr_layers.items():
        if is_fused(name):
            skipped.append(name)
            continue
        key = key_of_layer(name)
        for k, v in layer.export_tensors().items():
            out[f"{key}.{k}"] = v.to(torch.bfloat16).cpu().contiguous()
        out[f"{key}.alpha"] = torch.tensor(float(alpha))
    return out, skipped
