# -*- coding: utf-8 -*-
"""
lora_options.py — Opciones de entrenamiento comunes: rsLoRA, LoRA+, LoKr y el LR scheduler.
Shared training options: rsLoRA, LoRA+, LoKr and the LR scheduler.

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
from contextlib import contextmanager

import torch
from safetensors.torch import save_file

LOKR_WEIGHTS = "lokr_model.safetensors"
LOKR_CONFIG = "lokr_config.json"


def export_alpha(alpha, rank, rslora):
    """Alpha que se escribe en el LoRA exportado. ComfyUI y el resto aplican alpha/rank; con rsLoRA
    el LoRA se entrenó con alpha/√rank, así que se guarda alpha·√rank para que actúe igual."""
    return alpha * math.sqrt(rank) if rslora else alpha


def loraplus_groups(named_params, ratio):
    """LoRA+: lora_B en su propio grupo, con el LR multiplicado por ratio. El bucle aplica
    lr_mult en cada paso (group["lr"] = lr * group["lr_mult"])."""
    a, b = [], []
    for name, p in named_params:
        if p.requires_grad:
            (b if "lora_B" in name else a).append(p)
    return [g for g in ({"params": a, "lr_mult": 1.0}, {"params": b, "lr_mult": ratio}) if g["params"]]


def lr_factor(scheduler, step, total, warmup, min_ratio, cycles):
    """Multiplicador del LR en el paso step. constant no tiene warmup ni bajada; el resto sube lineal
    durante warmup y luego baja hasta min_ratio (constant_with_warmup se queda arriba).
    cosine_with_restarts repite el coseno cycles veces, como el de transformers / kohya."""
    if scheduler == "constant":
        return 1.0
    if step < warmup:
        return step / max(1, warmup)
    if scheduler == "constant_with_warmup":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    if scheduler == "cosine":
        decay = 0.5 * (1 + math.cos(math.pi * prog))
    elif scheduler == "linear":
        decay = 1 - prog
    elif scheduler == "cosine_with_restarts":
        decay = 0.0 if prog >= 1 else 0.5 * (1 + math.cos(math.pi * (cycles * prog % 1)))
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler}")
    return min_ratio + (1 - min_ratio) * decay


def lr_scheduler_label(scheduler, warmup, cycles):
    """Texto del scheduler para el resumen de la consola: "cosine (warmup 100)"."""
    if scheduler == "constant":
        return scheduler
    if scheduler == "cosine_with_restarts":
        return f"{scheduler} (warmup {warmup}, {cycles} cycles)"
    return f"{scheduler} (warmup {warmup})"


def dora_scales(model, adapter="default", base_norms=None):
    """
    DoRA (Liu et al., 2024) para exportar: {nombre del módulo PEFT: dora_scale [out, 1]} ([out, 1, 1, 1]
    en una convolución).
    PEFT entrena W' = m / ||W + s·B·A|| · (W + s·B·A), con la norma por fila de salida. ComfyUI lee
    <capa>.dora_scale, pero con un dora_scale por fila de salida divide por ||W|| (el peso sin el LoRA),
    no por ||W + s·B·A|| (comfy/weight_adapter/base.py, weight_decompose). Para que dé lo mismo que el
    entrenamiento se guarda m · ||W|| / ||W + s·B·A||.
    ||W|| es la del peso con el que se entrenó (NF4 descomprimido). ComfyUI usa la del peso que carga, y
    la de una fila NF4 se aparta de la original lo bastante para notarse junto a lo que aprende DoRA: el
    trainer que tiene el peso original al cargar pasa sus normas en base_norms ({nombre de la capa sin
    base_model.model.: norma por fila}) y entonces el resultado es exacto.
    DoRA for export. ComfyUI divides a per-output-row dora_scale by ||W||, not ||W + s·B·A||, so the
    exported scale is m · ||W|| / ||W + s·B·A||, which makes ComfyUI reproduce the trained weight.
    base_norms: row norms of the original (unquantized) weights, when the trainer has them.
    """
    from peft.utils.integrations import dequantize_module_weight
    out = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            magnitudes = getattr(module, "lora_magnitude_vector", None)
            if not magnitudes or adapter not in magnitudes:
                continue
            base = module.get_base_layer()
            weight = dequantize_module_weight(base).float()
            shape = weight.shape
            weight = weight.reshape(shape[0], -1)            # una convolución: cada canal de salida es una fila
            a = module.lora_A[adapter].weight.float().to(weight.device)
            b = module.lora_B[adapter].weight.float().to(weight.device)
            merged = weight + (b.reshape(shape[0], -1) @ a.reshape(a.shape[0], -1)) * module.scaling[adapter]
            m = magnitudes[adapter].weight.float().to(weight.device).view(-1)
            norm = weight.norm(dim=1)
            own = (base_norms or {}).get(name.replace("base_model.model.", "", 1))
            if own is None:
                own = getattr(base, "_dora_row_norm", None)  # de attach_dora_norms (NormFix)
            if own is not None and own.numel() == shape[0]:
                norm = own.float().to(weight.device).view(-1)
            scale = m * norm / merged.norm(dim=1).clamp_min(torch.finfo(torch.float32).eps)
            # FP32: un valor por fila, apenas ocupa
            out[name] = scale.view(shape[0], *[1] * (len(shape) - 1)).cpu().contiguous()
            del weight, merged
    return out


DORA_NORMS_FILE = "dora_norms.safetensors"


def _hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and os.path.exists("settings/HF_token.json"):
        try:
            with open("settings/HF_token.json", "r", encoding="utf-8") as f:
                token = str(json.load(f).get("token", "")).strip()
        except Exception:
            token = ""
    return token or None


DORA_NORMS_REPO = "AcademiaSD/dora_norms_for_training"   # normas de los NF4 de repos de terceros


def ensure_dora_norms(folders, repo_id=None, root=None, prefix=None):
    """
    Devuelve la ruta de dora_norms.safetensors en la primera carpeta que lo tenga; si no está en ninguna,
    lo descarga del repo NF4 (modelos bajados antes de que el repo lo tuviera). root: carpeta local del
    repo; la ruta dentro del repo es la de la carpeta relativa a root. prefix: carpeta del modelo dentro
    de un repo de normas aparte (DORA_NORMS_REPO), para los NF4 de terceros. None si no hay.
    Returns the local dora_norms.safetensors, downloading it from the NF4 repo when it is missing.
    """
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    folders = [str(f) for f in folders if f]
    for f in folders:
        if os.path.isfile(os.path.join(f, DORA_NORMS_FILE)):
            return os.path.join(f, DORA_NORMS_FILE)
    if not repo_id or not root:
        return None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:
        return None
    root = str(root)
    for f in folders:
        rel = os.path.relpath(f, root).replace("\\", "/")
        if rel.startswith(".."):
            continue
        name = DORA_NORMS_FILE if rel in (".", "") else f"{rel}/{DORA_NORMS_FILE}"
        if prefix:
            name = prefix.strip("/") + "/" + name
        print(f"  DoRA NormFix: downloading {name} from {repo_id}", flush=True)
        try:
            if prefix:                      # repo de normas aparte: se baja a la caché y se copia junto al NF4
                import shutil
                path = os.path.join(f, DORA_NORMS_FILE)
                shutil.copyfile(hf_hub_download(repo_id, name, token=_hf_token()), path)
            else:
                path = hf_hub_download(repo_id, name, local_dir=root, token=_hf_token())
        except EntryNotFoundError:
            print(f"  DoRA NormFix: {repo_id} has no {name}", flush=True)
            continue
        except Exception as e:
            print(f"  [!] DoRA NormFix: download failed ({type(e).__name__}: {e})", flush=True)
            return None
        if os.path.isfile(path):
            return path
    return None


def attach_dora_norms(model, folders, enabled=True, adapter="default", repo_id=None, root=None, prefix=None):
    """
    NormFix: da a cada capa DoRA la norma por fila del peso ORIGINAL (bf16), la que divide ComfyUI, en vez
    de la del NF4 con el que se entrena. La de una fila NF4 se aparta de la original ~0,2 %, más que lo que
    aprende DoRA (0,02-0,05 %), y sin esto el LoRA pierde el parecido en ComfyUI aunque la preview salga bien.
    folders: carpeta(s) del transformer NF4 donde buscar dora_norms.safetensors (lo crea
    tools/dora_norms/make_dora_norms.py); vale la primera que lo tenga. Si no está y se pasan repo_id y
    root (carpeta local del repo NF4), se descarga antes de seguir. Llamar después de get_peft_model.
    NormFix: gives each DoRA layer the row norms of the original weights, which ComfyUI divides by, instead of
    the NF4 ones it trains on. Returns the number of layers that got them.
    """
    if not enabled:
        print("  DoRA NormFix: off (the export uses the NF4 norms)", flush=True)
        return 0
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    path = ensure_dora_norms(folders, repo_id, root, prefix)   # si falta, lo baja del repo NF4 (repo_id + root)
    if path is None:
        print(f"  [!] DoRA NormFix: {DORA_NORMS_FILE} not found in {', '.join(map(str, folders))}. "
              "The export uses the NF4 norms: in ComfyUI the LoRA may look weaker than in the previews.",
              flush=True)
        return 0
    from safetensors.torch import load_file
    norms = load_file(path)
    total = done = 0
    for name, module in model.named_modules():
        magnitudes = getattr(module, "lora_magnitude_vector", None)
        if not magnitudes or adapter not in magnitudes:
            continue
        total += 1
        name = name.replace("base_model.model.", "", 1)
        own = norms.get(name)
        if own is None:                     # el modelo puede ir dentro de otro (p. ej. "transformer.")
            parts = name.split(".")
            own = next((norms[".".join(parts[i:])] for i in range(1, len(parts)) if ".".join(parts[i:]) in norms), None)
        base = module.get_base_layer()
        rows = getattr(base, "out_features", None) or base.weight.shape[0]
        if own is not None and own.numel() == rows:
            base._dora_row_norm = own.float()
            done += 1
    if done < total:
        print(f"  [!] DoRA NormFix: original norms for {done}/{total} layers ({path}); "
              "the others use the NF4 norms.", flush=True)
    else:
        print(f"  DoRA NormFix: original norms for {done} layers", flush=True)
    return done


def dora_export(model, key_of, swapped=(), base_norms=None):
    """Claves <capa>.dora_scale que leen ComfyUI y Forge (vacío si el LoRA no es DoRA). key_of convierte el
    nombre de la capa en el prefijo que espera el cargador del modelo, el mismo de sus lora_A / lora_B.
    swapped: capas donde el cargador intercambia las dos mitades de filas (scale/shift de Flux).
    base_norms: ver dora_scales."""
    out = {}
    for name, scale in dora_scales(model, base_norms=base_norms).items():
        name = name.replace("base_model.model.", "", 1)
        if name in swapped:
            scale = torch.cat(scale.chunk(2)[::-1]).contiguous()
        out[key_of(name) + ".dora_scale"] = scale
    return out


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
        self._disabled = 0
        # Inicialización de LyCORIS: w1 y w2_a aleatorias, w2_b a cero (el LoKr empieza sin efecto).
        self.lokr_w1 = torch.nn.Parameter(torch.empty(self.a, self.c))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.b, rank))
        self.lokr_w2_b = torch.nn.Parameter(torch.zeros(rank, self.d))
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))

    @property
    def weight(self):
        # El peso de la capa base, como las capas de PEFT: hay modelos que lo consultan (dispositivo, dtype).
        return self.base.weight

    @property
    def base_layer(self):
        return self.base

    def forward(self, x, *args, **kwargs):
        base_dtype = getattr(self.base, "compute_dtype", None) or self.base.weight.dtype
        out = self.base(x.to(base_dtype), *args, **kwargs)
        if self._disabled:
            return out
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
            # El dispositivo del primer parámetro: no todos los modelos tienen .device (el DiT de DiffSynth no).
            setattr(parent, child, LoKrLinear(getattr(parent, child), rank, alpha, factor).to(next(transformer.parameters()).device))
        self.config = {"r": rank, "lora_alpha": alpha, "lokr_factor": factor}

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)

    def get_base_model(self):
        return self.base_model.model

    @contextmanager
    def disable_adapter(self):
        layers = list(self.lokr_layers().values())
        for layer in layers:
            layer._disabled += 1
        try:
            yield
        finally:
            for layer in layers:
                layer._disabled -= 1

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


