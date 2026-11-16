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


# ---------------------------------------------------------------------------------------------------------
# Precisión de cálculo (selector Precision). Las GPUs sin BF16 por hardware (RTX 20 / Quadro RTX / Turing,
# GTX 16) lo emulan y entrenan varias veces más lento: ahí FP16, con la loss escalada (GradScaler) y el LoRA
# en FP32 (GradScaler no desescala gradientes FP16). El modelo sigue en NF4 en los dos casos.
# Compute precision. GPUs without BF16 in hardware emulate it; there FP16 with loss scaling is used.
# ---------------------------------------------------------------------------------------------------------
def gpu_has_bf16():
    try:
        return not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return True


def resolve_precision(value, fp16_ok=True):
    """'auto' | 'bf16' | 'fp16' → 'bf16' o 'fp16'. Auto: FP16 solo si la GPU no tiene BF16 y el modelo cabe
    en FP16 (fp16_ok: sus activaciones no pasan de 65504)."""
    value = str(value or "auto").lower()
    if value in ("bf16", "fp16"):
        return value
    return "fp16" if fp16_ok and not gpu_has_bf16() else "bf16"


def precision_dtypes(precision):
    """(dtype de cálculo, dtype del LoRA entrenable)."""
    return (torch.float16, torch.float32) if precision == "fp16" else (torch.bfloat16, torch.bfloat16)


def to_compute_dtype(module, dtype):
    """Pasa a dtype los pesos BF16 del modelo base (no los del LoRA), el compute_dtype de las capas NF4 y el
    dtype al que descomprimen (quant_state.dtype; si no, la ruta gemv de una sola fila sale en BF16).
    Llamar antes de añadir el LoRA. Las normas en FP32 se quedan como están."""
    if dtype == torch.bfloat16:
        return module
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if tensor.dtype == torch.bfloat16 and "lora_" not in name:
            tensor.data = tensor.data.to(dtype)
    for m in module.modules():
        if getattr(m, "compute_dtype", None) == torch.bfloat16:
            m.compute_dtype = dtype
        qs = getattr(getattr(m, "weight", None), "quant_state", None)
        if qs is not None and getattr(qs, "dtype", None) == torch.bfloat16:
            qs.dtype = dtype
    return module


def fp16_rescale(model, pattern, factor, on="input"):
    """
    FP16 sin desbordes donde el modelo pasa de 65504: divide por factor la entrada (on="input") o la salida
    (on="output") de los módulos cuyo nombre encaja con pattern. Solo vale cuando todo lo que va de ahí a la
    siguiente RMSNorm es lineal y sin bias (la norma quita la escala: el resultado es el mismo, LoRA incluido).
    Llamar después de añadir el LoRA, para que el gancho quede en la capa con LoRA. Devuelve cuántos.
    Divides the input or output of matching modules by factor; exact when only bias-free linear maps lead to
    the next RMSNorm.
    """
    import re
    rx = re.compile(pattern)
    inv = 1.0 / float(factor)

    def pre(module, args):
        if args and torch.is_tensor(args[0]) and args[0].is_floating_point():
            return (args[0] * inv,) + tuple(args[1:])
        return None

    def post(module, args, output):
        return output * inv if torch.is_tensor(output) else output

    count = 0
    for name, m in model.named_modules():
        if rx.fullmatch(name):
            if on == "input":
                m.register_forward_pre_hook(pre)
            else:
                m.register_forward_hook(post)
            count += 1
    return count


def fp32_output(model, pattern):
    """
    Capas cuya salida no cabe en FP16 ni escalando (p. ej. la última del MLP, antes de una RMSNorm): se
    calculan en FP32 y su salida sigue en FP32 hasta la norma, que la devuelve a FP16 (sus pesos son FP16).
    Llamar después de añadir el LoRA. Devuelve cuántas.
    Layers whose output does not fit in FP16 run in FP32 and hand an FP32 output to the next norm.
    """
    import re
    rx = re.compile(pattern)

    def pre(module, args):
        if args and torch.is_tensor(args[0]) and args[0].is_floating_point():
            return (args[0].float(),) + tuple(args[1:])
        return None

    count = 0
    for name, m in model.named_modules():
        if not rx.fullmatch(name):
            continue
        for sub in m.modules():
            if getattr(sub, "compute_dtype", None) is not None:
                sub.compute_dtype = torch.float32
            qs = getattr(getattr(sub, "weight", None), "quant_state", None)
            if qs is not None:
                qs.dtype = torch.float32
        m.register_forward_pre_hook(pre)
        count += 1
    return count


def grad_scaler(precision):
    """GradScaler activo solo en FP16; en BF16 scale/unscale_/step/update no hacen nada."""
    return torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, enabled=precision == "fp16")


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
