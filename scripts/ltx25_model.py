# -*- coding: utf-8 -*-
"""
ltx25_model.py — piezas comunes del entrenador LTX-2.5 / shared pieces of the LTX-2.5 trainer.

* load_nf4_transformer(): construye LTX2VideoTransformer3DModel vacío y lo rellena desde Models/LTX25-NF4
  (Linear de los bloques en NF4, el resto de Linear en BF16, others.safetensors). Mismo formato que LTX23-NF4.
* BlockSwap: block swap al estilo de MiniMax-H3 para entrenar en 8 GB.
    - Los N primeros bloques quedan residentes en VRAM; el resto se "aparca" en RAM.
    - Al aparcar, el dato NF4 empaquetado (uint8) se queda en una copia fija en CPU (`home`, pinned si se puede)
      y en la GPU solo vive mientras el bloque corre: subir = copiar `home` a CUDA; bajar = soltar la copia CUDA.
      Nunca se descarga de vuelta (los pesos NF4 están congelados): la idea de `nf4_cpu_home` de H3.
    - quant_state (absmax, code; ~1.5 % del bloque) y los bias, normas y tablas se quedan SIEMPRE en CUDA.
    - Con checkpoint reentrante (obligatorio: bitsandbytes guarda su peso en ctx, el no reentrante lo fijaría
      en VRAM) cada bloque aparcado sube 2 veces por paso: forward (sin grad) y recompute+backward.
    - Prefetch: mientras corre el bloque i se sube el siguiente en un stream aparte.
* plan_resident_blocks(): cuántos bloques caben dado un presupuesto, al estilo de los perfiles de H3.
"""
import json
import os

import torch
from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from safetensors.torch import load_file

COMPUTE_DTYPE = torch.bfloat16


# ----------------------------------------------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------------------------------------------
def _parent(root, name):
    parts = name.split(".")
    mod = root
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]


class CastLinear(torch.nn.Linear):
    """Linear BF16 que se comporta como Linear4bit: calcula en el dtype del peso y devuelve el de la entrada."""

    def forward(self, x):
        y = torch.nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)
        return y.to(x.dtype)


def load_nf4_transformer(model_dir, device="cuda", compute_dtype=COMPUTE_DTYPE, blocks_on_cpu=False, log=print):
    """
    Devuelve el transformer listo. Con `blocks_on_cpu=True` el dato NF4 empaquetado de los bloques se queda en RAM
    (su quant_state, bias y todo lo demás van a `device`): así cabe en 8 GB, y BlockSwap sube después los residentes.
    Nunca hay que llamar a .to("cuda") sobre el modelo entero después: subiría todos los bloques. Usar to_device_except_nf4().
    """
    from diffusers import LTX2VideoTransformer3DModel

    with open(os.path.join(model_dir, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(model_dir))
    weights_dir = os.path.join(model_dir, "weights")

    for name, info in index.get("quantized", {}).items():
        path = os.path.join(weights_dir, info["file"])
        with safe_open(path, framework="pt", device="cpu") as f:
            data = f.get_tensor("weight")
            bias = f.get_tensor("bias") if info.get("bias") else None
            qs = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}
        with torch.device("meta"):
            layer = Linear4bit(int(info["in_features"]), int(info["out_features"]), bias=bool(info.get("bias")),
                               quant_type="nf4", compute_dtype=compute_dtype)
        on_cpu = blocks_on_cpu and name.startswith("transformer_blocks.")
        layer.weight = Params4bit.from_prequantized(data=data, quantized_stats=qs, requires_grad=False,
                                                    device="cpu" if on_cpu else device, module=layer)
        if on_cpu:
            layer.weight.quant_state.to(device)   # absmax/code en GPU; el dato empaquetado se queda en RAM
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias.to(device, dtype=compute_dtype), requires_grad=False)
        parent, child = _parent(model, name)
        setattr(parent, child, layer)

    for name, info in index.get("unquantized", {}).items():
        path = os.path.join(weights_dir, info["file"])
        with safe_open(path, framework="pt", device="cpu") as f:
            w = f.get_tensor("weight")
            b = f.get_tensor("bias") if info.get("bias") else None
        with torch.device("meta"):
            layer = CastLinear(int(info["in_features"]), int(info["out_features"]), bias=b is not None)
        layer.weight = torch.nn.Parameter(w.to(device, dtype=compute_dtype), requires_grad=False)
        if b is not None:
            layer.bias = torch.nn.Parameter(b.to(device, dtype=compute_dtype), requires_grad=False)
        parent, child = _parent(model, name)
        setattr(parent, child, layer)

    others = load_file(os.path.join(model_dir, "others.safetensors"), device=str(device))
    model.load_state_dict(others, strict=False, assign=True)

    # Buffers que no vienen en el checkpoint (rope, etc.): se recrean en CPU por from_config -> a device.
    missing = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError("NF4 model incomplete / modelo NF4 incompleto: {}".format(missing[:5]))
    for n, b in list(model.named_buffers()):
        if b.device.type != torch.device(device).type:
            parent, child = _parent(model, n)
            parent.register_buffer(child, b.to(device), persistent=child in getattr(parent, "_buffers", {}))

    model.requires_grad_(False)
    model.eval()
    nq = len(index.get("quantized", {}))
    log("NF4 layers / capas NF4: {} · BF16 Linear: {}".format(nq, len(index.get("unquantized", {}))))
    return model


def to_device_except_nf4(model, device="cuda"):
    """Como model.to(device) pero sin tocar el dato NF4 de las Linear4bit (lo gestiona BlockSwap)."""
    for mod in model.modules():
        for name, p in list(mod._parameters.items()):
            if p is None or isinstance(p, Params4bit) or p.device.type == torch.device(device).type:
                continue
            p.data = p.data.to(device)
            if p.grad is not None:
                p.grad = p.grad.to(device)
        for name, b in list(mod._buffers.items()):
            if b is not None and b.device.type != torch.device(device).type:
                mod._buffers[name] = b.to(device)
    return model


# ----------------------------------------------------------------------------------------------------------------
# Block swap
# ----------------------------------------------------------------------------------------------------------------
def block_nf4_bytes(block):
    return sum(m.weight.data.numel() * m.weight.data.element_size()
               for m in block.modules() if isinstance(m, Linear4bit))


ACTIVE_SWAP = None


class BlockSwap:
    """
    Sube a VRAM los bloques [:resident] para siempre; aparca los [resident:] en RAM y los sube justo a tiempo
    (con prefetch del siguiente). Acepta bloques cargados en CPU (blocks_on_cpu) o ya en CUDA.
    """

    def __init__(self, blocks, resident, pin=True, prefetch=True, log=print):
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.resident = max(0, min(int(resident), self.n))
        self.prefetch = prefetch
        self.stream = torch.cuda.Stream() if prefetch else None
        self.events = {}
        self.uploads = 0
        self.pinned = False
        self._order = {id(b): i for i, b in enumerate(self.blocks)}
        parked_bytes = 0
        for i, block in enumerate(self.blocks):
            mods = [m for m in block.modules() if isinstance(m, Linear4bit)]
            block._swap_mods = mods
            block._swap_index = i
            block._swap_parked = i >= self.resident
            if not block._swap_parked:
                for m in mods:
                    if m.weight.data.device.type != "cuda":
                        m.weight.data = m.weight.data.to("cuda")
                continue
            for m in mods:
                cpu = m.weight.data.to("cpu")
                if pin:
                    try:
                        cpu = cpu.pin_memory()
                        self.pinned = True
                    except RuntimeError:
                        pin = False
                m._swap_home = cpu
                m.weight.data = cpu  # en CPU: la copia CUDA se libera
                parked_bytes += cpu.numel()
            block._swap_on_gpu = False
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)
            block.register_full_backward_hook(self._bwd_hook)
        global ACTIVE_SWAP
        ACTIVE_SWAP = self
        self._resident_offloaded = False
        torch.cuda.empty_cache()
        log("[SWAP] resident blocks / bloques residentes: {}/{} · parked / aparcados: {} ({:.2f} GB {})".format(
            self.resident, self.n, self.n - self.resident, parked_bytes / 1e9, "pinned" if self.pinned else "RAM"))

    # -- movimiento --------------------------------------------------------------------------------------------
    def _upload(self, block, stream=None):
        if block._swap_on_gpu:
            return
        ctx = torch.cuda.stream(stream) if stream is not None else _null()
        with ctx:
            for m in block._swap_mods:
                m.weight.data = m._swap_home.to("cuda", non_blocking=True)
        if stream is not None:
            ev = torch.cuda.Event()
            ev.record(stream)
            self.events[id(block)] = ev
        block._swap_on_gpu = True
        self.uploads += 1

    def _release(self, block):
        if not block._swap_on_gpu:
            return
        ev = self.events.pop(id(block), None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        for m in block._swap_mods:
            g = m.weight.data
            m.weight.data = m._swap_home
            # La copia CUDA puede seguir en uso por kernels encolados: el allocator lo respeta en el stream actual.
            del g
        block._swap_on_gpu = False

    def _wait(self, block):
        ev = self.events.pop(id(block), None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            # La memoria se reservó en el stream de prefetch y se usa en el actual.
            for m in block._swap_mods:
                m.weight.data.record_stream(torch.cuda.current_stream())

    def _neighbor(self, block, step):
        j = block._swap_index + step
        if 0 <= j < self.n and self.blocks[j]._swap_parked:
            return self.blocks[j]
        return None

    # -- dejar sitio al VAE de la preview -------------------------------------------------------------------------
    @property
    def swapping(self):
        return self.resident < self.n

    def offload_resident(self):
        """Baja a RAM los bloques residentes (para decodificar con el VAE en 8 GB). La copia en RAM se guarda."""
        if self._resident_offloaded:
            return
        for block in self.blocks[:self.resident]:
            for m in block._swap_mods:
                if getattr(m, "_res_home", None) is None:
                    m._res_home = m.weight.data.to("cpu")
                m.weight.data = m._res_home
        self._resident_offloaded = True
        torch.cuda.empty_cache()

    def restore_resident(self):
        if not self._resident_offloaded:
            return
        for block in self.blocks[:self.resident]:
            for m in block._swap_mods:
                m.weight.data = m._res_home.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        self._resident_offloaded = False

    # -- hooks ---------------------------------------------------------------------------------------------------
    def _pre_hook(self, block, args):
        self._upload(block)
        self._wait(block)
        if self.prefetch:
            # Forward (sin grad): el siguiente es i+1. Recompute del backward: el siguiente en recomputarse es i-1.
            nxt = self._neighbor(block, +1 if not torch.is_grad_enabled() else -1)
            if nxt is not None:
                self._upload(nxt, self.stream)

    def _post_hook(self, block, args, output):
        # Con checkpoint reentrante el forward va sin grad: se suelta ya. En el recompute (con grad) el peso hace
        # falta hasta el backward del bloque, y lo suelta _bwd_hook.
        if not torch.is_grad_enabled():
            self._release(block)
        return output

    def _bwd_hook(self, block, grad_input, grad_output):
        self._release(block)
        return None


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------------------------------------------
# Plan de VRAM (perfiles al estilo H3)
# ----------------------------------------------------------------------------------------------------------------
VRAM_PROFILES_GB = {"8": 8.0, "10": 10.0, "12": 12.0, "16": 16.0, "24": 24.0, "32": 32.0}


def plan_resident_blocks(card_gb, block_gb, fixed_gb, activation_gb, n_blocks, headroom_gb=0.6, swap_slots=2):
    """
    Bloques que caben residentes:  card - headroom - fijo - activaciones - slots de swap (bloque actual + prefetch).
    `fixed_gb`: resto del transformer + contexto CUDA + LoRA/optimizador. Los coeficientes reales se ajustan con
    mediciones (como en H3); esto es solo la forma de la cuenta.
    """
    free = card_gb - headroom_gb - fixed_gb - activation_gb
    if free < swap_slots * block_gb:
        return 0, free
    resident = int((free - swap_slots * block_gb) // block_gb)
    if resident >= n_blocks - swap_slots:
        resident = n_blocks  # sin swap no hacen falta slots
    return max(0, min(resident, n_blocks)), free


# ----------------------------------------------------------------------------------------------------------------
# Gemma 4 (text encoder) para 8 GB
# ----------------------------------------------------------------------------------------------------------------
def load_text_encoder_lean(path, device="cuda", compute_dtype=COMPUTE_DTYPE, log=print):
    """
    Gemma 4 NF4 con solo las capas de texto en la GPU (~5.6 GB en vez de ~7.7):
      * la tabla de embeddings (262144 x 3840, 2 GB en BF16, NF4 no la comprime) se queda en RAM: la búsqueda se
        hace en CPU y solo el resultado [tokens x 3840] sube a la GPU;
      * lm_head (copia atada de esa tabla) no hace falta para sacar estados ocultos: se sustituye por Identity;
      * las torres de visión y audio no se usan y se quedan en CPU.
    Se carga en CPU y se sube a mano porque el dispatch de accelerate vuelve a subir la tabla a la GPU.
    """
    import transformers

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())   # bitsandbytes necesita el índice
    cfg = transformers.AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    model = cls.from_pretrained(path, device_map={"": "cpu"}, torch_dtype=compute_dtype)
    lm = model.model.language_model
    emb = lm.embed_tokens

    # Cada capa NF4 sube con Params4bit.to (mueve también su quant_state). Si transformers la dejó como nn.Linear
    # con un Params4bit dentro (pasa al cargar en CPU), se envuelve en Linear4bit para que haga el matmul NF4.
    moved = 0
    for name, mod in list(lm.named_modules()):
        for cname, child in list(mod.named_children()):
            w = getattr(child, "weight", None)
            if not isinstance(w, Params4bit):
                continue
            if isinstance(child, Linear4bit):
                child.weight = w.to(device)
                if child.bias is not None:
                    child.bias.data = child.bias.data.to(device)
            else:
                out_f, in_f = w.quant_state.shape
                with torch.device("meta"):
                    new = Linear4bit(in_f, out_f, bias=child.bias is not None, quant_type="nf4",
                                     compute_dtype=compute_dtype)
                new.weight = w.to(device)
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.to(device), requires_grad=False)
                setattr(mod, cname, new)
            moved += 1
    for cname, child in lm.named_children():
        if child is emb:
            continue
        for mod in child.modules():
            for k, p in list(mod._parameters.items()):
                if p is not None and not isinstance(p, Params4bit) and p.device.type != "cuda":
                    p.data = p.data.to(device)
            for k, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[k] = b.to(device)
    for k, b in list(lm._buffers.items()):
        if b is not None:
            lm._buffers[k] = b.to(device)

    orig = emb.forward
    emb.forward = lambda ids, *a, **kw: orig(ids.to("cpu"), *a, **kw).to(device)
    model.lm_head = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    torch.cuda.empty_cache()
    log("[TE] Gemma 4 NF4: {} NF4 layers in VRAM, embeddings in RAM ({:.2f} GB)".format(
        moved, torch.cuda.memory_allocated() / 1e9))
    return model
