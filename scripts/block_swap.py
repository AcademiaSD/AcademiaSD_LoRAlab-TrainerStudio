# -*- coding: utf-8 -*-
"""
block_swap.py — block swap para entrenar modelos NF4 en poca VRAM / block swap to train NF4 models on little VRAM.

Lo usan LTX 2.5 y FLUX.2 Klein 9B. Al estilo de MiniMax-H3:
* BlockSwap: los N primeros bloques quedan residentes en VRAM; el resto se aparca en RAM fijada o se queda
  mapeado a su .safetensors (disco), y se sube justo a tiempo, con prefetch del siguiente en un stream aparte.
    - Solo se mueve el dato NF4 empaquetado (uint8). quant_state, bias, normas y tablas se quedan en CUDA.
    - Nunca se baja nada de vuelta: los pesos NF4 están congelados.
    - Necesita checkpoint REENTRANTE (bitsandbytes guarda su peso en ctx; el no reentrante lo fijaría en VRAM):
      cada bloque aparcado sube 2 veces por paso, en el forward (sin grad) y en el recompute + backward.
* plan_resident_blocks / plan_resident_sized: cuántos bloques caben, al estilo de los perfiles de H3.
* apply_vram_profile: tope duro de VRAM que simula una tarjeta menor.
* SaveOnCpu: offload de activaciones a RAM fijada que copia una sola vez cada tensor guardado varias veces.
* nf4_on_gpu / patch_peft_dora_init: DoRA con los bloques en RAM.
"""
import json
import os
import weakref

import torch
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open


def mmap_tensor(path, key="weight"):
    """
    Tensor de un .safetensors mapeado en memoria (sin copiarlo a RAM): sus páginas las carga y las suelta el
    sistema según haga falta. Los bytes NF4 de cada capa ya están en Models/LTX25-NF4/weights; esto evita tener
    una segunda copia en RAM. Solo lectura.
    """
    import struct
    import warnings
    import numpy as np
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    info = header[key]
    if info["dtype"] != "U8":
        raise ValueError("mmap_tensor: only U8 tensors ({} is {})".format(key, info["dtype"]))
    start, end = info["data_offsets"]
    arr = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + start, shape=(end - start,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # tensor de solo lectura: nunca se escribe
        return torch.from_numpy(arr).view(*info["shape"])


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


class nf4_on_gpu:
    """
    Con block swap el dato NF4 empaquetado de un bloque aparcado vive en RAM y su quant_state en la GPU: así no
    se puede descomprimir. Dentro de este contexto las Linear4bit de `module` con el dato en CPU lo tienen en la
    GPU, y al salir vuelven a su copia en RAM (sin copiar nada de vuelta: los pesos están congelados).
    Lo usa DoRA, que descomprime el peso para su norma al crear el LoRA y al exportarlo.
    """

    def __init__(self, module):
        self.module = module
        self.moved = []

    def __enter__(self):
        for m in self.module.modules():
            w = getattr(m, "weight", None)
            qs = getattr(w, "quant_state", None)
            if isinstance(w, Params4bit) and qs is not None and w.data.device.type == "cpu":
                self.moved.append((w, w.data))
                w.data = w.data.to(qs.absmax.device, non_blocking=True)
        return self.module

    def __exit__(self, *exc):
        for w, home in self.moved:
            w.data = home
        self.moved = []
        return False


_DORA_PROGRESS = {"done": 0, "total": 0, "t0": 0.0}


def patch_peft_dora_init(total_layers=0):
    """
    Magnitud inicial de DoRA (m = ||W + s*B*A|| por fila), como DoraLinearLayer.update_layer de PEFT pero:
      * en la GPU: con block swap el dato NF4 de la capa está en RAM; se sube solo para descomprimirlo;
      * sin la copia profunda de la capa NF4 que hace PEFT (un resto para FSDP: ~1,1 s por capa, 10 min en
        las 480 capas de LTX 2.5) y sin bajar el peso descomprimido a la CPU para la norma;
      * con una línea de progreso en la consola.
    El resultado es el mismo número (se comprueba contra PEFT en tools/tests). Idempotente.
    """
    import time
    from peft.tuners.lora import dora
    from peft.utils.integrations import dequantize_module_weight

    cls = dora.DoraLinearLayer
    _DORA_PROGRESS.update(done=0, total=int(total_layers), t0=time.time())
    if getattr(cls, "_block_swap_patched", False):
        return

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False):
        with torch.no_grad(), nf4_on_gpu(base_layer):
            weight = dequantize_module_weight(base_layer)
            dev = weight.device
            lora_weight = lora_B.to(dev, torch.float32) @ lora_A.to(dev, torch.float32)
            weight_norm = self.get_weight_norm(weight=weight, lora_weight=lora_weight, scaling=scaling)
            del weight, lora_weight
        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = torch.nn.Parameter(weight_norm.to(lora_A.device), requires_grad=True)

        p = _DORA_PROGRESS
        p["done"] += 1
        if p["total"] and (p["done"] % 16 == 0 or p["done"] == p["total"]):
            el = time.time() - p["t0"]
            print("\r[DoRA] magnitudes {}/{} ({:.0f} s)".format(p["done"], p["total"], el),
                  end="\n" if p["done"] == p["total"] else "", flush=True)

    cls.update_layer = update_layer
    cls._block_swap_patched = True


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

    def __init__(self, blocks, resident, pin=True, prefetch=True, log=print, park="ram", ram_reserve_gb=8.0,
                 ram_limit_gb=0.0):
        """
        park: dónde viven los bloques aparcados.
          "ram"  copia en RAM fijada (pinned): subidas asíncronas, lo más rápido;
          "disk" mapeados al .safetensors del modelo: no ocupan RAM propia (el sistema suelta sus páginas
                 cuando falta memoria); cada subida lee del fichero o de la caché del sistema;
          "auto" RAM + disco, como MiniMax-H3: cada bloque va a RAM mientras la RAM total del sistema no pase de
                 ram_limit_gb (0 = sin límite) y queden ram_reserve_gb libres; el resto, a disco.
        """
        self.blocks = list(blocks)
        self.n = len(self.blocks)
        self.resident = max(0, min(int(resident), self.n))
        self.prefetch = prefetch
        self.stream = torch.cuda.Stream() if prefetch else None
        self.events = {}
        self.uploads = 0
        self.pinned = False
        self._order = {id(b): i for i, b in enumerate(self.blocks)}
        self.park = park if park in ("ram", "disk", "auto") else "ram"
        parked_bytes = 0
        ram_bytes = disk_bytes = 0
        try:
            import psutil
        except Exception:
            psutil = None
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
            nbytes = block_nf4_bytes(block)
            to_ram = self.park == "ram"
            if self.park == "auto":
                vm = psutil.virtual_memory() if psutil else None
                avail = vm.available if vm else 0
                to_ram = avail - nbytes > ram_reserve_gb * 1e9
                if to_ram and ram_limit_gb and ram_limit_gb > 0 and vm is not None:
                    to_ram = (vm.total - avail) + nbytes <= ram_limit_gb * 1e9
            for m in mods:
                mm = getattr(m, "_mmap_home", None)
                if not to_ram and mm is not None:
                    home = mm                              # disco: mapeado, sin RAM propia
                else:
                    home = m.weight.data.to("cpu")
                    if home.data_ptr() == getattr(mm, "data_ptr", lambda: -1)():
                        home = home.clone()                # de mapeado a RAM: copia real
                    if pin:
                        try:
                            home = home.pin_memory()
                            self.pinned = True
                        except RuntimeError:
                            pin = False
                m._swap_home = home
                m.weight.data = home  # en CPU: la copia CUDA se libera
                parked_bytes += home.numel()
            if to_ram or all(getattr(m, "_mmap_home", None) is None for m in mods):
                ram_bytes += nbytes
            else:
                disk_bytes += nbytes
            block._swap_on_gpu = False
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)
            block.register_full_backward_hook(self._bwd_hook)
        global ACTIVE_SWAP
        ACTIVE_SWAP = self
        self._resident_offloaded = False
        torch.cuda.empty_cache()
        log("[SWAP] resident blocks / bloques residentes: {}/{} | parked / aparcados: {} ({:.2f} GB: {:.2f} GB {}, "
            "{:.2f} GB disk / disco) | storage: {}".format(
                self.resident, self.n, self.n - self.resident, parked_bytes / 1e9, ram_bytes / 1e9,
                "pinned RAM" if self.pinned else "RAM", disk_bytes / 1e9, self.park))

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
        """
        Saca de la VRAM los bloques residentes (para decodificar con el VAE) sin guardar copia en RAM: si el
        bloque está mapeado, su origen es el fichero mapeado; si no, al volver se relee de su .safetensors. Solo
        las capas sin fichero conocido se copian a RAM, y solo mientras dura la preview.
        """
        if self._resident_offloaded:
            return
        for block in self.blocks[:self.resident]:
            for m in block._swap_mods:
                mm = getattr(m, "_mmap_home", None)
                if mm is not None:
                    m._res_home = mm
                    m.weight.data = mm
                elif getattr(m, "_nf4_file", None):
                    m._res_home = None
                    m._res_shape = (m.weight.data.shape, m.weight.data.dtype)
                    m.weight.data = torch.empty(0, dtype=m.weight.data.dtype)
                else:
                    m._res_home = m.weight.data.to("cpu")
                    m.weight.data = m._res_home
        self._resident_offloaded = True
        torch.cuda.empty_cache()

    def restore_resident(self):
        if not self._resident_offloaded:
            return
        for block in self.blocks[:self.resident]:
            for m in block._swap_mods:
                if m._res_home is None:
                    with safe_open(m._nf4_file, framework="pt", device="cpu") as f:
                        data = f.get_tensor("weight")
                    shape, dtype = m._res_shape
                    m.weight.data = data.view(dtype).reshape(shape).to("cuda")
                else:
                    m.weight.data = m._res_home.to("cuda", non_blocking=True)
                m._res_home = None
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


def plan_resident_sized(free_gb, sizes_gb, swap_slots=2):
    """
    Como plan_resident_blocks, pero con bloques de tamaños distintos (FLUX.2: dobles y simples), en el orden en
    que corren: quedan residentes los primeros mientras quepan, dejando sitio para `swap_slots` de los aparcados
    más grandes (el que corre + el prefetch). Devuelve el número de residentes.
    """
    n = len(sizes_gb)
    if sum(sizes_gb) <= free_gb:
        return n
    resident, used = 0, 0.0
    for i, s in enumerate(sizes_gb):
        rest = sorted(sizes_gb[i + 1:], reverse=True)[:swap_slots]
        if used + s + sum(rest) > free_gb:
            break
        used += s
        resident += 1
    return resident


NON_TORCH_VRAM_GB = 1.0


def apply_vram_profile(profile, log=print, t=lambda s, **k: s.format(**k)):
    """
    Tope duro de VRAM del perfil (como H3): simular una tarjeta menor es fiable porque el proceso no puede pasar.
    Lo que PyTorch puede usar de una tarjeta es su VRAM menos ~1 GB que no ve el allocator (contexto CUDA, cuBLAS,
    escritorio de Windows): el tope se pone ahí, así el perfil de 8 GB cabe en una tarjeta real de 8 GB.
    Devuelve (GB de la tarjeta, GB para PyTorch).
    """
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    try:
        card = total if str(profile).strip().lower() in ("", "auto") else min(float(profile), total)
    except ValueError:
        card = total
    usable = card - NON_TORCH_VRAM_GB
    if card < total:
        torch.cuda.set_per_process_memory_fraction(usable / total, 0)
        log("[VRAM] " + t("Profile: {cap:.0f} GB card simulated on a {gb:.1f} GB GPU (hard cap {u:.2f} GB for PyTorch).",
                          cap=card, gb=total, u=usable))
    else:
        log("[VRAM] " + t("Profile: auto ({gb:.1f} GB, {u:.2f} GB for PyTorch)", gb=total, u=usable))
    return card, usable


class SaveOnCpu(torch.autograd.graph.saved_tensors_hooks):
    """
    Como torch.autograd.graph.save_on_cpu, pero un tensor que se guarda varias veces se copia a RAM una sola.
    Con checkpoint reentrante cada bloque guarda sus entradas, y las que comparten todos (la modulación por
    token de LTX 2.5, las de timestep de FLUX.2) se copiaban una vez por bloque: en LTX 2.5, 10-15 GB de RAM
    fijada por paso que el caché de PyTorch retenía para cada tamaño de muestra. Solo se reutiliza la copia si
    es el mismo objeto, vivo y sin cambios.
    """

    def __init__(self, pin_memory=True):
        seen = {}
        self._seen = seen

        def pack(t):
            if t.device.type != "cuda":
                return (None, t)
            key = (t.data_ptr(), t._version, tuple(t.shape), t.stride(), t.dtype)
            hit = seen.get(key)
            if hit is not None and hit[0]() is t:
                return (t.device, hit[1])
            cpu = torch.empty(t.size(), dtype=t.dtype, layout=t.layout, pin_memory=pin_memory)
            cpu.copy_(t)
            seen[key] = (weakref.ref(t), cpu)
            return (t.device, cpu)

        def unpack(packed):
            device, cpu = packed
            return cpu if device is None else cpu.to(device, non_blocking=pin_memory)

        super().__init__(pack, unpack)

    def __exit__(self, *args):
        self._seen.clear()   # las copias siguen vivas mientras el grafo las use
        return super().__exit__(*args)


def enable_reentrant_checkpointing(transformer):
    import functools
    transformer.enable_gradient_checkpointing(
        gradient_checkpointing_func=functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=True))


def ram_report(label=""):
    """RAM propia del proceso, su working set (cuenta también páginas mapeadas que el sistema puede soltar) y la del sistema."""
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        own = getattr(mi, "private", None) or getattr(mi, "uss", None) or mi.rss
        vm = psutil.virtual_memory()
        return "[RAM] {}own {:.1f} GB | working set {:.1f} GB | system {:.1f} / {:.1f} GB".format(
            (label + ": ") if label else "", own / 1e9, mi.rss / 1e9, (vm.total - vm.available) / 1e9, vm.total / 1e9)
    except Exception:
        return "[RAM] -"
