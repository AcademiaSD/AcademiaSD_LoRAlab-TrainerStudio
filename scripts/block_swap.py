# -*- coding: utf-8 -*-
"""
block_swap.py — block swap para entrenar modelos NF4 en poca VRAM / block swap to train NF4 models on little VRAM.

Lo usan LTX 2.5, FLUX.2 Klein 9B, FLUX.1 Dev, Fibo 2.0, Ideogram 4, Ming-Image, Wan 2.2 y LTX-2.3. Al estilo de MiniMax-H3:
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
        self._pin = pin
        self._ram_reserve_gb = ram_reserve_gb
        self._ram_limit_gb = ram_limit_gb
        self.log = log
        self.ram_bytes = self.disk_bytes = 0
        for i, block in enumerate(self.blocks):
            mods = [m for m in block.modules() if isinstance(m, Linear4bit)]
            block._swap_mods = mods
            block._swap_index = i
            block._swap_parked = i >= self.resident
            block._swap_hooked = False
            if not block._swap_parked:
                for m in mods:
                    if m.weight.data.device.type != "cuda":
                        m.weight.data = m.weight.data.to("cuda")
                continue
            self._park(block)
        global ACTIVE_SWAP
        ACTIVE_SWAP = self
        self._resident_offloaded = False
        self._suspended = None
        torch.cuda.empty_cache()
        log("[SWAP] resident blocks / bloques residentes: {}/{} | parked / aparcados: {} ({:.2f} GB: {:.2f} GB {}, "
            "{:.2f} GB disk / disco) | storage: {}".format(
                self.resident, self.n, self.n - self.resident, (self.ram_bytes + self.disk_bytes) / 1e9,
                self.ram_bytes / 1e9, "pinned RAM" if self.pinned else "RAM", self.disk_bytes / 1e9, self.park))

    def _park(self, block):
        """Aparca un bloque: su dato NF4 pasa a su sitio en RAM o en disco y se sube solo mientras corre."""
        try:
            import psutil
        except Exception:
            psutil = None
        mods = block._swap_mods
        nbytes = block_nf4_bytes(block)
        to_ram = self.park == "ram"
        if self.park == "auto":
            vm = psutil.virtual_memory() if psutil else None
            avail = vm.available if vm else 0
            to_ram = avail - nbytes > self._ram_reserve_gb * 1e9
            if to_ram and self._ram_limit_gb and self._ram_limit_gb > 0 and vm is not None:
                to_ram = (vm.total - avail) + nbytes <= self._ram_limit_gb * 1e9
        for m in mods:
            mm = getattr(m, "_mmap_home", None)
            if not to_ram and mm is not None:
                home = mm                              # disco: mapeado, sin RAM propia
            else:
                home = m.weight.data.to("cpu")
                if home.data_ptr() == getattr(mm, "data_ptr", lambda: -1)():
                    home = home.clone()                # de mapeado a RAM: copia real
                if self._pin:
                    try:
                        home = home.pin_memory()
                        self.pinned = True
                    except RuntimeError:
                        self._pin = False
            m._swap_home = home
            m.weight.data = home  # en CPU: la copia CUDA se libera
        if to_ram or all(getattr(m, "_mmap_home", None) is None for m in mods):
            self.ram_bytes += nbytes
        else:
            self.disk_bytes += nbytes
        block._swap_parked = True
        block._swap_on_gpu = False
        if not block._swap_hooked:
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)
            block.register_full_backward_hook(self._bwd_hook)
            block._swap_hooked = True

    def reset(self):
        """Tras un error a mitad de paso (OOM): suelta los bloques aparcados que se quedaron subidos."""
        for block in self.blocks:
            if block._swap_parked:
                block._swap_on_gpu = True
                self._release(block)
        self.events.clear()

    def park_more(self, k=2):
        """
        Aparca los k últimos bloques residentes (como MiniMax-H3 cuando falta VRAM: más lento, pero sigue).
        Devuelve cuántos aparcó.
        """
        self.reset()
        k = min(int(k), self.resident)
        for block in self.blocks[self.resident - k:self.resident]:
            self._park(block)
        self.resident -= k
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return k

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
                        data = f.get_tensor(getattr(m, "_nf4_key", "weight"))
                    shape, dtype = m._res_shape
                    m.weight.data = data.view(dtype).reshape(shape).to("cuda")
                else:
                    m.weight.data = m._res_home.to("cuda", non_blocking=True)
                m._res_home = None
        torch.cuda.synchronize()
        self._resident_offloaded = False

    # -- hooks ---------------------------------------------------------------------------------------------------
    def suspend(self):
        """
        Para una preview (va sin grad): aparca también los bloques residentes, así la preview tiene la VRAM que
        ocupaban (Ideogram 4 sube además su transformer incondicional). resume() los vuelve a subir.
        """
        if self._suspended is not None or not self.resident:
            return
        self.reset()
        counters = (self.ram_bytes, self.disk_bytes)
        for block in self.blocks[:self.resident]:
            self._park(block)
        self.ram_bytes, self.disk_bytes = counters
        self._suspended = self.resident
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def resume(self):
        if self._suspended is None:
            return
        self.reset()
        for block in self.blocks[:self._suspended]:
            for m in block._swap_mods:
                m.weight.data = m._swap_home.to("cuda")
                m._swap_home = None
            block._swap_parked = False
            block._swap_on_gpu = True
        self._suspended = None
        torch.cuda.synchronize()

    def _pre_hook(self, block, args):
        if not block._swap_parked:
            return
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
        if block._swap_parked and not torch.is_grad_enabled():
            self._release(block)
        return output

    def _bwd_hook(self, block, grad_input, grad_output):
        if block._swap_parked:
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


NON_TORCH_VRAM_GB = 1.0   # mínimo que se deja fuera de PyTorch (contexto CUDA, cuBLAS, escritorio)
LATE_CONTEXT_GB = 0.3     # lo que el contexto CUDA de este proceso crece después (kernels que se cargan al usarlos)


def apply_vram_profile(profile, log=print, t=lambda s, **k: s.format(**k)):
    """
    Tope duro de VRAM del perfil (como H3): simular una tarjeta menor es fiable porque el proceso no puede pasar.
    PyTorch puede usar la VRAM de la tarjeta menos lo que ya está ocupado al arrancar: el escritorio y las demás
    aplicaciones (Windows, el navegador... medido: 0,5-1,3 GB) y el contexto CUDA de este proceso, más lo que ese
    contexto crece después. Se mide al arrancar (mem_get_info), con 1 GB como mínimo: así el perfil de 8 GB cabe
    en una tarjeta real de 8 GB con ese mismo escritorio. Devuelve (GB de la tarjeta, GB para PyTorch).
    """
    gib = 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / gib
    try:
        card = total if str(profile).strip().lower() in ("", "auto") else min(float(profile), total)
    except ValueError:
        card = total
    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
        in_use = (total_b - free_b) / gib     # las demás aplicaciones + el contexto CUDA de este proceso
    except Exception:
        in_use = 0.0
    outside = max(NON_TORCH_VRAM_GB, in_use + LATE_CONTEXT_GB)
    usable = max(1.0, card - outside)
    if card < total:
        torch.cuda.set_per_process_memory_fraction(usable / total, 0)
        log("[VRAM] " + t("Profile: {cap:.0f} GB card simulated on a {gb:.1f} GB GPU (hard cap {u:.2f} GB for PyTorch; "
                          "{o:.2f} GB already in use by other apps and the CUDA context).",
                          cap=card, gb=total, u=usable, o=in_use))
    else:
        log("[VRAM] " + t("Profile: auto ({gb:.1f} GB, {u:.2f} GB for PyTorch; {o:.2f} GB already in use by other "
                          "apps and the CUDA context)", gb=total, u=usable, o=in_use))
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
            # Con los mismos strides: algún backward (la atención de Fibo 2.0) lee el tensor con la disposición en
            # memoria con que se guardó, y uno contiguo le daba NaN. Solo si hay solapes (stride 0) va contiguo.
            if all(st != 0 or sz <= 1 for st, sz in zip(t.stride(), t.size())):
                cpu = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, layout=t.layout, pin_memory=pin_memory)
            else:
                cpu = torch.empty(t.size(), dtype=t.dtype, layout=t.layout, pin_memory=pin_memory)
            cpu.copy_(t)
            seen[key] = (weakref.ref(t), cpu)
            return (t.device, cpu)

        def unpack(packed):
            device, cpu = packed
            if device is None:
                return cpu
            gpu = torch.empty_strided(cpu.size(), cpu.stride(), dtype=cpu.dtype, layout=cpu.layout, device=device)
            return gpu.copy_(cpu, non_blocking=pin_memory)

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


def inputs_require_grad(module):
    """
    Con checkpoint reentrante un bloque solo da gradiente si alguna de sus entradas lo pide. Se engancha a la capa
    de entrada (x_embedder...): su salida lo pide aunque esa capa no tenga LoRA.
    """
    def hook(mod, inputs, output):
        if torch.is_grad_enabled() and torch.is_tensor(output):
            output.requires_grad_(True)
    return module.register_forward_hook(hook)


def load_bnb_nf4(model_cls, transformer_dir, block_prefixes, device="cuda", compute_dtype=torch.bfloat16):
    """
    Construye un transformer de diffusers guardado en NF4 por bitsandbytes (un .safetensors con `x.weight`,
    `x.weight.absmax`, `x.weight.quant_map`, `x.weight.quant_state.bitsandbytes__nf4`) sin cargarlo en la GPU:
    el dato NF4 de las capas de los bloques (`block_prefixes`) queda mapeado al fichero (BlockSwap decide qué
    sube), y todo lo demás va a `device`.
    """
    import glob
    from accelerate import init_empty_weights
    from bitsandbytes.functional import QuantState
    cfg = model_cls.load_config(transformer_dir)
    cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
    with init_empty_weights():
        model = model_cls.from_config(cfg)
    files = sorted(glob.glob(os.path.join(transformer_dir, "diffusion_pytorch_model*.safetensors")))
    if not files:
        raise FileNotFoundError(transformer_dir)
    block_prefixes = tuple(block_prefixes)
    marker = ".weight.quant_state.bitsandbytes__nf4"
    plain = {}
    n_q = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            quantized = [k[:-len(marker)] for k in keys if k.endswith(marker)]
            qset = set(quantized)
            for name in quantized:
                pre = name + ".weight."
                qs = {k[len(pre):]: f.get_tensor(k) for k in keys if k.startswith(pre)}
                in_block = name.startswith(block_prefixes)
                data = mmap_tensor(path, name + ".weight") if in_block else f.get_tensor(name + ".weight").to(device)
                old = model.get_submodule(name)
                with torch.device("meta"):
                    layer = Linear4bit(old.in_features, old.out_features, bias=old.bias is not None,
                                       quant_type="nf4", compute_dtype=compute_dtype)
                w = Params4bit(data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
                w.quant_state = QuantState.from_dict(qs, device=device)
                w.bnb_quantized = True
                layer.weight = w
                if old.bias is not None:
                    layer.bias = torch.nn.Parameter(f.get_tensor(name + ".bias").to(device, compute_dtype),
                                                    requires_grad=False)
                layer._nf4_file = path
                layer._nf4_key = name + ".weight"
                if in_block:
                    layer._mmap_home = data
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, child, layer)
                n_q += 1
            for k in keys:
                if k.endswith(".bias"):
                    owner = k[:-5]
                elif ".weight" in k:
                    owner = k[:k.rindex(".weight")]
                else:
                    owner = None
                if owner in qset:
                    continue                      # ya está en su Linear4bit
                plain[k] = f.get_tensor(k).to(device)
    model.load_state_dict(plain, strict=False, assign=True)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_meta:
            raise RuntimeError("NF4 model incomplete: " + n)
    to_device_except_nf4(model, device)
    model.requires_grad_(False)
    return model, n_q, len(plain)


def setup(blocks, model, card_gb, usable_gb, act_gb, resident_blocks=-1, park="auto", ram_limit_gb=0.0,
          ram_reserve_gb=8.0, dora=False, headroom_gb=0.8, dora_margin_gb=0.5, log=print,
          t=lambda s, **k: s.format(**k)):
    """
    Plan del block swap y BlockSwap, como en FLUX.2 Klein 9B. Lo que ya hay en VRAM (resto del transformer, LoRA,
    contexto) se mide; el optimizador (AdamW: 2 estados FP32 + el gradiente por parámetro) y las activaciones del
    paso (`act_gb`, medidas en cada trainer) se estiman. `headroom_gb` deja margen para la fragmentación: con
    muchos tamaños de bucket la VRAM queda en trozos; si aun así falta, el paso aparca más bloques (park_more).
    """
    gib = 1024 ** 3
    sizes = [block_nf4_bytes(b) / gib for b in blocks]
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    in_use = torch.cuda.memory_allocated() / gib
    optim_gb = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad) * 3 / gib
    free = usable_gb - headroom_gb - in_use - optim_gb - act_gb
    if resident_blocks is not None and int(resident_blocks) >= 0:
        resident = min(int(resident_blocks), len(blocks))
    else:
        resident = plan_resident_sized(free, sizes)
    # Ni con todos los bloques aparcados queda sitio para el que corre y el siguiente. Con DoRA, en Klein 9B hizo
    # falta medio GB más: sus pesos descomprimidos, de tamaños distintos, fragmentan la VRAM.
    need = 2 * max(sizes) + (dora_margin_gb if dora else 0.0)
    if free < need:
        log("[VRAM] " + t("Warning: this run does not fit the {c:.0f} GB profile even with every block in RAM "
                          "(~{need:.1f} GB more). Pre-cache at a lower resolution, turn DoRA off or choose a larger "
                          "VRAM Profile.", c=card_gb, need=need - free))
    log("[VRAM] " + t("Plan: card {c:.1f} GB | in use {f:.2f} GB | optimizer ~{o:.2f} GB | activations ~{a:.2f} GB "
                      "| blocks {b:.2f} GB -> {r}/{n} resident",
                      c=card_gb, f=in_use, o=optim_gb, a=act_gb, b=sum(sizes), r=resident, n=len(blocks)))
    swap = BlockSwap(blocks, resident, pin=True, log=log, park=park, ram_reserve_gb=ram_reserve_gb,
                     ram_limit_gb=ram_limit_gb)
    torch.cuda.synchronize()
    log("[VRAM] " + t("After placing the blocks: {gb:.2f} GB allocated", gb=torch.cuda.memory_allocated() / 1e9))
    log(ram_report())
    return swap


def oom_recover(swap, optimizer, step, log=print, t=lambda s, **k: s.format(**k)):
    """
    Un paso se quedó sin VRAM (fragmentación, un bucket mayor): como MiniMax-H3, se aparcan 2 bloques más y el
    paso se repite. Los gradientes a medias del paso no valen: se descarta la acumulación en curso. Devuelve
    False si ya no queda nada que aparcar (el error se relanza).
    """
    if swap is None or swap.resident == 0:
        return False
    optimizer.zero_grad(set_to_none=True)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    parked = swap.park_more(2)
    log("\n[VRAM] " + t("Out of VRAM at step {s}: {k} more blocks to RAM ({r}/{n} resident now), and the "
                        "step is repeated.", s=step, k=parked, r=swap.resident, n=swap.n))
    return True


# Ajustes de memoria comunes (GUI: VRAM Profile, Resident Blocks, Block Swap Storage, RAM Limit).
SETTINGS_DEFAULTS = {
    "vram_profile": "auto",     # simula una tarjeta menor con un tope duro
    "resident_blocks": -1,      # -1 = lo decide el plan
    "park_mode": "auto",        # ram / disk / auto (RAM + disco)
    "ram_limit_gb": 0.0,        # techo de RAM total del sistema en auto (0 = sin límite)
    "ram_reserve_gb": 8.0,      # RAM que queda libre en auto
    "activation_offload": True, # entradas de cada bloque a RAM fijada mientras hay block swap
}


def settings(cfg):
    """Lee los ajustes de memoria del JSON del trainer (con los valores por defecto de arriba)."""
    from types import SimpleNamespace
    g = lambda k: cfg.get(k, SETTINGS_DEFAULTS[k])
    park = str(g("park_mode") or "auto").strip().lower()
    try:
        limit = max(0.0, float(g("ram_limit_gb") or 0))
    except (TypeError, ValueError):
        limit = 0.0
    try:
        resident = int(g("resident_blocks") if g("resident_blocks") is not None else -1)
    except (TypeError, ValueError):
        resident = -1
    return SimpleNamespace(
        vram_profile=str(g("vram_profile") or "auto").strip().lower(),
        resident_blocks=resident,
        park_mode=park if park in ("auto", "ram", "disk") else "auto",
        ram_limit_gb=limit,
        ram_reserve_gb=float(g("ram_reserve_gb")),
        activation_offload=bool(g("activation_offload")),
    )


def describe(s):
    """Línea de la cabecera del trainer."""
    return "{} | Resident Blocks {} | Block Swap Storage {}{}".format(
        s.vram_profile, "auto" if s.resident_blocks < 0 else s.resident_blocks, s.park_mode,
        " (RAM Limit {:g} GB)".format(s.ram_limit_gb) if s.park_mode == "auto" and s.ram_limit_gb > 0 else "")


def peak_report(t=lambda s, **k: s.format(**k)):
    return "[VRAM] " + t("Peak: {a:.2f} GB allocated · {r:.2f} GB reserved by PyTorch",
                         a=torch.cuda.max_memory_allocated() / 1e9, r=torch.cuda.max_memory_reserved() / 1e9)


def load_index_nf4(model_cls, cache_dir, block_prefixes, device="cuda", compute_dtype=torch.bfloat16):
    """
    Como load_bnb_nf4, para las cachés NF4 propias (transformer/index.json + weights/<capa>.safetensors +
    others.safetensors, las de tools/): el dato NF4 de las capas de los bloques queda mapeado a su fichero.
    """
    from accelerate import init_empty_weights
    from bitsandbytes.functional import QuantState
    from safetensors.torch import load_file
    with open(os.path.join(cache_dir, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    with init_empty_weights():
        model = model_cls.from_config(model_cls.load_config(cache_dir))
    weights_dir = os.path.join(cache_dir, "weights")
    block_prefixes = tuple(block_prefixes)

    def set_module(name, module):
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, module)

    for name, info in index["quantized"].items():
        path = os.path.join(weights_dir, info["file"])
        in_block = name.startswith(block_prefixes)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            data = None if in_block else f.get_tensor("weight").to(device)
            qs = {k[len("quant_state."):]: f.get_tensor(k) for k in keys if k.startswith("quant_state.")}
            bias = f.get_tensor("bias") if "bias" in keys else None
        if in_block:
            data = mmap_tensor(path, "weight")
        w = Params4bit(data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        w.quant_state = QuantState.from_dict(qs, device=device)
        w.bnb_quantized = True
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=bias is not None,
                               quant_type="nf4", compute_dtype=compute_dtype)
        layer.weight = w
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias.to(device, compute_dtype), requires_grad=False)
        layer._nf4_file = path
        if in_block:
            layer._mmap_home = data
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"], bias=info.get("bias", False),
                                    dtype=compute_dtype)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"]), device=device), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    model.load_state_dict(load_file(os.path.join(cache_dir, index["others"]), device=device), strict=False, assign=True)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_meta:
            raise RuntimeError("NF4 cache incomplete: " + n)
    to_device_except_nf4(model, device)
    model.requires_grad_(False)
    return model, len(index["quantized"]), len(index["unquantized"])


class preview_room:
    """Contexto para las previews: con block swap, todos los bloques del entrenamiento quedan aparcados mientras dura."""

    def __enter__(self):
        self.swap = ACTIVE_SWAP if (ACTIVE_SWAP is not None and ACTIVE_SWAP.swapping) else None
        if self.swap is not None:
            self.swap.suspend()
        return self

    def __exit__(self, *exc):
        if self.swap is not None:
            self.swap.resume()
        return False
