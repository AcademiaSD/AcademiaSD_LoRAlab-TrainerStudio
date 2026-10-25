# -*- coding: utf-8 -*-
"""
qwen_image2512_model.py — Piezas compartidas del entrenador de Qwen-Image-2512 / Qwen-Image-Edit-2511.
Shared pieces of the Qwen-Image-2512 / Qwen-Image-Edit-2511 trainer.

- Plantillas del text encoder y tamaños de imagen, los del pipeline oficial (diffusers QwenImagePipeline
  y QwenImageEditPlusPipeline): el pre-caché y las previews tienen que codificar igual que la inferencia.
- Carga del transformer desde la caché NF4 (5_conversor_QwenImage2512_NF4.py).
- BlockSwap: 20B en NF4 son ~11 GB. Los primeros bloques se quedan en la GPU y el resto vive en RAM;
  cada bloque sube sus pesos NF4 justo antes de usarse y los suelta al terminar.
"""
import json
import math
import os

import psutil
import torch
from accelerate import init_empty_weights
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from diffusers import QwenImageTransformer2DModel
from safetensors import safe_open
from safetensors.torch import load_file

# ---------------------------------------------------------------------------------------------
# Text encoder (Qwen2.5-VL-7B): última capa oculta, sin los tokens de la plantilla.
# ---------------------------------------------------------------------------------------------
PROMPT_T2I = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
              "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
              "<|im_start|>assistant\n")
DROP_T2I = 34
PROMPT_EDIT = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
               "background), then explain how the user's text instruction should alter or modify the image. Generate "
               "a new image that meets the user's requirements while maintaining consistency with the original input "
               "where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
DROP_EDIT = 64
EDIT_IMAGE_PROMPT = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"


def edit_image_prompt(count):
    return "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(count))
MAX_TEXT_TOKENS = 1024
# Edit-2511: la imagen de antes entra al text encoder a ~384² y al transformer (latente) a ~1024².
CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024

TRANSFORMER_FOLDERS = {"t2i": "transformer", "edit": "transformer_edit"}


def calculate_dimensions(target_area, ratio):
    """Ancho y alto múltiplos de 32 con ese área y esa proporción (igual que el pipeline de edición)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def pack_latents(x):
    """[B, 16, H, W] -> [B, H/2·W/2, 64]: parches 2x2, como _pack_latents del pipeline."""
    B, C, H, W = x.shape
    return x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(x, H, W):
    B, _, D = x.shape
    C = D // 4
    return x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


# ---------------------------------------------------------------------------------------------
# Caché NF4
# ---------------------------------------------------------------------------------------------
def load_nf4_transformer(folder):
    """
    Construye el transformer vacío desde config.json y lo rellena con la caché NF4: weights/ (una capa
    Linear4bit por fichero) y others.safetensors (normas, embedders y los bias de las capas NF4).
    """
    with open(os.path.join(folder, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(QwenImageTransformer2DModel.load_config(folder))

    weights_dir = os.path.join(folder, "weights")
    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt", device="cpu") as f:
            data = f.get_tensor("weight")
            qs = QuantState.from_dict({k[len("quant_state."):]: f.get_tensor(k)
                                       for k in f.keys() if k.startswith("quant_state.")}, device="cpu")
        weight = Params4bit(data, requires_grad=False, quant_type="nf4", quant_storage=torch.uint8)
        weight.quant_state = qs
        weight.bnb_quantized = True
        # En meta: nn.Linear.__init__ reservaría y rellenaría una matriz FP32 que se tira al momento.
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"],
                               quant_type="nf4", compute_dtype=torch.bfloat16)
        layer.weight = weight
        parent, _, child = name.rpartition(".")
        setattr(transformer.get_submodule(parent), child, layer)

    transformer.load_state_dict(load_file(os.path.join(folder, index["others"])), strict=False, assign=True)
    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError(f"NF4 cache incomplete: {missing[:5]}")
    transformer.requires_grad_(False)
    return transformer


# ---------------------------------------------------------------------------------------------
# BlockSwap
# ---------------------------------------------------------------------------------------------
def block_nf4_bytes(block):
    return sum(m.weight.numel() for m in block.modules() if isinstance(m, Linear4bit))


class BlockSwap:
    """
    Los bloques [0, resident) se quedan en la GPU. En el resto, solo los pesos NF4 de las Linear4bit
    viven en RAM (fijada si hay memoria de sobra): un pre-hook los sube justo antes del forward del
    bloque y un post-hook los suelta. Todo lo demás (bias, normas, modulación del bloque, LoRA) se
    queda en la GPU porque pesa poco. QuantState también vive en la GPU: es ~3% del peso NF4.

    Entrenando hace falta el checkpoint REENTRANTE (checkpoint_func): bitsandbytes guarda el peso en
    ctx.tensors, fuera de los saved tensors, así que con el no reentrante cada bloque dejaría su
    peso en la GPU hasta el backward. Con el reentrante el primer forward no guarda grafo y cada
    bloque se recalcula (sube, calcula, retropropaga, se suelta) por separado en el backward.
    """

    def __init__(self, transformer, resident, device="cuda"):
        self.device = device
        blocks = transformer.transformer_blocks
        self.resident = max(0, min(resident, len(blocks)))
        swapped = blocks[self.resident:]
        need_gb = sum(block_nf4_bytes(b) for b in swapped) / 1e9
        # Memoria fijada: la copia a la GPU es asíncrona y el doble de rápida, pero esa RAM deja de
        # estar disponible para el sistema. Solo si sobra con holgura.
        pin = psutil.virtual_memory().available / 1e9 > need_gb + 8
        # Los pesos NF4 de los bloques que se intercambian se apartan ANTES de mover el modelo: si no,
        # .to() subiría los 11 GB enteros. Durante el .to() se quedan con un tensor vacío.
        self.layers = {}
        for block in swapped:
            mods = [m for m in block.modules() if isinstance(m, Linear4bit)]
            cpu = []
            for m in mods:
                host = m.weight.data
                cpu.append(host.pin_memory() if pin else host)
                m.weight.data = torch.empty(0, dtype=host.dtype)
            self.layers[block] = (mods, cpu)
        transformer.to(device)  # el resto, y todos los QuantState, a la GPU
        for mods, cpu in self.layers.values():
            for m, host in zip(mods, cpu):
                m.weight.data = host
        self.order = list(self.layers)
        for block in self.order:
            block.register_forward_pre_hook(self._onload)
            block.register_forward_hook(self._offload)
        self.swapped_gb, self.pinned = need_gb, pin
        # Precarga: mientras un bloque calcula, el siguiente se copia en otro stream. Solo con RAM fijada
        # (desde RAM normal la copia no es asíncrona).
        self.stream = torch.cuda.Stream() if pin else None
        self.prefetched = {}
        torch.cuda.empty_cache()

    def _copy(self, block):
        return [host.to(self.device, non_blocking=True) for host in self.layers[block][1]]

    def _prefetch(self, block):
        # Hacia delante (forward y previews, sin gradiente) el siguiente es el de después; en el backward
        # del checkpoint reentrante los bloques se recalculan al revés, con gradiente.
        i = self.order.index(block) + (-1 if torch.is_grad_enabled() else 1)
        if not 0 <= i < len(self.order) or self.order[i] in self.prefetched:
            return
        nxt = self.order[i]
        # El destino se reserva en el stream principal y el otro solo copia: si la memoria se pidiera
        # desde el stream de copia, el allocator la guardaría en un pool aparte (+1.3 GB reservados).
        gpu = [torch.empty_like(host, device=self.device) for host in self.layers[nxt][1]]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for dst, host in zip(gpu, self.layers[nxt][1]):
                dst.copy_(host, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        self.prefetched[nxt] = (gpu, event)

    def _onload(self, block, args):
        mods, _ = self.layers[block]
        if block in self.prefetched:
            gpu, event = self.prefetched.pop(block)
            torch.cuda.current_stream().wait_event(event)
        else:
            gpu = self._copy(block)
        for m, t in zip(mods, gpu):
            m.weight.data = t
        if self.stream is not None:
            self._prefetch(block)

    def _offload(self, block, args, output):
        # Solo se cambia el puntero: si el autograd del recálculo sigue usando el peso en la GPU,
        # esa copia vive hasta que termina el backward del bloque y entonces se libera.
        mods, cpu = self.layers[block]
        for m, host in zip(mods, cpu):
            m.weight.data = host
        return output

    @staticmethod
    def checkpoint_func(module, *args):
        return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=True)


def training_reserve_gb(latent_hw, edit, rank):
    """
    VRAM del entrenamiento sin contar los bloques residentes, ajustada con lo medido en una RTX 5080
    (todos los bloques desde RAM, memoria reservada por PyTorch):
        imagen 512²: 3.9 GB · imagen 1024²: 5.9 GB · edición 512²: 4.6 GB · edición 1024²: 9.0 GB (rank 16)
    Fijo ~1.6 GB (embedders, QuantState, contexto CUDA) + LoRA en FP32 con su gradiente y los dos
    momentos de AdamW (16 bytes por parámetro, ~0.1 GB por punto de rank) + activaciones, que crecen
    con los tokens de imagen (en edición, el antes también cuenta) + 1 GB de margen.
    """
    k = latent_hw / 4 * (2 if edit else 1) / 1000
    return 1.6 + 0.106 * rank + 0.7 * k + 0.006 * k * k + 1.0


def auto_resident_blocks(transformer, free_gb, reserve_gb):
    """Cuántos bloques caben enteros en la GPU dejando reserve_gb para activaciones, LoRA y optimizador."""
    blocks = transformer.transformer_blocks
    per_block = sum(block_nf4_bytes(b) for b in blocks) / len(blocks) / 1e9
    return max(0, min(len(blocks), int((free_gb - reserve_gb) / per_block)))
