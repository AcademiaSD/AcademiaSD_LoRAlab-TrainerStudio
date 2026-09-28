# -*- coding: utf-8 -*-
"""
5_conversor_QwenImage21_NF4.py

Convierte Qwen-Image 2.1 a NF4 y guarda una carpeta autosuficiente
junto a este script:

    Qwen-Image21-NF4/
        transformer/            caché NF4 (formato Krea2 LoRAlab)
            config.json
            index.json
            metadata.json
            weights/            pesos NF4 + QuantState (packed=True) y Linear excluidas en BF16
            others.safetensors  resto de parámetros (normas RMSNorm, etc.)
        text_encoder_BF16/      Qwen3-VL-8B en BF16  (~15.2 GB)  formato nativo de transformers
        text_encoder_INT8/      Qwen3-VL-8B en INT8  (~8.8 GB)
        text_encoder_NF4/       Qwen3-VL-8B en NF4   (~5.4 GB)  también captioner
        ...                     resto del modelo descargado, copiado sin cambios
                                (vae, scheduler, processor, model_index.json, ...)

La conversión pesada se ejecuta UNA SOLA VEZ.

Transformer:
    A diferencia de Krea2, la caché incluye todos los parámetros, así que
    el trainer puede crear el Transformer vacío desde transformer/config.json
    y rellenarlo sin cargar los 14 GB en BF16.

Text encoder:
    - Solo se usa en el pre-cache, así que no ocupa VRAM durante el entrenamiento.
      Tres variantes para elegir precisión frente a VRAM. Error por token frente a
      BF16 medido en captions reales: INT8 ~9%, NF4 ~25%.
    - embed_tokens queda en BF16 (bitsandbytes no cuantiza embeddings).
    - Es exactamente Qwen3-VL-8B-Instruct, así que NF4 también sirve de captioner
      (0_caption_qwen_image21.py) y conserva lm_head. En BF16/INT8 lm_head no se
      guarda: se ata a embed_tokens porque el pre-cache nunca usa los logits.
    - Se carga directamente con:
        Qwen3VLForConditionalGeneration.from_pretrained(".../text_encoder_<precisión>")

IMPORTANTE:
La cuantización se realiza en GPU porque bitsandbytes necesita
materializar el QuantState real (bnb_quantized=True).

Requisitos: diffusers desde GitHub (QwenImage21Transformer2DModel),
transformers>=5.17, bitsandbytes, accelerate.
"""

import os
import gc
import json
import time
import shutil

import torch
from safetensors.torch import save_file


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_ID = r"F:\models\Qwen-Image-2.1"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen-Image21-NF4")

DTYPE = torch.bfloat16

# Módulos de primer nivel del Transformer que NO se cuantizan.
# Debe coincidir EXACTAMENTE con el trainer.
#   modulation: una sola capa compartida por los 32 bloques
#   txt_in:     proyección del texto (incluye la RMSNorm centrada en cero)
SKIP_QUANT = (
    "img_in",
    "time_text_embed",
    "txt_in",
    "modulation",
    "norm_out",
    "proj_out",
)

# Variantes del text encoder, cada una en text_encoder_<precisión>/.
# El pre-cache elige cuál usar según la VRAM disponible.
TEXT_ENCODER_PRECISIONS = ("BF16", "INT8", "NF4")

# Se copia todo el modelo descargado excepto lo que se cuantiza.
COPY_IGNORE = ("transformer", "text_encoder", ".cache")


# ============================================================================
# UTILIDADES
# ============================================================================

def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value):
    """
    Convierte objetos de configuración de torch a tipos serializables
    por JSON.
    """

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def is_skipped(full_name):
    return full_name.split(".")[0] in SKIP_QUANT


def confirm_overwrite(path, label):
    """
    Devuelve True si hay que (re)generar el componente.
    """

    if not os.path.exists(path):
        return True

    print()
    print(f"[AVISO] Ya existe {label}:")
    print(" ", os.path.abspath(path))

    respuesta = input("\n¿Quieres sobrescribirlo? [s/N]: ").strip().lower()

    if respuesta in ("s", "si", "sí", "y", "yes"):
        return True

    print(f"Se conserva {label}.")
    return False


# ============================================================================
# CUANTIZACIÓN DEL TRANSFORMER
# ============================================================================

def quantize_to_nf4_(module, prefix=""):
    """
    Recorre recursivamente el Transformer y reemplaza Linear por Linear4bit.

    IMPORTANTE:
    No guardamos aquí.

    Esta función únicamente realiza la conversión.
    """

    from bitsandbytes.nn import Linear4bit, Params4bit

    for name, child in list(module.named_children()):

        full = f"{prefix}.{name}" if prefix else name

        if isinstance(child, torch.nn.Linear) and not is_skipped(full):

            print(f"  NF4 -> {full}")

            # Copia FP32 temporal.
            # Es el mismo procedimiento utilizado por el trainer.
            w = child.weight.data.float().contiguous()

            new_layer = Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )

            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")

            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)

            setattr(module, name, new_layer)

            del child
            del w

        elif not is_skipped(full):
            quantize_to_nf4_(child, full)


# ============================================================================
# EXTRACCIÓN DE LA CACHÉ NF4
# ============================================================================

def extract_nf4_cache(transformer, output_dir):

    from bitsandbytes.nn import Linear4bit

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    index = {
        "quantized": {},
        "unquantized": {},
        "others": "others.safetensors",
    }

    # Parámetros ya guardados por capa; el resto va a others.safetensors.
    saved_params = set()

    print()
    print("=" * 70)
    print("EXTRAYENDO CACHÉ NF4")
    print("=" * 70)

    for name, layer in transformer.named_modules():

        filename = name.replace(".", "__") + ".safetensors"
        filepath = os.path.join(weights_dir, filename)

        # ------------------------------------------------------------
        # CAPAS NF4
        # ------------------------------------------------------------

        if isinstance(layer, Linear4bit):

            weight = layer.weight
            qs = weight.quant_state

            if not getattr(weight, "bnb_quantized", False) or qs is None:
                raise RuntimeError(f"{name}: Linear4bit sin QuantState válido.")

            qs_dict = qs.as_dict(packed=True)

            # Peso NF4 real: uint8 con shape [N, 1].
            tensors = {"weight": weight.data.detach().cpu().contiguous()}

            if layer.bias is not None:
                tensors["bias"] = layer.bias.detach().to(torch.bfloat16).cpu().contiguous()

            metadata = {
                "layer_name": name,
                "type": "Linear4bit",
                "quant_type": "nf4",
                "bnb_quantized": "true",
                "in_features": str(layer.in_features),
                "out_features": str(layer.out_features),
                "bias": str(layer.bias is not None),
            }

            # as_dict(packed=True) puede devolver absmax, quant_map,
            # nested_absmax, nested_quant_map y quant_state.bitsandbytes__nf4.
            # Los tensores se guardan junto al peso; el resto como metadata JSON.
            for key, value in qs_dict.items():

                if torch.is_tensor(value):
                    # Evitamos colisión con "weight" y "bias".
                    tensors["quant_state." + key] = value.detach().cpu().contiguous()
                else:
                    metadata["qs_" + key] = json.dumps(json_safe(value))

            save_file(tensors, filepath, metadata=metadata)

            index["quantized"][name] = {
                "file": filename,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "quant_state_keys": list(qs_dict.keys()),
            }

            saved_params.update(f"{name}.{p}" for p in ("weight", "bias"))

            if len(index["quantized"]) % 25 == 0:
                print(f"  Guardadas {len(index['quantized'])} capas NF4...")

        # ------------------------------------------------------------
        # CAPAS LINEAR EXCLUIDAS (BF16)
        # ------------------------------------------------------------

        elif isinstance(layer, torch.nn.Linear):

            tensors = {"weight": layer.weight.detach().to(torch.bfloat16).cpu().contiguous()}

            if layer.bias is not None:
                tensors["bias"] = layer.bias.detach().to(torch.bfloat16).cpu().contiguous()

            save_file(
                tensors,
                filepath,
                metadata={"layer_name": name, "type": "Linear", "quantized": "false"},
            )

            index["unquantized"][name] = {
                "file": filename,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            }

            saved_params.update(f"{name}.{p}" for p in ("weight", "bias"))

    # ------------------------------------------------------------------------
    # RESTO DE PARÁMETROS (normas con pesos)
    # ------------------------------------------------------------------------
    # txt_in.text_norm.weight y attn.norm_q / attn.norm_k de cada bloque.
    # Se guardan en su dtype original.

    others = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in list(transformer.named_parameters()) + list(transformer.named_buffers())
        if name not in saved_params
    }

    save_file(others, os.path.join(output_dir, "others.safetensors"))

    print(f"  Otros parámetros: {len(others)} tensores")

    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return len(index["quantized"]), len(index["unquantized"]), len(others)


# ============================================================================
# TRANSFORMER
# ============================================================================

def convert_transformer(output_dir):

    from bitsandbytes.nn import Linear4bit
    from diffusers import QwenImage21Transformer2DModel

    print()
    print("=" * 70)
    print("TRANSFORMER")
    print("=" * 70)

    print("Cargando Transformer de Qwen-Image 2.1...")

    transformer = QwenImage21Transformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=DTYPE,
    )

    print("Parámetros:", sum(p.numel() for p in transformer.parameters()))

    print()
    print("Cuantizando a NF4 (excepto " + "/".join(SKIP_QUANT) + ")")

    quantize_to_nf4_(transformer)

    # ------------------------------------------------------------------------
    # MOVER A CUDA: aquí bitsandbytes materializa el QuantState
    # ------------------------------------------------------------------------

    print()
    print("Moviendo Transformer a CUDA...")

    transformer.to("cuda")
    free_memory()

    print("VRAM:", f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    verified = sum(
        1
        for layer in transformer.modules()
        if isinstance(layer, Linear4bit)
        and getattr(layer.weight, "bnb_quantized", False)
        and layer.weight.quant_state is not None
    )

    print("Capas Linear4bit verificadas:", verified)

    if verified == 0:
        raise RuntimeError(
            "No se encontró ninguna Linear4bit con bnb_quantized=True y QuantState válido."
        )

    quantized_count, unquantized_count, others_count = extract_nf4_cache(transformer, output_dir)

    metadata = {
        "format": "QwenImage21-NF4",
        "version": 1,
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "quantized_layers": quantized_count,
        "unquantized_layers": unquantized_count,
        "other_tensors": others_count,
        "skip_quant": list(SKIP_QUANT),
        "bitsandbytes_prequantized": True,
        "quant_state_serialization": "as_dict(packed=True)",
        "loader": "Params4bit.from_prequantized",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Capas NF4:", quantized_count)
    print("Capas BF16:", unquantized_count)

    del transformer
    free_memory()


# ============================================================================
# TEXT ENCODER
# ============================================================================

def text_encoder_quant_config(precision):
    from transformers import BitsAndBytesConfig

    if precision == "INT8":
        return BitsAndBytesConfig(load_in_8bit=True)

    if precision == "NF4":
        # Los mismos ajustes que Params4bit por defecto en el Transformer.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Lista vacía: transformers dejaría lm_head en BF16 (+1.2 GB); en NF4 basta para el captioner.
            llm_int8_skip_modules=[],
        )

    return None


def convert_text_encoder(output_dir, precision):

    from transformers import Qwen3VLForConditionalGeneration

    print()
    print("=" * 70)
    print(f"TEXT ENCODER (Qwen3-VL-8B) {precision}")
    print("=" * 70)

    quant_config = text_encoder_quant_config(precision)

    # BF16 no necesita GPU: se carga en RAM para no depender de la VRAM disponible.
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        quantization_config=quant_config,
        dtype=DTYPE,
        device_map="cuda" if quant_config is not None else "cpu",
    )

    # El pre-cache solo lee hidden states: en BF16/INT8 lm_head se ata a embed_tokens
    # y no se guarda (-1.2 GB). NF4 lo conserva (+0.3 GB) porque es el captioner.
    if precision != "NF4":
        text_encoder.config.tie_word_embeddings = True
        text_encoder.lm_head.weight = text_encoder.model.language_model.embed_tokens.weight

    print("Guardando...")

    text_encoder.save_pretrained(output_dir)

    del text_encoder
    free_memory()


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print(" QWEN-IMAGE 2.1 NF4 CONVERTER")
    print("=" * 70)

    print()
    print("Modelo:", MODEL_ID)
    print("Salida:", OUTPUT_DIR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. La cuantización NF4 requiere CUDA.")

    print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Copiando el modelo sin " + ", ".join(COPY_IGNORE) + "...")

    shutil.copytree(
        MODEL_ID,
        OUTPUT_DIR,
        ignore=shutil.ignore_patterns(*COPY_IGNORE),
        dirs_exist_ok=True,
    )

    transformer_dir = os.path.join(OUTPUT_DIR, "transformer")

    if confirm_overwrite(os.path.join(transformer_dir, "index.json"), "la caché NF4 del Transformer"):
        shutil.rmtree(transformer_dir, ignore_errors=True)
        convert_transformer(transformer_dir)
        shutil.copy2(os.path.join(MODEL_ID, "transformer", "config.json"), transformer_dir)

    for precision in TEXT_ENCODER_PRECISIONS:
        text_encoder_dir = os.path.join(OUTPUT_DIR, f"text_encoder_{precision}")
        if confirm_overwrite(os.path.join(text_encoder_dir, "config.json"), f"el text encoder {precision}"):
            shutil.rmtree(text_encoder_dir, ignore_errors=True)
            convert_text_encoder(text_encoder_dir, precision)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("CONVERSIÓN COMPLETADA")
    print("=" * 70)
    print("Tiempo:", f"{elapsed / 60:.1f} minutos")
    print()
    print("Carpeta NF4:")
    print(os.path.abspath(OUTPUT_DIR))
    print()
    print("IMPORTANTE:")
    print("No borres el modelo original todavía.")
    print("Primero debemos probar la reconstrucción")
    print("de la caché NF4 y comparar el modelo.")
    print("=" * 70)


if __name__ == "__main__":
    main()
