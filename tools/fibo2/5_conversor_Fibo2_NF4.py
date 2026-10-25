# -*- coding: utf-8 -*-
"""
5_conversor_Fibo2_NF4.py

Convierte Fibo 2.0 Dev (BRIA) a NF4 y guarda una carpeta autosuficiente
junto al modelo original:

    Fibo2_Dev-Raw/            (origen, formato diffusers)
    Fibo2_Dev-NF4/            (destino)
        transformer/            caché NF4 (formato LoRAlab)
            config.json
            index.json
            metadata.json
            weights/            pesos NF4 + QuantState (packed=True) y Linear excluidas en BF16
            others.safetensors  resto de parámetros (RMSNorm, adaLN, perceptor, ...)
        text_encoder_BF16/      Qwen3-VL-4B en BF16  (~8.5 GB)
        text_encoder_INT8/      Qwen3-VL-4B en INT8  (~4.9 GB)
        text_encoder_NF4/       Qwen3-VL-4B en NF4   (~2.9 GB)  también captioner
        ...                     resto del modelo copiado sin cambios
                                (vae, scheduler, processor, fonts, model_index.json, ...)

La conversión pesada se ejecuta UNA SOLA VEZ.

Transformer (8.1 B, 44 bloques):
    A diferencia de Krea2, la caché incluye todos los parámetros, así que el trainer
    puede crear el Transformer vacío desde transformer/config.json y rellenarlo sin
    cargar los 15.5 GB en BF16.

    Se cuantizan a NF4 las	lineales de la atención, del MLP y de la atención cruzada
    de los 44 bloques y de los 2 bloques refiner. Se dejan en BF16:
      - x_embedder, t_embedder, final_layer: la entrada, el embedder del tiempo y la salida.
      - adaLN_modulation (uno por bloque): es la modulación que escala cada bloque; un
        error de cuantización ahí se nota en el resultado final.
      - perceiver: el resampler que convierte la salida del VLM en tokens "gist";
        cuantizarlo añadiría ruido justo en la vía que conditioner el texto.
    Todas las lineales cuantizadas no tienen bias (el sesgo va en las que se dejan
    en BF16), pero el formato lo soporta por si cambia el modelo.

Text encoder (Qwen3-VL-4B-Instruct, el mismo que usa el captioner):
    - Solo se usa en el pre-cache, así que no ocupa VRAM durante el entrenamiento.
      Tres variantes para elegir precisión frente a VRAM.
    - embed_tokens queda en BF16 (bitsandbytes no cuantiza embeddings) y lm_head va atado
      a él: el pre-cache nunca usa los logits, y el captioner funciona igual.
    - Se carga directamente con:
        Qwen3VLForConditionalGeneration.from_pretrained(".../text_encoder_<precisión>")

IMPORTANTE:
La cuantización se realiza en GPU porque bitsandbytes necesita
materializar el QuantState real (bnb_quantized=True).

Requisitos: diffusers con BriaFibo2Transformer2DModel (main >= 53788704),
transformers>=5.17, bitsandbytes, accelerate.
"""

import os
import gc
import json
import time
import shutil
import argparse

import torch
from safetensors.torch import save_file


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# tools/fibo2/5_conversor_Fibo2_NF4.py -> raíz del estudio (parents[2])
STUDIO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_ID = os.path.join(STUDIO_DIR, "Fibo2_Dev-Raw")
DEFAULT_OUTPUT_DIR = os.path.join(STUDIO_DIR, "Fibo2_Dev-NF4")

DTYPE = torch.bfloat16

# Módulos de primer nivel del Transformer que NO se cuantizan.
# Debe coincidir EXACTAMENTE con el trainer.
SKIP_TOP = (
    "perceiver",     # resampler de texto -> tokens gist
    "x_embedder",    # parche de entrada
    "t_embedder",    # embedder del tiempo
    "final_layer",   # capa de salida
)

# Sub-módulos que NO se cuantizan, en cualquier parte del árbol.
SKIP_SUB = (
    "adaLN_modulation",   # modulación de cada bloque
)

# Variantes del text encoder, cada una en text_encoder_<precisión>/.
# El pre-cache elige cuál usar según la VRAM disponible.
TEXT_ENCODER_PRECISIONS = ("BF16", "INT8", "NF4")

# Se copia todo el modelo descargado excepto lo que se cuantiza.
COPY_IGNORE = ("transformer", "text_encoder", ".cache", "fine_tuning", "assets")


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
    """True si el módulo (por su ruta) se queda en BF16."""
    parts = full_name.split(".")
    if parts[0] in SKIP_TOP:
        return True
    return any(part in SKIP_SUB for part in parts)


def confirm_overwrite(path, label):
    """
    Devuelve True si hay que (re)generar el componente.
    """

    if not os.path.exists(path):
        return True

    print()
    print(f"[AVISO] Ya existe {label}:")
    print(" ", os.path.abspath(path))

    respuesta = input("\nQuieres sobrescribirlo? [s/N]: ").strip().lower()

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

        # Un módulo entero en la lista de exclusión: ni se toca ni se baja dentro.
        if is_skipped(full):
            continue

        if isinstance(child, torch.nn.Linear):

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

            # module=new_layer: el Linear4bit guarda su propio quant_state.
            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4", module=new_layer)

            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)

            setattr(module, name, new_layer)

            del child
            del w

        else:
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

            if len(index["quantized"]) % 50 == 0:
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
    # RESTO DE PARÁMETROS (normas RMSNorm, adaLN, perceiver, gates...)
    # Se guardan en su dtype original.
    # ------------------------------------------------------------------------

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
    from diffusers import BriaFibo2Transformer2DModel

    print()
    print("=" * 70)
    print("TRANSFORMER (BriaFibo2)")
    print("=" * 70)

    print("Cargando Transformer de Fibo 2.0...")

    transformer = BriaFibo2Transformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        dtype=DTYPE,
    )

    print("Parámetros:", sum(p.numel() for p in transformer.parameters()))

    print()
    print("Cuantizando a NF4 (excepto " + "/".join(SKIP_TOP) + " y " + "/".join(SKIP_SUB) + ")")

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
        "format": "Fibo2-NF4",
        "version": 1,
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "quantized_layers": quantized_count,
        "unquantized_layers": unquantized_count,
        "other_tensors": others_count,
        "skip_quant_top": list(SKIP_TOP),
        "skip_quant_sub": list(SKIP_SUB),
        "bitsandbytes_prequantized": True,
        "quant_state_serialization": "as_dict(packed=True)",
        "loader": "Params4bit.quant_state + QuantState.from_dict",
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

    # lm_head se queda en BF16: después se ata a embed_tokens (que bitsandbytes no cuantiza),
    # y una capa cuantizada con un peso atado revienta al hacer la matmul.
    if precision == "INT8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])

    if precision == "NF4":
        # Los mismos ajustes que Params4bit por defecto en el Transformer.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # lm_head atado a embed_tokens: 0.8 GB menos y la misma matriz para el captioner.
            llm_int8_skip_modules=["lm_head"],
        )

    return None


def convert_text_encoder(output_dir, precision):

    from transformers import Qwen3VLForConditionalGeneration

    print()
    print("=" * 70)
    print(f"TEXT ENCODER (Qwen3-VL-4B) {precision}")
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

    # El pre-cache solo lee hidden states: lm_head se ata a embed_tokens y no se guarda
    # (0.8 GB menos). El captioner sigue funcionando: usa la misma matriz.
    try:
        text_encoder.config.tie_word_embeddings = True
        text_encoder.lm_head.weight = text_encoder.model.language_model.embed_tokens.weight
    except Exception as exc:  # pragma: no cover - depende de la versión de transformers
        print(f"[AVISO] No se pudo atar lm_head a embed_tokens: {exc}")

    print("Guardando...")

    text_encoder.save_pretrained(output_dir)

    del text_encoder
    free_memory()


# ============================================================================
# MAIN
# ============================================================================

MODEL_ID = DEFAULT_MODEL_ID
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def main():

    global MODEL_ID, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Convierte Fibo 2.0 Dev a NF4.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Modelo original en formato diffusers.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Carpeta de salida.")
    parser.add_argument("--skip-text-encoder", action="store_true",
                        help="No convierte el text encoder (solo el transformer).")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="No convierte el transformer (solo el text encoder).")
    parser.add_argument("--yes", action="store_true", help="Sobrescribe sin preguntar.")
    args = parser.parse_args()

    MODEL_ID = os.path.abspath(args.model)
    OUTPUT_DIR = os.path.abspath(args.output)

    start_time = time.time()

    print()
    print("=" * 70)
    print(" FIBO 2.0 DEV NF4 CONVERTER")
    print("=" * 70)

    print()
    print("Modelo:", MODEL_ID)
    print("Salida:", OUTPUT_DIR)

    if not os.path.isdir(MODEL_ID):
        raise SystemExit(f"[ERROR] No existe el modelo de origen: {MODEL_ID}")

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

    if not args.skip_transformer:
        if args.yes or confirm_overwrite(os.path.join(transformer_dir, "index.json"), "la caché NF4 del Transformer"):
            shutil.rmtree(transformer_dir, ignore_errors=True)
            convert_transformer(transformer_dir)
            shutil.copy2(os.path.join(MODEL_ID, "transformer", "config.json"), transformer_dir)

    if not args.skip_text_encoder:
        for precision in TEXT_ENCODER_PRECISIONS:
            text_encoder_dir = os.path.join(OUTPUT_DIR, f"text_encoder_{precision}")
            if args.yes or confirm_overwrite(os.path.join(text_encoder_dir, "config.json"),
                                             f"el text encoder {precision}"):
                shutil.rmtree(text_encoder_dir, ignore_errors=True)
                convert_text_encoder(text_encoder_dir, precision)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("CONVERSIÓN COMPLETADA")
    print("=" * 70)
    print("Tiempo:", f"{elapsed / 60:.1f} minutos")
    # La tarjeta de Hugging Face del modelo NF4 (no la del modelo original de BRIA).
    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README_HF_Fibo2.md")
    if os.path.exists(card):
        shutil.copy2(card, os.path.join(OUTPUT_DIR, "README.md"))

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
