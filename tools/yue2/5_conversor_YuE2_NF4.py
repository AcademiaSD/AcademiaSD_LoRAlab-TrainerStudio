# -*- coding: utf-8 -*-
"""
5_conversor_YuE2_NF4.py

Convierte el backbone YuE2-3B (pesos Raw de m-a-p/YuE2-3B) a NF4 y genera una
carpeta autosuficiente en la raíz de LoRAlab Trainer Studio:

    YuE2-3B_NF4/
        model/              caché NF4 del backbone (formato caché NF4 del studio)
            config.json
            index.json      {"quantized": {...}, "unquantized": {...}, "others": …}
            metadata.json
            others.safetensors
            weights/        capa por fichero: peso NF4 (uint8) + QuantState
                            (as_dict(packed=True)); Linear excluidas en BF16
        vae/                YuE2-VAE copiado tal cual
        modeling_yue2.py    copia del modeling (consumo offline sin trust_remote_code)
        yue2_generation_config.json y ficheros del tokenizador (copias)

Reglas de cuantización EXACTAS a las del trainer (2_train_lora_yue2.py,
is_quantized): NF4 solo de atención+MLP de las dos ramas (AR y NAR) de los 28
bloques; llm2vae/vae2llm/time_embedder, normas, embed_tokens, lm_head y
latent_pos_embed quedan en BF16: la escala del latente del VAE y el flow
matching son la parte sensible.

Se cuantiza en GPU (bitsandbytes materializa ahí el QuantState real). El
script termina VERIFICANDO la reconstrucción: recarga la caché con el mismo
cargador del trainer y compara los logits AR del modelo NF4 contra el modelo
BF16 de referencia (NF4 es cuantización con pérdida: se espera un error
pequeño, no cero; lo que se quiere acotar es que el QuantState se reconstruya
bien y que el modelo cargue y funcione).

La subida a HF (AcademiaSD/YuE2-3B-NF4-for-LoRA-Training) se hace aparte con
huggingface-cli: esta carpeta ya está lista para subir.
"""
import os
import sys
import gc
import json
import time
import shutil
import random

import torch

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/yue2/..
sys.path.insert(0, os.path.join(STUDIO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(STUDIO_ROOT, "tools", "yue2"))

from safetensors import safe_open
from safetensors.torch import save_file, load_file

from bitsandbytes.nn import Linear4bit, Params4bit
from bitsandbytes.functional import QuantState
from accelerate import init_empty_weights

from project_layout import resolve_model_dir

# Los modelos viven en Models/ desde la migracion del layout.
# The models live in Models/ since the layout migration.
RAW_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_Raw", STUDIO_ROOT))
VAE_DIR = str(resolve_model_dir(None, "Models/YuE2-Vae", STUDIO_ROOT))
OUTPUT_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_NF4", STUDIO_ROOT))
REPO_ID = "AcademiaSD/YuE2-3B-NF4-for-LoRA-Training"

DTYPE = torch.bfloat16
TOP_MODEL_DIR = os.path.join(OUTPUT_DIR, "model")   # caché NF4 del backbone
sys.path.insert(0, RAW_DIR)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value):
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


def confirm_overwrite(path, label):
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


def is_quantized_name(full_name):
    """Espejo EXACTO de trainer is_quantized()."""
    return (".norm" not in full_name
            and not full_name.startswith(("llm2vae", "vae2llm", "time_embedder", "lm_head",
                                          "model.embed_tokens", "latent_pos_embed")))


# ══════════════════════════════════════════════════════════════════════════════
# CUANTIZACIÓN DEL BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

def quantize_to_nf4_(model):
    """Mismo recorrido que el trainer (load_model, precision nf4):
    named_modules completo + repliegue por rpartition."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and is_quantized_name(name):
            w = module.weight.data.float().contiguous()
            layer = Linear4bit(module.in_features, module.out_features,
                               bias=module.bias is not None,
                               quant_type="nf4", compute_dtype=DTYPE)
            layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")
            if module.bias is not None:
                layer.bias = torch.nn.Parameter(module.bias.data.clone(), requires_grad=False)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, layer)
            count += 1
            if count % 50 == 0:
                print(f"  {count} capas NF4...")


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE LA CACHÉ NF4 (formato studio/Qwen-Image 2.1)
# ══════════════════════════════════════════════════════════════════════════════

def extract_nf4_cache(model, output_dir):
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    index = {"quantized": {}, "unquantized": {}, "others": "others.safetensors"}
    saved_params = set()

    print()
    print("=" * 70)
    print("EXTRAYENDO CACHÉ NF4")
    print("=" * 70)

    for name, layer in model.named_modules():
        filename = name.replace(".", "__") + ".safetensors"
        filepath = os.path.join(weights_dir, filename)

        if isinstance(layer, Linear4bit):
            weight = layer.weight
            qs = weight.quant_state
            if not getattr(weight, "bnb_quantized", False) or qs is None:
                raise RuntimeError(f"{name}: Linear4bit sin QuantState válido.")

            qs_dict = qs.as_dict(packed=True)
            tensors = {"weight": weight.data.detach().cpu().contiguous()}
            if layer.bias is not None:
                tensors["bias"] = layer.bias.detach().to(DTYPE).cpu().contiguous()

            metadata = {
                "layer_name": name,
                "type": "Linear4bit",
                "quant_type": "nf4",
                "bnb_quantized": "true",
                "in_features": str(layer.in_features),
                "out_features": str(layer.out_features),
                "bias": str(layer.bias is not None),
            }
            for key, value in qs_dict.items():
                if torch.is_tensor(value):
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

        elif isinstance(layer, torch.nn.Linear):
            tensors = {"weight": layer.weight.detach().to(DTYPE).cpu().contiguous()}
            if layer.bias is not None:
                tensors["bias"] = layer.bias.detach().to(DTYPE).cpu().contiguous()
            save_file(tensors, filepath,
                      metadata={"layer_name": name, "type": "Linear", "quantized": "false"})
            index["unquantized"][name] = {
                "file": filename,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            }
            saved_params.update(f"{name}.{p}" for p in ("weight", "bias"))

    others = {
        name: tensor.detach().to(DTYPE).cpu().contiguous()
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if name not in saved_params
    }
    save_file(others, os.path.join(output_dir, "others.safetensors"))
    print(f"  Otros parámetros: {len(others)} tensores")

    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return len(index["quantized"]), len(index["unquantized"]), len(others)


# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE: cargar -> cuantizar -> extraer
# ══════════════════════════════════════════════════════════════════════════════

def deterministic_inputs(vocab_size):
    random.seed(7)
    ids = [151643, 151644, 151852, 151853] + [random.randrange(0, 33000) for _ in range(64)]
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def convert_model(model_dir):
    from modeling_yue2 import YuE2ForCausalLM

    print()
    print("=" * 70)
    print("BACKBONE YuE2-3B")
    print("=" * 70)
    print("Cargando YuE2ForCausalLM BF16...")

    model = YuE2ForCausalLM.from_pretrained(RAW_DIR, local_files_only=True, torch_dtype=DTYPE)
    print("Parámetros:", f"{sum(p.numel() for p in model.parameters()):,}")

    # Referencia BF16 ANTES de cuantizar: logits AR de una secuencia fija
    model.to("cuda")
    model.eval()
    tokens = deterministic_inputs(model.config.vocab_size)
    with torch.inference_mode():
        hidden, _ = model.model(input_ids=tokens,
                                position_ids=torch.arange(tokens.shape[-1], device="cuda")[None],
                                use_cache=False)
        ref_logits = model.lm_head(hidden[:, -8:, :]).float().cpu()
    del model
    free_memory()

    model = YuE2ForCausalLM.from_pretrained(RAW_DIR, local_files_only=True, torch_dtype=DTYPE)
    print("Cuantizando a NF4...")
    quantize_to_nf4_(model)

    print("Moviendo a CUDA (bnb materializa el QuantState)...")
    model.to("cuda")
    free_memory()
    print("VRAM:", f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    verified = sum(1 for layer in model.modules()
                   if isinstance(layer, Linear4bit)
                   and getattr(layer.weight, "bnb_quantized", False)
                   and layer.weight.quant_state is not None)
    print("Capas Linear4bit verificadas:", verified)
    if verified == 0:
        raise RuntimeError("Ninguna Linear4bit quedó con QuantState materializado.")

    quantized_count, unquantized_count, others_count = extract_nf4_cache(model, model_dir)

    metadata = {
        "format": "YuE2-NF4",
        "version": 1,
        "model_id": "m-a-p/YuE2-3B",
        "repo_output": REPO_ID,
        "dtype": "bfloat16",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "quantized_layers": quantized_count,
        "unquantized_layers": unquantized_count,
        "other_tensors": others_count,
        "skip_quant": ["llm2vae", "vae2llm", "time_embedder", "latent_pos_embed",
                       "lm_head", "model.embed_tokens", "*.norm"],
        "bitsandbytes_prequantized": True,
        "quant_state_serialization": "as_dict(packed=True)",
        "loader": "Params4bit.from_prequantized",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"Capas NF4: {quantized_count} | BF16: {unquantized_count}")

    del model
    free_memory()
    return ref_logits


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN: mismo cargador que el trainer
# ══════════════════════════════════════════════════════════════════════════════

def verify_cache(ref_logits):
    from modeling_yue2 import YuE2ForCausalLM

    print()
    print("=" * 70)
    print("VERIFICACIÓN DE RECONSTRUCCIÓN")
    print("=" * 70)

    index_path = os.path.join(TOP_MODEL_DIR, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        cfg = YuE2ForCausalLM.config_class.from_pretrained(TOP_MODEL_DIR)
        test = YuE2ForCausalLM(cfg)
    test.to_empty(device="cpu")

    def set_module(name, module):
        parent_name, _, child_name = name.rpartition(".")
        setattr(test.get_submodule(parent_name) if parent_name else test, child_name, module)

    weights_dir = os.path.join(TOP_MODEL_DIR, "weights")
    for name, info in index["quantized"].items():
        with safe_open(os.path.join(weights_dir, info["file"]), framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias = f.get_tensor("bias") if info["bias"] else None
            meta = f.metadata()
            qs_dict = {k[len("quant_state."):]: f.get_tensor(k)
                       for k in f.keys() if k.startswith("quant_state.")}
        for k, v in (meta or {}).items():
            if k.startswith("qs_"):
                qs_dict[k[3:]] = json.loads(v)      # campos no tensoriales del QuantState

        weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4",
                            quant_storage=torch.uint8)
        weight.quant_state = QuantState.from_dict(qs_dict, device="cpu")
        weight.bnb_quantized = True
        with torch.device("meta"):
            layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"],
                               quant_type="nf4", compute_dtype=DTYPE)
        layer.weight = weight
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        set_module(name, layer)

    for name, info in index["unquantized"].items():
        with torch.device("meta"):
            layer = torch.nn.Linear(info["in_features"], info["out_features"],
                                    bias=info["bias"], dtype=DTYPE)
        layer.load_state_dict(load_file(os.path.join(weights_dir, info["file"])), assign=True)
        layer.requires_grad_(False)
        set_module(name, layer)

    test.load_state_dict(load_file(os.path.join(TOP_MODEL_DIR, index["others"])),
                         strict=False, assign=True)
    missing = [n for n, t in list(test.named_parameters()) + list(test.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError("Caché NF4 incompleta: " + ", ".join(missing[:6]))

    test.to("cuda")
    test.eval()
    free_memory()

    tokens = deterministic_inputs(test.config.vocab_size)
    with torch.inference_mode():
        hidden, _ = test.model(input_ids=tokens,
                               position_ids=torch.arange(tokens.shape[-1], device="cuda")[None],
                               use_cache=False)
        nf4_logits = test.lm_head(hidden[:, -8:, :]).float().cpu()

    rel = ((nf4_logits - ref_logits).norm() / ref_logits.norm().clamp_min(1e-6)).item()
    print(f"  Error relativo logits AR (NF4 reconstruido vs BF16): {rel:.4%}")
    print("  (NF4 es con pérdida: el error pequeño y acotado es el esperado)")
    if rel > 0.15:
        print("[AVISO] error > 15%: algo raro en la cuantización o carga; revisar.")

    del test
    free_memory()
    return rel


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()

    print()
    print("=" * 70)
    print(" YuE2-3B NF4 CONVERTER")
    print("=" * 70)
    print("Modelo:", RAW_DIR)
    print("Salida :", OUTPUT_DIR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible: la cuantización NF4 requiere CUDA.")
    print("GPU:", torch.cuda.get_device_name(0))

    if confirm_overwrite(os.path.join(TOP_MODEL_DIR, "index.json"), "la caché NF4 del backbone"):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(TOP_MODEL_DIR, exist_ok=True)

        # Cached config para el cargador
        from modeling_yue2 import YuE2ForCausalLM
        cfg = YuE2ForCausalLM.config_class.from_pretrained(RAW_DIR)
        with open(os.path.join(TOP_MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        ref_logits = convert_model(TOP_MODEL_DIR)
        verify_cache(ref_logits)

        # VAE tal cual + ficheros no tensoriales del Raw
        shutil.copytree(VAE_DIR, os.path.join(OUTPUT_DIR, "vae"),
                        ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        for fname in os.listdir(RAW_DIR):
            path = os.path.join(RAW_DIR, fname)
            if os.path.isfile(path) and fname != "model.safetensors" and fname != "weights_manifest.json":
                shutil.copy2(path, os.path.join(OUTPUT_DIR, fname))

        with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join([
                "# YuE2-3B NF4 (AcademiaSD)",
                "",
                f"Convertido de `m-a-p/YuE2-3B` a NF4 (bitsandbytes, cuantizado en GPU). Repo destino: `{REPO_ID}`.",
                "",
                "- `model/` — caché NF4 del backbone: `index.json` + `weights/` (peso uint8 + QuantState `as_dict(packed=True)`) + `others.safetensors`.",
                "  NF4 solo en atención+MLP de las ramas AR y NAR de los 28 bloques; `llm2vae/vae2llm/time_embedder`, normas, `embed_tokens`, `lm_head` y `latent_pos_embed` quedan en BF16.",
                "- `vae/` — YuE2-VAE sin cambios.",
                "- `modeling_yue2.py` + config/tokenizador — para carga offline sin `trust_remote_code`.",
                "",
                "Consume el LoRAlab Trainer Studio (`2_train_lora_yue2.py`, fase 5 del plan YuE2).",
            ]) + "\n")

    elapsed = time.time() - start
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
    print("No borres el modelo original todavía: primero prueba la reconstrucción")
    print("(el propio script ya la ejecuta al final) y después el trainer NF4.")
    print("=" * 70)


if __name__ == "__main__":
    main()
