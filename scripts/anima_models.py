# -*- coding: utf-8 -*-
"""
anima_models.py — Los dos trainers de Anima: Anima (Base, 28 bloques) y Anima 2.9B (40 bloques)
The two Anima trainers: Anima (Base, 28 blocks) and Anima 2.9B (40 blocks)

Anima-2.9B (Gazingstars123) es Anima-Base con 12 bloques más, mismo ancho, mismo VAE y mismo
Qwen3-0.6B, pero con su propio adaptador LLM (text_conditioner): la caché de texto de uno no sirve
para el otro, y un LoRA de uno tampoco (los bloques nuevos van intercalados, 2, 5, 8..., y los
originales cambian de índice). Por eso es un trainer aparte, con sus settings, su caché y sus
salidas, pero con el mismo código: server_anima29b.py pone LORALAB_ANIMA_VARIANT=anima29b y carga
server_anima.py, y los scripts 0/1/2 (que heredan esa variable) leen aquí su modelo y sus nombres.

Anima-2.9B (Gazingstars123) is Anima-Base with 12 more blocks, same width, same VAE and same
Qwen3-0.6B, but with its own LLM adapter (text_conditioner): one model's text cache does not work
for the other, nor does a LoRA (the new blocks are interleaved, 2, 5, 8..., and the original blocks
change index). So it is a separate trainer with its own settings, cache and outputs, but the same
code: server_anima29b.py sets LORALAB_ANIMA_VARIANT=anima29b and loads server_anima.py, and scripts
0/1/2 (which inherit that variable) read their model and names here.
"""
import os

ENV = "LORALAB_ANIMA_VARIANT"

VARIANTS = {
    "anima": {
        "name": "Anima",
        "repo": "circlestone-labs/Anima-Base-v1.0-Diffusers",
        "model_dir": "Models/Anima-Base",
        "label": "Anima-Base-v1.0",
        "blocks": 28,
        "size_gb": 5.6,
        "file_prefix": "Anima",           # Anima_LoRA_step_N / Anima_FINAL_LoRA
        "lora_name": "anima_lora",
        "defaults": {"pre_cache": {}, "train": {}},
    },
    # Versión diffusers de Gazingstars123/Anima-2.9B (preview v1), la que recomiendan en OneTrainer.
    "anima29b": {
        "name": "Anima 2.9B",
        "repo": "v2ray/Anima-2.9B-preview-v1-diffusers",
        "model_dir": "Models/Anima-2.9B",
        "label": "Anima-2.9B-preview-v1",
        "blocks": 40,
        "size_gb": 7.3,
        "file_prefix": "Anima29B",
        "lora_name": "anima29b_lora",
        # Valores por defecto propios, los de un entrenamiento de personaje probado (NF4, < 6 GB de VRAM).
        # Los scripts los usan cuando falta la clave y el servidor rellena con ellos lo que el json no
        # tiene, así la GUI los muestra en un proyecto nuevo. LoRA+ queda desmarcado, como en Anima.
        "defaults": {
            "pre_cache": {"target_area": 512 * 512, "max_side": 1024, "multiple": 32},
            "train": {"total_steps": 800, "lr": 2e-5, "lr_scheduler": "cosine", "warmup_steps": 50, "lr_cycles": 3,
                      "lora_rank": 16, "lora_alpha": 16, "use_rslora": False, "use_loraplus": False,
                      "use_lokr": False, "use_dora": False, "lora_targets": "blocks", "precision": "nf4",
                      "batch_size": 1, "grad_accum_steps": 4, "save_every": 50, "seed": 42,
                      "preview_every": 50, "preview_caption_mode": "random", "preview_steps": 30, "preview_cfg": 4,
                      "preview_size": 768, "preview_aspect_ratio": "16:9", "preview_width": 1024, "preview_height": 576},
        },
    },
}


def current():
    """(id, datos) de la variante de este proceso; sin la variable es Anima (Base), como siempre."""
    vid = os.environ.get(ENV, "anima").strip().lower()
    if vid not in VARIANTS:
        vid = "anima"
    return vid, {**VARIANTS[vid],
                 "id": vid,
                 "cache_prefix": f"cached_data_{vid}",
                 "output_prefix": f"{vid}_lora_output",
                 "precache_settings": f"pre_cache_settings_{vid}.json",
                 "train_settings": f"train_settings_{vid}.json",
                 "caption_settings": f"caption_settings_{vid}.json"}


VARIANT_ID, VARIANT = current()
