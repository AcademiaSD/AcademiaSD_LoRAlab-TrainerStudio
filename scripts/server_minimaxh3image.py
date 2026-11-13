# -*- coding: utf-8 -*-
"""
server_minimaxh3image.py - MiniMax-H3-Image trainer.

MiniMax-H3-Image shares its text encoder, its latent space and therefore its
whole pre-cache with MiniMax-H3, so this is server_minimaxh3.py opened in image
mode: the same GUI with the model pinned to MiniMax-H3-Image, its own settings
files (train/pre_cache_settings_minimaxh3image.json) and its own output folders
(Outputs/minimaxh3image_lora_output_<project>). The trainer it launches is
2_train_lora_minimaxh3image.py.

MiniMax-H3-Image comparte text encoder, espacio latente y pre-cache con
MiniMax-H3: esto es server_minimaxh3.py en modo imagen, con ajustes y carpetas
de salida propios.
"""
import os
from pathlib import Path

_SETTINGS = Path(__file__).resolve().parent.parent / "settings"
os.environ["LORALAB_H3_MODEL"] = "image"
# The pre-cache and trainer scripts read these, so they follow this card's settings.
# Los scripts de pre-cache y entrenamiento leen estas variables.
os.environ.setdefault("LORALAB_TRAIN_CONFIG", str(_SETTINGS / "train_settings_minimaxh3image.json"))
os.environ.setdefault("LORALAB_PRECACHE_CONFIG", str(_SETTINGS / "pre_cache_settings_minimaxh3image.json"))

import server_minimaxh3  # noqa: E402

app = server_minimaxh3.app

if __name__ == "__main__":
    server_minimaxh3.main()
