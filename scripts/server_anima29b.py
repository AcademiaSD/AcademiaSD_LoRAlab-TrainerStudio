# -*- coding: utf-8 -*-
"""
server_anima29b.py — Trainer Anima 2.9B: el servidor de Anima con la variante anima29b
Anima 2.9B trainer: the Anima server with the anima29b variant

Anima-2.9B es Anima-Base con 40 bloques en vez de 28 y su propio adaptador LLM. Comparte todo el
código con Anima (server_anima.py, 0/1/2_*_anima.py, trainer_ui_anima.html); lo que cambia (modelo,
settings, caché, salidas) está en anima_models.py. Los scripts que lanza heredan la variable.

Anima-2.9B is Anima-Base with 40 blocks instead of 28 and its own LLM adapter. It shares all the
code with Anima (server_anima.py, 0/1/2_*_anima.py, trainer_ui_anima.html); what changes (model,
settings, cache, outputs) lives in anima_models.py. The scripts it launches inherit the variable.
"""
import os
import runpy
from pathlib import Path

os.environ["LORALAB_ANIMA_VARIANT"] = "anima29b"
runpy.run_path(str(Path(__file__).resolve().with_name("server_anima.py")), run_name="__main__")
