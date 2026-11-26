# -*- coding: utf-8 -*-
"""
output_preservation.py — Protección de lo que el modelo ya sabe (output preservation).
Output preservation: keeps the LoRA from changing what the model already knows.

Es la idea de la Differential Output Preservation (DOP) de ai-toolkit (ostris), reimplementada aquí. En cada
paso (o cada N pasos), además de la loss normal, se toma el caption de la imagen con el trigger cambiado por
una palabra de clase ("woman", "man", "dog", "style"...) y se compara la salida del modelo con y sin LoRA
sobre el mismo latente con ruido: la diferencia se castiga. El LoRA aprende el trigger, pero "woman" sigue
siendo la mujer del modelo base, así que no contamina a otras personas, se quema menos y combina mejor con
otros LoRAs.

Los embeddings de esos prompts los codifica el script de pre-caché del trainer (--dop-prompts) antes de
cargar el modelo, y se guardan en la caché como <muestra>_dop_embed.pt, con su texto en _dop_prompts.json
para recodificar solo lo que cambie.
"""
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager

INDEX = "_dop_prompts.json"
DEFAULTS = {"output_preservation": False, "preservation_class": "", "preservation_strength": 1.0,
            "preservation_every": 1}


def settings(cfg):
    """(activa, palabra de clase, fuerza, cada N pasos). Sin palabra de clase no se usa."""
    word = str(cfg.get("preservation_class") or "").strip()
    on = bool(cfg.get("output_preservation")) and bool(word)
    try:
        weight = max(0.0, min(10.0, float(cfg.get("preservation_strength", 1.0))))
    except (TypeError, ValueError):
        weight = 1.0
    try:
        every = max(1, min(100, int(cfg.get("preservation_every", 1))))
    except (TypeError, ValueError):
        every = 1
    return on and weight > 0, word, weight, every


def class_prompt(caption, trigger, word):
    """El caption con el trigger cambiado por la palabra de clase (sin trigger: la palabra delante)."""
    trigger = (trigger or "").strip()
    caption = (caption or "").strip()
    if trigger and re.search(re.escape(trigger), caption, re.I):
        return re.sub(re.escape(trigger), word, caption, flags=re.I)
    return "{}, {}".format(word, caption).strip(", ")


def embed_name(sample):
    return "{}_dop_embed.pt".format(sample)


def prepare(cfg, cache_dir, samples, trigger, precache_script, log=print, device="cuda"):
    """
    Deja en la caché el embedding del prompt de clase de cada muestra. Solo codifica los que faltan o
    cuyo texto ha cambiado (otra palabra, otro caption). Devuelve {muestra: prompt} o None si no hay que usarla
    o no se pudo preparar (el entrenamiento sigue sin ella).
    """
    on, word, _, _ = settings(cfg)
    if not on:
        if cfg.get("output_preservation"):
            log("[!] Output preservation is on but the class word is empty: it is not used.")
        return None
    import preview_prompts
    prompts = {s: class_prompt(preview_prompts.dataset_caption(cfg, s, trigger), trigger, word) for s in samples}
    index_path = os.path.join(cache_dir, INDEX)
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = {}
    todo = [[s, p] for s, p in prompts.items()
            if index.get(s) != p or not os.path.isfile(os.path.join(cache_dir, embed_name(s)))]
    if todo:
        log("[PRESERVATION] Encoding {} class prompts (\"{}\")...".format(len(todo), word))
        job = os.path.join(cache_dir, "_dop_job.json")
        with open(job, "w", encoding="utf-8") as f:
            json.dump(todo, f, ensure_ascii=False)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-u", precache_script, "--dop-prompts", cache_dir, device, job],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        try:
            os.remove(job)
        except OSError:
            pass
        missing = [s for s, _ in todo if not os.path.isfile(os.path.join(cache_dir, embed_name(s)))]
        if result.returncode != 0 or missing:
            log(result.stdout[-2000:])
            log("[!] Output preservation: the class prompts could not be encoded; training without it.")
            return None
        index.update({s: p for s, p in todo})
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
    example = next(iter(prompts.values()), "")
    log("[PRESERVATION] On: \"{}\" keeps the base model's output (e.g. \"{}\").".format(word, example[:90]))
    return prompts


@contextmanager
def lora_off(model):
    """El modelo sin el LoRA (PEFT: disable_adapter, también DoRA; LoKr y LoRAs propios: fuerza 0)."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        import xy_grid
        with xy_grid.strength(model, 0.0):
            yield


def run_encoder_job(argv, load_text_pipe, encode_text):
    """
    Para el script de pre-caché: --dop-prompts <carpeta de caché> <cuda|cpu> <json [[muestra, prompt], ...]>.
    Devuelve True si era este modo (y ya ha terminado).
    """
    if len(argv) != 5 or argv[1] != "--dop-prompts":
        return False
    import torch
    cache_dir, device, job = argv[2], argv[3], argv[4]
    with open(job, encoding="utf-8") as f:
        items = json.load(f)
    pipe = load_text_pipe(device)
    for sample, prompt in items:
        embed = encode_text(pipe, prompt, device)
        tmp = os.path.join(cache_dir, embed_name(sample) + ".tmp")
        torch.save(embed, tmp)
        os.replace(tmp, os.path.join(cache_dir, embed_name(sample)))
    print("[PRESERVATION] {} class prompts encoded.".format(len(items)), flush=True)
    return True
