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
DEFAULTS = {"output_preservation": False, "preservation_class": "", "preservation_strength": 0.5,
            "preservation_every": 1}


def settings(cfg):
    """(activa, palabra de clase, fuerza, cada N pasos). Sin palabra de clase no se usa."""
    word = str(cfg.get("preservation_class") or "").strip()
    on = bool(cfg.get("output_preservation")) and bool(word)
    try:
        weight = max(0.0, min(10.0, float(cfg.get("preservation_strength", 0.5))))
    except (TypeError, ValueError):
        weight = 0.5
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


# ── Mapa de calor ──
# Dónde cambia el LoRA la salida del modelo cuando el prompt no lleva el trigger. Cada paso de protección
# suma el error por posición del latente (media de canales) a la muestra que se ha usado; cada 100 pasos se
# guarda el de la muestra con más pasos acumulados, encima de su imagen del dataset. El color es relativo
# (rojo = lo que más cambia en ese mapa); el número de abajo es la media absoluta.
MAP_DIR = "preservation_maps"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class HeatMaps:
    def __init__(self, cfg, output_dir, log=print):
        self.on = bool(cfg.get("preservation_map"))
        self.cfg = cfg
        self.folder = os.path.join(str(output_dir), MAP_DIR)
        self.log = log
        self.acc = {}

    def add(self, names, pred, prior):
        """pred, prior: [B, C, H, W] (u otra forma con canales en la dimensión 1)."""
        if not self.on:
            return
        import torch
        with torch.no_grad():
            diff = (pred.detach().float() - prior.float()).pow(2).mean(1)
            while diff.dim() > 3:
                diff = diff.mean(1)
            for name, d in zip(names, diff.cpu()):
                total, n = self.acc.get(name, (None, 0))
                if total is not None and total.shape != d.shape:
                    continue
                self.acc[name] = (d if total is None else total + d, n + 1)

    def save(self, step):
        if not self.on or not self.acc:
            return None
        name, (total, n) = max(self.acc.items(), key=lambda kv: kv[1][1])
        self.acc = {}
        try:
            path = self._draw(step, name, (total / n).numpy(), n)
        except Exception as e:
            self.log("[!] Preservation heatmap: {}".format(e))
            return None
        return path

    def _draw(self, step, name, m, n):
        import numpy as np
        from PIL import Image, ImageDraw
        h, w = m.shape
        folder = self.cfg.get("dataset_path") or "./dataset"
        src = next((os.path.join(folder, name + e) for e in IMAGE_EXTS if os.path.isfile(os.path.join(folder, name + e))), None)
        side = 640
        W, H = (side, max(1, round(side * h / w))) if w >= h else (max(1, round(side * w / h)), side)
        if src:
            img = Image.open(src).convert("RGB")
            ratio = max(W / img.width, H / img.height)
            img = img.resize((max(W, round(img.width * ratio)), max(H, round(img.height * ratio))), Image.LANCZOS)
            left, top = (img.width - W) // 2, (img.height - H) // 2
            img = img.crop((left, top, left + W, top + H))
        else:
            img = Image.new("RGB", (W, H), (40, 40, 40))
        from PIL import ImageFilter
        # Mapa suave: se recortan los picos sueltos (percentil 99), se sube a la imagen con bicúbica y se
        # difumina, para que se vean zonas y no los bloques del latente.
        v = m / (np.percentile(m, 99) + 1e-12)
        v = np.clip(v, 0, 1) ** 0.8
        g = Image.fromarray((v * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        # Difuminado amplio y normalizado otra vez: cada zona caliente se ensancha sin perder intensidad, y el
        # ruido suelto del fondo se reparte en vez de crecer.
        g = g.filter(ImageFilter.GaussianBlur(radius=max(3, W // 40)))
        v = np.asarray(g, dtype=np.float32)
        v = np.clip(v / (v.max() + 1e-6), 0, 1) ** 0.75
        color = _inferno(v)
        # La foto, algo apagada y desaturada, para que el calor resalte; el color entra con el valor.
        base = np.asarray(img, dtype=np.float32) / 255
        base = (base * 0.55 + base.mean(-1, keepdims=True) * 0.45) * 0.62
        alpha = (0.18 + 0.72 * v ** 0.9)[..., None]
        out = base * (1 - alpha) + color * alpha
        out = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))
        # Pie: barra con el degradado (menos → más cambio) y los datos.
        foot = 44
        canvas = Image.new("RGB", (W, H + foot), (11, 18, 34))
        canvas.paste(out, (0, 0))
        bar_w = min(220, W // 3)
        bar = (_inferno(np.tile(np.linspace(0, 1, bar_w, dtype=np.float32), (8, 1))) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(bar), (W - bar_w - 10, H + 12))
        draw = ImageDraw.Draw(canvas)
        draw.text((W - bar_w - 10, H + 24), "less", fill=(148, 163, 184))
        draw.text((W - 40, H + 24), "more", fill=(148, 163, 184))
        draw.text((10, H + 8), "Step {}  ·  {}".format(step, name), fill=(226, 232, 240))
        draw.text((10, H + 24), "{} passes  ·  mean change {:.5f}".format(n, float(m.mean())), fill=(148, 163, 184))
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, "map_step_{:06d}.png".format(int(step)))
        tmp = path + ".tmp.png"
        canvas.save(tmp)
        os.replace(tmp, path)
        return path


def _inferno(v):
    """Escala de color tipo inferno (morado oscuro → magenta → naranja → amarillo claro), interpolada."""
    import numpy as np
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                      [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32) / 255
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(np.int32), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def clear_maps(output_dir):
    """Un entrenamiento nuevo (paso 0) empieza sin los mapas del anterior."""
    import shutil
    shutil.rmtree(os.path.join(str(output_dir), MAP_DIR), ignore_errors=True)


def register(app, get_output_dir):
    """Lista y sirve los mapas de calor del proyecto (la GUI los enseña encima de la galería)."""
    from pathlib import Path
    from flask import jsonify, send_from_directory, abort

    @app.route("/api/preservation-maps")
    def preservation_maps():
        folder = Path(get_output_dir()) / MAP_DIR
        items = []
        if folder.is_dir():
            for f in folder.glob("map_step_*.png"):
                st = f.stat()
                if st.st_size > 0:
                    items.append({"name": f.name, "step": int(f.stem.rsplit("_", 1)[1]), "version": int(st.st_mtime * 1000)})
        items.sort(key=lambda i: i["step"], reverse=True)
        return jsonify({"maps": items[:24]})

    @app.route("/api/preservation-map/<name>")
    def preservation_map(name):
        folder = Path(get_output_dir()) / MAP_DIR
        if not (name.startswith("map_step_") and name.endswith(".png")) or not (folder / name).is_file():
            abort(404)
        return send_from_directory(str(folder), name, max_age=0)
