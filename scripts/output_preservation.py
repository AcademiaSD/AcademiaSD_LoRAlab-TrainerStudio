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


def dop_name(sample):
    """Nombre de muestra con que se guarda el prompt de clase: sus ficheros son los de una muestra más
    (<muestra>_dop_embed.pt y, según el modelo, máscara, pooled...), en el formato de ese trainer."""
    return "{}_dop".format(sample)


def raw_caption(cfg, sample):
    """El .txt de la muestra tal cual (sin trigger), o cadena vacía."""
    path = os.path.join(cfg.get("dataset_path") or "./dataset", "{}.txt".format(sample))
    try:
        from text_io import read_text_file
        return read_text_file(path).strip()
    except Exception:
        return ""


def embed_name(sample):
    return "{}_embed.pt".format(dop_name(sample))


def prepare(cfg, cache_dir, samples, trigger, precache_script, log=print, device="cuda", edit=False, caption=None):
    """
    Deja en la caché el embedding del prompt de clase de cada muestra. Solo codifica los que faltan o
    cuyo texto ha cambiado (otra palabra, otro caption). Devuelve {muestra: prompt} o None si no hay que usarla
    o no se pudo preparar (el entrenamiento sigue sin ella). edit=True (LoRA de edición): no se usa.
    caption(muestra): el caption con el trigger tal como lo codifica el pre-caché de ese trainer (Fibo lo mete
    dentro del JSON); por defecto, el del .txt con el trigger delante.
    """
    on, word, _, _ = settings(cfg)
    if not on:
        if cfg.get("output_preservation"):
            log("[!] Output preservation is on but the class word is empty: it is not used.")
        return None
    if edit:
        log("[!] Output preservation is for text-to-image LoRAs: it is not used with an edit LoRA.")
        return None
    import preview_prompts
    prompts = {s: class_prompt(caption(s) if caption else preview_prompts.dataset_caption(cfg, s, trigger), trigger, word)
               for s in samples}
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


def dop_job(argv):
    """argv de --dop-prompts <carpeta de caché> <cuda|cpu> <json> -> (carpeta, device, [(muestra, prompt)]) o None."""
    if len(argv) != 5 or argv[1] != "--dop-prompts":
        return None
    with open(argv[4], encoding="utf-8") as f:
        return argv[2], argv[3], [(str(s), str(p)) for s, p in json.load(f)]


def run_encoder_job(argv, load_text_pipe, encode_text=None, save=None):
    """
    Para el script de pre-caché: --dop-prompts <carpeta de caché> <cuda|cpu> <json [[muestra, prompt], ...]>.
    Cada prompt de clase se guarda como la muestra <muestra>_dop, con la misma función que guarda el caption
    de una muestra: save(handle, nombre, prompt), con handle = load_text_pipe(device). Si solo se da
    encode_text(handle, prompt, device), se guarda su tensor como <nombre>_embed.pt.
    Devuelve True si era este modo (y ya ha terminado).
    """
    job = dop_job(argv)
    if job is None:
        return False
    import torch
    cache_dir, device, items = job
    handle = load_text_pipe(device)
    for sample, prompt in items:
        name = dop_name(sample)
        if save is not None:
            save(handle, name, prompt)
        else:
            path = os.path.join(cache_dir, "{}_embed.pt".format(name))
            torch.save(encode_text(handle, prompt, device), path + ".tmp")
            os.replace(path + ".tmp", path)
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


# ── Lo que usa cada trainer ──
class Preservation:
    """
    Todo lo común de la protección en un trainer. Uso:

        pres = output_preservation.Preservation(cfg, CACHE_DIR, OUTPUT_DIR, TRIGGER_WORD, PRECACHE_SCRIPT)
            # antes de cargar el modelo: codifica los prompts de clase con la GPU libre
        pres.begin(start_step)              # tras reanudar (un entrenamiento nuevo borra los mapas viejos)
        cache_data[n]["dop"] = ...          # el condicionamiento de dop_name(n), cargado como el de n
        ...
        if pres.due(step):                  # después del backward de la loss normal
            cls = ...                       # el condicionamiento del lote, como el normal pero con "dop"
            pres.step(model, lambda: forward(cls), names, scaler, GRAD_ACCUM_STEPS)
        pres.report(step)

    forward() hace la misma pasada que la loss normal (mismo latente con ruido y sigma) y devuelve la
    predicción. to_map(t) pasa esa predicción a [B, C, H, W] para el mapa de calor (por defecto, tal cual si
    ya tiene 4 dimensiones; con 5, la media del eje de tiempo).
    """

    def __init__(self, cfg, cache_dir, output_dir, trigger, precache_script, log=print, edit=False,
                 samples=None, to_map=None, device="cuda", caption=None):
        self.log, self.output_dir, self.to_map = log, output_dir, to_map
        self.on, self.word, self.weight, self.every = settings(cfg)
        if samples is None:
            import cache_names
            try:
                samples = sorted(f[:-len("_latent.pt")] for f in os.listdir(cache_dir)
                                 if f.endswith("_latent.pt") and not cache_names.is_reserved(f))
            except OSError:
                samples = []
        self.prompts = prepare(cfg, cache_dir, samples, trigger, precache_script, log=log, device=device, edit=edit,
                               caption=caption)
        self.active = bool(self.prompts)
        self.heat = HeatMaps(cfg, output_dir, log) if self.active else None
        self.sum, self.n = 0.0, 0

    def begin(self, start_step):
        if self.heat and self.heat.on and int(start_step) == 0:
            clear_maps(self.output_dir)

    @staticmethod
    def name(sample):
        return dop_name(sample)

    def due(self, step):
        return self.active and step % self.every == 0

    def step(self, model, forward, names, scaler=None, accum=1, to_map=None):
        """Pasada sin LoRA (referencia) y con LoRA, con el prompt de clase; backward de su loss.
        to_map: como el del constructor, para este paso (p. ej. desempaquetar tokens con el tamaño del lote).
        Sin VRAM para la pasada extra, ese paso se salta (el paso normal ya está hecho) y se avisa."""
        import torch
        import torch.nn.functional as F
        prior = kept = loss = scaled = None
        try:
            with torch.no_grad(), lora_off(model):
                prior = forward().float()
            kept = forward().float()
            loss = F.mse_loss(kept, prior)
            scaled = loss * self.weight / max(1, accum)
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
        except torch.OutOfMemoryError:
            prior = kept = loss = scaled = None
            torch.cuda.empty_cache()
            self.oom = getattr(self, "oom", 0) + 1
            if self.oom in (1, 10, 100):
                self.log("\n[!] Output preservation: out of VRAM in its extra pass, skipped this step ({} so far). "
                         "Use Every N steps 2 or more, or more block swap.".format(self.oom))
            return
        self.sum, self.n = self.sum + loss.item(), self.n + 1
        mapper = to_map or self.to_map
        if self.heat and self.heat.on:
            try:
                a, b = (mapper(kept), mapper(prior)) if mapper else (kept, prior)
                if a.dim() == 5:
                    a, b = a.mean(2), b.mean(2)
                if a.dim() == 4:
                    self.heat.add(names, a, b)
            except Exception:
                pass
        del prior, kept, loss, scaled

    def report(self, step):
        every = int(os.environ.get("LORALAB_PRESERVATION_REPORT", "100") or 100)
        if self.n and step % max(1, every) == 0:
            try:
                from i18n import t
                text = t("Step {n}: preservation loss {loss:.5f} (average of the last {k})",
                         n=step, loss=self.sum / self.n, k=self.n)
            except Exception:
                text = "Step {}: preservation loss {:.5f}".format(step, self.sum / self.n)
            print("\n[PRESERVATION] " + text, flush=True)
            self.sum, self.n = 0.0, 0
            if self.heat:
                self.heat.save(step)
