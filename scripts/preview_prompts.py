# -*- coding: utf-8 -*-
"""
preview_prompts.py — Hasta 4 prompts propios para las previews (Custom Prompts).
Up to 4 custom prompts for the previews.

Ajustes (claves planas para que Save JSON y los ajustes en caliente los comparen como cualquier otro):
    preview_custom_count     0-4. 0 = las previews usan el Caption Mode (First / Random / Rotate 4).
    preview_custom_prompt    prompt 1 (la clave de siempre)
    preview_custom_prompt_2  prompt 2 ... preview_custom_prompt_4
    preview_custom_random    True = cada ronda hace una sola preview, con uno de los prompts al azar.
Un json antiguo con preview_caption_mode = "custom" cuenta como 1 prompt.

Caché: el prompt i (0-3) se guarda en _custom_prompt.txt / _custom_embed.pt para el primero y con
_2, _3, _4 antes de la extensión para los demás (_custom_embed_2.pt...). Los nombres con más de un
fichero por prompt (p. ej. _custom_pooled.pt) siguen la misma regla con slot_name().

Codificación: el Pre-Cache los codifica en GPU con el resto de textos. Al guardar los ajustes, el
servidor codifica los que han cambiado en un proceso aparte (start_encoding): en GPU si no hay nada
en marcha y en CPU si se está entrenando, y el trainer relee cada embedding cuando cambia en disco.
Ese proceso es el Pre-Cache de cada modelo con:
    --prompts-only <carpeta de caché> <cuda|cpu> <json [[slot, prompt con trigger], ...]>
"""
import json
import os
import random
import subprocess
import sys
import threading

MAX = 4
PROMPT_KEYS = ["preview_custom_prompt"] + [f"preview_custom_prompt_{i}" for i in range(2, MAX + 1)]
DEFAULTS = {"preview_custom_count": 0, "preview_custom_random": False,
            "preview_custom_image_per_prompt": False,
            **{k: "" for k in PROMPT_KEYS}}


# --------------------------------------------------------------------------- ajustes
def count(cfg):
    """Cuántos prompts propios usar (0-4). Un json antiguo con Caption Mode = custom cuenta como 1."""
    raw = cfg.get("preview_custom_count")
    if raw is None:
        return 1 if str(cfg.get("preview_caption_mode", "")).strip().lower() == "custom" else 0
    try:
        return max(0, min(MAX, int(raw)))
    except (TypeError, ValueError):
        return 0


def prompts(cfg):
    """[(slot, prompt)] de los prompts activos y no vacíos, sin el trigger."""
    out = []
    for slot, key in enumerate(PROMPT_KEYS[:count(cfg)]):
        text = str(cfg.get(key) or "").strip()
        if text:
            out.append((slot, text))
    return out


def random_one(cfg):
    return bool(cfg.get("preview_custom_random", False))


def per_prompt(cfg):
    """
    Edición con varias imágenes de referencia: True = cada prompt edita la suya (prompt 1 -> imagen 1,
    prompt 2 -> imagen 2...; si hay menos imágenes que prompts, se vuelve a empezar). False = todas son
    <image1>, <image2>... de una misma edición, para todos los prompts.
    """
    return bool(cfg.get("preview_custom_image_per_prompt", False))


def slot_sources(sources, slot, one_per_prompt):
    """Las imágenes de referencia que ve el prompt de ese slot."""
    if not one_per_prompt or not sources:
        return list(sources)
    return [sources[slot % len(sources)]]


def ctrl_file(slot, index, one_per_prompt):
    """
    Fichero del latente de referencia: compartidas, _custom_ctrl.pt, _custom_ctrl_2.pt... (<image1>,
    <image2>...); una por prompt, la del prompt en su propio nombre (_custom_ctrl.pt, _custom2_ctrl.pt...).
    """
    if one_per_prompt:
        return f"{prefix(slot)}_ctrl.pt"
    return "_custom_ctrl.pt" if index == 0 else f"_custom_ctrl_{index + 1}.pt"


def ref_plan(sources, slots, one_per_prompt):
    """{slot: [(imagen, fichero del latente)]}: las referencias que ve cada prompt y dónde van."""
    return {slot: [(src, ctrl_file(slot, i, one_per_prompt))
                   for i, src in enumerate(slot_sources(sources, slot, one_per_prompt))]
            for slot in slots}


def images_key(images, one_per_prompt=False):
    """Texto de _custom_image.txt: las rutas y, si cada prompt usa la suya, una marca."""
    return "\n".join(images) + ("\n[one image per prompt]" if one_per_prompt else "")


def with_trigger(prompt, trigger):
    trigger = (trigger or "").strip()
    if trigger and prompt and trigger.lower() not in prompt.lower():
        return f"{trigger}, {prompt}"
    return prompt


def wanted(cfg, trigger=None, add_trigger=None):
    """
    [(slot, prompt con trigger)] tal como se codifican. add_trigger(prompt) = la regla propia de un
    modelo (Fibo 2: en un caption JSON el trigger va dentro); sin ella, «trigger, prompt».
    """
    trigger = cfg.get("trigger_word", "") if trigger is None else trigger
    if add_trigger is not None:
        return [(slot, add_trigger(p)) for slot, p in prompts(cfg)]
    return [(slot, with_trigger(p, trigger)) for slot, p in prompts(cfg)]


def settings(cfg):
    """Copia de los ajustes que usan estas funciones, para guardarla en el trainer."""
    keys = list(DEFAULTS) + ["preview_caption_mode", "trigger_word"]
    return {k: cfg[k] for k in keys if k in cfg}


def live_update(current, raw, changes):
    """
    Ajustes en caliente del trainer: copia en current (de settings()) los valores nuevos de raw (el
    json releído) y añade a changes una línea por cada uno que cambia. Devuelve True si cambió algo.
    trigger_word no se toca: el trainer lo fija al arrancar.
    """
    changed = False
    for key in list(DEFAULTS) + ["preview_caption_mode"]:
        if key not in raw or raw[key] == current.get(key):
            continue
        shown = lambda v: (str(v)[:40] + "...") if len(str(v)) > 40 else v
        changes.append(f"{key}: {shown(current.get(key, ''))} -> {shown(raw[key])}")
        current[key] = raw[key]
        changed = True
    return changed


# --------------------------------------------------------------------------- caché
def slot_name(name, slot):
    """'_custom_embed.pt', 2 -> '_custom_embed_3.pt'. El slot 0 conserva el nombre de siempre."""
    if slot == 0:
        return name
    base, ext = os.path.splitext(name)
    return f"{base}_{slot + 1}{ext}"


def encoded_text(cache_dir, slot):
    path = os.path.join(cache_dir, slot_name("_custom_prompt.txt", slot))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def pending(cache_dir, cfg, trigger=None, images=None, add_trigger=None):
    """
    [(slot, prompt)] que faltan en la caché o cuyo texto codificado ya no es el de los ajustes. En
    edición (images = lista de referencias) cambiar las imágenes obliga a recodificar todos: el text
    encoder también las ve.
    """
    items = wanted(cfg, trigger, add_trigger)
    if images is not None and items and encoded_images(cache_dir) != images_key(images, per_prompt(cfg)):
        return items
    return [(slot, p) for slot, p in items if encoded_text(cache_dir, slot) != p]


def prefix(slot):
    """
    Modelos de edición: cada prompt usa su propio nombre de muestra (_custom, _custom2... _custom4), así
    su función de codificar y la de cargar sirven igual. Las imágenes de referencia (_custom_ctrl*.pt y
    _custom_image.txt) se codifican una vez y las comparten los cuatro prompts.
    """
    return "_custom" if slot == 0 else f"_custom{slot + 1}"


def remove_unused_prefixed(cache_dir, cfg, suffixes=("embed", "mask", "imgmask")):
    """Modelos de edición: borra los ficheros <prefix>_<sufijo>.pt de los prompts que ya no se usan."""
    used = {slot for slot, _ in prompts(cfg)}
    for slot in range(MAX):
        if slot in used:
            continue
        for suffix in suffixes:
            path = os.path.join(cache_dir, f"{prefix(slot)}_{suffix}.pt")
            if os.path.exists(path):
                os.remove(path)
        path = os.path.join(cache_dir, slot_name("_custom_prompt.txt", slot))
        if os.path.exists(path):
            os.remove(path)


def encoded_images(cache_dir):
    try:
        with open(os.path.join(cache_dir, "_custom_image.txt"), "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def save_images_text(cache_dir, images, one_per_prompt=False):
    def write(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(images_key(images, one_per_prompt))
    write_atomic(cache_dir, "_custom_image.txt", write)


def remove_slot_refs(cache_dir, keep=()):
    """
    Borra las imágenes de referencia propias de cada prompt (_custom2_ctrl.pt, _custom2_neg_*...): con
    imágenes compartidas, si quedaran, ese prompt seguiría usando la suya. keep = prefijos que se usan.
    """
    for f in os.listdir(cache_dir):
        for slot in range(1, MAX):
            name = prefix(slot)
            if name in keep:
                continue
            if f.startswith((name + "_ctrl", name + "_neg_", name + "_uncond")):
                os.remove(os.path.join(cache_dir, f))


def remove_slot(cache_dir, slot, names=("_custom_embed.pt",)):
    for name in ("_custom_prompt.txt",) + tuple(names):
        path = os.path.join(cache_dir, slot_name(name, slot))
        if os.path.exists(path):
            os.remove(path)


def remove_unused(cache_dir, cfg, names=("_custom_embed.pt",)):
    """Borra de la caché los prompts que ya no están en los ajustes."""
    used = {slot for slot, _ in prompts(cfg)}
    for slot in range(MAX):
        if slot not in used:
            remove_slot(cache_dir, slot, names)


def write_atomic(cache_dir, name, write):
    """write(ruta temporal) y luego os.replace: el trainer nunca lee un fichero a medias."""
    final = os.path.join(cache_dir, name)
    tmp = final + ".tmp"
    write(tmp)
    os.replace(tmp, final)


def save_text(cache_dir, slot, prompt):
    def write(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)
    write_atomic(cache_dir, slot_name("_custom_prompt.txt", slot), write)


def parse_prompts_only(argv):
    """
    argv de --prompts-only -> (carpeta, device, [(slot, prompt)]) o, en edición, con un 4º elemento:
    la lista de imágenes de referencia. None si no es ese modo.
    """
    if len(argv) in (5, 6) and argv[1] == "--prompts-only":
        out = (argv[2], argv[3], [(int(s), str(p)) for s, p in json.loads(argv[4])])
        return out + (json.loads(argv[5]),) if len(argv) == 6 else out
    return None


# --------------------------------------------------------------------------- trainer
class Embeds:
    """
    Embeddings de los prompts propios dentro del trainer. load(slot) lee los ficheros de un slot y
    devuelve lo que el trainer guarde para él (un tensor, un dict...); refresh() relee los que han
    cambiado en disco (el servidor los recodifica en CPU durante el entrenamiento).
    """

    def __init__(self, cache_dir, load, watch="_custom_embed.pt", prefixed=False):
        # prefixed=True (edición): el fichero vigilado de cada prompt es <prefix(slot)>_embed.pt.
        self.cache_dir, self.load, self.watch, self.prefixed = cache_dir, load, watch, prefixed
        self.items, self.mtimes = {}, {}

    def path(self, slot):
        if self.prefixed:
            return os.path.join(self.cache_dir, f"{prefix(slot)}_embed.pt")
        return os.path.join(self.cache_dir, slot_name(self.watch, slot))

    def refresh(self, cfg):
        """Relee lo que ha cambiado; devuelve los slots releídos."""
        changed = []
        active = {slot for slot, _ in prompts(cfg)}
        for slot in range(MAX):
            path = self.path(slot)
            if slot not in active or not os.path.exists(path):
                self.items.pop(slot, None)
                self.mtimes.pop(slot, None)
                continue
            mtime = os.path.getmtime(path)
            if self.mtimes.get(slot) != mtime:
                try:
                    self.items[slot] = self.load(slot)
                except Exception as exc:   # p. ej. un fichero que se está escribiendo
                    print(f"\n[PREVIEW] {os.path.basename(path)}: {exc}", flush=True)
                    continue
                self.mtimes[slot] = mtime
                changed.append(slot)
        return changed

    def round(self, cfg):
        """Slots para esta ronda de previews: todos los listos, o uno al azar con Random."""
        ready = [slot for slot, _ in prompts(cfg) if slot in self.items]
        if ready and random_one(cfg):
            return [random.choice(ready)]
        return ready

    def missing(self, cfg):
        """Prompts activos que aún no tienen embedding en la caché."""
        return [slot for slot, _ in prompts(cfg) if slot not in self.items]


def preview_name(step, index, ext=".png"):
    """Nombre de la preview: la primera de la ronda es preview_step_N.png, las demás _2, _3, _4."""
    return f"preview_step_{step}{ext}" if index == 0 else f"preview_step_{step}_{index + 1}{ext}"


def custom_text(cache_dir, slot):
    """Texto exacto (con trigger) con el que se codificó el prompt propio de ese slot."""
    return encoded_text(cache_dir, slot) or ""


def dataset_caption(cfg, name, trigger=None):
    """
    Caption de una imagen del dataset tal como la codifica el Pre-Cache (con el trigger delante), para
    guardarla en su preview. Cadena vacía si no se encuentra el .txt.
    """
    folder = cfg.get("dataset_path") or "./dataset"
    trigger = cfg.get("trigger_word", "") if trigger is None else trigger
    path = os.path.join(folder, f"{name}.txt")
    try:
        from text_io import read_text_file
        return with_trigger(read_text_file(path).strip(), trigger)
    except Exception:
        return ""


def save_png(image, path, prompt):
    """
    Guarda la preview (PIL.Image o array HxWx3 uint8) con el prompt dentro del PNG (texto «prompt»,
    como ComfyUI), para que la galería pueda enseñarlo. Escritura atómica.
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    info = PngInfo()
    if prompt:
        info.add_text("prompt", prompt)
    tmp = path + ".tmp"
    image.save(tmp, format="PNG", pnginfo=info)
    os.replace(tmp, path)


def tag_png(path, prompt):
    """Añade el prompt a un PNG ya guardado por otro código (lo reescribe con el texto «prompt»)."""
    if not prompt or not path or not str(path).lower().endswith(".png") or not os.path.exists(path):
        return
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        image = im.copy()
    save_png(image, str(path), prompt)


def save_sidecar(path, prompt):
    """Previews de vídeo o audio (mp4, wav): el prompt va en <preview>.txt, al lado."""
    side = path + ".txt"
    if not prompt:
        if os.path.exists(side):
            os.remove(side)
        return
    with open(side, "w", encoding="utf-8") as f:
        f.write(prompt)


_png_prompts = {}


def png_prompt(path):
    """
    Prompt de una preview: el texto «prompt» del PNG (save_png) o, en vídeo y audio, su <preview>.txt
    (save_sidecar). "" si no tiene. Se recuerda por fecha del fichero.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""
    hit = _png_prompts.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        if path.lower().endswith(".png"):
            from PIL import Image
            with Image.open(path) as im:
                prompt = (getattr(im, "text", None) or im.info or {}).get("prompt", "")
        else:
            with open(path + ".txt", "r", encoding="utf-8") as f:
                prompt = f.read()
    except Exception:
        prompt = ""
    _png_prompts[path] = (mtime, prompt)
    return prompt


def describe(cfg, cache_dir, trigger=None, add_trigger=None):
    """Líneas para la consola del trainer: qué prompt va en cada slot y si está codificado."""
    lines = []
    for slot, p in wanted(cfg, trigger, add_trigger):
        state = "OK" if encoded_text(cache_dir, slot) == p else "..."
        lines.append(f"  [{state}] #{slot + 1}: '{p}'")
    return lines


# --------------------------------------------------------------------------- servidor
class Encoder:
    """
    Un único proceso de codificación por servidor. Si se guarda otra vez mientras codifica, se
    relanza con la lista nueva. stage(): "loading" / "prompt" mientras trabaja, None si no.
    """

    def __init__(self, precache_script, base_dir):
        self.script, self.base_dir = str(precache_script), str(base_dir)
        self.process, self.args, self._stage, self.device = None, None, None, None

    def stage(self):
        return self._stage if self.process is not None and self.process.poll() is None else None

    def start(self, cache_dir, items, device, images=None):
        args = [sys.executable, "-u", self.script, "--prompts-only", str(cache_dir), device,
                json.dumps([[s, p] for s, p in items], ensure_ascii=False)]
        if images is not None:
            args.append(json.dumps(list(images), ensure_ascii=False))
        if self.stage() is not None:
            if self.args == args:
                return
            self.process.terminate()
            self.process.wait()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else None
        process = subprocess.Popen(args, cwd=self.base_dir, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        self.process, self.args, self._stage = process, args, "loading"
        self.device = "GPU" if device == "cuda" else "CPU"

        def follow():
            # Reenvía la salida a la consola del servidor y actualiza la fase que muestra la GUI.
            for line in process.stdout:
                print(line, end="", flush=True)
                if self.process is process and line.startswith("[Custom Prompt] Prompt"):
                    self._stage = "prompt"
                elif self.process is process and line.startswith("[Custom Prompt] Image"):
                    self._stage = "image"

        threading.Thread(target=follow, daemon=True).start()

    def update(self, cache_dir, cfg, training, ready, images=None, add_trigger=None):
        """
        Tras Save JSON: codifica los prompts que han cambiado. ready = la caché ya existe (si no,
        lo hará el Pre-Cache). training = hay un entrenamiento en marcha (entonces en CPU).
        images = en edición, la lista de imágenes de referencia de la preview.
        """
        items = pending(cache_dir, cfg, images=images, add_trigger=add_trigger)
        if items and ready:
            self.start(cache_dir, items, "cpu" if training else "cuda", images)
        return self.stage()
