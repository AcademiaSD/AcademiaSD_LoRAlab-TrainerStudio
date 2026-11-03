# -*- coding: utf-8 -*-
"""
dataset_forge.py — Dataset Forge: un dataset de personaje a partir de pocas imágenes (desde 1).
Dataset Forge: a character dataset from a few images (from 1).

Flujo / Flow
  1. Analizar (0_forge_dataset.py analyze): con el captioner compartido (Qwen3-VL-8B) escribe el
     caption de las referencias que no lo tienen y una ficha del personaje (sujeto + rasgos fijos).
  2. Plan: combina encuadre, ropa, lugar (interior / exterior), luz, ángulo y expresión sin repetir
     ropa ni lugar hasta agotarlos. Es Python puro: el servidor lo rehace al instante y la página
     deja editarlo antes de generar.
  3. Generar (0_forge_dataset.py generate): Qwen-Image 2.1 (o Qwen-Image-Edit-2511) en NF4 con su LoRA
     turbo de 4 pasos recibe las referencias y la instrucción de cada línea del plan. Cada imagen pasa por el parecido facial
     (face_match.py) y, si no llega al umbral, se repite con otra semilla.
  El resultado va a una carpeta NUEVA (la de origen no se toca): referencias + generadas, cada una
  con su .txt. El caption de entrenamiento es el del plan SIN los rasgos fijos: así el trigger se
  queda con la identidad.

Servidor:  register(app, get_dataset_dir) sirve /forge (la página), /dataset_forge_button.js y /api/forge/*.
Estado:    settings/dataset_forge_status.json (lo escribe el script) y settings/dataset_forge_console.log.
"""
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "GUI"
SETTINGS_DIR = BASE_DIR / "settings"
SETTINGS_FILE = SETTINGS_DIR / "dataset_forge_settings.json"
STATUS_FILE = SETTINGS_DIR / "dataset_forge_status.json"
LOG_FILE = SETTINGS_DIR / "dataset_forge_console.log"
SCRIPT = Path(__file__).resolve().parent / "0_forge_dataset.py"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULTS = {
    "source_dir": "",
    "references": [],
    "output_dir": "",
    "trigger_word": "",
    # flux3klein = FLUX.3 Klein (FLUX Kommunity) · fibo2 = Fibo 2.0 (CC BY-NC) · qwen21 = Qwen-Image 2.1
    # (Qwen Research) · klein9b = FLUX.2 Klein 9B (FLUX Non-Commercial) · edit2511 = Qwen-Image-Edit-2511
    # (Apache 2.0). quality (2.1 y FLUX.3): best = pasos completos, fast = LoRA turbo (2.1: 4 pasos; FLUX.3: 8).
    "model": "flux3klein",
    "quality": "fast",
    "steps": 8,   # solo FLUX.3: Rápida 4 u 8, Mejor calidad 12-50 (0 = los de la calidad: 8 / 28)
    "subject": "person",
    "identity": "",
    "count": 20,
    "resolution": 1024,
    "threshold": 60,
    "max_tries": 2,
    "seed": 0,
    "include_references": True,
    "ref_captions": {},
    "plan": [],
}

# ---------------------------------------------------------------------------------------------
# Plan de variaciones / Variation plan
# ---------------------------------------------------------------------------------------------
# (encuadre, proporción ancho/alto, peso). Más primeros planos: de ahí aprende la cara el LoRA.
FRAMINGS = [
    ("close-up portrait of a {s}, head and shoulders,", 4 / 5, 5),
    ("medium shot of a {s} from the waist up,", 3 / 4, 3),
    ("full body shot of a {s} standing,", 2 / 3, 1),
    ("full body shot of a {s} sitting,", 3 / 4, 1),
]
# Proporción de una línea editada a mano, por palabras clave (la primera que aparezca manda).
RATIO_KEYS = (("close-up", 4 / 5), ("portrait", 4 / 5), ("sitting", 3 / 4), ("full body", 2 / 3), ("medium shot", 3 / 4))
# Conjuntos completos: con «a linen shirt» a secas, en cuerpo entero sale sin pantalón.
OUTFITS = [
    "a plain white t-shirt and blue jeans", "a black leather jacket and dark jeans", "a navy blue business suit",
    "a cozy knitted sweater and jeans", "a grey hoodie and joggers", "a denim jacket over a white top and black jeans",
    "a long winter coat, a scarf and boots", "a light linen shirt and beige chinos", "a sports tracksuit",
    "a checked flannel shirt and jeans", "a black turtleneck sweater and grey trousers",
    "a beige trench coat over a sweater and trousers", "a smart blazer with a shirt and trousers",
    "a yellow raincoat and jeans", "a striped polo shirt and chinos", "a green bomber jacket, a t-shirt and jeans",
]
OUTFITS_BY_SUBJECT = {
    "woman": ["a floral summer dress", "an elegant red evening dress", "a white blouse and a black skirt"],
    "man": ["a black tuxedo with a bow tie", "a white shirt with rolled-up sleeves"],
}
INDOOR = [
    "in a cozy living room", "in a modern kitchen", "in a busy coffee shop", "in a library with tall bookshelves",
    "in a bright office", "in a photo studio with a plain grey backdrop", "in a bedroom with soft morning light",
    "in an elegant restaurant", "in a gym", "in an art gallery", "in a train station hall", "in a home office",
]
# Sin hora del día: la pone la luz (si no, salen «beach at sunset» + «bright midday sun»).
OUTDOOR = [
    "on a busy city street", "in a green park with trees", "on a sandy beach", "on a mountain trail",
    "in a snowy forest", "on a rooftop terrace with a city view", "in a field of flowers",
    "by a calm lake", "in a narrow old town alley", "in a desert with sand dunes", "at a street market",
    "on a wooden pier by the sea",
]
LIGHTS = {
    "in": ["soft natural window light", "warm indoor lighting", "dramatic side lighting", "bright even lighting"],
    "out": ["golden hour sunlight", "overcast daylight", "bright midday sun", "soft evening light"],
}
ANGLES = ["facing the camera", "three-quarter view", "looking slightly to the side", "seen from a low angle"]
EXPRESSIONS = ["smiling", "with a neutral expression", "laughing", "with a serious expression", "with a calm, thoughtful look"]


class _Deck:
    """Saca elementos sin repetir hasta agotar la lista; entonces la baraja de nuevo."""

    def __init__(self, items, rng):
        self.items, self.rng, self.left = list(items), rng, []

    def draw(self):
        if not self.left:
            self.left = self.items[:]
            self.rng.shuffle(self.left)
        return self.left.pop()


def build_plan(count, subject="person", seed=0):
    """[{caption_part, ratio}] con count variaciones equilibradas (mitad interiores, mitad exteriores)."""
    rng = random.Random(seed or 1234)
    subject = (subject or "person").strip().lower()
    outfits = _Deck(OUTFITS + OUTFITS_BY_SUBJECT.get(subject, []), rng)
    places = {"in": _Deck(INDOOR, rng), "out": _Deck(OUTDOOR, rng)}
    lights = {k: _Deck(v, rng) for k, v in LIGHTS.items()}
    angles, expressions = _Deck(ANGLES, rng), _Deck(EXPRESSIONS, rng)
    framings = _Deck([f for f in FRAMINGS for _ in range(f[2])], rng)
    plan = []
    for i in range(max(1, int(count))):
        framing, ratio, _ = framings.draw()
        side = "in" if i % 2 == 0 else "out"
        text = (f"{framing.format(s=subject)} wearing {outfits.draw()}, {places[side].draw()}, "
                f"{lights[side].draw()}, {angles.draw()}, {expressions.draw()}")
        plan.append({"text": text, "ratio": round(ratio, 4)})
    return plan


def plan_ratio(text):
    """Proporción de una línea editada a mano: la de su encuadre, o 3:4."""
    low = text.lower()
    for key, ratio in RATIO_KEYS:
        if key in low:
            return round(ratio, 4)
    return 0.75


def instruction(line, subject, identity, refs, tags=False):
    """
    Lo que recibe el modelo de edición: mismo personaje que en las referencias + la línea del plan.
    tags: cómo nombra cada modelo sus referencias. True = Qwen-Image 2.1 (<image1>, <image2>...);
    "reference" = FLUX.2 Klein (su text encoder no ve las imágenes); False = Edit-2511 («Picture 1»).
    """
    if tags == "reference":
        pics = "the reference image" if refs <= 1 else "the reference images"
    elif tags:
        pics = " and ".join(f"<image{i}>" for i in range(1, refs + 1)) if refs <= 2 else f"<image1> to <image{refs}>"
    else:
        pics = "Picture 1" if refs <= 1 else "the reference pictures"
    text = (f"The same {subject} as in {pics}, keeping exactly the same face, facial features and identity. "
            f"New photo: {line}. Use a new pose and expression as described, not the ones in the reference.")
    if identity:
        text += f" Features: {identity.rstrip('.')}."
    return text + " Realistic photo, natural skin texture, sharp focus."


def training_caption(trigger, line):
    trigger = (trigger or "").strip().rstrip(",")
    return f"{trigger}, {line}" if trigger else line


# ---------------------------------------------------------------------------------------------
# Estado / State
# ---------------------------------------------------------------------------------------------
def read_json(path, default=None):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else (default or {})
    except (OSError, ValueError):
        return dict(default or {})


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


_proc = None
_stopped = False


def running():
    if _proc is not None:
        return _proc.poll() is None
    # La lanzó otro entrenador abierto (otro puerto) o este antes de reiniciarse: su PID está en el estado.
    pid = read_json(STATUS_FILE).get("pid")
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(pid) and "0_forge_dataset" in " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return False


def _gpu_busy():
    try:
        import project_layout
        return project_layout.training_in_progress()
    except Exception:
        return False


def start(mode):
    global _proc, _stopped
    if running():
        return False, "running"
    _stopped = False
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    write_json(STATUS_FILE, {"mode": mode, "phase": "starting", "started": time.time()})
    _proc = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT), mode], cwd=str(BASE_DIR), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    log.close()
    st = read_json(STATUS_FILE)
    st["pid"] = _proc.pid
    write_json(STATUS_FILE, st)
    return True, None


def stop():
    global _stopped
    if not running():
        return False
    _stopped = True
    if _proc is None:
        # De otro servidor: no es hijo nuestro, solo se puede terminar.
        try:
            import psutil
            psutil.Process(read_json(STATUS_FILE).get("pid")).terminate()
        except Exception:
            return False
        return True
    try:
        _proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except Exception:
        _proc.terminate()
    return True


# ---------------------------------------------------------------------------------------------
# Servidor / Server
# ---------------------------------------------------------------------------------------------
def register(app, get_dataset_dir):
    from flask import jsonify, request, send_file, send_from_directory
    from i18n import t

    @app.route("/forge")
    def forge_page():
        resp = send_from_directory(str(GUI_DIR), "dataset_forge.html")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/dataset_forge_button.js")
    def forge_button_js():
        return send_from_directory(str(GUI_DIR), "dataset_forge_button.js")

    @app.route("/api/forge/settings")
    def forge_settings():
        cfg = {**DEFAULTS, **read_json(SETTINGS_FILE)}
        if not cfg["source_dir"]:
            cfg["source_dir"] = str(get_dataset_dir())
        try:
            import face_match
            fm = face_match.installed()
        except Exception:
            fm = False
        return jsonify({"settings": cfg, "face_match": fm})

    @app.route("/api/forge/list")
    def forge_list():
        folder = Path(request.args.get("path", "")).expanduser()
        items = []
        for p in list_images(folder)[:200]:
            txt = p.with_suffix(".txt")
            caption = ""
            if txt.is_file():
                try:
                    caption = txt.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
            items.append({"file": p.name, "caption": caption, "mtime": int(p.stat().st_mtime)})
        return jsonify({"ok": folder.is_dir(), "path": str(folder.resolve()) if folder.is_dir() else str(folder),
                        "images": items, "default_output": str(folder.resolve()) + "_forge" if folder.is_dir() else ""})

    @app.route("/api/forge/image")
    def forge_image():
        path = Path(request.args.get("path", "")).expanduser()
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            return "", 404
        resp = send_file(str(path))
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/api/forge/plan", methods=["POST"])
    def forge_plan():
        data = request.get_json(silent=True) or {}
        return jsonify({"plan": build_plan(data.get("count", 20), data.get("subject", "person"), data.get("seed", 0))})

    @app.route("/api/forge/run", methods=["POST"])
    def forge_run():
        data = request.get_json(silent=True) or {}
        mode = data.get("mode")
        if mode not in ("analyze", "generate"):
            return jsonify({"ok": False, "error": "mode"}), 400
        if running():
            return jsonify({"ok": False, "error": t("Dataset Forge is already running.")}), 409
        if _gpu_busy():
            return jsonify({"ok": False, "error": t("The GPU is busy with a training, pre-cache or caption job. Wait until it finishes.")}), 409
        cfg = {**DEFAULTS, **{k: v for k, v in (data.get("settings") or {}).items() if k in DEFAULTS}}
        src = Path(cfg["source_dir"]).expanduser()
        refs = [r for r in cfg["references"] if (src / r).is_file()]
        if not refs:
            return jsonify({"ok": False, "error": t("Select at least one reference image.")}), 400
        cfg["references"] = refs
        if mode == "generate":
            if data.get("consent") is not True:
                return jsonify({"ok": False, "error": t("Confirm that you have the right or the permission to use the images of this person.")}), 400
            out = Path(cfg["output_dir"]).expanduser() if cfg["output_dir"] else Path(str(src.resolve()) + "_forge")
            if out.resolve() == src.resolve():
                return jsonify({"ok": False, "error": t("The output folder must be different from the source folder.")}), 400
            cfg["output_dir"] = str(out)
            cfg["plan"] = [{"text": str(p.get("text", "")).strip(), "ratio": float(p.get("ratio") or plan_ratio(p.get("text", "")))}
                           for p in cfg["plan"] if str(p.get("text", "")).strip()]
            if not cfg["plan"]:
                return jsonify({"ok": False, "error": t("The plan is empty.")}), 400
        write_json(SETTINGS_FILE, cfg)
        ok, err = start(mode)
        return jsonify({"ok": ok, "error": err})

    @app.route("/api/forge/stop", methods=["POST"])
    def forge_stop():
        return jsonify({"ok": stop()})

    @app.route("/api/forge/status")
    def forge_status():
        since = request.args.get("since", 0, type=int)
        lines = []
        try:
            with open(LOG_FILE, encoding="utf-8", errors="replace", newline="") as f:
                # Las barras de progreso se reescriben con \r: de cada línea, solo su último estado.
                # Lo que va tras el último salto está a medias: sale en el siguiente sondeo.
                lines = [ln.split("\r")[-1] for ln in f.read().replace("\r\n", "\n").split("\n")[:-1]]
        except OSError:
            pass
        st = read_json(STATUS_FILE)
        st["running"] = running()
        # Proceso muerto sin cerrar su estado (lo mataron, o se reinició el servidor): se cierra aquí y
        # se guarda, para que nadie lo dé por vivo.
        if not st["running"] and st.get("phase") not in (None, "done", "error", "stopped"):
            st["phase"] = "stopped" if _proc is None or _stopped else ("error" if _proc.returncode else "done")
            if st["phase"] == "stopped":
                st["message"] = t("Stopped.")
            write_json(STATUS_FILE, {k: v for k, v in st.items() if k != "running"})
        return jsonify({"status": st, "lines": lines[since:], "total": len(lines)})
