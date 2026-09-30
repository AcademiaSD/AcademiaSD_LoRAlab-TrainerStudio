# -*- coding: utf-8 -*-
"""
launcher.py — Lanzador común de AcademiaSD LoRAlab Trainer Studio
Common launcher for AcademiaSD LoRAlab Trainer Studio

Sirve GUI/launcher.html en http://127.0.0.1:4990. Los entrenadores salen de GUI/launcher.json:
al pulsar uno se abre su code/Run_LoRAlab-*.bat en una ventana nueva, que arranca su servidor
en http://127.0.0.1:5000. Para añadir un LoRAlab basta con una entrada más en GUI/launcher.json
(id, name, description, image, run). Las tarjetas se colocan solas: hasta 3 en una fila y, a partir
de ahí, dos filas con la sobrante arriba (4 -> 2+2, 5 -> 3+2, 6 -> 3+3). image_scale fija el tamaño
de la portada (0.2 = 20 %) y rows fuerza el número de filas.
"""
import json
import logging
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "GUI"
CODE_DIR = BASE_DIR / "code"
CONFIG_FILE = GUI_DIR / "launcher.json"
PORT = 4990
TRAINER_PORT = 5000

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def trainer_running():
    # Todos los entrenadores usan el puerto 5000: solo puede haber uno abierto a la vez.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", TRAINER_PORT)) == 0


@app.route("/")
def index():
    return send_from_directory(str(GUI_DIR), "launcher.html")


@app.route("/launcher.json")
def config():
    return jsonify(load_config())


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(BASE_DIR / "assets"), filename)


@app.route("/api/launch", methods=["POST"])
def launch():
    trainer_id = str((request.get_json(force=True) or {}).get("id", ""))
    trainer = next((t for t in load_config()["trainers"] if t["id"] == trainer_id), None)
    if trainer is None:
        abort(404)

    run = (CODE_DIR / trainer["run"]).resolve()
    if run.parent != CODE_DIR or run.suffix.lower() != ".bat" or not run.is_file():
        return jsonify({"status": "error", "error": f"Not found / No existe: {trainer['run']}"}), 404

    if trainer_running():
        return jsonify({"status": "busy", "url": f"http://127.0.0.1:{TRAINER_PORT}"})

    # "start" abre el .bat en su propia consola: cerrarla detiene ese entrenador.
    subprocess.Popen(["cmd", "/c", "start", "", str(run)], cwd=str(BASE_DIR))
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 70)
    print("  ACADEMIASD — LORALAB TRAINER STUDIO LAUNCHER")
    print("=" * 70)
    print(f"  URL : http://127.0.0.1:{PORT}")
    print("  Cierra esta ventana para cerrar el lanzador / Close this window to close the launcher.")
    print("=" * 70)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
