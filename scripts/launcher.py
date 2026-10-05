# -*- coding: utf-8 -*-
"""
launcher.py — Lanzador común de AcademiaSD LoRAlab Trainer Studio
Common launcher for AcademiaSD LoRAlab Trainer Studio

Sirve GUI/launcher.html en http://127.0.0.1:4990 (IP y puertos en settings/network.json, ver remote_access.py). Los entrenadores salen de GUI/launcher.json:
al pulsar uno se abre su code/Run_LoRAlab-* (.bat en Windows, .sh equivalente en Linux)
en una ventana nueva, que arranca su servidor
en el puerto 5000. Para añadir un LoRAlab basta con una entrada más en GUI/launcher.json
(id, name, description, image, run). Las tarjetas se colocan solas: hasta 3 en una fila y, a partir
de ahí, dos filas con la sobrante arriba (4 -> 2+2, 5 -> 3+2, 6 -> 3+3). image_scale fija el tamaño
de la portada (0.2 = 20 %) y rows fuerza el número de filas.
"""
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

import i18n
import remote_access
from i18n import t

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "GUI"
CODE_DIR = BASE_DIR / "code"
CONFIG_FILE = GUI_DIR / "launcher.json"

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
i18n.register(app)


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def trainer_running():
    # Todos los entrenadores usan el mismo puerto: solo puede haber uno abierto a la vez.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", remote_access.load()["trainer_port"])) == 0


def trainer_url():
    # La URL pública si hay un proxy delante; si no, la misma dirección con la que se abrió el
    # lanzador, para que sirva también desde otro equipo.
    cfg = remote_access.load()
    return cfg["trainer_url"] or f"http://{request.host.rsplit(':', 1)[0]}:{cfg['trainer_port']}"


def _launch_linux(run: Path):
    # Abre el .sh en su propia terminal (cerrarla detiene ese entrenador),
    # igual que "start" hace con el .bat en Windows.
    # Opens the .sh in its own terminal (closing it stops that trainer),
    # just like "start" does with the .bat on Windows.
    terminals = [
        (["konsole", "-e"], True),
        (["gnome-terminal", "--"], True),
        (["xfce4-terminal", "-e"], True),
        (["x-terminal-emulator", "-e"], True),
        (["xterm", "-e"], True),
    ]
    for cmd, append in terminals:
        if shutil.which(cmd[0]) is not None:
            try:
                subprocess.Popen(
                    cmd + ["bash", str(run)] if append else [cmd[0], "bash", str(run)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except OSError:
                continue
    # Sin terminal grafica (p. ej. SSH): proceso independiente con log.
    # No graphical terminal (e.g. SSH): detached process with log file.
    log = BASE_DIR / "settings" / "trainer_console.log"
    log.parent.mkdir(exist_ok=True)
    fh = log.open("ab")
    subprocess.Popen(
        ["bash", str(run)],
        cwd=str(BASE_DIR),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


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

    if os.name == "nt":
        run = (CODE_DIR / trainer["run"]).resolve()
        if run.parent != CODE_DIR or run.suffix.lower() != ".bat" or not run.is_file():
            return jsonify({"status": "error", "error": t("Not found: {name}", name=trainer["run"])}), 404
    else:
        # En Linux se usa el .sh con el mismo nombre / On Linux use the same-named .sh
        run = (CODE_DIR / Path(trainer["run"]).with_suffix(".sh")).resolve()
        if run.parent != CODE_DIR or run.suffix.lower() != ".sh" or not run.is_file():
            return jsonify({"status": "error", "error": t("Not found: {name}", name=run.name)}), 404

    if trainer_running():
        return jsonify({"status": "busy", "url": trainer_url()})

    if os.name == "nt":
        # "start" abre el .bat en su propia consola: cerrarla detiene ese entrenador.
        subprocess.Popen(["cmd", "/c", "start", "", str(run)], cwd=str(BASE_DIR))
    else:
        _launch_linux(run)
    return jsonify({"status": "ok", "url": trainer_url(), "remote": not remote_access.is_local()})


# La configuración de red solo se cambia desde este PC: un equipo remoto no puede abrir más el acceso.
@app.route("/api/network", methods=["GET", "POST"])
def network():
    if not remote_access.is_local():
        abort(403)
    if request.method == "POST":
        req = request.get_json(force=True) or {}
        try:
            ports = int(req.get("launcher_port", 4990)), int(req.get("trainer_port", 5000))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "error": t("Invalid port")}), 400
        if not all(1024 <= p <= 65535 for p in ports) or ports[0] == ports[1]:
            return jsonify({"status": "error", "error": t("Ports: 1024-65535 and different")}), 400
        listen, user, password = bool(req.get("listen")), str(req.get("user", "")).strip(), str(req.get("password", ""))
        external_auth, public_url = bool(req.get("external_auth")), str(req.get("trainer_url", "")).strip()
        if public_url and not public_url.startswith(("http://", "https://")):
            return jsonify({"status": "error", "error": t("Trainer URL must start with http:// or https://")}), 400
        if listen and not external_auth and (not user or not (password or remote_access.load()["password_hash"])):
            return jsonify({"status": "error", "error": t("Network access needs a user and a password")}), 400
        remote_access.save(listen, ports[0], ports[1], user, password, external_auth, public_url)
    cfg = remote_access.load()
    return jsonify({"status": "ok", "listen": cfg["listen"], "launcher_port": cfg["launcher_port"],
                    "trainer_port": cfg["trainer_port"], "user": cfg["user"],
                    "has_password": bool(cfg["password_hash"]), "external_auth": cfg["external_auth"],
                    "trainer_url": cfg["trainer_url"], "lan_ip": remote_access.lan_address()})


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 70)
    print("  ACADEMIASD — LORALAB TRAINER STUDIO LAUNCHER")
    print("=" * 70)
    url = remote_access.local_url("launcher_port")
    print(f"  URL : {url}")
    print("  " + t("Close this window to close the launcher."))
    print("=" * 70)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    remote_access.serve(app, "launcher_port")
