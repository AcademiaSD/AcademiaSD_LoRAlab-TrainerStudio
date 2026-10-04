# -*- coding: utf-8 -*-
"""
remote_access.py — Acceso desde otros equipos de la red al lanzador y a los entrenadores.
Access to the launcher and the trainers from other machines on the network.

settings/network.json (se edita desde el lanzador / edited from the launcher):
    listen         false = solo este PC (127.0.0.1); true = toda la red (0.0.0.0)
    launcher_port  4990
    trainer_port   5000
    user, password_hash   usuario y contraseña que se piden a los equipos remotos

Las peticiones desde este mismo PC nunca piden contraseña. Las remotas la piden siempre, y sin
contraseña configurada se rechazan: la interfaz lanza procesos y escribe archivos, así que abrirla
a la red sin contraseña no es una opción. Es HTTP sin cifrar, pensado para la red de casa; para
entrar desde Internet, Tailscale o un túnel SSH.
"""
import json
import socket
from pathlib import Path

from flask import Response, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings" / "network.json"
DEFAULTS = {"listen": False, "launcher_port": 4990, "trainer_port": 5000, "user": "", "password_hash": ""}

# Los diálogos nativos se abrirían en la pantalla de este PC, no en la del navegador remoto.
NATIVE_DIALOGS = ("/api/select-folder", "/api/select-file")


def load():
    cfg = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        cfg.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    return cfg


def save(listen, launcher_port, trainer_port, user, password):
    """password vacío conserva la contraseña guardada. / An empty password keeps the saved one."""
    cfg = load()
    cfg.update(listen=bool(listen), launcher_port=int(launcher_port), trainer_port=int(trainer_port), user=user.strip())
    if password:
        cfg["password_hash"] = generate_password_hash(password)
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def is_local():
    return request.remote_addr in ("127.0.0.1", "::1")


def lan_address():
    """IP de este PC en la red local. connect() en UDP no envía nada: solo elige la interfaz."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _check():
    if is_local():
        return None
    cfg = load()
    if not cfg["password_hash"]:
        return Response("Remote access needs a password: set it in the launcher, on the training PC.\n"
                        "El acceso remoto necesita contraseña: ponla en el lanzador, en el PC de entrenamiento.\n",
                        403, mimetype="text/plain")
    auth = request.authorization
    if not auth or auth.username != cfg["user"] or not check_password_hash(cfg["password_hash"], auth.password or ""):
        return Response("Login required / Inicio de sesión necesario\n", 401,
                        {"WWW-Authenticate": 'Basic realm="AcademiaSD LoRAlab", charset="UTF-8"'}, mimetype="text/plain")
    if request.path in NATIVE_DIALOGS:
        return jsonify({"status": "error", "error": "Remote access: type the path. / Acceso remoto: escribe la ruta."})
    return None


def serve(app, port_key):
    """Arranca el servidor Flask con la IP y el puerto de network.json. port_key: launcher_port o trainer_port."""
    cfg = load()
    app.before_request(_check)
    host = "0.0.0.0" if cfg["listen"] else "127.0.0.1"
    port = cfg[port_key]
    if cfg["listen"]:
        print(f"  Network / Red: http://{lan_address()}:{port}"
              + ("" if cfg["password_hash"] else "  [!] No password: remote access refused / Sin contraseña: acceso remoto rechazado"))
    app.run(host=host, port=port, debug=False, threaded=True)


def local_url(port_key):
    return f"http://127.0.0.1:{load()[port_key]}"
