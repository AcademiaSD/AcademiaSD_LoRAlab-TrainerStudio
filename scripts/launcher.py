# -*- coding: utf-8 -*-
"""
launcher.py — Lanzador común de AcademiaSD LoRAlab Trainer Studio
Common launcher for AcademiaSD LoRAlab Trainer Studio

Sirve GUI/launcher.html en http://127.0.0.1:4990 (IP y puertos en settings/network.json, ver remote_access.py). Los entrenadores salen de GUI/launcher.json:
al pulsar uno se abre su code/Run_LoRAlab-* (.bat en Windows, .sh equivalente en Linux)
en una ventana nueva (la consola, que mantiene vivo el entrenador), que arranca su servidor
en el puerto 5000; la pestaña del lanzador navega a él y su tarjeta queda con un brillo
pulsante mientras siga abierto (el brillo exige que el proceso guardado siga vivo, así una
entrada vieja en active_trainer.json no lo enciende). Elegir otro entrenador avisa de que el
activo se cerrará y, al aceptar, lo cierra antes de abrir el nuevo (se guarda su PID; sin PID
guardado, solo se cierra un server_*.py de esta carpeta scripts). Si el activo está entrenando
no se cierra: se avisa de que hay que detener el entrenamiento antes de cambiar.
Para añadir un LoRAlab basta con una entrada más en GUI/launcher.json
(id, name, description, image, run). Las tarjetas se colocan solas: hasta 3 en una fila y, a partir
de ahí, dos filas con la sobrante arriba (4 -> 2+2, 5 -> 3+2, 6 -> 3+3). image_scale fija el tamaño
de la portada (0.2 = 20 %) y rows fuerza el número de filas.
"""
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

import gpu_stats
import i18n
import project_layout
import remote_access
from i18n import t

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "GUI"
CODE_DIR = BASE_DIR / "code"
SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = GUI_DIR / "launcher.json"

project_layout.ensure_layout_and_log(BASE_DIR)

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


def trainer_training():
    # ¿El entrenador que escucha en el puerto está ejecutando un script? Se le pregunta a él
    # mismo; las peticiones locales no piden contraseña ni con el acceso remoto activado.
    # Is the trainer listening on the port running a script? Asked to the trainer itself;
    # local requests never ask for a password, not even with remote access on.
    # None = no ha contestado (ocupado o no es un entrenador): ante la duda no se cierra.
    # None = no answer (busy or not a trainer): when in doubt, nothing is closed.
    port = remote_access.load()["trainer_port"]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None
    return bool(data.get("running"))


ACTIVE_FILE = BASE_DIR / "settings" / "active_trainer.json"


def _active_data():
    # Entrada guardada al abrir un entrenador ({"id", "pid"}), sin comprobar nada.
    # Entry saved when a trainer was opened ({"id", "pid"}), unchecked.
    try:
        data = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
        return {"id": data.get("id") or None, "pid": int(data.get("pid") or 0)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"id": None, "pid": 0}


def active_pid():
    return _active_data()["pid"]


def active_running():
    # El id activo solo cuenta si su proceso sigue vivo y es el del entrenador; el archivo no
    # se borra al cerrarlo, así que una entrada vieja (p. ej. otro entrenador arrancado a mano
    # en su lugar) no debe encender su tarjeta ni devolverla como si siguiera abierta.
    # The active id only counts while its process is alive and is a trainer's; the file is not
    # removed on close, so a stale entry (e.g. another trainer started by hand instead) must
    # not light its card up nor be returned as still open.
    data = _active_data()
    if data["id"] and data["pid"] and _pid_alive(data["pid"]) and _pid_is_trainer(data["pid"]):
        return data["id"]
    return None


def save_active(trainer_id, pid=None):
    # El PID de la consola permite cerrar el entrenador al cambiar a otro.
    # The console PID allows closing the trainer when switching to another one.
    ACTIVE_FILE.parent.mkdir(exist_ok=True)
    ACTIVE_FILE.write_text(json.dumps({"id": trainer_id, "pid": pid}), encoding="utf-8")


def _pids_on_port(port: int):
    # Quién escucha en el puerto (Linux, sin herramientas externas): inodo del socket en
    # /proc/net/tcp, buscado entre los descriptores de cada proceso.
    # Who listens on the port (Linux, no external tools): socket inode in /proc/net/tcp,
    # looked up among every process' file descriptors.
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) > 9 and parts[3] == "0A" and int(parts[1].split(":")[1], 16) == port:
            inodes.add(f"socket:[{parts[9]}]")
    pids = []
    if inodes:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                if any(os.readlink(fd) in inodes for fd in (d / "fd").iterdir()):
                    pids.append(int(d.name))
            except OSError:
                continue
    return pids


def _pid_alive(pid: int):
    # En Windows os.kill no sirve para esto: por documentación, cualquier señal que no sea
    # CTRL_C/CTRL_BREAK termina el proceso. Se mira con OpenProcess/WaitForSingleObject.
    # os.kill is not a liveness check on Windows: per docs, any signal other than
    # CTRL_C/CTRL_BREAK terminates the process. Checked with OpenProcess/WaitForSingleObject.
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _pid_cmdline_nt(pid: int):
    # Línea de comandos del proceso en Windows. wmic ya no existe en Windows 11 24H2+,
    # así que hay respaldo por la API CIM (WMI) de PowerShell.
    # Process command line on Windows. wmic is gone in Windows 11 24H2+, so there is a
    # fallback through PowerShell's CIM (WMI) API.
    for cmd in (["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine"],
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.strip():
            return out
    return ""


_CMDLINE_CACHE = {}  # pid -> (hora, línea de comandos)


def _pid_cmdline(pid: int):
    # Con caché: /api/status la consulta cada pocos segundos y en Windows cuesta un proceso
    # aparte (wmic o PowerShell).
    # Cached: /api/status asks every few seconds and on Windows it costs a separate process
    # (wmic or PowerShell).
    hit = _CMDLINE_CACHE.get(pid)
    if hit and time.time() - hit[0] < 15:
        return hit[1]
    if os.name == "nt":
        cmd = _pid_cmdline_nt(pid)
    else:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            cmd = ""
    if len(_CMDLINE_CACHE) > 32:
        _CMDLINE_CACHE.clear()
    _CMDLINE_CACHE[pid] = (time.time(), cmd)
    return cmd


def _pid_is_repo_server(pid: int):
    # Servidor de un entrenador de este repo: la línea de comandos trae server_*.py de esta
    # carpeta scripts. Así no se cierra otro programa que estuviera usando el mismo puerto.
    # A server of a trainer from this repo: the command line carries server_*.py from this
    # scripts folder. This keeps another program using the same port from being closed.
    cmd = _pid_cmdline(pid)
    if not cmd or "server_" not in cmd:
        return False
    if str(SCRIPTS_DIR).replace("\\", "/").lower() in cmd.replace("\\", "/").lower():
        return True
    if os.name == "nt":
        return False
    try:
        return os.path.realpath(f"/proc/{pid}/cwd") == str(SCRIPTS_DIR)
    except OSError:
        return False


def _pid_is_trainer(pid: int):
    # La consola que abre el lanzador (Run_LoRAlab) o un servidor de este repo: solo se cierra
    # algo con esa pinta, por si el PID se reutilizó.
    # The console the launcher opens (Run_LoRAlab) or a server of this repo: only something
    # looking like that is closed, in case the PID was reused.
    return "Run_LoRAlab" in _pid_cmdline(pid) or _pid_is_repo_server(pid)


def _trainer_pids_on_port(port: int):
    # Lo que escucha en el puerto y es un servidor de este repo (nunca otro programa).
    # Whatever listens on the port and is a server of this repo (never another program).
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
        pids = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 4 and "LISTENING" in line and parts[1].endswith(f":{port}"):
                try:
                    pids.append(int(parts[-1]))
                except ValueError:
                    continue
    else:
        pids = _pids_on_port(port)
    return sorted({p for p in pids if _pid_is_repo_server(p)})


def _kill_pid(pid: int):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def kill_trainer():
    """Cierra el entrenador en marcha y espera a que suelte el puerto.
    Closes the running trainer and waits for it to release the port."""
    port = remote_access.load()["trainer_port"]
    pid = active_pid()
    if pid and _pid_alive(pid) and _pid_is_trainer(pid):
        # La consola se lleva el servidor (y su ventana) por delante.
        # The console takes the server (and its window) with it.
        _kill_pid(pid)
    else:
        # Sin PID guardado (p. ej. arranque directo): lo que escuche en el puerto, y solo si
        # es un servidor de este repo.
        # No saved PID (e.g. direct run): whatever listens on the port, and only if it is a
        # server of this repo.
        for p in _trainer_pids_on_port(port):
            _kill_pid(p)

    for _ in range(25):  # hasta 5 s de cortesía / up to 5 s of grace
        if not trainer_running():
            return True
        time.sleep(0.2)
    if os.name != "nt":
        # Último recurso: lo que siga con el puerto, sin cortesía.
        # Last resort: whatever still holds the port, no grace.
        for p in _trainer_pids_on_port(port):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        for _ in range(15):
            if not trainer_running():
                return True
            time.sleep(0.2)
    return not trainer_running()


def _launch_linux(run: Path, env: dict):
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
                proc = subprocess.Popen(
                    cmd + ["bash", str(run)] if append else [cmd[0], "bash", str(run)],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                return proc.pid
            except OSError:
                continue
    # Sin terminal grafica (p. ej. SSH): proceso independiente con log.
    # No graphical terminal (e.g. SSH): detached process with log file.
    log = BASE_DIR / "settings" / "trainer_console.log"
    log.parent.mkdir(exist_ok=True)
    fh = log.open("ab")
    proc = subprocess.Popen(
        ["bash", str(run)],
        cwd=str(BASE_DIR),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return proc.pid


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
    req = request.get_json(force=True) or {}
    trainer_id = str(req.get("id", ""))
    force = bool(req.get("force"))
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
        active = active_running()
        # Tarjeta con brillo: el entrenador activo ya va corriendo, volver a él.
        # Glowing card: the active trainer is already running, jump back to it.
        if active == trainer_id:
            return jsonify({"status": "ok", "url": trainer_url(), "already": True})
        training = trainer_training()
        current = next((x["name"] for x in load_config()["trainers"] if x["id"] == active), None) or t("Another trainer")
        if training is None:
            return jsonify({"status": "training", "url": trainer_url(),
                            "error": t("Could not check whether {current} is training. Close it from its console window, then try again.",
                                       current=current)}), 409
        if training:
            # El script de entrenamiento es hijo del servidor: cerrarlo ahora cortaría el
            # entrenamiento, y lo avanzado desde el último checkpoint.
            # The training script is a child of the server: closing it now would cut the
            # training, and everything since the last checkpoint.
            return jsonify({"status": "training", "url": trainer_url(),
                            "error": t("{current} is training. Stop the training first, then switch.",
                                       current=current)}), 409
        if not force:
            return jsonify({"status": "busy", "url": trainer_url()})
        # Cambiar de entrenador: cerrar el activo antes de abrir el nuevo.
        # Switching trainers: close the active one before opening the new one.
        if not kill_trainer():
            return jsonify({"status": "error",
                            "error": t("Could not close the running trainer.")}), 409

    # La GPU elegida pasa al entrenador por herencia: el .bat/.sh, su servidor y los scripts la ven.
    gpu_stats.apply_saved_gpu(os.environ)

    # La pestaña del lanzador abre el navegador; el servidor no debe abrir otro.
    # The launcher tab opens the browser; the server must not open another one.
    env = dict(os.environ)
    env["LORALAB_SKIP_BROWSER"] = "1"

    if os.name == "nt":
        # Consola propia para el .bat, con su PID a mano para poder cerrarla luego.
        # Own console for the .bat, PID at hand so it can be closed later.
        proc = subprocess.Popen(["cmd", "/c", str(run)], cwd=str(BASE_DIR), env=env,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        pid = proc.pid
    else:
        pid = _launch_linux(run, env)
    save_active(trainer_id, pid)
    return jsonify({"status": "ok", "url": trainer_url(), "remote": not remote_access.is_local()})


@app.route("/api/status")
def status():
    running = trainer_running()
    return jsonify({"trainer": running, "active": active_running() if running else None,
                    "url": trainer_url()})


# La configuración de red solo se cambia desde este PC: un equipo remoto no puede abrir más el acceso.
@app.route("/api/gpu", methods=["GET", "POST"])
def gpu():
    """GET: tarjetas NVIDIA (o AMD) y la elegida. POST {"gpu": "<índice>" | ""}: la guarda ("" = la primera)."""
    gpus = gpu_stats.list_gpus()
    if request.method == "POST":
        if not remote_access.is_local():
            return jsonify({"status": "error", "error": t("Only from the training PC.")}), 403
        chosen = str((request.get_json(force=True) or {}).get("gpu", ""))
        if chosen and not any(g["index"] == chosen for g in gpus):
            return jsonify({"status": "error", "error": t("GPU not found: {name}", name=chosen)}), 400
        gpu_stats.save_gpu(chosen)
    return jsonify({"status": "ok", "gpus": gpus, "selected": gpu_stats.saved_gpu()})


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
