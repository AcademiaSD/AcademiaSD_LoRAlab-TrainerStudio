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
no se cierra: se ofrece abrir el otro en un puerto libre (trainer_port + 1...) para preparar un
trabajo y añadirlo a la cola (job_queue.py); esos se recuerdan en trainer_instances.json y su
tarjeta lleva a ellos. /api/status incluye la cola: el trabajo en curso y los pendientes.
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
import job_queue
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


# Cola de trabajos de todos los entrenadores (ver job_queue.py): rutas /api/queue... y su hilo.
QUEUE = job_queue.register(app, lambda: load_config()["trainers"])


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


# Entrenadores abiertos en otro puerto mientras el del puerto principal entrena, para preparar
# trabajos de la cola: [{"id", "port", "pid"}]. Solo cuentan los que siguen vivos y contestan.
# Trainers opened on another port while the main one trains, to prepare queue jobs.
INSTANCES_FILE = BASE_DIR / "settings" / "trainer_instances.json"


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def extra_instances():
    try:
        data = json.loads(INSTANCES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = []
    alive = [i for i in data if isinstance(i, dict) and _pid_alive(int(i.get("pid") or 0)) and _port_open(i.get("port") or 0)]
    if len(alive) != len(data):
        # Un borrador cerrado ya no necesita su copia de ajustes (lo que se añadió a la cola tiene la suya).
        for gone in (i for i in data if isinstance(i, dict) and i not in alive and i.get("draft")):
            shutil.rmtree(DRAFTS_DIR / f"{gone.get('id')}-{gone.get('port')}", ignore_errors=True)
        # Editor de un trabajo cerrado sin guardar: el trabajo vuelve a como estaba y puede arrancar.
        for gone in (i for i in data if isinstance(i, dict) and i not in alive and i.get("edit")):
            QUEUE.end_edit(gone["edit"], save=False)
        _save_instances(alive)
    return alive


def _save_instances(items):
    INSTANCES_FILE.parent.mkdir(exist_ok=True)
    INSTANCES_FILE.write_text(json.dumps(items), encoding="utf-8")


def url_for_port(port):
    # Sin proxy delante: la misma dirección con la que se abrió el lanzador, con otro puerto.
    return f"http://{request.host.rsplit(':', 1)[0]}:{port}"


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


def _run_script(trainer):
    """(ruta del .bat / .sh que abre el entrenador, None) o (None, error)."""
    if os.name == "nt":
        run = (CODE_DIR / trainer["run"]).resolve()
        if run.parent != CODE_DIR or run.suffix.lower() != ".bat" or not run.is_file():
            return None, t("Not found: {name}", name=trainer["run"])
    else:
        # En Linux se usa el .sh con el mismo nombre / On Linux use the same-named .sh
        run = (CODE_DIR / Path(trainer["run"]).with_suffix(".sh")).resolve()
        if run.parent != CODE_DIR or run.suffix.lower() != ".sh" or not run.is_file():
            return None, t("Not found: {name}", name=run.name)
    return run, None


@app.route("/api/launch", methods=["POST"])
def launch():
    req = request.get_json(force=True) or {}
    trainer_id = str(req.get("id", ""))
    force = bool(req.get("force"))
    trainer = next((t for t in load_config()["trainers"] if t["id"] == trainer_id), None)
    if trainer is None:
        abort(404)

    run, error = _run_script(trainer)
    if error:
        return jsonify({"status": "error", "error": error}), 404

    # Borrador: el mismo entrenador en otro puerto con su propia copia de los ajustes, para preparar otro
    # entrenamiento del mismo modelo mientras este entrena (el que entrena relee los ajustes normales).
    # Draft: the same trainer on another port with its own settings copy, to prepare another training.
    if req.get("draft"):
        extra = next((i for i in extra_instances() if i["id"] == trainer_id and i.get("draft")), None)
        if extra:
            return jsonify({"status": "ok", "url": url_for_port(extra["port"]), "port": extra["port"], "already": True})
        return launch_second(trainer_id, run, draft=req.get("settings") or {})

    # Ya abierto en otro puerto (para preparar un trabajo de la cola): volver a él.
    # Already open on another port (to prepare a queue job): jump back to it.
    extra = next((i for i in extra_instances() if i["id"] == trainer_id and not i.get("edit")), None)
    if extra:
        return jsonify({"status": "ok", "url": url_for_port(extra["port"]), "already": True})
    if req.get("second"):
        return launch_second(trainer_id, run)

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
            # entrenamiento, y lo avanzado desde el último checkpoint. Se ofrece abrir el otro en
            # un puerto distinto, para preparar un trabajo y añadirlo a la cola.
            # The training script is a child of the server: closing it now would cut the
            # training. Offer to open the other one on another port, to prepare a queue job.
            return jsonify({"status": "queue_offer", "current": current, "url": trainer_url()})
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


DRAFTS_DIR = BASE_DIR / "settings" / "queue" / "drafts"


@app.route("/api/queue-edit/<jid>", methods=["POST"])
def queue_edit(jid):
    """Editar un trabajo en espera: su entrenador en otro puerto, con los ajustes del trabajo."""
    extra = next((i for i in extra_instances() if i.get("edit") == jid), None)
    if extra:
        return jsonify({"status": "ok", "url": url_for_port(extra["port"]), "port": extra["port"], "already": True})
    job, error = QUEUE.begin_edit(jid)
    if error:
        return jsonify({"status": "error", "error": error}), 409
    trainer = next((x for x in load_config()["trainers"] if x["id"] == job["trainer"]), None)
    run, error = _run_script(trainer) if trainer else (None, t("Not found: {name}", name=job["trainer"]))
    if error:
        QUEUE.end_edit(jid, save=False)
        return jsonify({"status": "error", "error": error}), 404
    resp = launch_second(job["trainer"], run, edit=jid)
    if resp.get_json().get("status") != "ok":
        QUEUE.end_edit(jid, save=False)
    return resp


@app.route("/api/queue-edit/<jid>/done", methods=["POST"])
def queue_edit_done(jid):
    """Guardar o cancelar la edición: el trabajo vuelve a poder arrancar y su editor se cierra."""
    error = QUEUE.end_edit(jid, save=bool((request.get_json(silent=True) or {}).get("save")))
    editor = next((i for i in extra_instances() if i.get("edit") == jid), None)
    if editor:
        _save_instances([i for i in extra_instances() if i is not editor and i.get("port") != editor["port"]])
        # Después de contestar: la petición llega a través del propio editor.
        threading.Timer(1.5, _kill_pid, args=(int(editor["pid"]),)).start()
    if error:
        return jsonify({"status": "error", "error": error}), 404
    return jsonify({"status": "ok"})


def launch_second(trainer_id, run, draft=None, edit=None):
    """Abre el entrenador en un puerto libre (trainer_port + 1...), con su consola, sin cerrar el del
    puerto principal, que está entrenando. Lo que se prepare ahí va a la cola.
    draft (dict con "train" / "pre_cache"): borrador con su propia copia de los ajustes, que parte de esos
    valores; lo que se guarde en él no toca los ajustes normales del entrenador.
    edit (id de un trabajo en espera): editor de ese trabajo, que guarda en su copia de los ajustes."""
    port = QUEUE._free_port(exclude={i["port"] for i in extra_instances()})
    if port is None:
        return jsonify({"status": "error", "error": t("No free port for the job.")}), 409
    gpu_stats.apply_saved_gpu(os.environ)
    env = dict(os.environ)
    env.update(LORALAB_SKIP_BROWSER="1", LORALAB_TRAINER_PORT=str(port))
    if draft is not None:
        folder = DRAFTS_DIR / f"{trainer_id}-{port}"
        folder.mkdir(parents=True, exist_ok=True)
        for key, name in (("train", f"train_settings_{trainer_id}.json"), ("pre_cache", f"pre_cache_settings_{trainer_id}.json")):
            data = draft.get(key) if isinstance(draft.get(key), dict) else None
            if data is None:
                src = BASE_DIR / "settings" / name
                data = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {}
            (folder / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        env.update(LORALAB_DRAFT="1",
                   LORALAB_TRAIN_CONFIG=str(folder / f"train_settings_{trainer_id}.json"),
                   LORALAB_PRECACHE_CONFIG=str(folder / f"pre_cache_settings_{trainer_id}.json"))
    elif edit is not None:
        folder = job_queue.JOBS_DIR / edit
        env.update(LORALAB_EDIT_JOB=edit,
                   LORALAB_TRAIN_CONFIG=str(folder / f"train_settings_{trainer_id}.json"),
                   LORALAB_PRECACHE_CONFIG=str(folder / f"pre_cache_settings_{trainer_id}.json"))
    if os.name == "nt":
        proc = subprocess.Popen(["cmd", "/c", str(run)], cwd=str(BASE_DIR), env=env,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        pid = proc.pid
    else:
        pid = _launch_linux(run, env)
    _save_instances(extra_instances() + [{"id": trainer_id, "port": port, "pid": pid, "draft": draft is not None,
                                          **({"edit": edit} if edit else {})}])
    if edit:
        QUEUE.begin_edit(edit, port)
    return jsonify({"status": "ok", "url": url_for_port(port), "port": port, "second": True})


@app.route("/api/status")
def status():
    # ?port=N: ¿contesta ya el entrenador abierto en ese puerto? (espera tras abrir uno en otro puerto)
    port = request.args.get("port", type=int)
    if port:
        return jsonify({"trainer": _port_open(port)})
    running = trainer_running()
    q = QUEUE.snapshot()
    job = q["running"]
    by_trainer = {}
    waiting = [j for j in q["pending"] if j["state"] != "failed"]   # un fallido no arranca solo
    for j in waiting:
        by_trainer[j["trainer"]] = by_trainer.get(j["trainer"], 0) + 1
    return jsonify({"trainer": running, "active": active_running() if running else None,
                    "url": trainer_url(),
                    "instances": [{"id": i["id"], "port": i["port"]} for i in extra_instances()],
                    "queue": {"running": {k: job.get(k) for k in ("id", "trainer", "trainer_name", "project", "kind",
                                                                  "progress", "port")} if job else None,
                              "pending": len(waiting), "by_trainer": by_trainer, "paused": q["paused"]}})


# ------------------------------------------------------------------ pausar todo y salir
# Desde la interfaz, sin tocar ventanas: pausa la cola y el trabajo en curso (con checkpoint), detiene lo
# que siga ejecutándose, cierra todos los entrenadores con sus consolas y, en este PC, también el lanzador.
# En la nube (RunPod...) o desde otro equipo el lanzador se queda abierto: cerrado ya no se podría volver.
# Pause all and exit, from the UI: pauses the queue and the running job, closes every trainer and, on
# this PC, the launcher too. In the cloud or from another device the launcher stays open.
SHUTDOWN = {"state": None, "step": "", "keep_launcher": False}
PAUSE_TIMEOUT = 300   # segundos para que el trabajo en curso guarde su checkpoint
STOP_TIMEOUT = 60


def in_cloud():
    return bool(os.environ.get("RUNPOD_POD_ID") or os.environ.get("LORALAB_CLOUD"))


def _trainer_ports():
    """Puertos de todos los entrenadores abiertos: el principal, los de otros puertos y los de trabajos."""
    ports = {int(remote_access.load()["trainer_port"])}
    ports |= {int(i["port"]) for i in extra_instances()}
    with QUEUE._lock:
        ports |= {int(j["port"]) for j in QUEUE.data["jobs"] if j.get("port")}
    return sorted(p for p in ports if _port_open(p))


def _shutdown(keep_launcher):
    def step(text):
        SHUTDOWN["step"] = text
        print(f"[SHUTDOWN] {text}", flush=True)

    step(t("Pausing the queue..."))
    QUEUE.set_options(paused=True)
    job = QUEUE.running()
    if job:
        step(t("Pausing {name} and saving its checkpoint...", name=f"{job['trainer_name']} · {job['project'] or '-'}"))
        QUEUE.pause(job["id"])
        end = time.time() + PAUSE_TIMEOUT
        while QUEUE.running() and time.time() < end:
            time.sleep(1)
    # Lo que aún se ejecute fuera de la cola (unos subtítulos, por ejemplo) se detiene igual que con Stop.
    busy = [p for p in _trainer_ports() if (job_queue._http(p, "/api/status")[1] or {}).get("running")]
    if busy:
        step(t("Stopping what is still running..."))
        for p in busy:
            job_queue._http(p, "/api/stop", {})
        end = time.time() + STOP_TIMEOUT
        while time.time() < end and any((job_queue._http(p, "/api/status")[1] or {}).get("running") for p in busy):
            time.sleep(1)

    step(t("Closing the trainers..."))
    QUEUE._reap_idle_servers()                         # servidores de los trabajos (sin ventana)
    for inst in extra_instances():                     # abiertos en otros puertos, con su consola
        _kill_pid(int(inst["pid"]))
    _save_instances([])
    shutil.rmtree(DRAFTS_DIR, ignore_errors=True)      # borradores: lo añadido a la cola tiene su copia
    if trainer_running():
        kill_trainer()                                 # el del puerto principal, con su consola

    if keep_launcher:
        SHUTDOWN.update(state="done", step=t("Everything is paused and closed. The launcher stays available."))
        return
    SHUTDOWN.update(state="done", step=t("Everything is closed. You can close this tab."))
    time.sleep(2.5)                                    # la página llega a leer el último paso
    _close_own_console()


def _close_own_console():
    """Cierra el lanzador y, en Windows, la consola de Start_LoRAlab (si no, se quedaría en «pause»)."""
    if os.name == "nt":
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            for parent in proc.parents():
                if parent.name().lower() == "cmd.exe":
                    subprocess.run(["taskkill", "/PID", str(parent.pid), "/T", "/F"], capture_output=True)
                    break
        except Exception:
            pass
    os._exit(0)


@app.route("/api/shutdown", methods=["GET", "POST"])
def shutdown():
    if request.method == "POST" and SHUTDOWN["state"] != "running":
        keep = in_cloud() or not remote_access.is_local()
        SHUTDOWN.update(state="running", step="", keep_launcher=keep)
        threading.Thread(target=_shutdown, args=(keep,), daemon=True).start()
    return jsonify({"status": "ok", **SHUTDOWN, "cloud": in_cloud(), "local": remote_access.is_local()})


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
