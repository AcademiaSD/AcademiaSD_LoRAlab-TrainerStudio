# -*- coding: utf-8 -*-
"""
job_queue.py — Cola de trabajos (pre-caché y entrenamiento) de todos los entrenadores.
Job queue (pre-cache and training) shared by every trainer.

Vive en el lanzador, que es la única fuente de verdad: la GPU tiene como mucho un trabajo a la vez.

- settings/queue.json guarda la cola: orden, estados y opciones. settings/queue/<id>/ guarda la copia
  de los ajustes de cada trabajo (train_settings_<entrenador>.json y pre_cache_settings_...json),
  tomada al añadirlo: los ajustes que recuerda cada entrenador no se tocan nunca.
- Cada trabajo se ejecuta en su propio servidor del entrenador, abierto sin ventana en un puerto libre
  (trainer_port + 1, + 2...) con LORALAB_TRAINER_PORT y LORALAB_TRAIN_CONFIG / LORALAB_PRECACHE_CONFIG,
  así que conserva toda su interfaz (consola, gráfica, previews, ajustes en caliente) en ese puerto.
- Un entrenamiento lanzado directamente desde un entrenador con la GPU libre se registra como trabajo
  «externo» (register_external): corre en ese servidor, con sus ajustes, y la cola solo lo sigue.
- Un hilo sigue el trabajo en curso cada POLL segundos (estado y progreso de su servidor), decide cómo
  acabó y lanza el siguiente. Pausar = detener guardando checkpoint (el script ya lo hace con Stop);
  continuar = volver a lanzarlo: el script reanuda solo desde ese checkpoint.

Estados: queued -> running -> done | failed, y paused (se salta hasta continuar). Al reiniciar el
lanzador, un trabajo «running» cuyo servidor ya no contesta pasa a paused con la nota «interrupted».
Un trabajo externo detenido a medias pasa a stopped: se reanuda desde su entrenador, no desde la cola.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import gpu_stats
import remote_access
from i18n import t

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_DIR = BASE_DIR / "settings"
SCRIPTS_DIR = Path(__file__).resolve().parent
QUEUE_FILE = SETTINGS_DIR / "queue.json"
JOBS_DIR = SETTINGS_DIR / "queue"

KINDS = ("precache", "train")
ACTIVE = ("queued", "running", "paused")
HISTORY_MAX = 50
POLL = 2.0
SAVE_EVERY = 10.0        # mientras entrena, el progreso se guarda en disco como mucho cada 10 s
START_TIMEOUT = 120      # segundos que se espera a que el servidor de un trabajo conteste
LOST_TICKS = 5           # sondeos sin respuesta antes de dar el trabajo por interrumpido
PORT_SPAN = 100          # puertos para trabajos: trainer_port + 1 ... + PORT_SPAN


def _settings_names(trainer_id):
    return {"train": f"train_settings_{trainer_id}.json", "pre_cache": f"pre_cache_settings_{trainer_id}.json"}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _http(port, path, payload=None, timeout=3):
    """GET (payload None) o POST JSON a un servidor local. Devuelve (código, json) o (None, None)."""
    url = f"http://127.0.0.1:{port}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    # X-LoRAlab-Queue: el servidor de un trabajo solo arranca cuando se lo pide la cola (queue_link.py).
    headers = {"X-LoRAlab-Queue": "1", **({"Content-Type": "application/json"} if data else {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace") or "{}")
        except ValueError:
            return exc.code, {}
    except (OSError, ValueError):
        return None, None


def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True).stdout
        return str(int(pid)) in out
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _log_tail(job, lines=3):
    """Últimas líneas del registro del servidor de un trabajo, para explicar un fallo al arrancar."""
    try:
        text = (JOBS_DIR / job["id"] / "server.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " | ".join(l.strip() for l in text.strip().splitlines()[-lines:])


def _kill_tree(pid):
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass


class JobQueue:
    def __init__(self, load_trainers):
        self.load_trainers = load_trainers      # función: lista de entrenadores de launcher.json
        self._lock = threading.RLock()
        self.data = {"paused": False, "stop_on_failure": False, "jobs": []}
        self.data.update(_read_json(QUEUE_FILE))
        self._thread = None
        self._saved_at = 0.0

    # ------------------------------------------------------------------ datos
    def _save(self):
        _write_json(QUEUE_FILE, self.data)
        self._saved_at = time.time()

    def _trainer(self, trainer_id):
        return next((x for x in self.load_trainers() if x["id"] == trainer_id), None)

    def _job(self, jid):
        return next((j for j in self.data["jobs"] if j["id"] == jid), None)

    def running(self):
        with self._lock:
            return next((dict(j) for j in self.data["jobs"] if j["state"] == "running"), None)

    def snapshot(self):
        """La cola para la interfaz: el trabajo en curso, los pendientes en orden y el historial."""
        with self._lock:
            jobs = self.data["jobs"]
            pending = [j for j in jobs if j["state"] in ("queued", "paused")]
            history = sorted((j for j in jobs if j["state"] not in ACTIVE),
                             key=lambda j: j.get("finished") or 0, reverse=True)
            return {"paused": self.data["paused"], "stop_on_failure": self.data["stop_on_failure"],
                    "running": next((j for j in jobs if j["state"] == "running"), None),
                    "pending": pending, "history": history,
                    "trainer_port": remote_access.load()["trainer_port"]}

    # ------------------------------------------------------------- operaciones
    def add(self, trainer_id, kind, settings=None):
        """Añade un trabajo con una copia de los ajustes (los que manda el entrenador o, si no manda
        nada, los que tiene guardados). Devuelve (job, None) o (None, error)."""
        trainer = self._trainer(trainer_id)
        if trainer is None:
            return None, t("Not found: {name}", name=trainer_id)
        if kind not in KINDS:
            return None, t("Unknown job type: {name}", name=kind)
        names = _settings_names(trainer_id)
        settings = settings or {}
        copies = {key: (settings.get(key) if isinstance(settings.get(key), dict) else _read_json(SETTINGS_DIR / name))
                  for key, name in names.items()}
        train, pre = copies["train"], copies["pre_cache"]
        project = (train.get("project_name") or pre.get("project_name") or "").strip()
        with self._lock:
            for j in self.data["jobs"]:
                if (j["state"] in ACTIVE and j["trainer"] == trainer_id and j["project"] == project
                        and j["kind"] == kind):
                    return None, t("{trainer} / {project} is already in the queue.",
                                   trainer=trainer["name"], project=project or "-")
            jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            job_dir = JOBS_DIR / jid
            for key, name in names.items():
                _write_json(job_dir / name, copies[key])
            total = int(train.get("total_steps") or 0) if kind == "train" else 0
            job = {"id": jid, "trainer": trainer_id, "trainer_name": trainer["name"], "kind": kind,
                   "project": project, "state": "queued", "external": False, "created": time.time(),
                   "started": None, "finished": None, "port": None, "pid": None, "code": None,
                   "progress": {"step": 0, "total": total} if total else None, "note": "",
                   "phase": None, "requested": None}
            self.data["jobs"].append(job)
            self._save()
            return dict(job), None

    def register_external(self, trainer_id, kind, port, project="", total=0):
        """Un entrenador lanzó un trabajo directamente (GPU libre): la cola lo sigue como uno más."""
        trainer = self._trainer(trainer_id)
        if trainer is None or kind not in KINDS:
            return None, t("Not found: {name}", name=trainer_id)
        with self._lock:
            if self.running():
                return None, t("The GPU is busy with another job.")
            jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            job = {"id": jid, "trainer": trainer_id, "trainer_name": trainer["name"], "kind": kind,
                   "project": project, "state": "running", "external": True, "created": time.time(),
                   "started": time.time(), "finished": None, "port": int(port), "pid": None, "code": None,
                   "progress": {"step": 0, "total": int(total)} if total else None, "note": "",
                   "phase": "run", "requested": None, "lost": 0}
            self.data["jobs"].append(job)
            self._save()
            return dict(job), None

    def pause(self, jid):
        with self._lock:
            job = self._job(jid)
            if job is None:
                return t("Job not found.")
            if job["state"] == "queued":
                job["state"] = "paused"
            elif job["state"] == "running":
                job["requested"] = "pause"
                port = job["port"]
            self._save()
        if job["state"] == "running":
            _http(port, "/api/stop", {})
        return None

    def resume(self, jid):
        with self._lock:
            job = self._job(jid)
            if job is None:
                return t("Job not found.")
            if job["state"] in ("paused", "failed") and not job["external"]:
                # Un pausado conserva su sitio; un fallido sale del historial al final de los pendientes.
                if job["state"] == "failed":
                    self.data["jobs"].remove(job)
                    self.data["jobs"].append(job)
                job.update(state="queued", note="", code=None, finished=None)
                self._save()
            return None

    def remove(self, jid):
        with self._lock:
            job = self._job(jid)
            if job is None:
                return t("Job not found.")
            if job["state"] == "running":
                job["requested"] = "remove"
                self._save()
                port = job["port"]
            else:
                self._drop(job)
                self._save()
                return None
        _http(port, "/api/stop", {})
        return None

    def move(self, jid, index):
        """Coloca un pendiente en la posición index de la lista de pendientes (0 = el siguiente)."""
        with self._lock:
            job = self._job(jid)
            if job is None or job["state"] not in ("queued", "paused"):
                return t("Job not found.")
            pending = [j for j in self.data["jobs"] if j["state"] in ("queued", "paused")]
            pending.remove(job)
            index = max(0, min(int(index), len(pending)))
            pending.insert(index, job)
            others = [j for j in self.data["jobs"] if j["state"] not in ("queued", "paused")]
            self.data["jobs"] = others + pending
            self._save()
            return None

    def set_options(self, paused=None, stop_on_failure=None):
        with self._lock:
            if paused is not None:
                self.data["paused"] = bool(paused)
            if stop_on_failure is not None:
                self.data["stop_on_failure"] = bool(stop_on_failure)
            self._save()

    def clear_history(self):
        with self._lock:
            for job in [j for j in self.data["jobs"] if j["state"] not in ACTIVE]:
                self._drop(job)
            self._save()

    def _drop(self, job):
        if job in self.data["jobs"]:
            self.data["jobs"].remove(job)
        shutil.rmtree(JOBS_DIR / job["id"], ignore_errors=True)

    # --------------------------------------------------------------- ejecución
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="job-queue", daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            try:
                self.tick()
            except Exception as exc:          # el bucle no puede morir por un fallo puntual
                print(f"[QUEUE] {exc}", flush=True)
            time.sleep(POLL)

    def tick(self):
        with self._lock:
            running = next((j for j in self.data["jobs"] if j["state"] == "running"), None)
        if running is not None:
            self._follow(running)
            return
        with self._lock:
            if self.data["paused"]:
                return
            nxt = next((j for j in self.data["jobs"] if j["state"] == "queued"), None)
        if nxt is not None:
            self._launch(nxt)

    def _free_port(self, exclude=()):
        """Primer puerto libre desde trainer_port + 1, sin los de otros trabajos ni los de exclude."""
        base = int(remote_access.load()["trainer_port"])
        with self._lock:
            taken = {j["port"] for j in self.data["jobs"] if j.get("port")} | set(exclude)
        for port in range(base + 1, base + 1 + PORT_SPAN):
            if port not in taken and not _port_in_use(port):
                return port
        return None

    def _launch(self, job):
        """Abre el servidor del trabajo sin ventana; _follow le manda la orden de arrancar al contestar."""
        self._reap_idle_servers()
        port = self._free_port()
        if port is None:
            self._finish(job, "failed", note=t("No free port for the job."))
            return
        script = SCRIPTS_DIR / f"server_{job['trainer']}.py"
        job_dir = JOBS_DIR / job["id"]
        names = _settings_names(job["trainer"])
        env = dict(os.environ)
        env.update(LORALAB_SKIP_BROWSER="1", LORALAB_TRAINER_PORT=str(port), LORALAB_JOB_ID=job["id"],
                   LORALAB_TRAIN_CONFIG=str(job_dir / names["train"]),
                   LORALAB_PRECACHE_CONFIG=str(job_dir / names["pre_cache"]))
        gpu_stats.apply_saved_gpu(env)
        log = open(job_dir / "server.log", "a", encoding="utf-8")
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=str(BASE_DIR), env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=flags, start_new_session=(os.name != "nt"))
        except OSError as exc:
            self._finish(job, "failed", note=str(exc))
            return
        finally:
            log.close()
        with self._lock:
            job.update(state="running", port=port, pid=proc.pid, started=time.time(), finished=None,
                       code=None, note="", phase="boot", lost=0, requested=None)
            self._save()

    def _follow(self, job):
        code, st = _http(job["port"], "/api/status")
        with self._lock:
            if st is None:
                # El servidor aún arranca, o se ha cerrado (consola cerrada, PC reiniciado...).
                if job["phase"] == "boot":
                    if not job["external"] and not _pid_alive(job["pid"]):
                        # Murió al arrancar: el final de su registro dice por qué.
                        self._finish(job, "failed", note=t("Its trainer could not start.") + " " + _log_tail(job))
                        return
                    if time.time() - (job["started"] or 0) < START_TIMEOUT:
                        return
                job["lost"] = job.get("lost", 0) + 1
                if job["lost"] >= LOST_TICKS:
                    requested = job.get("requested")
                    if requested == "remove":
                        self._drop(job)
                        self._save()
                    else:
                        self._finish(job, "paused", note=t("Interrupted: its trainer stopped answering."))
                return
            job["lost"] = 0
            if st.get("progress"):
                job["progress"] = st["progress"]
            phase, requested = job["phase"], job.get("requested")
        if phase == "boot":
            if requested in ("pause", "remove"):
                # Pausado o quitado antes de llegar a arrancar: no se lanza nada.
                self._end_without_run(job, requested)
                return
            if st.get("encoding"):
                return                        # codificando el prompt de la preview: se reintenta
            code, res = _http(job["port"], "/api/run", {"script": job["kind"]})
            if code == 200:
                with self._lock:
                    job["phase"] = "run"
                    self._save()
            else:
                self._finish(job, "failed", note=(res or {}).get("error") or t("The trainer did not start the job."))
            return
        if st.get("running"):
            with self._lock:
                if time.time() - self._saved_at >= SAVE_EVERY:
                    self._save()
            return
        last = st.get("last")
        if last is None:
            return                            # el proceso acaba de salir: el código llega en el siguiente sondeo
        self._conclude(job, int(last.get("code") or 0), st.get("progress"))

    def _end_without_run(self, job, requested):
        with self._lock:
            if requested == "remove":
                self._drop(job)
                self._save()
            else:
                self._finish(job, "paused")
        self._stop_server(job)

    def _conclude(self, job, code, progress):
        requested = job.get("requested")
        if requested == "remove":
            with self._lock:
                self._drop(job)
                self._save()
            return
        if requested == "pause":
            self._finish(job, "paused", code=code)
        elif code != 0:
            self._finish(job, "failed", code=code, note=t("Exited with code {code}.", code=code))
            if self.data["stop_on_failure"]:
                self.set_options(paused=True)
        elif job["kind"] == "train" and progress and progress["step"] < progress["total"]:
            # Detenido desde su entrenador antes del final: queda pausado, con su checkpoint.
            self._finish(job, "paused", code=code, note=t("Stopped from its trainer."))
        else:
            self._finish(job, "done", code=code)

    def _finish(self, job, state, code=None, note=""):
        with self._lock:
            if state == "paused" and job["external"]:
                state = "stopped"             # se reanuda desde su entrenador, no desde la cola
            job.update(state=state, finished=None if state == "paused" else time.time(), code=code,
                       requested=None, phase=None, note=note or job.get("note", ""))
            history = sorted((j for j in self.data["jobs"] if j["state"] not in ACTIVE),
                             key=lambda j: j.get("finished") or 0)
            while len(history) > HISTORY_MAX:
                self._drop(history.pop(0))
            self._save()

    def _stop_server(self, job):
        if not job.get("external") and job.get("pid"):
            _kill_tree(job["pid"])

    def _reap_idle_servers(self):
        """Cierra los servidores de trabajos ya terminados para liberar memoria y puertos. El último
        se deja abierto hasta que arranca el siguiente, para poder mirar su consola y sus previews."""
        with self._lock:
            ended = [j for j in self.data["jobs"]
                     if j["state"] not in ("running",) and not j["external"] and j.get("pid")]
        for job in ended:
            code, st = _http(job["port"], "/api/status") if job.get("port") else (None, None)
            if st is not None and st.get("running"):
                continue                      # alguien lanzó algo a mano en ese servidor: no se toca
            _kill_tree(job["pid"])
            with self._lock:
                job.update(pid=None, port=None)
                self._save()


def register(app, load_trainers):
    """Rutas /api/queue... del lanzador y arranque del hilo de la cola. Devuelve la JobQueue."""
    from flask import jsonify, request

    jq = JobQueue(load_trainers)

    def reply(err):
        if err:
            return jsonify({"status": "error", "error": err}), 400
        return jsonify({"status": "ok", **jq.snapshot()})

    @app.route("/api/queue")
    def queue_state():
        return jsonify({"status": "ok", **jq.snapshot()})

    @app.route("/api/queue/add", methods=["POST"])
    def queue_add():
        req = request.get_json(force=True) or {}
        job, err = jq.add(str(req.get("trainer", "")), str(req.get("kind", "")), req.get("settings"))
        if err:
            return jsonify({"status": "error", "error": err}), 400
        return jsonify({"status": "ok", "job": job, **jq.snapshot()})

    @app.route("/api/queue/register", methods=["POST"])
    def queue_register():
        req = request.get_json(force=True) or {}
        job, err = jq.register_external(str(req.get("trainer", "")), str(req.get("kind", "")),
                                        int(req.get("port") or 0), str(req.get("project", "")),
                                        int(req.get("total") or 0))
        if err:
            return jsonify({"status": "error", "error": err}), 409
        return jsonify({"status": "ok", "job": job})

    @app.route("/api/queue/<jid>/<action>", methods=["POST"])
    def queue_action(jid, action):
        ops = {"pause": jq.pause, "resume": jq.resume, "remove": jq.remove}
        if action == "move":
            return reply(jq.move(jid, (request.get_json(force=True) or {}).get("index", 0)))
        if action not in ops:
            return jsonify({"status": "error", "error": t("Unknown action: {name}", name=action)}), 404
        return reply(ops[action](jid))

    @app.route("/api/queue/options", methods=["POST"])
    def queue_options():
        req = request.get_json(force=True) or {}
        jq.set_options(req.get("paused"), req.get("stop_on_failure"))
        return reply(None)

    @app.route("/api/queue/clear-history", methods=["POST"])
    def queue_clear():
        jq.clear_history()
        return reply(None)

    @app.route("/queue")
    def queue_page():
        from flask import send_from_directory
        return send_from_directory(str(BASE_DIR / "GUI"), "queue.html")

    @app.route("/api/queue/<jid>/metrics")
    def queue_metrics(jid):
        """La curva de loss del trabajo en curso, pedida a su entrenador (otro puerto) desde aquí."""
        with jq._lock:
            job = jq._job(jid)
            port = job and job.get("port")
        if not port:
            return jsonify({"points": [], "run": None})
        code, res = _http(port, f"/api/metrics?since={request.args.get('since', 0, type=int)}", timeout=4)
        if code != 200 or not res:
            return jsonify({"points": [], "run": None})
        return jsonify({"run": res.get("run"), "points": [{"s": p["s"], "l": p["l"], "dt": p.get("dt")}
                                                          for p in res.get("points", [])]})

    jq.start()
    return jq
