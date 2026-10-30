# -*- coding: utf-8 -*-
"""
queue_link.py — Enlace de cada entrenador con la cola de trabajos del lanzador (ver job_queue.py).
Each trainer's link with the launcher's job queue.

register(app, trainer_id, runtime, train_config, precache_config) en cada server_*.py:

- Envuelve su /api/run. Con la GPU ocupada por un trabajo de otro puerto, pre-caché y entrenamiento
  no arrancan (409 "queue"): la página ofrece añadirlo a la cola. Con la GPU libre arrancan como
  siempre y se registran en la cola como trabajo externo, para verlos allí.
- Un servidor abierto por la cola para un trabajo (LORALAB_JOB_ID) solo arranca cuando se lo pide la
  cola (cabecera X-LoRAlab-Queue); su página lleva un aviso y se controla desde la cola.
- Sin lanzador (entrenador abierto con su .bat sin el lanzador) todo funciona como antes.
- /api/queue-gate dice a la página qué hacer al pulsar Start, /api/queue-add añade un trabajo con una
  copia de los ajustes que este entrenador tiene guardados (los suyos, o la copia de su trabajo) y
  /api/queue-info alimenta el botón «Cola» y el aviso de la página. Sirve también /queue_client.js.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from flask import jsonify, request, send_from_directory

import remote_access
from i18n import t

GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"
KINDS = ("precache", "train")
JOB_ID = os.environ.get("LORALAB_JOB_ID") or None


def _launcher(path, payload=None, timeout=2):
    """Llamada al lanzador en este PC (las locales no piden contraseña). (código, json) o (None, None)."""
    port = remote_access.load()["launcher_port"]
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
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


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def register(app, trainer_id, runtime, train_config, precache_config):
    my_port = remote_access.port_of("trainer_port")

    def queue_state():
        code, st = _launcher("/api/queue")
        return st if code == 200 else None

    def busy_elsewhere(st):
        running = (st or {}).get("running")
        return running if running and running.get("port") != my_port else None

    def own_job(st):
        if not JOB_ID or not st:
            return None
        jobs = ([st["running"]] if st.get("running") else []) + st.get("pending", []) + st.get("history", [])
        return next((j for j in jobs if j and j.get("id") == JOB_ID), None)

    # ---------------------------------------------------------- puerta de /api/run
    endpoint = next(r.endpoint for r in app.url_map.iter_rules() if r.rule == "/api/run")
    original_run = app.view_functions[endpoint]

    def gated_run():
        data = request.get_json(silent=True) or {}
        script = data.get("script")
        if script not in KINDS:
            return original_run()
        if JOB_ID:
            if request.headers.get("X-LoRAlab-Queue") != "1":
                return jsonify({"status": "job", "error": t("This trainer runs a queue job: pause or resume it from the queue.")}), 409
            return original_run()
        st = queue_state()
        if busy_elsewhere(st):
            return jsonify({"status": "queue", "running": st["running"],
                            "error": t("Another job is training. Add this one to the queue.")}), 409
        resp = original_run()
        status = resp[1] if isinstance(resp, tuple) else getattr(resp, "status_code", 200)
        if st is not None and status == 200:
            train, pre = _read(train_config), _read(precache_config)
            _launcher("/api/queue/register", {
                "trainer": trainer_id, "kind": script, "port": my_port,
                "project": (train.get("project_name") or pre.get("project_name") or "").strip(),
                "total": int(train.get("total_steps") or 0) if script == "train" else 0})
        return resp

    app.view_functions[endpoint] = gated_run

    # -------------------------------------------------------------- rutas nuevas
    @app.route("/queue_client.js")
    def queue_client_js():
        return send_from_directory(str(GUI_DIR), "queue_client.js")

    @app.route("/api/queue-gate", methods=["POST"])
    def queue_gate():
        script = (request.get_json(force=True) or {}).get("script")
        if JOB_ID and script in KINDS:
            return jsonify({"mode": "job", "job_id": JOB_ID})
        st = queue_state()
        if st is None:
            return jsonify({"mode": "direct", "launcher": False})
        running = busy_elsewhere(st)
        if running and script in KINDS:
            return jsonify({"mode": "queue", "running": running})
        if running and script == "caption":
            return jsonify({"mode": "caption_busy", "running": running})
        return jsonify({"mode": "direct", "launcher": True})

    @app.route("/api/queue-add", methods=["POST"])
    def queue_add():
        kind = (request.get_json(force=True) or {}).get("kind")
        if kind not in KINDS:
            return jsonify({"status": "error", "error": t("Unknown job type: {name}", name=kind)}), 400
        code, res = _launcher("/api/queue/add", {
            "trainer": trainer_id, "kind": kind,
            "settings": {"train": _read(train_config), "pre_cache": _read(precache_config)}}, timeout=5)
        if code is None:
            return jsonify({"status": "error", "error": t("The launcher is not running: the queue needs it.")}), 503
        return jsonify(res), code

    @app.route("/api/queue-info")
    def queue_info():
        st = queue_state()
        if st is None:
            return jsonify({"launcher": False, "job_id": JOB_ID})
        pending = st.get("pending", [])
        return jsonify({"launcher": True, "job_id": JOB_ID, "job": own_job(st), "port": my_port,
                        "running": st.get("running"), "queued": len(pending)})
