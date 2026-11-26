# -*- coding: utf-8 -*-
"""
queue_link.py — Enlace de cada entrenador con la cola de trabajos del lanzador (ver job_queue.py).
Each trainer's link with the launcher's job queue.

register(app, trainer_id, runtime, train_config, precache_config) en cada server_*.py:

- Envuelve su /api/run. Con la GPU ocupada por un trabajo de otro puerto, pre-caché y entrenamiento
  no arrancan (409 "queue"): la página ofrece añadirlo a la cola. Con la GPU libre arrancan como
  siempre y se registran en la cola como trabajo externo, para verlos allí.
- Un servidor abierto por la cola para un trabajo (LORALAB_JOB_ID) arranca cuando se lo pide la
  cola (cabecera X-LoRAlab-Queue), o desde su página con el trabajo pausado y la GPU libre; si no, se
  controla desde la cola.
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
# Borrador abierto con «Nuevo entrenamiento en cola»: guarda en su propia copia de los ajustes.
DRAFT = os.environ.get("LORALAB_DRAFT") == "1"
# Ese borrador parte de los ajustes de un trabajo del historial (⧉ en la cola).
TEMPLATE = DRAFT and os.environ.get("LORALAB_TEMPLATE") == "1"
# Editor de un trabajo en espera (✎ en la cola): guarda en la copia de ajustes de ese trabajo; Start pasa
# a ser «Guardar cambios en la cola» y aquí nunca se entrena.
EDIT_JOB = os.environ.get("LORALAB_EDIT_JOB") or None


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


# ------------------------------------------------------------ estado de la pre-caché
# Start Training avisa si la pre-caché de este proyecto no existe o es anterior a los últimos cambios del
# dataset (imágenes, clips, audio o captions), y ofrece hacerla antes de entrenar (queue_client.js).
# Start Training warns when this project's pre-cache is missing or older than the dataset's last changes.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".mp4", ".mov", ".webm",
                ".mkv", ".avi", ".m4v", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".txt", ".json")
# Margen para relojes y sistemas de archivos que guardan la hora con poca resolución (FAT, exFAT: 2 s).
MTIME_SLACK = 2.0
WALK_LIMIT = 50000


def _project_dir(raw, cache=False):
    """Ruta de un ajuste (relativa al proyecto). Las cachés viejas en la raíz se movieron a Cache_Data."""
    raw = str(raw or "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if cache and not path.exists():
        moved = BASE_DIR / "Cache_Data" / path.name
        if moved.exists():
            return moved
    return path


def _walk(root, depth=3):
    """(ruta, mtime) de los archivos bajo root, sin carpetas ocultas y hasta depth niveles."""
    found = 0
    stack = [(Path(root), 0)]
    while stack:
        folder, level = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    if level < depth:
                        stack.append((Path(entry.path), level + 1))
                elif entry.is_file():
                    yield Path(entry.path), entry.stat().st_mtime
                    found += 1
                    if found >= WALK_LIMIT:
                        return
            except OSError:
                continue


def _is_cache_data(path):
    """Lo que escribe la pre-caché por muestra. Fuera: _neg/_custom (también los escribe el servidor al
    guardar el prompt de las previews), la copia de los ajustes y los .tmp a medio escribir."""
    name = path.name
    return not (name.startswith("_") or name.startswith("pre_cache_settings_") or name.endswith(".tmp"))


def cache_status(train_config, precache_config):
    """
    {"state": "ok" | "missing" | "stale" | "unknown", "cache_dir", "dataset", "changed": [nombres],
     "changed_count", "removed": bool, "precache_matches": bool}
    missing: la carpeta de caché no tiene ninguna muestra. stale: algún archivo del dataset es más nuevo
    que la última muestra cacheada, o la carpeta del dataset cambió después (archivos quitados o
    renombrados). precache_matches: los ajustes de la pre-caché apuntan a la misma caché y el mismo
    dataset que el entrenamiento, así que Start Pre-Cache rehace justo la caché que se va a entrenar.
    """
    train, pre = _read(train_config), _read(precache_config)
    cache_dir = _project_dir(train.get("cache_dir") or pre.get("cache_dir"), cache=True)
    dataset = _project_dir(train.get("dataset_path") or pre.get("dataset_path"))
    pre_cache = _project_dir(pre.get("cache_dir"), cache=True)
    pre_dataset = _project_dir(pre.get("dataset_path"))

    def same(a, b):
        return (a is not None and b is not None
                and os.path.normcase(str(a)).rstrip("\\/") == os.path.normcase(str(b)).rstrip("\\/"))

    out = {"state": "unknown", "cache_dir": str(cache_dir or ""), "dataset": str(dataset or ""),
           "changed": [], "changed_count": 0, "removed": False,
           "precache_matches": same(cache_dir, pre_cache) and same(dataset, pre_dataset)}
    if cache_dir is None:
        return out
    newest = max((m for p, m in _walk(cache_dir) if _is_cache_data(p)), default=None) if cache_dir.is_dir() else None
    if newest is None:
        out["state"] = "missing"
        return out
    out["state"] = "ok"
    if dataset is None or not dataset.is_dir():
        return out
    # Solo el primer nivel: es lo que leen las pre-cachés (_rejected de Dataset Forge y demás, fuera).
    changed = sorted((m, p) for p, m in _walk(dataset, depth=0)
                     if p.suffix.lower() in DATASET_EXTS and m > newest + MTIME_SLACK)
    out["changed_count"] = len(changed)
    out["changed"] = [str(p.relative_to(dataset)) for _, p in changed[-5:]][::-1]
    try:
        # Quitar o renombrar un archivo no deja nada más nuevo en el dataset, pero sí cambia su carpeta.
        out["removed"] = not changed and dataset.stat().st_mtime > newest + MTIME_SLACK
    except OSError:
        pass
    if changed or out["removed"]:
        out["state"] = "stale"
    return out


def register(app, trainer_id, runtime, train_config, precache_config):
    my_port = remote_access.port_of("trainer_port")

    def queue_state():
        code, st = _launcher("/api/queue")
        return st if code == 200 else None

    def busy_elsewhere(st):
        running = (st or {}).get("running")
        return running if running and running.get("port") != my_port else None

    def register_here(script):
        """La cola sigue lo que se ha lanzado en esta página (trabajo «externo»). Devuelve el trabajo o None."""
        train, pre = _read(train_config), _read(precache_config)
        code, res = _launcher("/api/queue/register", {
            "trainer": trainer_id, "kind": script, "port": my_port,
            "project": (train.get("project_name") or pre.get("project_name") or "").strip(),
            "total": int(train.get("total_steps") or 0) if script == "train" else 0,
            # Copia de los ajustes: pausado desde la cola, se queda en ella y se puede continuar.
            "settings": {"train": train, "pre_cache": pre}}, timeout=10)
        return (res or {}).get("job") if code == 200 else None

    def own_job(st):
        jid = JOB_ID or EDIT_JOB
        if not jid or not st:
            return None
        jobs = ([st["running"]] if st.get("running") else []) + st.get("pending", []) + st.get("history", [])
        return next((j for j in jobs if j and j.get("id") == jid), None)

    # ---------------------------------------------------------- puerta de /api/run
    endpoint = next(r.endpoint for r in app.url_map.iter_rules() if r.rule == "/api/run")
    original_run = app.view_functions[endpoint]

    def gated_run():
        data = request.get_json(silent=True) or {}
        script = data.get("script")
        if script not in KINDS:
            return original_run()
        if EDIT_JOB:
            return jsonify({"status": "edit", "error": t("This trainer edits a queued job: save the changes to the queue.")}), 409
        if JOB_ID:
            if request.headers.get("X-LoRAlab-Queue") != "1":
                # Start/Resume en su propia página: con el trabajo pausado y la GPU libre arranca aquí mismo,
                # y la cola lo sigue como en marcha. Si no, se controla desde la cola.
                code, res = _launcher(f"/api/queue/{JOB_ID}/claim", {"port": my_port, "kind": script}, timeout=5)
                if code != 200:
                    error = (res or {}).get("error") or t("This trainer runs a queue job: pause or resume it from the queue.")
                    return jsonify({"status": "job", "error": error}), 409
                resp = original_run()
                status = resp[1] if isinstance(resp, tuple) else getattr(resp, "status_code", 200)
                if status != 200:
                    _launcher(f"/api/queue/{JOB_ID}/release", {})
                return resp
            return original_run()
        st = queue_state()
        if busy_elsewhere(st):
            return jsonify({"status": "queue", "running": st["running"],
                            "error": t("Another job is training. Add this one to the queue.")}), 409
        resp = original_run()
        status = resp[1] if isinstance(resp, tuple) else getattr(resp, "status_code", 200)
        if st is not None and status == 200:
            register_here(script)
        return resp

    app.view_functions[endpoint] = gated_run

    # -------------------------------------------------------------- rutas nuevas
    @app.route("/queue_client.js")
    def queue_client_js():
        return send_from_directory(str(GUI_DIR), "queue_client.js")

    @app.route("/number_limits.js")
    def number_limits_js():
        return send_from_directory(str(GUI_DIR), "number_limits.js")

    @app.route("/preview_prompts.js")
    def preview_prompts_js():
        return send_from_directory(str(GUI_DIR), "preview_prompts.js")

    @app.route("/lr_curve.js")
    def lr_curve_js():
        return send_from_directory(str(GUI_DIR), "lr_curve.js")

    @app.route("/optimizer.js")
    def optimizer_js():
        return send_from_directory(str(GUI_DIR), "optimizer.js")

    @app.route("/api/queue-handover", methods=["POST"])
    def queue_handover():
        """La cola arranca aquí el entrenamiento que esperaba a la pre-caché lanzada desde esta página, con
        la copia de ajustes del trabajo, para que se vea en esta misma ventana (ver JobQueue._handover)."""
        data = request.get_json(force=True) or {}
        if request.remote_addr not in ("127.0.0.1", "::1"):
            return jsonify({"status": "error", "error": "local only"}), 403
        if JOB_ID or EDIT_JOB or data.get("script") not in KINDS or runtime.status().get("running"):
            return jsonify({"status": "error", "error": t("The GPU is busy with another job.")}), 409
        settings = data.get("settings") or {}
        for key, path in (("train", train_config), ("pre_cache", precache_config)):
            if isinstance(settings.get(key), dict) and settings[key]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(str(path) + ".tmp")
                tmp.write_text(json.dumps(settings[key], indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
        return original_run()

    @app.route("/api/queue-gate", methods=["POST"])
    def queue_gate():
        script = (request.get_json(force=True) or {}).get("script")
        if EDIT_JOB and script in KINDS:
            return jsonify({"mode": "edit", "job_id": EDIT_JOB})
        if JOB_ID and script in KINDS:
            st = queue_state()
            own = own_job(st)
            if own and own.get("state") in ("paused", "failed") and own.get("kind") == script \
                    and own.get("port") == my_port and not (st or {}).get("running"):
                # Su trabajo está parado y la GPU libre: Start/Resume arranca aquí (ver gated_run).
                return jsonify({"mode": "direct", "launcher": True})
            return jsonify({"mode": "job", "job_id": JOB_ID})
        st = queue_state()
        if st is None:
            return jsonify({"mode": "direct", "launcher": False})
        own = runtime.status()
        if own.get("running") and script in KINDS:
            # Este mismo entrenador está ocupado (p. ej. con la pre-caché): lo nuevo va a la cola y arranca
            # en cuanto acabe, en lugar de esperar a que se libere la página.
            return jsonify({"mode": "queue_self", "script": own.get("script")})
        running = busy_elsewhere(st)
        if running and script in KINDS:
            return jsonify({"mode": "queue", "running": running})
        if running and script == "caption":
            return jsonify({"mode": "caption_busy", "running": running})
        return jsonify({"mode": "direct", "launcher": True})

    @app.route("/api/queue-add", methods=["POST"])
    def queue_add():
        req = request.get_json(force=True) or {}
        kind = req.get("kind")
        if kind not in KINDS:
            return jsonify({"status": "error", "error": t("Unknown job type: {name}", name=kind)}), 400
        after = req.get("after") or None
        if after == "self_precache":
            # El entrenamiento espera a la pre-caché que este entrenador acaba de lanzar (Start sin caché).
            st = queue_state() or {}
            running = st.get("running") or {}
            if (running.get("port") != my_port or running.get("kind") != "precache") and not running:
                own = runtime.status()
                if own.get("running") and own.get("script") == "precache":
                    running = register_here("precache") or {}
            if running.get("port") != my_port or running.get("kind") != "precache":
                return jsonify({"status": "error", "error": t("The pre-cache is not running in the queue.")}), 409
            after = running.get("id")
        code, res = _launcher("/api/queue/add", {
            "trainer": trainer_id, "kind": kind, "after": after,
            "settings": {"train": _read(train_config), "pre_cache": _read(precache_config)}}, timeout=5)
        if code is None:
            return jsonify({"status": "error", "error": t("The launcher is not running: the queue needs it.")}), 503
        return jsonify(res), code

    @app.route("/api/cache-check")
    def cache_check():
        """¿Hay pre-caché para lo que se va a entrenar, y es posterior a los cambios del dataset?"""
        own = runtime.status()
        if own.get("running") and own.get("script") == "precache":
            return jsonify({"state": "running"})
        try:
            return jsonify(cache_status(train_config, precache_config))
        except Exception as exc:              # la comprobación nunca impide entrenar
            return jsonify({"state": "unknown", "error": str(exc)})

    @app.route("/api/queue-info")
    def queue_info():
        st = queue_state()
        if st is None:
            return jsonify({"launcher": False, "job_id": JOB_ID, "draft": DRAFT, "template": TEMPLATE, "edit_job": EDIT_JOB})
        pending = [j for j in st.get("pending", []) if j.get("state") != "failed"]
        own = runtime.status()
        return jsonify({"launcher": True, "job_id": JOB_ID, "draft": DRAFT, "template": TEMPLATE, "edit_job": EDIT_JOB,
                        "job": own_job(st), "port": my_port,
                        "running": st.get("running"), "queued": len(pending),
                        "self_script": own.get("script") if own.get("running") else None})

    @app.route("/api/queue-draft", methods=["POST"])
    def queue_draft():
        """Nuevo entrenamiento del mismo modelo para la cola mientras este entrena: el lanzador abre un
        borrador en otro puerto que parte de los ajustes de aquí, con su propia copia."""
        code, res = _launcher("/api/launch", {"id": trainer_id, "draft": True,
                                              "settings": {"train": _read(train_config), "pre_cache": _read(precache_config)}},
                              timeout=15)
        if code is None:
            return jsonify({"status": "error", "error": t("The launcher is not running: the queue needs it.")}), 503
        return jsonify(res), code

    @app.route("/api/queue-resume", methods=["POST"])
    def queue_resume():
        """Reanudar la cola entera (estaba pausada al añadir un trabajo)."""
        code, res = _launcher("/api/queue/options", {"paused": False}, timeout=5)
        if code is None:
            return jsonify({"status": "error", "error": t("The launcher is not running: the queue needs it.")}), 503
        return jsonify(res), code

    @app.route("/api/queue-edit-done", methods=["POST"])
    def queue_edit_done():
        """Guardar (o cancelar) la edición del trabajo: el lanzador lo deja listo y cierra este editor."""
        if not EDIT_JOB:
            return jsonify({"status": "error", "error": t("Job not found.")}), 404
        save = bool((request.get_json(silent=True) or {}).get("save"))
        code, res = _launcher(f"/api/queue-edit/{EDIT_JOB}/done", {"save": save}, timeout=10)
        if code is None:
            return jsonify({"status": "error", "error": t("The launcher is not running: the queue needs it.")}), 503
        return jsonify(res), code

    @app.route("/api/queue-draft-ready")
    def queue_draft_ready():
        import socket
        port = request.args.get("port", type=int) or 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return jsonify({"ready": bool(port) and s.connect_ex(("127.0.0.1", port)) == 0})
