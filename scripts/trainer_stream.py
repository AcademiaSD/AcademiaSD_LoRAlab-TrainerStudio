# -*- coding: utf-8 -*-
"""
trainer_stream.py — Entrenamiento desacoplado del navegador.
Training decoupled from the browser.

El script de entrenamiento y la lectura de su consola viven en un hilo del servidor, no en la
petición HTTP que abre la pestaña: cerrarla ya no cierra la tubería (el proceso hijo moría al
cerrarse el pipe) y la salida queda accesible para cualquier navegador (backlog en memoria +
settings/trainer_console_<id>.log en disco).
"""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from flask import Response, jsonify, request, send_file

from console_stream import read_console
from i18n import t

MAX_LINES = 4000        # backlog en memoria para reengancharse
PING_SECONDS = 15.0     # keep-alive del SSE a través de proxies


def _sse(kind, payload):
    return f"data: {json.dumps({'type': kind, **payload}, ensure_ascii=False)}\n\n"


class TrainerStream:
    """Ejecuta un script a la vez y mantiene su consola accesible para cualquier cliente."""

    def __init__(self, base_dir, settings_dir, trainer_id):
        self.base_dir = Path(base_dir)
        self.log_path = Path(settings_dir) / f"trainer_console_{trainer_id}.log"
        self._proc = None
        self._script = None
        self._lines = deque(maxlen=MAX_LINES)  # (seq, texto, reemplaza)
        self._seq = 0
        self._done = None                      # (seq, script, código) del último proceso
        self._lock = threading.Lock()
        self._subs = set()
        self._log = None

    # ------------------------------------------------------------------ estado
    def status(self):
        with self._lock:
            proc, script = self._proc, self._script
            if proc is None or proc.poll() is not None:
                return {"running": False, "script": None, "pid": None}
            return {"running": True, "script": script, "pid": proc.pid}

    def last_code(self):
        """Código de salida de la última ejecución terminada (0 si aún no hubo ninguna)."""
        with self._lock:
            return self._done[2] if self._done is not None else 0

    # -------------------------------------------------------- arranque / parada
    def start(self, script_name, script_path):
        """Lanza el script; devuelve (True, None) o (False, error)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, t("Process already running: {name}", name=self._script)
            self._lines.clear()
            self._done = None
            self._script = script_name
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(script_path)],
                cwd=str(self.base_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            proc = self._proc
        self._open_log(script_name)
        threading.Thread(target=self._pump, args=(proc, script_name), daemon=True).start()
        return True, None

    def stop(self):
        """Mismo protocolo que el /api/stop actual (SIGINT; CTRL_BREAK en Windows)."""
        with self._lock:
            proc, script = self._proc, self._script
        if proc is None or proc.poll() is not None:
            return {"status": "not_running"}
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
            return {"status": "terminating", "script": script}
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
            return {"status": "terminated", "script": script}

    # ----------------------------------------------------------------- consola
    def _open_log(self, script_name):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("w", encoding="utf-8")
            self._log.write(f"[SYSTEM] {t('Launching {script}...', script=script_name)}\n")
        except OSError:
            self._log = None

    def _pump(self, proc, script_name):
        # Único lector del pipe: backlog + .log + terminal + suscriptores SSE.
        try:
            if proc.stdout is not None:
                for text, replace in read_console(proc.stdout):
                    self._emit(text, replace)
        finally:
            self._emit_done(script_name, proc.wait())
            if self._log is not None:
                try:
                    self._log.close()
                except OSError:
                    pass
                self._log = None

    def _emit(self, text, replace=False):
        print(text, end="\r" if replace else "\n", flush=True)
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._lines.append((seq, text, replace))
            subs = list(self._subs)
            log = self._log
        if log is not None:
            try:
                log.write(text + ("\r" if replace else "\n"))
                log.flush()
            except OSError:
                pass
        for q in subs:
            q.put(("line", {"seq": seq, "text": text, "replace": replace}))

    def _emit_done(self, script_name, code):
        with self._lock:
            self._seq += 1
            self._done = (self._seq, script_name, code)
            subs = list(self._subs)
            done = self._done
        for q in subs:
            q.put(("done", {"seq": done[0], "script": script_name, "code": code}))

    def sse_stream(self, since=0):
        def gen():
            q = queue.Queue()
            with self._lock:
                self._subs.add(q)
                backlog = [item for item in self._lines if item[0] > since]
                done = self._done
            try:
                for seq, text, replace in backlog:
                    yield _sse("line", {"seq": seq, "text": text, "replace": replace})
                if done is not None:
                    # La ejecución ya terminó: se reenvía el final (si no se vio) y se cierra.
                    if done[0] > since:
                        yield _sse("done", {"seq": done[0], "script": done[1], "code": done[2]})
                    return
                while True:
                    try:
                        kind, payload = q.get(timeout=PING_SECONDS)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                    yield _sse(kind, payload)
                    if kind == "done":
                        return
            except GeneratorExit:
                # La pestaña se cerró: solo se quita este suscriptor, el proceso sigue.
                pass
            finally:
                with self._lock:
                    self._subs.discard(q)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ rutas
    def register(self, app):
        @app.route("/api/console/stream")
        def console_stream():
            return self.sse_stream(request.args.get("since", 0, type=int))

        @app.route("/api/console-log")
        def console_log():
            if not self.log_path.exists():
                return jsonify({"status": "error", "error": t("No log yet.")}), 404
            return send_file(self.log_path, as_attachment=True, download_name=self.log_path.name)
