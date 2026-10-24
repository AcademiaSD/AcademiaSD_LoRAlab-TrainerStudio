# -*- coding: utf-8 -*-
"""
training_metrics.py — Historial de loss para la gráfica del entrenamiento.
Loss history for the training chart.

Entrenador:  metrics = MetricsLog(OUTPUT_DIR, start_step)
             metrics.log(step, loss_de_este_paso, lr, gnorm, segundos, pasos_totales)
             Una línea JSON por paso en <OUTPUT_DIR>/metrics.jsonl (coste despreciable frente a un paso).
             Al reanudar se conservan los pasos anteriores a start_step; empezar de cero borra el historial.
             name: otro archivo por etapa (YuE2 entrena AR y NAR en la misma carpeta); se enseña el más reciente.
Servidor:    register(app, get_output_dir) sirve /loss_chart.js y /api/metrics?since=<paso>, con los
             previews (preview_step_N.*) y checkpoints (*_step_N.safetensors) de la carpeta como marcadores.
"""
import json
import os
import re
import time
from pathlib import Path

METRICS_FILE = "metrics.jsonl"
GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"

PREVIEW_RE = re.compile(r"^preview_step_0*(\d+).*\.(png|jpg|jpeg|webp|wav|mp3|mp4|webm)$", re.I)
CHECKPOINT_RE = re.compile(r"_step_(\d+)\.safetensors$", re.I)


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


class MetricsLog:
    """Escritor del entrenador. Nunca rompe el entrenamiento: si el disco falla, deja de escribir."""

    def __init__(self, output_dir, start_step=0, name=METRICS_FILE):
        self.path = os.path.join(str(output_dir), name)
        self._file = None
        try:
            os.makedirs(str(output_dir), exist_ok=True)
            kept = []
            if start_step > 0 and os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if "run" in rec or rec.get("s", 0) <= start_step:
                            kept.append(line.rstrip("\n"))
            with open(self.path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
                # "run" cambia en cada arranque: el navegador sabe que debe pedir el historial entero.
                f.write(json.dumps({"run": time.time(), "start": int(start_step)}) + "\n")
            self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError:
            self._file = None

    def log(self, step, loss, lr=None, gnorm=None, sec=None, total=None):
        if self._file is None:
            return
        rec = {"s": int(step), "l": _num(loss), "lr": _num(lr), "g": _num(gnorm),
               "dt": _num(sec), "n": int(total) if total else None, "t": round(time.time(), 1)}
        try:
            self._file.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
        except OSError:
            self._file = None


# ---------------------------------------------------------------------- servidor
_cache = {}   # ruta -> (mtime, tamaño, run, puntos)


def _read_points(path):
    try:
        st = path.stat()
    except OSError:
        return None, []
    hit = _cache.get(str(path))
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2], hit[3]
    run, by_step = None, {}
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "run" in rec:
                    run = rec["run"]
                elif "s" in rec and rec.get("l") is not None:
                    by_step[rec["s"]] = rec   # un paso repetido tras reanudar: gana el último
    except OSError:
        return None, []
    points = [by_step[s] for s in sorted(by_step)]
    _cache[str(path)] = (st.st_mtime, st.st_size, run, points)
    return run, points


def _markers(output_dir, since_time):
    previews, checkpoints = [], []
    try:
        entries = list(os.scandir(output_dir))
    except OSError:
        return previews, checkpoints
    for e in entries:
        if not e.is_file():
            continue
        try:
            # Archivos de un entrenamiento anterior del mismo proyecto no son de esta curva.
            if since_time and e.stat().st_mtime < since_time - 5:
                continue
        except OSError:
            continue
        m = PREVIEW_RE.match(e.name)
        if m:
            previews.append([int(m.group(1)), e.name])
            continue
        m = CHECKPOINT_RE.search(e.name)
        if m:
            checkpoints.append(int(m.group(1)))
    previews.sort()
    return previews, sorted(set(checkpoints))


def register(app, get_output_dir):
    from flask import jsonify, request, send_from_directory

    @app.route("/loss_chart.js")
    def loss_chart_js():
        return send_from_directory(str(GUI_DIR), "loss_chart.js")

    @app.route("/api/metrics")
    def api_metrics():
        output_dir = Path(get_output_dir())
        since = request.args.get("since", 0, type=int)
        logs = [p for p in output_dir.glob("metrics*.jsonl") if p.is_file()] if output_dir.is_dir() else []
        newest = max(logs, key=lambda p: p.stat().st_mtime, default=output_dir / METRICS_FILE)
        run, points = _read_points(newest)
        first_time = points[0].get("t") if points else None
        previews, checkpoints = _markers(output_dir, first_time) if points else ([], [])
        return jsonify({
            "run": run,
            "points": [p for p in points if p["s"] > since],
            "previews": previews,
            "checkpoints": checkpoints,
        })
