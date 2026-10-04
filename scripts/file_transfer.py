# -*- coding: utf-8 -*-
"""
file_transfer.py — Subir el dataset y descargar los LoRAs desde el navegador.
Upload the dataset and download the LoRAs from the browser.

Hace falta cuando el entrenador no está en el PC que tiene el navegador: acceso remoto, RunPod.
Los archivos sueltos o el contenido de un .zip van a la carpeta del dataset sin subcarpetas; solo se
aceptan las extensiones del dataset y .txt, y cada nombre se comprueba contra la carpeta destino.
"""
import zipfile
from pathlib import Path, PurePath

from flask import abort, jsonify, request, send_file, send_from_directory

GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"


def _target(folder, name):
    """Ruta dentro de folder para un nombre recibido, o None si no vale (ruta, oculto, fuera de la carpeta)."""
    base = PurePath(name.replace("\\", "/")).name.strip()
    if not base or base.startswith("."):
        return None
    path = (folder / base).resolve()
    return path if path.parent == folder.resolve() else None


def register(app, get_dataset_dir, get_output_dir, dataset_exts):
    allowed = tuple(dataset_exts) + (".txt",)

    @app.route("/file_transfer.js")
    def file_transfer_js():
        return send_from_directory(str(GUI_DIR), "file_transfer.js")

    @app.route("/api/upload-dataset", methods=["POST"])
    def upload_dataset():
        folder = get_dataset_dir()
        folder.mkdir(parents=True, exist_ok=True)
        saved, skipped = 0, []
        for item in request.files.getlist("files"):
            if item.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(item.stream) as z:
                    for info in z.infolist():
                        dest = None if info.is_dir() or "__MACOSX" in info.filename else _target(folder, info.filename)
                        if dest is None or dest.suffix.lower() not in allowed:
                            if not info.is_dir():
                                skipped.append(info.filename)
                            continue
                        with z.open(info) as src, open(dest, "wb") as out:
                            while chunk := src.read(1 << 20):
                                out.write(chunk)
                        saved += 1
                continue
            dest = _target(folder, item.filename)
            if dest is None or dest.suffix.lower() not in allowed:
                skipped.append(item.filename)
                continue
            item.save(dest)
            saved += 1
        return jsonify({"status": "ok", "saved": saved, "skipped": skipped[:50], "path": str(folder)})

    @app.route("/api/output-files")
    def output_files():
        folder = get_output_dir()
        files = sorted(folder.glob("*.safetensors"), key=lambda f: f.stat().st_mtime, reverse=True) if folder.is_dir() else []
        return jsonify({"status": "ok", "path": str(folder),
                        "files": [{"name": f.name, "size": f.stat().st_size} for f in files]})

    @app.route("/api/download-output/<path:name>")
    def download_output(name):
        dest = _target(get_output_dir(), name)
        if dest is None or dest.suffix.lower() != ".safetensors" or not dest.is_file():
            abort(404)
        return send_file(dest, as_attachment=True, download_name=dest.name)
