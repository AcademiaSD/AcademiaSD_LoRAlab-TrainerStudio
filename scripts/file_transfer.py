# -*- coding: utf-8 -*-
"""
file_transfer.py — Subir el dataset y descargar los LoRAs desde el navegador.
Upload the dataset and download the LoRAs from the browser.

Hace falta cuando el entrenador no está en el PC que tiene el navegador: acceso remoto, RunPod.
Los archivos sueltos o el contenido de un .zip van a la carpeta del dataset sin subcarpetas; solo se
aceptan las extensiones del dataset y .txt, y cada nombre se comprueba contra la carpeta destino.
"""
import json
import re
import struct
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


def lora_options_suffix(path):
    """_rs (rsLoRA), _plus (LoRA+) o _lokr (LoKr) según los metadatos del LoRA; "" si no lleva ninguno.
    Se lee solo la cabecera del .safetensors (8 bytes de longitud + JSON), sin cargar los pesos."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        return ""
    return (("_rs" if meta.get("rslora") == "true" else "") + ("_plus" if meta.get("loraplus_ratio") else "")
            + ("_lokr" if meta.get("lokr_factor") else ""))


def with_lora_options_suffix(name, path):
    """name.safetensors con el sufijo que corresponde al LoRA de path. Un sufijo que no corresponda (la GUI
    lo pone según las casillas, que pueden cambiar después de entrenar) se sustituye por el correcto."""
    stem = name[:-len(".safetensors")] if name.lower().endswith(".safetensors") else name
    stem = re.sub(r"(_rs)?(_plus)?(_lokr)?$", "", stem, flags=re.IGNORECASE) or stem
    return stem + lora_options_suffix(path) + ".safetensors"


def register(app, get_dataset_dir, get_output_dir, dataset_exts):
    allowed = tuple(dataset_exts) + (".txt",)

    @app.route("/file_transfer.js")
    def file_transfer_js():
        return send_from_directory(str(GUI_DIR), "file_transfer.js")

    @app.route("/notify.js")
    def notify_js():
        return send_from_directory(str(GUI_DIR), "notify.js")

    @app.route("/api/upload-dataset", methods=["POST"])
    def upload_dataset():
        folder = get_dataset_dir()
        folder.mkdir(parents=True, exist_ok=True)
        saved, skipped, failed = [], [], []

        def save_stream(src, dest, name):
            try:
                with open(dest, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                saved.append(dest.name)
            except OSError as exc:
                failed.append({"name": name, "error": str(exc)})

        for item in request.files.getlist("files"):
            name = item.filename or ""
            if name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(item.stream) as z:
                        for info in z.infolist():
                            if info.is_dir() or "__MACOSX" in info.filename:
                                continue
                            dest = _target(folder, info.filename)
                            if dest is None or dest.suffix.lower() not in allowed:
                                skipped.append(info.filename)
                                continue
                            try:
                                with z.open(info) as src:
                                    save_stream(src, dest, info.filename)
                            except OSError as exc:
                                failed.append({"name": info.filename, "error": str(exc)})
                except (zipfile.BadZipFile, OSError) as exc:
                    failed.append({"name": name, "error": str(exc)})
                continue

            dest = _target(folder, name)
            if dest is None or dest.suffix.lower() not in allowed:
                skipped.append(name)
                continue
            try:
                item.save(dest)
                saved.append(dest.name)
            except OSError as exc:
                failed.append({"name": name, "error": str(exc)})

        return jsonify({
            "status": "ok" if not failed else "partial",
            "saved": len(saved),
            "saved_names": saved,
            "skipped": skipped[:50],
            "failed": failed[:50],
            "path": str(folder),
        })

    def download_name(f):
        # Solo cambia el nombre de los LoRAs entrenados con rsLoRA, LoRA+ o LoKr.
        return with_lora_options_suffix(f.name, f) if lora_options_suffix(f) else f.name

    @app.route("/api/output-files")
    def output_files():
        folder = get_output_dir()
        files = sorted(folder.glob("*.safetensors"), key=lambda f: f.stat().st_mtime, reverse=True) if folder.is_dir() else []
        return jsonify({"status": "ok", "path": str(folder),
                        "files": [{"name": f.name, "size": f.stat().st_size, "download_name": download_name(f)}
                                  for f in files]})

    @app.route("/api/download-output/<path:name>")
    def download_output(name):
        dest = _target(get_output_dir(), name)
        if dest is None or dest.suffix.lower() != ".safetensors" or not dest.is_file():
            abort(404)
        return send_file(dest, as_attachment=True, download_name=download_name(dest))
