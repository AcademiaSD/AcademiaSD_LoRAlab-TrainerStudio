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

from i18n import t

GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"
# Carpeta propia para las imágenes de preview de un dataset de edición subidas por HTTP: nunca
# dentro del dataset, o el entrenamiento las tomaría por pares de edición. / Its own folder for
# the edit-dataset preview images uploaded over HTTP: never inside the dataset, or training
# would take them for edit pairs.
PREVIEW_EDIT_DIR = GUI_DIR.parent / "preview_edit_images"


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

    @app.route("/api/upload-preview-image", methods=["POST"])
    def upload_preview_image():
        """Imagen de preview de un dataset de edición subida desde el navegador: es la vía cuando el
        diálogo nativo no sirve (RunPod, acceso remoto), porque abriría el explorador del PC que
        entrena. Se guarda en el servidor y se devuelve su ruta. / The edit dataset's preview image
        uploaded from the browser: the way in when the native dialog is useless (RunPod, remote
        access), since it would open on the training PC. Saved on the server, its path returned."""
        item = (request.files.getlist("files") or [None])[0]
        if item is None or not item.filename:
            return jsonify({"status": "error", "error": t("No file received")}), 400
        PREVIEW_EDIT_DIR.mkdir(parents=True, exist_ok=True)
        dest = _target(PREVIEW_EDIT_DIR, item.filename)
        if dest is None or dest.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            return jsonify({"status": "error", "error": t("Only images (.png, .jpg, .jpeg, .webp)")}), 400
        try:
            item.save(dest)
        except OSError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 500
        return jsonify({"status": "ok", "path": str(dest.resolve())})

    def download_name(f, final_name=""):
        """Nombre con el que se descarga un LoRA. El final usa el 'Final LoRA Filename' de la GUI
        (que ya lleva el Project Name dentro) y los checkpoints intermedios añaden _<pasos>_steps.
        El sufijo _rs / _plus / _lokr del LoRA se conserva. Sin final_name (campo vacío) se mantiene
        el nombre del archivo tal cual. / Download name. The final LoRA uses the GUI's 'Final LoRA
        Filename' (Project Name already inside) and intermediate checkpoints add _<steps>_steps.
        The LoRA's _rs / _plus / _lokr suffix is kept. Empty final_name keeps the file name."""
        opts = lora_options_suffix(f)
        if final_name:
            stem = final_name[:-len(".safetensors")] if final_name.lower().endswith(".safetensors") else final_name
            # El campo ya lleva el sufijo que la GUI pone según las casillas: se quita y se pone el que
            # dicen los metadatos del LoRA, para no duplicarlo (_lokr_lokr).
            stem = re.sub(r"(_rs)?(_plus)?(_lokr)?$", "", stem, flags=re.IGNORECASE) or stem
            steps = re.search(r"_step_(\d+)$", f.stem)
            if steps:
                return f"{stem}_{steps.group(1)}_steps{opts}.safetensors"
            if "final" in f.stem.lower():
                return f"{stem}{opts}.safetensors"
        return with_lora_options_suffix(f.name, f) if opts else f.name

    @app.route("/api/output-files")
    def output_files():
        folder = get_output_dir()
        files = sorted(folder.glob("*.safetensors"), key=lambda f: f.stat().st_mtime, reverse=True) if folder.is_dir() else []
        # El nombre solo decora la descarga: nunca toca el archivo de la carpeta de salida.
        # The name only decorates the download: it never touches the output folder's file.
        final_name = PurePath(request.args.get("final_name") or "").name.strip()
        return jsonify({"status": "ok", "path": str(folder),
                        "files": [{"name": f.name, "size": f.stat().st_size, "download_name": download_name(f, final_name)}
                                  for f in files]})

    @app.route("/api/download-output/<path:name>")
    def download_output(name):
        dest = _target(get_output_dir(), name)
        if dest is None or dest.suffix.lower() != ".safetensors" or not dest.is_file():
            abort(404)
        final_name = PurePath(request.args.get("final_name") or "").name.strip()
        return send_file(dest, as_attachment=True, download_name=download_name(dest, final_name))
