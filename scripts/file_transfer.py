# -*- coding: utf-8 -*-
"""
file_transfer.py — Subir el dataset y descargar los LoRAs desde el navegador.
Upload the dataset and download the LoRAs from the browser.

Hace falta cuando el entrenador no está en el PC que tiene el navegador: acceso remoto, RunPod.
Los archivos sueltos o el contenido de un .zip van a la carpeta del dataset sin subcarpetas; solo se
aceptan las extensiones del dataset y .txt, y cada nombre se comprueba contra la carpeta destino.

Los LoRAs también se pueden subir a un repo de Hugging Face (token con permiso de escritura), a mano
desde el diálogo de descarga o solos según se guardan (Colab, RunPod: no se pierden al cerrar).
The LoRAs can also go to a Hugging Face repo (write token), by hand from the download dialog or on
their own as they are saved (Colab, RunPod: nothing is lost when the machine goes away).
"""
import json
import os
import re
import struct
import threading
import time
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
SETTINGS_DIR = GUI_DIR.parent / "settings"
HF_TOKEN_FILE = SETTINGS_DIR / "HF_token.json"
# Compartido por todos los entrenadores: {"repo", "private", "auto", "auto_since", "final_name", "uploaded"}.
# Shared by every trainer.
HF_UPLOAD_FILE = SETTINGS_DIR / "hf_upload.json"
HF_AUTO_INTERVAL = 30   # s entre vistazos a la carpeta de salida / s between looks at the output folder
_hf_lock = threading.Lock()


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _hf_token():
    """El de la GUI (settings/HF_token.json) o, si está vacío, HF_TOKEN del entorno (Docker / RunPod)."""
    return (_read_json(HF_TOKEN_FILE).get("token") or "").strip() or os.environ.get("HF_TOKEN", "").strip()


def _hf_upload(path, path_in_repo, repo, private):
    """Sube un LoRA a repo (lo crea si no existe) y devuelve la URL del archivo. Un solo envío a la vez."""
    token = _hf_token()
    if not token:
        raise RuntimeError(t("No Hugging Face token: save a token with write access first."))
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    with _hf_lock:
        api.create_repo(repo, repo_type="model", private=bool(private), exist_ok=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo,
                        repo_type="model", commit_message=f"Upload {path_in_repo} (LoRAlab Trainer Studio)")
    return f"https://huggingface.co/{repo}/blob/main/{path_in_repo}"


def _valid_repo(repo):
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""))


def _target(folder, name):
    """Ruta dentro de folder para un nombre recibido, o None si no vale (ruta, oculto, fuera de la carpeta)."""
    base = PurePath(name.replace("\\", "/")).name.strip()
    if not base or base.startswith("."):
        return None
    path = (folder / base).resolve()
    return path if path.parent == folder.resolve() else None


def _strip_options_suffix(stem, opts):
    """Quita del nombre los sufijos que se ponen según los metadatos (_rs/_plus/_lokr), para no
    duplicarlos. _AR / _NAR solo en los LoRA de YuE2: en otro trainer puede ser parte del nombre."""
    pattern = r"(_rs)?(_plus)?(_lokr)?(_n?ar)?$" if opts.endswith(("_AR", "_NAR")) else r"(_rs)?(_plus)?(_lokr)?$"
    return re.sub(pattern, "", stem, flags=re.IGNORECASE) or stem


def lora_options_suffix(path):
    """_rs (rsLoRA), _plus (LoRA+) o _lokr (LoKr) según los metadatos del LoRA, y _AR / _NAR si es una
    etapa de YuE2 (cada etapa se carga en un sitio distinto de ComfyUI); "" si no lleva ninguno.
    Se lee solo la cabecera del .safetensors (8 bytes de longitud + JSON), sin cargar los pesos."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        return ""
    return (("_rs" if meta.get("rslora") == "true" else "") + ("_plus" if meta.get("loraplus_ratio") else "")
            + ("_lokr" if meta.get("lokr_factor") else "")
            + {"ar": "_AR", "nar": "_NAR"}.get(meta.get("yue2_stage"), ""))


def with_lora_options_suffix(name, path):
    """name.safetensors con el sufijo que corresponde al LoRA de path. Un sufijo que no corresponda (la GUI
    lo pone según las casillas, que pueden cambiar después de entrenar) se sustituye por el correcto."""
    stem = name[:-len(".safetensors")] if name.lower().endswith(".safetensors") else name
    opts = lora_options_suffix(path)
    return _strip_options_suffix(stem, opts) + opts + ".safetensors"


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
        stem = _strip_options_suffix(stem, opts)
        steps = re.search(r"_step_(\d+)$", f.stem)
        if steps:
            return f"{stem}_{steps.group(1)}_steps{opts}.safetensors"
        if "final" in f.stem.lower():
            return f"{stem}{opts}.safetensors"
    return with_lora_options_suffix(f.name, f) if opts else f.name


def register(app, get_dataset_dir, get_output_dir, dataset_exts):
    allowed = tuple(dataset_exts) + (".txt",)

    @app.route("/file_transfer.js")
    def file_transfer_js():
        return send_from_directory(str(GUI_DIR), "file_transfer.js")

    @app.route("/notify.js")
    def notify_js():
        return send_from_directory(str(GUI_DIR), "notify.js")

    @app.route("/preview_size.js")
    def preview_size_js():
        return send_from_directory(str(GUI_DIR), "preview_size.js")

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

    # --- Hugging Face ---------------------------------------------------------------
    hf_state = {"last": "", "pending": {}}

    def mark_uploaded(f, mtime):
        cfg = _read_json(HF_UPLOAD_FILE)
        cfg.setdefault("uploaded", {})[str(f.resolve())] = mtime
        _write_json(HF_UPLOAD_FILE, cfg)

    @app.route("/api/hf-upload-settings", methods=["GET"])
    def hf_upload_settings():
        cfg = _read_json(HF_UPLOAD_FILE)
        return jsonify({"repo": cfg.get("repo", ""), "private": cfg.get("private", True),
                        "auto": bool(cfg.get("auto")), "has_token": bool(_hf_token()), "last": hf_state["last"]})

    @app.route("/api/save-hf-upload-settings", methods=["POST"])
    def save_hf_upload_settings():
        req = request.get_json(force=True) or {}
        repo = str(req.get("repo", "")).strip()
        if repo and not _valid_repo(repo):
            return jsonify({"status": "error", "error": t("The repo must look like user/name.")}), 400
        cfg = _read_json(HF_UPLOAD_FILE)
        auto = bool(req.get("auto")) and bool(repo)
        # Solo se suben solos los LoRAs guardados desde que se activa: los de antes, a mano.
        # Only LoRAs saved after it is switched on go up by themselves: older ones, by hand.
        if auto and not cfg.get("auto"):
            cfg["auto_since"] = time.time()
        cfg.update({"repo": repo, "private": bool(req.get("private", True)), "auto": auto,
                    "final_name": PurePath(str(req.get("final_name") or "")).name.strip()})
        _write_json(HF_UPLOAD_FILE, cfg)
        return jsonify({"status": "ok", "auto": auto})

    @app.route("/api/hf-upload", methods=["POST"])
    def hf_upload():
        req = request.get_json(force=True) or {}
        dest = _target(get_output_dir(), str(req.get("name", "")))
        if dest is None or dest.suffix.lower() != ".safetensors" or not dest.is_file():
            return jsonify({"status": "error", "error": t("File not found")}), 404
        cfg = _read_json(HF_UPLOAD_FILE)
        repo = str(req.get("repo") or cfg.get("repo", "")).strip()
        if not _valid_repo(repo):
            return jsonify({"status": "error", "error": t("The repo must look like user/name.")}), 400
        final_name = PurePath(str(req.get("final_name") or "")).name.strip()
        mtime = dest.stat().st_mtime
        try:
            url = _hf_upload(dest, download_name(dest, final_name), repo, req.get("private", cfg.get("private", True)))
        except Exception as exc:
            return jsonify({"status": "error", "error": str(exc)}), 500
        mark_uploaded(dest, mtime)
        return jsonify({"status": "ok", "url": url})

    def hf_auto_pass():
        cfg = _read_json(HF_UPLOAD_FILE)
        folder = get_output_dir()
        if not cfg.get("auto") or not _valid_repo(cfg.get("repo")) or not folder.is_dir():
            return
        uploaded, pending = cfg.get("uploaded", {}), hf_state["pending"]
        for f in sorted(folder.glob("*.safetensors"), key=lambda p: p.stat().st_mtime):
            st, key = f.stat(), str(f.resolve())
            if st.st_mtime < cfg.get("auto_since", 0) or uploaded.get(key) == st.st_mtime:
                continue
            # Igual que en el vistazo anterior y con 10 s de calma: el entrenador ya no lo está escribiendo.
            # Same as on the previous look and quiet for 10 s: the trainer is no longer writing it.
            if pending.get(key) != (st.st_size, st.st_mtime) or time.time() - st.st_mtime < 10:
                pending[key] = (st.st_size, st.st_mtime)
                continue
            name = download_name(f, cfg.get("final_name", ""))
            try:
                url = _hf_upload(f, name, cfg["repo"], cfg.get("private", True))
            except Exception as exc:
                hf_state["last"] = t("Upload of {name} failed: {error}", name=name, error=exc)
                print("[HF] " + hf_state["last"])
                return   # se reintenta en el siguiente vistazo / retried on the next look
            pending.pop(key, None)
            mark_uploaded(f, st.st_mtime)
            hf_state["last"] = t("Uploaded {name}", name=name)
            print(f"[HF] {hf_state['last']}: {url}")

    def hf_auto_loop():
        while True:
            time.sleep(HF_AUTO_INTERVAL)
            try:
                hf_auto_pass()
            except Exception as exc:
                print(f"[HF] {exc}")

    threading.Thread(target=hf_auto_loop, name="hf-auto-upload", daemon=True).start()
