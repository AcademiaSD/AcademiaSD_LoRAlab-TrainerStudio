# -*- coding: utf-8 -*-
"""
server.py — Backend web para AcademiaSD FLUX.2 Klein 9B Trainer
Web backend for AcademiaSD FLUX.2 Klein 9B Trainer
"""

import json
import time
import os
import subprocess
import sys
import threading
import importlib.util
import signal
import shutil
import re
import string
import logging
import webbrowser

from console_stream import read_console
from trainer_stream import TrainerStream
import i18n
import remote_access
import file_transfer
from pathlib import Path
from i18n import t


# =============================================================================
# DEPENDENCIAS / DEPENDENCIES
# =============================================================================

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name

    if importlib.util.find_spec(import_name) is not None:
        return

    print()
    print("=" * 70)
    print("[INFO] " + t("Installing missing package: '{name}'...", name=package_name))
    print(f"[INFO] Python: {sys.executable}")
    print("=" * 70)
    print()

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError as exc:
        print("\n[ERROR] " + t("Failed to install '{name}'. Exit code: {code}", name=package_name, code=exc.returncode) + "\n")
        raise

    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(t("Package '{name}' installed but import failed.", name=package_name))

    print("[OK] " + t("'{name}' installed successfully.", name=package_name))


ensure_package("Flask", "flask")
ensure_package("psutil", "psutil")

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory
)

# Silenciar registros HTTP recurrentes de Werkzeug en consola
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# =============================================================================
# CONFIGURACIÓN / CONFIGURATION
# =============================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
SETTINGS_DIR = BASE_DIR / "settings"
SETTINGS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = BASE_DIR / "assets" / "klein9b"

UI_FILE = BASE_DIR / "GUI" / "trainer_ui_klein9b.html"
LOGO_FILE = ASSETS_DIR / "logo.png" if (ASSETS_DIR / "logo.png").exists() else BASE_DIR / "logo.png"

PRECACHE_CONFIG = SETTINGS_DIR / "pre_cache_settings_klein9b.json"
PREVIEW_KEYS = ("lora_type", "preview_custom_prompt", "preview_edit_image")
TRAIN_CONFIG = SETTINGS_DIR / "train_settings_klein9b.json"
HF_TOKEN_CONFIG = SETTINGS_DIR / "HF_token.json"
EXPORT_CONFIG = SETTINGS_DIR / "export_settings.json"  # compartido / shared by every trainer
CAPTION_CONFIG = SETTINGS_DIR / "caption_settings_klein9b.json"

DATASET_EXTS = (".png", ".jpg", ".jpeg", ".webp")

CAPTION_SCRIPT = SCRIPTS_DIR / "0_caption_klein9b.py"
PRECACHE_SCRIPT = SCRIPTS_DIR / "1_pre_cache_klein9b.py"
TRAIN_SCRIPT = SCRIPTS_DIR / "2_train_lora_klein9b.py"

app = Flask(__name__)

runtime = TrainerStream(BASE_DIR, SETTINGS_DIR, "klein9b")


# =============================================================================
# UTILIDADES / UTILS
# =============================================================================

def get_windows_drives():
    drives = []
    if os.name == "nt":
        try:
            from ctypes import windll
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(f"{letter}:\\")
                bitmask >>= 1
        except Exception:
            pass
    return drives


def read_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception as exc:
        print("[ERROR] " + t("Could not read {path}: {error}", path=path, error=exc))
        return default


def write_json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, path)


def resolve_config_path(value, default):
    if value is None or str(value).strip() == "":
        value = default
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def get_train_output_dir():
    cfg = read_json_file(TRAIN_CONFIG, {})
    proj = cfg.get("project_name", "").strip()
    if proj:
        return resolve_config_path(f"klein9b_lora_output_{proj}", "klein9b_lora_output")
    return resolve_config_path(cfg.get("output_dir"), "klein9b_lora_output")


def get_dataset_dir():
    cfg = read_json_file(PRECACHE_CONFIG, {"dataset_path": "./dataset"})
    return resolve_config_path(cfg.get("dataset_path"), "./dataset")


def get_precache_dir():
    cfg = read_json_file(PRECACHE_CONFIG, {})
    proj = cfg.get("project_name", "").strip()
    if proj:
        return resolve_config_path(f"cached_data_klein9b_{proj}", "cached_data_klein9b")
    return resolve_config_path(cfg.get("cache_dir"), "cached_data_klein9b")


def get_script_for_name(script_name):
    if script_name == "caption":
        return CAPTION_SCRIPT
    if script_name == "precache":
        return PRECACHE_SCRIPT
    if script_name == "train":
        return TRAIN_SCRIPT
    return None


def get_status():
    return runtime.status()


# =============================================================================
# HUGGINGFACE TOKEN API
# =============================================================================

@app.route("/api/hf-token", methods=["GET"])
def get_hf_token():
    data = read_json_file(HF_TOKEN_CONFIG, {"token": ""})
    return jsonify({"token": data.get("token", "")})


@app.route("/api/save-hf-token", methods=["POST"])
def save_hf_token():
    try:
        req = request.get_json(force=True) or {}
        token = req.get("token", "").strip()
        write_json_file(HF_TOKEN_CONFIG, {"token": token})
        return jsonify({"status": "ok", "file": HF_TOKEN_CONFIG.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# Carpeta de exportación: un único fichero para todos los entrenadores.
# Export folder: one file shared by every trainer.
@app.route("/api/save-export-settings", methods=["POST"])
def save_export_settings():
    try:
        req = request.get_json(force=True) or {}
        write_json_file(EXPORT_CONFIG, {"export_models_dir": str(req.get("export_models_dir", "")).strip()})
        return jsonify({"status": "ok", "file": EXPORT_CONFIG.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# REQUESTER NATIVO DE WINDOWS (POPUP SELECCIONAR CARPETA)
# =============================================================================

@app.route("/api/select-file", methods=["POST"])
def select_file_native():
    """Diálogo nativo para elegir una imagen (la de antes de las previews de edición)."""
    try:
        initial = str((request.get_json(force=True) or {}).get("initial_path", "")).strip()
        initial_dir = os.path.dirname(initial) if initial and os.path.exists(os.path.dirname(initial)) else str(BASE_DIR)

        selected_path = None
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            chosen = filedialog.askopenfilename(
                title=t("Select Image"),
                initialdir=initial_dir,
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
            )
            root.destroy()
            if chosen:
                selected_path = str(Path(chosen).resolve())
        except Exception:
            pass

        if not selected_path:
            try:
                ps_cmd = (
                    '[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null; '
                    '$dialog = New-Object System.Windows.Forms.OpenFileDialog; '
                    '$dialog.Filter = "Images|*.png;*.jpg;*.jpeg;*.webp"; '
                    f'$dialog.InitialDirectory = "{initial_dir}"; '
                    'if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileName }'
                )
                creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                output = subprocess.check_output(["powershell", "-command", ps_cmd], text=True, errors="ignore", creationflags=creation_flag).strip()
                if output:
                    selected_path = str(Path(output).resolve())
            except Exception:
                pass

        if selected_path:
            return jsonify({"status": "ok", "path": selected_path})
        return jsonify({"status": "cancelled", "path": None})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/select-folder", methods=["POST"])
def select_folder_native():
    try:
        data = request.get_json(force=True) or {}
        initial_dir = data.get("initial_dir", str(BASE_DIR)).strip()
        
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = str(BASE_DIR)

        selected_path = None
        # Sin diálogo nativo (Linux sin tkinter): la web abre su propio explorador de carpetas.
        dialog_shown = False

        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            chosen = filedialog.askdirectory(
                title=t("Select Folder"),
                initialdir=initial_dir
            )
            root.destroy()
            dialog_shown = True
            if chosen:
                selected_path = str(Path(chosen).resolve())
        except Exception:
            pass

        if not selected_path:
            try:
                ps_cmd = (
                    '[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null; '
                    '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; '
                    f'$dialog.SelectedPath = "{initial_dir}"; '
                    'if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.SelectedPath }'
                )
                creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                output = subprocess.check_output(["powershell", "-command", ps_cmd], text=True, errors="ignore", creationflags=creation_flag).strip()
                dialog_shown = True
                if output:
                    selected_path = str(Path(output).resolve())
            except Exception:
                pass

        if selected_path:
            return jsonify({"status": "ok", "path": selected_path})
        if not dialog_shown:
            return jsonify({"status": "error", "error": t("No native folder dialog")})
        else:
            return jsonify({"status": "cancelled", "path": None})

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# MONITOR VRAM, RAM & TEMPERATURA GPU DE HARDWARE
# =============================================================================

@app.route("/api/system-stats", methods=["GET"])
def get_system_stats():
    import psutil
    
    ram = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(ram.total / (1024**3), 2),
        "used_gb": round(ram.used / (1024**3), 2),
        "percent": ram.percent
    }

    vram_info = {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "percent": 0.0,
        "temp_c": 0,
        "gpu_name": "N/A"
    }

    try:
        creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,name,temperature.gpu", "--format=csv,nounits,noheader"]
        output = subprocess.check_output(cmd, text=True, errors="ignore", creationflags=creation_flag).strip().splitlines()[0]
        parts = [p.strip() for p in output.split(",")]
        total_m = float(parts[0])
        used_m = float(parts[1])
        vram_info["gpu_name"] = parts[2]
        vram_info["temp_c"] = int(float(parts[3]))
        vram_info["total_gb"] = round(total_m / 1024.0, 2)
        vram_info["used_gb"] = round(used_m / 1024.0, 2)
        vram_info["percent"] = round((used_m / total_m) * 100, 1) if total_m > 0 else 0.0
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                vram_info["gpu_name"] = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                used_b = total_b - free_b
                vram_info["total_gb"] = round(total_b / (1024**3), 2)
                vram_info["used_gb"] = round(used_b / (1024**3), 2)
                vram_info["percent"] = round((used_b / total_b) * 100, 1) if total_b > 0 else 0.0
        except Exception:
            pass

    return jsonify({"ram": ram_info, "vram": vram_info})


# =============================================================================
# EXPLORADOR DE DIRECTORIOS (WEB BACKUP)
# =============================================================================

@app.route("/api/browse-dir", methods=["GET"])
def browse_dir():
    requested_path = request.args.get("path", str(BASE_DIR))
    try:
        target = Path(requested_path).resolve()
        if not target.exists() or not target.is_dir():
            target = BASE_DIR
    except Exception:
        target = BASE_DIR

    parent = str(target.parent) if target.parent != target else str(target)
    dirs = []
    try:
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                dirs.append({"name": item.name, "path": str(item)})
    except Exception:
        pass

    image_count = 0
    try:
        image_count = sum(1 for f in target.iterdir() if f.is_file() and f.suffix.lower() in DATASET_EXTS)
    except Exception:
        pass

    return jsonify({
        "current_path": str(target),
        "parent_path": parent,
        "directories": dirs,
        "drives": get_windows_drives(),
        "image_count": image_count
    })


# =============================================================================
# EXPORTAR LORA A CARPETA MODELS
# =============================================================================

@app.route("/api/export-lora", methods=["POST"])
def export_lora():
    try:
        data = request.get_json(force=True) or {}
        target_dir_str = data.get("target_dir", "").strip()
        custom_name = data.get("final_name", "").strip()

        if not target_dir_str:
            return jsonify({"status": "error", "error": t("Please select a target folder.")}), 400

        target_dir = Path(target_dir_str).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"status": "error", "error": t("Target folder does not exist: {path}", path=target_dir)}), 400

        if not custom_name:
            custom_name = "klein9b_lora.safetensors"
        if not custom_name.lower().endswith(".safetensors"):
            custom_name += ".safetensors"

        output_dir = get_train_output_dir()
        if not output_dir.exists():
            return jsonify({"status": "error", "error": t("Output folder does not exist: {path}", path=output_dir)}), 404

        final_file = output_dir / "Klein9B_FINAL_LoRA.safetensors"
        source_file = None

        if final_file.exists():
            source_file = final_file
        else:
            candidates = []
            for f in output_dir.glob("*.safetensors"):
                match = re.search(r"step_(\d+)\.safetensors$", f.name, re.IGNORECASE)
                if match:
                    candidates.append((int(match.group(1)), f))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                source_file = candidates[0][1]

        if not source_file or not source_file.exists():
            return jsonify({"status": "error", "error": t("No .safetensors files found in: {path}", path=output_dir)}), 404

        custom_name = file_transfer.with_lora_options_suffix(custom_name, source_file)
        dest_file = target_dir / custom_name
        shutil.copy2(source_file, dest_file)

        return jsonify({
            "status": "ok",
            "source": source_file.name,
            "dest": str(dest_file),
            "filename": custom_name
        })

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# UI, ASSETS Y LOGO (FAVICON)
# =============================================================================

@app.route("/")
def index():
    if not UI_FILE.exists():
        return t("File not found: {path}", path=UI_FILE), 404
    return send_from_directory(str(UI_FILE.parent), UI_FILE.name)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    if ASSETS_DIR.exists():
        return send_from_directory(str(ASSETS_DIR), filename)
    return send_from_directory(str(BASE_DIR), filename)


@app.route("/favicon.ico")
@app.route("/logo.png")
def serve_logo():
    if (ASSETS_DIR / "logo.png").exists():
        return send_from_directory(str(ASSETS_DIR), "logo.png")
    if (BASE_DIR / "logo.png").exists():
        return send_from_directory(str(BASE_DIR), "logo.png")
    return "", 404


# =============================================================================
# CONFIGURACIÓN JSON (GUARDADO EN 2 SITIOS)
# =============================================================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "pre_cache": read_json_file(PRECACHE_CONFIG, {}),
        "train": read_json_file(TRAIN_CONFIG, {}),
        "export": read_json_file(EXPORT_CONFIG, {}),
        "base_dir": str(BASE_DIR)
    })


@app.route("/api/save-precache", methods=["POST"])
def save_precache():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": t("JSON object required.")}), 400
        
        proj = data.get("project_name", "").strip()
        cache_dir_name = f"cached_data_klein9b_{proj}" if proj else "cached_data_klein9b"
        output_dir_name = f"klein9b_lora_output_{proj}" if proj else "klein9b_lora_output"

        data["cache_dir"] = f"./{cache_dir_name}"

        # 1. Guardar en raíz con nombre genérico pre_cache_settings_klein9b.json (lectura por defecto)
        write_json_file(PRECACHE_CONFIG, data)
        saved_files = [PRECACHE_CONFIG.name]

        # 2. Guardar copia dentro de la carpeta del proyecto
        if proj:
            cache_dir_path = resolve_config_path(data["cache_dir"], cache_dir_name)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_json_file = cache_dir_path / f"pre_cache_settings_{proj}.json"
            write_json_file(cache_json_file, data)
            saved_files.append(f"{cache_dir_name}/{cache_json_file.name}")

        # Sincronizar train_settings_klein9b.json en raíz
        train_cfg = read_json_file(TRAIN_CONFIG, {})
        if "dataset_path" in data:
            train_cfg["dataset_path"] = data["dataset_path"]
        if "project_name" in data:
            train_cfg["project_name"] = data["project_name"]
            train_cfg["cache_dir"] = data["cache_dir"]
            train_cfg["output_dir"] = f"./{output_dir_name}"
        if "trigger_word" in data:
            train_cfg["trigger_word"] = data["trigger_word"]
        for key in PREVIEW_KEYS:
            if key in data:
                train_cfg[key] = data[key]

        write_json_file(TRAIN_CONFIG, train_cfg)

        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


prompt_job = {"process": None, "args": None, "stage": None, "device": None}


def encoding_stage():
    # "loading", "image" o "prompt" mientras se codifica la preview manual; None si no hay nada en marcha.
    process = prompt_job["process"]
    return prompt_job["stage"] if process is not None and process.poll() is None else None


def encode_preview_prompt(cache_dir, prompt, image):
    # Un solo proceso a la vez: si se guarda otro prompt mientras codifica, gana el último.
    # En GPU si está libre; con el entrenamiento en marcha, en CPU para no tocar su VRAM.
    device = "cpu" if get_status()["script"] == "train" else "cuda"
    args = [sys.executable, "-u", str(PRECACHE_SCRIPT), "--prompt-only", str(cache_dir), prompt, image, device]
    if encoding_stage() is not None:
        if prompt_job["args"] == args:
            return
        prompt_job["process"].terminate()
        prompt_job["process"].wait()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else None
    process = subprocess.Popen(args, cwd=str(BASE_DIR), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    prompt_job.update(process=process, args=args, stage="loading", device="GPU" if device == "cuda" else "CPU")

    def follow():
        # Reenvía la salida a la consola del servidor y actualiza la fase que muestra la GUI.
        for line in process.stdout:
            print(line, end="", flush=True)
            if prompt_job["process"] is process and line.startswith("[Custom Prompt] "):
                stage = line[len("[Custom Prompt] "):].split(":", 1)[0].lower()
                if stage in ("image", "prompt"):
                    prompt_job["stage"] = stage

    threading.Thread(target=follow, daemon=True).start()


@app.route("/api/save-train", methods=["POST"])
def save_train():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": t("JSON object required.")}), 400
        
        proj = data.get("project_name", "").strip()
        cache_dir_name = f"cached_data_klein9b_{proj}" if proj else "cached_data_klein9b"
        output_dir_name = f"klein9b_lora_output_{proj}" if proj else "klein9b_lora_output"

        data["cache_dir"] = f"./{cache_dir_name}"
        data["output_dir"] = f"./{output_dir_name}"

        # 1. Guardar en raíz con nombre genérico
        write_json_file(TRAIN_CONFIG, data)

        # El Pre-Cache codifica el prompt y la imagen de las previews desde su propio json.
        precache_cfg = read_json_file(PRECACHE_CONFIG, {})
        for key in PREVIEW_KEYS:
            precache_cfg[key] = data.get(key, "")
        write_json_file(PRECACHE_CONFIG, precache_cfg)

        # Si la caché no tiene el prompt o la imagen de preview guardados, se codifican en CPU en paralelo
        # (sin tocar la GPU del entrenamiento) y el trainer los relee antes de la siguiente preview.
        prompt = data.get("preview_custom_prompt", "").strip()
        trigger = data.get("trigger_word", "").strip()
        if trigger and prompt and trigger.lower() not in prompt.lower():
            prompt = f"{trigger}, {prompt}"
        wanted = [prompt, data.get("preview_edit_image", "").strip() if data.get("lora_type") == "edit" else ""]
        cache_dir = resolve_config_path(data["cache_dir"], cache_dir_name)
        encoded = [(cache_dir / f).read_text(encoding="utf-8") if (cache_dir / f).exists() else None
                   for f in ("_custom_prompt.txt", "_custom_image.txt")]
        if prompt and wanted != encoded and (cache_dir / "_neg_embed.pt").exists() and get_status()["script"] != "precache":
            encode_preview_prompt(cache_dir, *wanted)
        saved_files = [TRAIN_CONFIG.name]

        # 2. Guardar copia dentro de la carpeta del proyecto
        if proj:
            output_dir_path = resolve_config_path(data["output_dir"], output_dir_name)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_json_file = output_dir_path / f"train_settings_{proj}.json"
            write_json_file(output_json_file, data)
            saved_files.append(f"{output_dir_name}/{output_json_file.name}")

        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files, "encoding": encoding_stage(), "encoding_device": prompt_job["device"]})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# EJECUCIÓN DE SCRIPT & STREAMING
# =============================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({**get_status(), "encoding": encoding_stage(), "encoding_device": prompt_job["device"]})


@app.route("/api/checkpoint-info", methods=["GET"])
def checkpoint_info():
    output_dir = get_train_output_dir()
    step_file = output_dir / "current_step.txt"
    resume_dir = output_dir / "resume_checkpoint"
    
    has_checkpoint = step_file.exists() and resume_dir.exists()
    current_step = 0
    if step_file.exists():
        try:
            current_step = int(step_file.read_text(encoding="utf-8").strip())
        except Exception:
            pass
            
    return jsonify({
        "has_checkpoint": has_checkpoint,
        "current_step": current_step,
        "output_dir": str(output_dir)
    })


@app.route("/api/run", methods=["POST"])
def run_script():
    data = request.get_json(force=True) or {}
    script_name = data.get("script")
    script_path = get_script_for_name(script_name)

    if script_path is None or not script_path.exists():
        return jsonify({"status": "error", "error": t("Script not found: {name}", name=script_name)}), 404

    if encoding_stage() is not None:
        return jsonify({"status": "error", "error": t("Encoding the preview prompt, wait until it finishes.")}), 409

    ok, error = runtime.start(script_name, script_path)
    if not ok:
        return jsonify({"status": "error", "error": error}), 409
    return jsonify({"status": "ok", "script": script_name})

@app.route("/api/stop", methods=["POST"])
def stop_script():
    return jsonify(runtime.stop())

@app.route("/api/previews", methods=["GET"])
def get_previews():
    output_dir = get_train_output_dir()
    previews = []
    if output_dir.is_dir():
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith("preview_step_") and file_path.suffix.lower() == ".png":
                st = file_path.stat()
                # Un PNG recien escrito puede estar a medias: se ensena en el siguiente sondeo.
                if st.st_size > 0 and time.time() - st.st_mtime > 1.0:
                    previews.append((file_path.name, st.st_mtime))
    previews.sort(key=lambda p: p[1], reverse=True)
    previews = previews[:50]
    # versions = mtime en ms, para que el navegador no reutilice una imagen vieja con el mismo nombre.
    return jsonify({"output_dir": str(output_dir), "previews": [n for n, _ in previews],
                    "versions": {n: int(m * 1000) for n, m in previews}})


@app.route("/api/preview/<path:filename>")
def serve_preview(filename):
    output_dir = get_train_output_dir()
    requested = (output_dir / filename).resolve()
    try:
        requested.relative_to(output_dir.resolve())
    except ValueError:
        return "", 403
    if not requested.is_file():
        return "", 404
    response = send_from_directory(str(output_dir), requested.name)
    response.headers["Cache-Control"] = "no-store"
    return response


EDIT_SUFFIXES = ("_before", "_after")


def caption_path(image_path):
    # En un par de edición (nombre_before / nombre_after) el caption es uno solo: nombre.txt.
    stem = image_path.stem
    for suffix in EDIT_SUFFIXES:
        if stem.endswith(suffix):
            return image_path.with_name(stem[:-len(suffix)] + ".txt")
    return image_path.with_suffix(".txt")


def dataset_caption_paths(dataset_dir):
    # Un .txt por muestra: los dos lados de un par comparten el suyo y solo cuentan una vez.
    return list(dict.fromkeys(caption_path(f) for f in sorted(dataset_dir.iterdir())
                              if f.is_file() and f.suffix.lower() in DATASET_EXTS))


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    dataset_dir = get_dataset_dir()
    images = []
    if dataset_dir.is_dir():
        for file_path in sorted(dataset_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in DATASET_EXTS:
                txt_path = caption_path(file_path)
                caption = ""
                if txt_path.exists():
                    try:
                        caption = txt_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass
                # mtime va a la URL (?v=) para que el navegador no muestre una imagen reemplazada desde su caché.
                images.append({"file": file_path.name, "mtime": int(file_path.stat().st_mtime),
                               "has_txt": txt_path.exists(), "caption": caption})
    return jsonify({
        "path": str(dataset_dir),
        "image_count": len(images),
        "caption_count": sum(1 for t in dataset_caption_paths(dataset_dir) if t.exists()) if dataset_dir.is_dir() else 0,
        "images": images[:500]
    })


@app.route("/api/dataset-image/<path:filename>")
def serve_dataset_image(filename):
    dataset_dir = get_dataset_dir()
    try:
        requested = (dataset_dir / filename).resolve()
        requested.relative_to(dataset_dir.resolve())
    except Exception:
        return "", 403
    if not requested.is_file() or requested.suffix.lower() not in DATASET_EXTS:
        return "", 404
    resp = send_from_directory(str(dataset_dir), requested.name)
    # Sin Cache-Control el navegador aplica caché heurística y sigue mostrando una imagen reemplazada.
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/save-caption", methods=["POST"])
def save_caption():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        caption = data.get("caption", "").strip()
        dataset_dir = get_dataset_dir()
        
        img_path = (dataset_dir / filename).resolve()
        img_path.relative_to(dataset_dir.resolve())
        
        txt_path = caption_path(img_path)
        txt_path.write_text(caption, encoding="utf-8")
        
        return jsonify({"status": "ok", "file": txt_path.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/batch-caption", methods=["POST"])
def batch_caption():
    """
    Aplica a todos los captions: antepone el trigger donde falte y, si hay texto
    común, lo añade al final (append), sustituye el caption (replace) o lo quita (remove).
    """
    try:
        data = request.get_json(force=True)
        trigger = data.get("trigger_word", "").strip()
        common = str(data.get("common", "") or "").strip()
        mode = str(data.get("mode", "append") or "append").strip().lower()
        dataset_dir = get_dataset_dir()
        count = 0

        if not dataset_dir.is_dir() or not (trigger or common):
            return jsonify({"status": "ok", "updated_count": 0})

        for txt_path in dataset_caption_paths(dataset_dir):
            current = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""

            if not common:
                text = current
            elif mode == "remove":
                # Quita el texto y la puntuación que lo rodeaba, sin tocar el resto del caption.
                text = re.sub(re.escape(common), "", current, flags=re.IGNORECASE)
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\s+([,.;:!?])", r"\1", text)
                text = re.sub(r"^[\s,;:.]+", "", text)
                text = re.sub(r"[,;:]\s*$", "", text).strip()
            elif mode == "replace":
                text = common
            elif common.lower() in current.lower():
                # Pulsar dos veces no duplica la frase.
                text = current
            elif current:
                base = current if current[-1] in ".!?,;:" else current + "."
                text = f"{base} {common}"
            else:
                text = common

            # En modo remove no se antepone nada: se acaba de pedir quitar una palabra.
            if mode != "remove" and trigger and trigger.lower() not in text.lower():
                text = f"{trigger}, {text}".strip(", ")

            if text != current:
                txt_path.write_text(text, encoding="utf-8")
                count += 1

        return jsonify({"status": "ok", "updated_count": count})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/clear-dataset", methods=["POST"])
def clear_dataset():
    """
    Vacía la carpeta del dataset: sus imágenes (y vídeos y audios donde los hay) y sus .txt. Las
    subcarpetas y los demás ficheros no se tocan. Con un proceso en marcha no se borra nada (lo estaría leyendo).
    La GUI pide escribir el número de ficheros antes de llamar aquí.
    """
    try:
        if get_status()["running"]:
            return jsonify({"status": "error", "error": t("A process is running")}), 409
        dataset_dir = get_dataset_dir()
        removed, errors = 0, []
        if dataset_dir.is_dir():
            for f in dataset_dir.iterdir():
                if f.is_file() and f.suffix.lower() in DATASET_EXTS + (".txt",):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception as exc:
                        errors.append(f"{f.name}: {exc}")
        return jsonify({"status": "ok" if not errors else "partial", "removed": removed, "errors": errors, "path": str(dataset_dir)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/clear-captions", methods=["POST"])
def clear_captions():
    """
    Borra el .txt de todas las imágenes. Se borra el fichero en vez de vaciarlo:
    un .txt vacío contaría como caption y el pre-caché entrenaría con texto en blanco.
    """
    try:
        dataset_dir = get_dataset_dir()
        removed, errors = 0, []
        if dataset_dir.is_dir():
            for txt in dataset_caption_paths(dataset_dir):
                if txt.exists():
                    try:
                        txt.unlink()
                        removed += 1
                    except Exception as exc:
                        errors.append(f"{txt.name}: {exc}")
        return jsonify({"status": "ok" if not errors else "partial", "removed": removed, "errors": errors})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/delete-dataset-image", methods=["POST"])
def delete_dataset_image():
    """Borra una imagen del dataset y su .txt. En un par de edición borra las dos mitades: una sola no sirve."""
    try:
        filename = str(request.get_json(force=True).get("file", "")).strip()
        dataset_dir = get_dataset_dir()
        target = (dataset_dir / filename).resolve()

        # El nombre viene del navegador: tiene que seguir dentro del dataset.
        if not filename or dataset_dir.resolve() not in target.parents:
            return jsonify({"status": "error", "error": t("Path outside the dataset")}), 400
        if not target.is_file() or target.suffix.lower() not in DATASET_EXTS:
            return jsonify({"status": "error", "error": t("Not found: {name}", name=filename)}), 404

        files = [target]
        for suffix in EDIT_SUFFIXES:
            if target.stem.endswith(suffix):
                base = target.stem[:-len(suffix)]
                files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in DATASET_EXTS
                         and f.stem in (base + "_before", base + "_after")]
        files.append(caption_path(target))

        removed = []
        for f in files:
            if f.is_file():
                f.unlink()
                removed.append(f.name)
        return jsonify({"status": "ok", "removed": removed})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/delete-project-data", methods=["POST"])
def delete_project_data():
    """Vacía la pre-caché o la salida de entrenamiento del proyecto actual (sin borrar la carpeta)."""
    try:
        target = str(request.get_json(force=True).get("target", "")).strip().lower()
        if target == "precache":
            path = get_precache_dir()
        elif target == "training":
            path = get_train_output_dir()
        else:
            return jsonify({"status": "error", "error": t("Unknown target: {name}", name=target)}), 400

        # Un proceso en marcha tiene ficheros abiertos en esas carpetas.
        if get_status()["running"]:
            return jsonify({"status": "error",
                            "error": t("A process is running. Stop it first.")}), 409

        removed, errors = 0, []
        if path.is_dir():
            for entry in path.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    removed += 1
                except Exception as exc:
                    errors.append(f"{entry.name}: {exc}")
        return jsonify({"status": "ok" if not errors else "partial", "removed": removed, "errors": errors, "path": str(path)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/caption-settings", methods=["GET"])
def get_caption_settings():
    return jsonify(read_json_file(CAPTION_CONFIG, {}))


@app.route("/api/save-caption-settings", methods=["POST"])
def save_caption_settings():
    """Guarda los ajustes del auto-caption. Se fusiona para no perder lo editado a mano en el JSON."""
    try:
        merged = read_json_file(CAPTION_CONFIG, {})
        merged.update(request.get_json(force=True) or {})
        write_json_file(CAPTION_CONFIG, merged)
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def open_browser():
    try:
        webbrowser.open(remote_access.local_url("trainer_port"))
    except Exception:
        pass


i18n.register(app)
file_transfer.register(app, get_dataset_dir, get_train_output_dir, DATASET_EXTS)
runtime.register(app)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ACADEMIASD — FLUX.2 KLEIN 9B LORA TRAINER WEB SERVER")
    print("=" * 70)
    print(f"  {t('Base folder'):<20}: {BASE_DIR}")
    print(f"  {t('Python interpreter'):<20}: {sys.executable}")
    print(f"  URL                 : {remote_access.local_url('trainer_port')}")
    print("=" * 70 + "\n")

    threading.Timer(1.2, open_browser).start()

    remote_access.serve(app, "trainer_port")