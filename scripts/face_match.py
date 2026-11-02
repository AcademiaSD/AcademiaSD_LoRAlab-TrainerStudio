# -*- coding: utf-8 -*-
"""
face_match.py — Parecido de cara entre las previews y el dataset, en porcentaje.
Face likeness between the previews and the dataset, as a percentage.

Usa los modelos buffalo_l de InsightFace (detector SCRFD det_10g + ArcFace w600k_r50) con onnxruntime
en CPU: no toca la VRAM del entrenamiento y funciona igual en Windows, Linux, macOS y RunPod.
Los pesos de InsightFace solo se pueden usar para investigación no comercial, así que NO se descargan
por defecto: el usuario los pide desde la GUI aceptando antes, en una ventana, que los usa bajo su
responsabilidad y sin fines comerciales. La aceptación se guarda en settings/face_match.json.
Uses InsightFace's buffalo_l models with onnxruntime on CPU. Their weights are for non-commercial
research only, so they are NOT downloaded by default: the user asks for them from the GUI after
accepting, in a popup, that they use them under their own responsibility and not commercially.

Referencia: media de los embeddings de la cara más grande de cada imagen del dataset.
Parecido de una preview: coseno (x100) de su cara más parecida contra esa referencia.
Reference: mean embedding of the largest face in each dataset image. A preview's likeness: cosine
(x100) of its closest face against that reference.

Servidor:    register(app, get_dataset_dir, get_output_dir) sirve /face_match.js y /api/face-match/*.
"""
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from i18n import t
import project_layout

GUI_DIR = Path(__file__).resolve().parent.parent / "GUI"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings" / "face_match.json"
MODEL_DIR = project_layout.MODELS_DIR / "FaceMatch-InsightFace"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LICENSE_NOTE = (
    "InsightFace buffalo_l (SCRFD det_10g + ArcFace w600k_r50)\n"
    "Source: " + MODEL_URL + "\n\n"
    "The InsightFace pretrained models are available for non-commercial research purposes only.\n"
    "Downloaded by the user after accepting, under their own responsibility, not to use them commercially.\n"
    "See https://github.com/deepinsight/insightface#license\n"
)

# Puntos de la cara alineada de ArcFace (112x112): ojos, nariz y comisuras.
ARCFACE_DST = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041))
DET_SIZE = 640
DET_THRESH = 0.5
NMS_THRESH = 0.4

_state = {"downloading": False, "stage": "", "progress": 0.0, "error": ""}
_state_lock = threading.Lock()
_compute_lock = threading.Lock()
_engine = None
_reference = {"key": None, "emb": None, "faces": 0, "images": 0}
_scores = {}   # (ruta, mtime) -> porcentaje o None (sin cara)


def _read_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def installed():
    return all((MODEL_DIR / f).is_file() for f in MODEL_FILES) and importlib.util.find_spec("onnxruntime") is not None


def status():
    with _state_lock:
        st = dict(_state)
    st["installed"] = installed()
    st["accepted"] = bool(_read_settings().get("accepted"))
    return st


def _set(**kw):
    with _state_lock:
        _state.update(kw)


# =============================================================================
# DESCARGA / DOWNLOAD
# =============================================================================

def _download():
    try:
        if importlib.util.find_spec("onnxruntime") is None:
            _set(stage="onnxruntime", progress=0.0)
            print("[INFO] " + t("Face match: installing onnxruntime..."), flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
            importlib.invalidate_caches()
            if importlib.util.find_spec("onnxruntime") is None:
                raise RuntimeError(t("Package '{name}' installed but import failed.", name="onnxruntime"))

        if not all((MODEL_DIR / f).is_file() for f in MODEL_FILES):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tmp_zip = MODEL_DIR / "buffalo_l.zip.part"
            _set(stage="model", progress=0.0)
            print("[INFO] " + t("Face match: downloading {url}", url=MODEL_URL), flush=True)
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "LoRAlab-TrainerStudio"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_zip, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        _set(progress=done / total)
            # Solo hacen falta el detector y el reconocedor; el resto del zip no se guarda.
            with zipfile.ZipFile(tmp_zip) as zf:
                for name in zf.namelist():
                    base = os.path.basename(name)
                    if base in MODEL_FILES:
                        part = MODEL_DIR / (base + ".part")
                        with zf.open(name) as src, open(part, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        os.replace(part, MODEL_DIR / base)
            tmp_zip.unlink(missing_ok=True)
            (MODEL_DIR / "LICENSE_NOTE.txt").write_text(LICENSE_NOTE, encoding="utf-8")
            missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).is_file()]
            if missing:
                raise RuntimeError(t("Missing in the download: {files}", files=", ".join(missing)))
        print("[OK] " + t("Face match ready."), flush=True)
        _set(stage="done", progress=1.0, error="")
    except Exception as exc:
        print("[ERROR] " + t("Face match download failed: {error}", error=exc), flush=True)
        _set(error=str(exc))
    finally:
        _set(downloading=False)


def start_download():
    with _state_lock:
        if _state["downloading"]:
            return
        _state.update(downloading=True, stage="", progress=0.0, error="")
    threading.Thread(target=_download, daemon=True).start()


# =============================================================================
# CARAS / FACES
# =============================================================================

class _Engine:
    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        opts = ort.SessionOptions()
        # Pocos hilos: el entrenamiento comparte la CPU (carga de datos).
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        prov = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(MODEL_DIR / "det_10g.onnx"), opts, providers=prov)
        self.rec = ort.InferenceSession(str(MODEL_DIR / "w600k_r50.onnx"), opts, providers=prov)
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name

    def _detect(self, img):
        """img: PIL RGB. Devuelve [(score, bbox[4], kps[5,2]), ...] en coordenadas de la imagen."""
        from PIL import Image
        np = self.np
        w, h = img.size
        scale = DET_SIZE / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.float32)
        canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32)
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det_input: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            score = outs[i].reshape(-1)
            bbox = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride
            size = DET_SIZE // stride
            ys, xs = np.mgrid[:size, :size]
            centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, 2, axis=0)   # 2 anclas por punto / 2 anchors per point
            keep = np.where(score >= DET_THRESH)[0]
            if not len(keep):
                continue
            c, b, k = centers[keep], bbox[keep], kps[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=1))
            kpss.append(np.stack([c[:, 0:1] + k[:, 0::2], c[:, 1:2] + k[:, 1::2]], axis=-1))
            scores.append(score[keep])
        if not scores:
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)

        order = scores.argsort()[::-1]
        faces = []
        while len(order):
            i = order[0]
            faces.append((float(scores[i]), boxes[i], kpss[i]))
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
            iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
            order = rest[iou <= NMS_THRESH]
        return faces

    def _embed(self, img, kps):
        """Alinea la cara a 112x112 con sus 5 puntos y devuelve el embedding ArcFace normalizado."""
        from PIL import Image
        np = self.np
        src = np.asarray(kps, dtype=np.float64)
        dst = np.asarray(ARCFACE_DST, dtype=np.float64)
        # Transformación de semejanza (Umeyama) src -> dst.
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        cov = dc.T @ sc / len(src)
        u, s, vt = np.linalg.svd(cov)
        d = np.diag([1.0, 1.0 if np.linalg.det(u) * np.linalg.det(vt) > 0 else -1.0])
        rot = u @ d @ vt
        scale = np.trace(np.diag(s) @ d) / sc.var(0).sum()
        m = np.eye(3)
        m[:2, :2] = scale * rot
        m[:2, 2] = mu_d - scale * rot @ mu_s
        inv = np.linalg.inv(m)   # PIL pide la transformación de salida -> entrada
        face = img.transform((112, 112), Image.AFFINE, tuple(inv[:2].reshape(-1)), resample=Image.BILINEAR)
        blob = ((np.asarray(face, dtype=np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        emb = self.rec.run(None, {self.rec_input: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def faces(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        found = self._detect(img)
        # La más grande primero: en el dataset es la del personaje.
        found.sort(key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]), reverse=True)
        return [self._embed(img, kps) for _, _, kps in found[:8]]


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _Engine()
    return _engine


def _image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _reference_emb(dataset_dir):
    files = _image_files(dataset_dir)
    key = (str(dataset_dir), tuple((p.name, p.stat().st_mtime_ns) for p in files))
    if _reference["key"] != key:
        eng = _get_engine()
        embs = []
        for p in files:
            try:
                faces = eng.faces(p)
            except Exception:
                continue
            if faces:
                embs.append(faces[0])
        emb = None
        if embs:
            emb = eng.np.mean(embs, axis=0)
            emb = emb / (eng.np.linalg.norm(emb) + 1e-9)
        _reference.update(key=key, emb=emb, faces=len(embs), images=len(files))
        _scores.clear()
    return _reference["emb"]


def scores(dataset_dir, output_dir, names):
    """{nombre: porcentaje | None} para las previews pedidas (None = no se ve ninguna cara)."""
    with _compute_lock:
        ref = _reference_emb(dataset_dir)
        result = {}
        if ref is None:
            return result
        eng = _get_engine()
        out = Path(output_dir).resolve()
        for name in names:
            path = (out / name).resolve()
            if path.parent != out or not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            key = (str(path), path.stat().st_mtime_ns)
            if key not in _scores:
                try:
                    faces = eng.faces(path)
                except Exception:
                    faces = []
                best = max((float(f @ ref) for f in faces), default=None)
                _scores[key] = None if best is None else round(max(0.0, min(1.0, best)) * 100, 1)
            result[name] = _scores[key]
        return result


# =============================================================================
# SERVIDOR / SERVER
# =============================================================================

def register(app, get_dataset_dir, get_output_dir):
    from flask import jsonify, request, send_from_directory

    @app.route("/face_match.js")
    def face_match_js():
        return send_from_directory(str(GUI_DIR), "face_match.js")

    @app.route("/api/face-match/status")
    def face_match_status():
        return jsonify(status())

    @app.route("/api/face-match/download", methods=["POST"])
    def face_match_download():
        data = request.get_json(silent=True) or {}
        # Las dos casillas de la ventana; sin ellas no se descarga nada.
        if data.get("own_responsibility") is not True or data.get("non_commercial") is not True:
            return jsonify({"ok": False, "error": t("Both conditions must be accepted to download.")}), 400
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps({
            "accepted": True, "own_responsibility": True, "non_commercial": True,
            "accepted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model_url": MODEL_URL,
        }, indent=2) + "\n", encoding="utf-8")
        start_download()
        return jsonify({"ok": True, **status()})

    @app.route("/api/face-match/scores", methods=["POST"])
    def face_match_scores():
        if not installed():
            return jsonify({"ok": False, "error": t("Face match is not installed.")}), 409
        names = (request.get_json(silent=True) or {}).get("files") or []
        names = [n for n in names if isinstance(n, str)][:60]
        try:
            res = scores(get_dataset_dir(), get_output_dir(), names)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, "scores": res,
                        "reference": {"faces": _reference["faces"], "images": _reference["images"]}})
