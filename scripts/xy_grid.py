# -*- coding: utf-8 -*-
"""
xy_grid.py — Rejilla final de pruebas: checkpoints × fuerza del LoRA, al acabar el entrenamiento
Final test grid: checkpoints × LoRA strength, when the training ends

Con el modelo aún cargado, el trainer genera la misma imagen (mismo prompt y semilla) con varios
checkpoints (filas) y varias fuerzas del LoRA (columnas), y la monta en una sola imagen por prompt en
<salida>/xy_grid/. Así se ve qué paso y qué fuerza usar sin ir a ComfyUI.

- Checkpoints: durante el entrenamiento se guarda una instantánea de los pesos entrenables (LoRA, LoKr
  o DoRA, en BF16 como el LoRA exportado) en los pasos Total·k/N, k = 1..N-1; la última fila es el
  estado final, en memoria. Se guarda en cada paso que toca, aunque no coincida con Save Every.
- Fuerza: como la aplica ComfyUI, W + s·ΔW. En LoRA y LoKr basta con multiplicar su escala; en DoRA
  la salida se interpola entre la capa sin y con el LoRA (ComfyUI interpola el peso ya normalizado).
- Cada trainer aporta solo render(embed, size, seed) -> imagen HxWx3 uint8, y los prompts.

With the model still loaded, the trainer renders the same image (same prompt and seed) with several
checkpoints (rows) and LoRA strengths (columns), and joins them into one image per prompt in
<output>/xy_grid/. Snapshots of the trainable weights are taken at steps Total·k/N while training;
the last row is the final state. Strength works as in ComfyUI: W + s·ΔW.
"""
import json
import os
import shutil
import time
from contextlib import contextmanager

import torch

from i18n import t

DEFAULTS = {"xy_grid": False, "xy_checkpoints": 4, "xy_strengths": "0.75, 1.0, 1.25",
            "best_checkpoint": False, "best_checkpoints": 8}
DIR_NAME = "xy_grid"
SNAP_DIR = "snapshots"


def settings(cfg):
    """(activa, nº de checkpoints, [fuerzas]) desde el json de entrenamiento."""
    enabled = bool(cfg.get("xy_grid", DEFAULTS["xy_grid"]))
    try:
        rows = max(1, min(8, int(cfg.get("xy_checkpoints", DEFAULTS["xy_checkpoints"]))))
    except (TypeError, ValueError):
        rows = DEFAULTS["xy_checkpoints"]
    strengths = []
    for part in str(cfg.get("xy_strengths", DEFAULTS["xy_strengths"]) or "").replace(";", ",").split(","):
        try:
            value = float(part.strip())
        except ValueError:
            continue
        if 0.0 <= value <= 3.0 and value not in strengths:
            strengths.append(value)
    return enabled, rows, (strengths[:8] or [1.0])


def best_settings(cfg):
    """(activa, nº de checkpoints a probar, Save Every) para el mejor checkpoint (best_checkpoint.py)."""
    enabled = bool(cfg.get("best_checkpoint", DEFAULTS["best_checkpoint"]))
    try:
        count = max(2, min(16, int(cfg.get("best_checkpoints", DEFAULTS["best_checkpoints"]))))
    except (TypeError, ValueError):
        count = DEFAULTS["best_checkpoints"]
    try:
        save_every = max(0, int(cfg.get("save_every") or 0))
    except (TypeError, ValueError):
        save_every = 0
    return enabled, count, save_every


def _read_cfg(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


# ── Fuerza del LoRA ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def strength(model, s):
    """Aplica el LoRA con fuerza s mientras dura el bloque (s = 1: sin cambios)."""
    undo = []
    if s != 1.0:
        from peft.tuners.lora import LoraLayer
        from lora_options import LoKrLinear
        for module in model.modules():
            if isinstance(module, LoKrLinear):
                undo.append((module, "scale", module.scale))
                module.scale = module.scale * s
            elif isinstance(module, LoraLayer):
                adapters = [a for a in module.active_adapters if a in module.scaling]
                if any(module.use_dora.get(a, False) for a in adapters):
                    undo.append((module, "forward", module.__dict__.get("forward")))
                    module.forward = _interpolated(module, module.forward, s)
                else:
                    for a in adapters:
                        undo.append((module.scaling, a, module.scaling[a]))
                        module.scaling[a] = module.scaling[a] * s
            elif getattr(module, "lora_A", None) is not None and isinstance(getattr(module, "lora_scale", None), (int, float)):
                # LoRA propio de un trainer (MiniMax-H3 Image): salida = base + lora_scale·B·A.
                undo.append((module, "lora_scale", module.lora_scale))
                module.lora_scale = module.lora_scale * s
    try:
        yield
    finally:
        for target, key, old in reversed(undo):
            if isinstance(target, dict):
                target[key] = old
            elif key == "forward":
                if old is None:
                    target.__dict__.pop("forward", None)
                else:
                    target.forward = old
            else:
                setattr(target, key, old)


def _interpolated(module, full_forward, s):
    # DoRA: base(x) + s·(capa con DoRA(x) - base(x)), lo mismo que interpolar el peso como ComfyUI.
    def forward(x, *args, **kwargs):
        full = full_forward(x, *args, **kwargs)
        base = module.base_layer(x, *args, **kwargs)
        return base + (full - base) * s
    return forward


# ── Imágenes desde la preview de cada trainer ───────────────────────────────────────────────────

def preview_renderer(call, module_globals, seed_key="SEED"):
    """
    render(item, size, seed) para XYGrid.run a partir de la preview que ya tiene el trainer:
    call(item, size) genera una preview normal. Se fija su semilla (la variable global SEED del
    trainer) y se recoge la imagen que iba a guardar con preview_prompts.save_png, sin escribirla.
    Lo que imprime la preview se calla; si falla, se enseña antes de seguir con el error.
    """
    import contextlib
    import io
    import sys
    import numpy as np
    import preview_prompts

    def render(item, size, seed):
        box = []
        original = preview_prompts.save_png
        old_seed = module_globals.get(seed_key)
        module_globals[seed_key] = seed
        preview_prompts.save_png = lambda image, path, prompt="": box.append(image)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                call(item, size)
        except BaseException:
            sys.stdout.write(out.getvalue())
            raise
        finally:
            preview_prompts.save_png = original
            module_globals[seed_key] = old_seed
        if not box:
            raise RuntimeError("the preview did not produce an image")
        image = box[-1]
        return np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    return render


def _quiet(render):
    """La consola solo enseña el progreso de la rejilla: lo que imprime cada imagen se calla (y se
    enseña si falla)."""
    import contextlib
    import io
    import sys

    def run(item, size, seed):
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                return render(item, size, seed)
        except BaseException:
            sys.stdout.write(out.getvalue())
            raise
    return run


# ── La rejilla ──────────────────────────────────────────────────────────────────────────────────

class XYGrid:
    def __init__(self, output_dir, config_path, cfg, log=print):
        self.dir = os.path.join(output_dir, DIR_NAME)
        self.snap_dir = os.path.join(self.dir, SNAP_DIR)
        self.config_path = config_path
        self.enabled, self.rows, self.strengths = settings(cfg)
        self.best, self.best_n, self.save_every = best_settings(cfg)
        self.dataset = cfg.get("dataset_path")
        self.output_dir = output_dir
        self.log = log
        self._mtime = None

    def _refresh(self):
        # La GUI puede cambiar la rejilla con el entrenamiento en marcha (Save JSON): se relee el json.
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return
        if mtime != self._mtime:
            self._mtime = mtime
            fresh = _read_cfg(self.config_path)
            self.enabled, self.rows, self.strengths = settings(fresh)
            self.best, self.best_n, self.save_every = best_settings(fresh)
            self.dataset = fresh.get("dataset_path", self.dataset)

    def begin(self, start_step):
        """Un entrenamiento nuevo (paso 0) borra las instantáneas de uno anterior en la misma carpeta."""
        if start_step == 0 and os.path.isdir(self.snap_dir):
            shutil.rmtree(self.snap_dir, ignore_errors=True)

    def targets(self, total_steps):
        """Pasos de las filas anteriores a la última: Total·k/N, k = 1..N-1."""
        return sorted({max(1, round(total_steps * k / self.rows)) for k in range(1, self.rows)})

    def best_targets(self, total_steps):
        """
        Checkpoints que prueba el mejor checkpoint, antes del final: Total·k/N llevados al Save Every
        más cercano, para que cada uno tenga su LoRA guardado (el elegido se copia como _BEST).
        """
        if not self.save_every:
            return []
        steps = set()
        for k in range(1, self.best_n):
            step = round(total_steps * k / self.best_n / self.save_every) * self.save_every
            if 0 < step < total_steps:
                steps.add(step)
        return sorted(steps)

    def maybe_snapshot(self, model, step, total_steps):
        """Llamar en cada paso, después del optimizador. Solo guarda en los pasos de la rejilla y del
        mejor checkpoint."""
        self._refresh()
        wanted = ((self.enabled and step in self.targets(total_steps))
                  or (self.best and step in self.best_targets(total_steps)))
        if not wanted:
            return
        from safetensors.torch import save_file
        os.makedirs(self.snap_dir, exist_ok=True)
        tensors = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in _trainable(model).items()}
        tmp = os.path.join(self.snap_dir, f"step_{step}.tmp")
        save_file(tensors, tmp)
        os.replace(tmp, os.path.join(self.snap_dir, f"step_{step}.safetensors"))
        self.log("  ↳ " + t("XY grid: snapshot of step {n} saved.", n=step))

    def snapshot_rows(self, final_step, targets):
        """[(paso, ruta o None = estado final)]: la instantánea más cercana a cada objetivo, sin repetir."""
        found = self._snapshots(final_step)
        rows = []
        for target in targets:
            if not found:
                break
            best = min(found, key=lambda s: abs(s - target))
            rows.append((best, found.pop(best)))
        rows.sort()
        return rows + [(final_step, None)]

    def _snapshots(self, final_step):
        found = {}
        if os.path.isdir(self.snap_dir):
            for f in os.listdir(self.snap_dir):
                if f.startswith("step_") and f.endswith(".safetensors"):
                    try:
                        step = int(f[5:-12])
                    except ValueError:
                        continue
                    if 0 < step < final_step:
                        found[step] = os.path.join(self.snap_dir, f)
        return found

    def _rows(self, final_step):
        return self.snapshot_rows(final_step, self.targets(final_step))

    def run(self, model, final_step, prompts, render, seed):
        """
        prompts: [(texto, item, size)]; render(item, size, seed) -> HxWx3 uint8 (ver preview_renderer).
        Genera y guarda una rejilla por prompt. Devuelve las rutas guardadas. El LoRA final ya está
        exportado: si la rejilla falla (falta de VRAM...), se avisa y el entrenamiento acaba bien.
        """
        self._refresh()
        if not prompts:
            return []
        render = _quiet(render)
        saved = []
        if self.enabled:
            try:
                saved = self._run(model, final_step, prompts, render, seed)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log("[XY] " + t("The XY grid failed: {error}. The LoRA is saved.", error=e))
        if self.best:
            try:
                import best_checkpoint
                best_checkpoint.evaluate(self, model, final_step, prompts, render, seed)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log("[BEST] " + t("The best checkpoint search failed: {error}. The LoRA is saved.", error=e))
        return saved

    def _run(self, model, final_step, prompts, render, seed):
        rows = self._rows(final_step)
        total = len(rows) * len(self.strengths) * len(prompts)
        self.log("\n" + "=" * 65)
        self.log("[XY] " + t("XY grid: {r} checkpoints × {c} strengths × {p} prompts = {n} images.",
                             r=len(rows), c=len(self.strengths), p=len(prompts), n=total))
        self.log("[XY] " + t("Checkpoints: {steps} | Strengths: {s} | Seed: {seed}",
                             steps=", ".join(str(s) for s, _ in rows),
                             s=", ".join(f"{v:g}" for v in self.strengths), seed=seed))
        params = _trainable(model)
        final = {n: p.detach().clone() for n, p in params.items()}
        cells = [[[None] * len(self.strengths) for _ in rows] for _ in prompts]
        done, t0 = 0, time.time()
        was_training = model.training
        model.eval()
        try:
            for r, (step, path) in enumerate(rows):
                self._load(params, final if path is None else path)
                for c, s in enumerate(self.strengths):
                    with strength(model, s):
                        for p, (_, item, size) in enumerate(prompts):
                            cells[p][r][c] = render(item, size, seed)
                            done += 1
                            eta = (time.time() - t0) / done * (total - done)
                            self.log(f"[XY] {done}/{total} | {t('Step')} {step} | ×{s:g} | "
                                     f"ETA {int(eta // 60):02d}:{int(eta % 60):02d}")
        finally:
            self._load(params, final)
            if was_training:
                model.train()
        scores = self._likeness(cells, rows)
        saved = []
        os.makedirs(self.dir, exist_ok=True)
        for p, (text, _, _) in enumerate(prompts):
            path = os.path.join(self.dir, f"xy_grid_step_{final_step}_{p + 1}.png")
            compose(cells[p], [s for s, _ in rows], self.strengths, text, seed, path,
                    scores[p] if scores else None)
            saved.append(path)
            self.log("✓ " + t("XY grid saved to: {path}", path=path))
        return saved

    def _likeness(self, cells, rows):
        """
        % de parecido de cada celda con las caras del dataset, con el mismo modelo que la galería
        (face_match.py, InsightFace en CPU). None si no está instalado o el dataset no tiene caras.
        """
        try:
            import face_match
            from project_layout import resolve_project_path
            if not face_match.installed() or not self.dataset:
                return None
            dataset = resolve_project_path(self.dataset, "./dataset")
            refs = face_match._image_files(dataset)
            if not refs:
                return None
            self.log("[XY] " + t("Face likeness of each image ({n} dataset images as reference)...", n=len(refs)))
            scorer = face_match.Scorer(refs)
            if scorer.ref is None:
                return None
            from PIL import Image
            scores = [[[scorer.score(Image.fromarray(img)) for img in row] for row in grid] for grid in cells]
        except Exception as e:
            self.log("[XY] " + t("Face likeness not available: {error}", error=e))
            return None
        # La mejor celda de cada prompt, para no tener que buscarla a ojo.
        for p, grid in enumerate(scores):
            best = max(((v, r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v is not None), default=None)
            if best:
                self.log("[XY] " + t("Prompt {p}: best likeness {v}% at step {step}, ×{s}",
                                     p=p + 1, v=f"{best[0]:.0f}", step=rows[best[1]][0], s=f"{self.strengths[best[2]]:g}"))
        return scores

    @staticmethod
    def _load(params, source):
        if isinstance(source, dict):
            for n, p in params.items():
                p.data.copy_(source[n])
            return
        from safetensors import safe_open
        with safe_open(source, framework="pt", device="cpu") as f:
            for n, p in params.items():
                if n in f.keys():
                    p.data.copy_(f.get_tensor(n).to(p.dtype))


# ── Montaje ─────────────────────────────────────────────────────────────────────────────────────

def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit(draw, text, font, width):
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _badge(draw, x, y, value, font, pad):
    # Los mismos colores que la etiqueta de la galería (face_match.js): ≥45 verde, ≥30 ámbar, si no rojo.
    if value is None:
        text, fill, edge = "—", (15, 23, 42), (71, 85, 105)
    else:
        text = f"{value:.0f}%"
        fill, edge = (((5, 150, 105), (52, 211, 153)) if value >= 45 else
                      ((180, 83, 9), (251, 191, 36)) if value >= 30 else ((185, 28, 28), (248, 113, 113)))
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill, outline=edge, width=max(1, pad // 4))
    draw.text((x + pad - box[0], y + pad - box[1]), text, fill=(255, 255, 255), font=font)


def compose(cells, steps, strengths, prompt, seed, path, scores=None, col_labels=None, row_labels=None):
    """
    cells[fila][columna] = HxWx3 uint8. Filas = pasos, columnas = fuerzas; prompt y semilla arriba.
    scores[fila][columna] = % de parecido (o None): una etiqueta en la esquina de cada imagen.
    """
    from PIL import Image, ImageDraw
    import preview_prompts
    h, w = cells[0][0].shape[:2]
    scale = max(w, h) / 1024
    fs = max(14, int(26 * scale))
    font, small, badge_font = _font(fs), _font(max(12, int(fs * 0.85))), _font(int(fs * 1.25))
    left, top, gap = int(fs * 5), int(fs * 4.6), max(4, int(6 * scale))
    W = left + len(strengths) * (w + gap)
    H = top + len(steps) * (h + gap)
    sheet = Image.new("RGB", (W, H), (15, 23, 42))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, int(fs * 0.3)), _fit(draw, prompt, small, W - 2 * gap), fill=(203, 213, 225), font=small)
    draw.text((gap, int(fs * 1.6)), f"seed {seed}", fill=(148, 163, 184), font=small)
    for c, s in enumerate(strengths):
        x = left + c * (w + gap) + w // 2
        label = col_labels[c] if col_labels else f"LoRA ×{s:g}"
        draw.text((x, top - int(fs * 0.4)), label, fill=(226, 232, 240), font=font, anchor="ms")
    for r, step in enumerate(steps):
        y = top + r * (h + gap) + h // 2
        label = row_labels[r] if row_labels else f"{t('Step')}\n{step}"
        draw.text((left // 2, y), label, fill=(226, 232, 240), font=font, anchor="mm", align="center")
        for c in range(len(strengths)):
            x0, y0 = left + c * (w + gap), top + r * (h + gap)
            sheet.paste(Image.fromarray(cells[r][c]), (x0, y0))
            if scores is not None:
                _badge(draw, x0 + int(fs * 0.5), y0 + int(fs * 0.5), scores[r][c], badge_font, int(fs * 0.4))
    preview_prompts.save_png(sheet, path, prompt)


# ── Servidor ────────────────────────────────────────────────────────────────────────────────────

def register(app, get_output_dir):
    """/xy_grid.js y la lista de rejillas del proyecto (la GUI las enseña bajo la galería)."""
    from pathlib import Path
    from flask import jsonify, send_from_directory, abort
    import preview_prompts

    gui_dir = Path(__file__).resolve().parent.parent / "GUI"

    @app.route("/xy_grid.js")
    def xy_grid_js():
        return send_from_directory(str(gui_dir), "xy_grid.js")

    @app.route("/api/xy-grids")
    def xy_grids():
        folder = Path(get_output_dir()) / DIR_NAME
        items = []
        if folder.is_dir():
            for f in folder.glob("xy_grid_*.png"):
                st = f.stat()
                if st.st_size > 0:
                    items.append({"name": f.name, "version": int(st.st_mtime * 1000),
                                  "prompt": preview_prompts.png_prompt(str(f))})
        items.sort(key=lambda i: i["version"], reverse=True)
        return jsonify({"grids": items[:12]})

    @app.route("/api/xy-best")
    def xy_best():
        # Resultado del mejor checkpoint (best_checkpoint.py), o {} si no lo hay.
        path = Path(get_output_dir()) / DIR_NAME / "best_checkpoint.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return jsonify({})
        data["exists"] = bool(data.get("file")) and (Path(get_output_dir()) / data["file"]).is_file()
        return jsonify(data)

    @app.route("/api/xy-best/export", methods=["POST"])
    def xy_best_export():
        # «Send to Models» del mejor checkpoint: lo copia a la carpeta de LoRAs con el nombre elegido.
        from flask import request
        import file_transfer
        data = request.get_json(force=True) or {}
        target_dir = str(data.get("target_dir", "")).strip()
        name = str(data.get("final_name", "")).strip()
        try:
            best = json.loads((Path(get_output_dir()) / DIR_NAME / "best_checkpoint.json").read_text(encoding="utf-8"))
            source = Path(get_output_dir()) / best["file"]
        except (OSError, ValueError, KeyError, TypeError):
            return jsonify({"status": "error", "error": t("There is no best checkpoint yet.")}), 404
        if not source.is_file():
            return jsonify({"status": "error", "error": t("File not found: {path}", path=source)}), 404
        if not target_dir or not Path(target_dir).is_dir():
            return jsonify({"status": "error", "error": t("Please select a target folder.")}), 400
        name = name or source.name
        if not name.lower().endswith(".safetensors"):
            name += ".safetensors"
        name = file_transfer.with_lora_options_suffix(name, source)
        dest = Path(target_dir) / name
        shutil.copyfile(source, dest)
        return jsonify({"status": "ok", "dest": str(dest), "filename": name, "step": best.get("best")})

    @app.route("/api/xy-grid/<path:name>")
    def xy_grid_file(name):
        folder = Path(get_output_dir()) / DIR_NAME
        if "/" in name or "\\" in name or not name.startswith("xy_grid_") or not name.endswith(".png"):
            abort(404)
        return send_from_directory(str(folder), name)
