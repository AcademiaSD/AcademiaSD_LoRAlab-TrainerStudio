# -*- coding: utf-8 -*-
"""
best_checkpoint.py — El mejor checkpoint, elegido al acabar el entrenamiento
The best checkpoint, chosen when the training ends

Con el modelo aún cargado se generan 4 imágenes fijas (2 prompts × 2 semillas, o 1 prompt × 4
semillas) con el modelo sin LoRA y con N checkpoints guardados, repartidos por el entrenamiento
(xy_grid.XYGrid.best_targets). Cada checkpoint se puntúa con tres medidas:

- Parecido: con caras en el dataset (personajes) y Face likeness instalado, el % de parecido facial
  (InsightFace, face_match.py); si no (estilos, objetos), la similitud SigLIP de las imágenes con la
  media del dataset.
- Fidelidad al prompt: similitud SigLIP texto-imagen. Si cae respecto al modelo sin LoRA, el LoRA se
  está comiendo el prompt.
- Variedad: similitud SigLIP entre las semillas del mismo prompt. Si sube mucho respecto al modelo
  sin LoRA, todas las semillas salen iguales: el LoRA está sobreentrenado.

Puntuación = parecido normalizado (0 = sin LoRA, 1 = el máximo de los checkpoints) − caída de
fidelidad / 10 (en puntos de coseno ×100) − 2·subida de la similitud entre semillas (/100). Se recomienda el primer checkpoint a menos de 0,05 del
máximo: cuando el parecido ya no sube, el más temprano es el que menos se ha sobrecocido. Es una
heurística: la gráfica y la hoja de imágenes enseñan los datos para decidir. El elegido se copia como
<Modelo>_BEST_LoRA.safetensors junto a los demás.

SigLIP: google/siglip-base-patch16-224 (Apache 2.0, ~0,8 GB), en Models/SigLIP-base-patch16-224 la
primera vez que se usa.
"""
import glob
import json
import os
import re
import shutil
import time

import torch

from i18n import t

SIGLIP_REPO = "google/siglip-base-patch16-224"
SIGLIP_DIR = "Models/SigLIP-base-patch16-224"
SIGLIP_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
                "spiece.model", "tokenizer.json", "tokenizer_config.json"]
RESULT = "best_checkpoint.json"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ── Juez: SigLIP + parecido facial ──────────────────────────────────────────────────────────────

class Judge:
    def __init__(self, dataset_dir, log):
        from PIL import Image
        from transformers import SiglipModel, SiglipProcessor
        from transformers.utils import logging as hf_logging
        from project_layout import resolve_model_dir

        hf_logging.set_verbosity_error()      # avisos de config de SigLIP que no afectan a nada

        self.log = log
        self.files = sorted(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                            if f.lower().endswith(IMAGE_EXTS)) if dataset_dir and os.path.isdir(dataset_dir) else []
        folder = str(resolve_model_dir(None, SIGLIP_DIR))
        if not os.path.isfile(os.path.join(folder, "model.safetensors")):
            from huggingface_hub import snapshot_download
            log("[BEST] " + t("Downloading {name} (~{gb} GB, only the first time)...", name=SIGLIP_REPO, gb="0.8"))
            snapshot_download(SIGLIP_REPO, local_dir=folder, allow_patterns=SIGLIP_FILES)
        # En la GPU si queda sitio junto al modelo del entrenamiento; si no, en la CPU (más lento).
        self.device = "cpu"
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free > 2 * 1024 ** 3:
                self.device = "cuda"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SiglipModel.from_pretrained(folder, dtype=dtype).to(self.device).eval()
        self.proc = SiglipProcessor.from_pretrained(folder)

        refs = []
        for f in self.files:
            try:
                with Image.open(f) as im:
                    refs.append(im.convert("RGB"))
            except OSError:
                continue
        self.dataset = self.images(refs) if refs else None
        self.centroid = None
        if self.dataset is not None:
            c = self.dataset.mean(0)
            self.centroid = c / c.norm()

        # Personaje: la mayoría del dataset tiene cara y Face likeness está instalado.
        self.faces = None
        try:
            import face_match
            if face_match.installed() and self.files:
                scorer = face_match.Scorer(self.files)
                if scorer.ref is not None and scorer.faces >= max(1, len(self.files) // 2):
                    self.faces = scorer
        except Exception as e:
            log("[BEST] " + t("Face likeness not available: {error}", error=e))

    @torch.no_grad()
    def images(self, imgs):
        from PIL import Image
        out = []
        for i in range(0, len(imgs), 16):
            batch = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in imgs[i:i + 16]]
            inputs = self.proc(images=batch, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.model.dtype))
            emb = getattr(emb, "pooler_output", emb)
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def texts(self, texts):
        inputs = self.proc(text=texts, padding="max_length", truncation=True, max_length=64,
                           return_tensors="pt").to(self.device)
        emb = self.model.get_text_features(input_ids=inputs["input_ids"])
        emb = getattr(emb, "pooler_output", emb)
        return torch.nn.functional.normalize(emb.float(), dim=-1).cpu()

    def likeness(self, img):
        from PIL import Image
        v = self.faces.score(Image.fromarray(img))
        return 0.0 if v is None else v

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def measure(judge, imgs, tests, text_emb):
    """Parecido, fidelidad al prompt y similitud entre semillas (en %) de un juego de imágenes."""
    emb = judge.images(imgs)
    if judge.faces is not None:
        resemblance = sum(judge.likeness(im) for im in imgs) / len(imgs)
    elif judge.centroid is not None:
        resemblance = float((emb @ judge.centroid).mean()) * 100
    else:
        resemblance = 0.0
    adherence = float(sum(float(emb[i] @ text_emb[p]) for i, (p, _, _) in enumerate(tests)) / len(tests)) * 100
    pairs = [(i, j) for i in range(len(tests)) for j in range(i + 1, len(tests)) if tests[i][0] == tests[j][0]]
    seed_sim = (sum(float(emb[i] @ emb[j]) for i, j in pairs) / len(pairs) * 100) if pairs else 0.0
    return {"resemblance": round(resemblance, 2), "adherence": round(adherence, 2), "seed_sim": round(seed_sim, 2)}


def score(rows, base):
    """Puntuación de cada checkpoint y el recomendado (ver la cabecera)."""
    top = max(r["resemblance"] for r in rows)
    span = max(top - base["resemblance"], 1e-6)
    for r in rows:
        gain = max(0.0, min(1.0, (r["resemblance"] - base["resemblance"]) / span))
        # Fidelidad en puntos de coseno SigLIP ×100 (pequeños y a veces negativos): 3 puntos menos que
        # el modelo sin LoRA ya es mucho, y restan 0,3.
        drop = max(0.0, base["adherence"] - r["adherence"]) / 10
        collapse = max(0.0, r["seed_sim"] - base["seed_sim"]) / 100
        r["score"] = round(gain - 3 * drop - 2 * collapse, 3)
    best = max(r["score"] for r in rows)
    return next(r for r in rows if r["score"] >= best - 0.05)


# ── Ficheros ────────────────────────────────────────────────────────────────────────────────────

def lora_file(output_dir, step, final_step):
    pattern = "*_FINAL_LoRA.safetensors" if step == final_step else f"*_step_{step}.safetensors"
    found = [f for f in glob.glob(os.path.join(output_dir, pattern)) if "_BEST" not in os.path.basename(f)]
    return max(found, key=os.path.getmtime) if found else None


def copy_best(output_dir, step, final_step):
    for old in glob.glob(os.path.join(output_dir, "*_BEST_LoRA.safetensors")):
        os.remove(old)
    src = lora_file(output_dir, step, final_step)
    if not src:
        return None
    stem = os.path.splitext(os.path.basename(src))[0]
    name = re.sub(r"_LoRA_step_\d+$|_step_\d+$|_FINAL_LoRA$", "_BEST_LoRA", stem)
    if name == stem:
        name = stem + "_BEST_LoRA"
    dst = os.path.join(output_dir, name + ".safetensors")
    shutil.copyfile(src, dst)
    return dst


# ── Evaluación ──────────────────────────────────────────────────────────────────────────────────

def evaluate(grid, model, final_step, prompts, render, seed):
    """grid: el XYGrid del trainer (instantáneas y ajustes); prompts/render como en XYGrid.run."""
    from xy_grid import strength, _trainable, compose
    log = grid.log
    log("\n" + "=" * 65)
    if not grid.save_every:
        log("[BEST] " + t("The best checkpoint needs Save Every: without saved checkpoints there is nothing to choose."))
        return None
    rows = grid.snapshot_rows(final_step, grid.best_targets(final_step))
    if len(prompts) >= 2:
        tests = [(p, prompts[p], seed + s) for p in range(2) for s in range(2)]
    else:
        tests = [(0, prompts[0], seed + s) for s in range(4)]
    total = (len(rows) + 1) * len(tests)
    log("[BEST] " + t("Best checkpoint: {r} checkpoints and the model without LoRA, {n} images each ({total} images).",
                      r=len(rows), n=len(tests), total=total))
    log("[BEST] " + t("Checkpoints: {steps}", steps=", ".join(str(s) for s, _ in rows)))

    params = _trainable(model)
    final = {n: p.detach().clone() for n, p in params.items()}
    sets = []                       # [(paso o None = sin LoRA, [imágenes])]
    done, t0 = 0, time.time()
    was_training = model.training
    model.eval()

    def render_set(label):
        nonlocal done
        imgs = []
        for _, (_, item, size), s in tests:
            imgs.append(render(item, size, s))
            done += 1
            eta = (time.time() - t0) / done * (total - done)
            log(f"[BEST] {done}/{total} | {label} | ETA {int(eta // 60):02d}:{int(eta % 60):02d}")
        return imgs

    try:
        with strength(model, 0.0):
            sets.append((None, render_set(t("No LoRA"))))
        for step, path in rows:
            grid._load(params, final if path is None else path)
            sets.append((step, render_set(f"{t('Step')} {step}")))
    finally:
        grid._load(params, final)
        if was_training:
            model.train()

    log("[BEST] " + t("Scoring the images..."))
    dataset = None
    if grid.dataset:
        from project_layout import resolve_project_path
        dataset = str(resolve_project_path(grid.dataset, "./dataset"))
    judge = Judge(dataset, log)
    try:
        text_emb = judge.texts([text for text, _, _ in prompts[:2]])
        results = []
        for step, imgs in sets:
            m = measure(judge, imgs, tests, text_emb)
            m["step"] = step
            results.append(m)
        mode = "face" if judge.faces is not None else "style"
        cell_scores = ([[judge.likeness(im) for im in imgs] for _, imgs in sets] if judge.faces is not None else None)
    finally:
        judge.close()

    base, rows_m = results[0], results[1:]
    best = score(rows_m, base)
    label = t("Face likeness") if mode == "face" else t("Similarity to the dataset")
    log("[BEST] " + f"{t('Step'):>8} | {label:>24} | {t('Prompt adherence'):>16} | {t('Seed similarity'):>15} | {t('Score'):>6}")
    for r in [dict(base, score=None)] + rows_m:
        step = t("No LoRA") if r["step"] is None else str(r["step"])
        mark = "  ★" if r is best else ""
        sc = "" if r["score"] is None else f"{r['score']:6.2f}"
        log("[BEST] " + f"{step:>8} | {r['resemblance']:24.1f} | {r['adherence']:16.2f} | {r['seed_sim']:15.1f} | {sc:>6}{mark}")

    os.makedirs(grid.dir, exist_ok=True)
    copied = copy_best(grid.output_dir, best["step"], final_step)
    if copied:
        log("✓ " + t("Best checkpoint: step {n}. Copied as {file}", n=best["step"], file=os.path.basename(copied)))
    else:
        log("[BEST] " + t("Best checkpoint: step {n} (its LoRA file was not found to copy it).", n=best["step"]))

    # Hoja de imágenes: filas = sin LoRA y checkpoints, columnas = las 4 pruebas.
    sheet = os.path.join(grid.dir, f"xy_grid_best_step_{final_step}.png")
    col_labels = [f"P{p + 1} · seed {s}" for p, _, s in tests]
    row_labels = [t("No LoRA") if st is None else f"{t('Step')}\n{st}" + ("\nBEST" if st == best["step"] else "")
                  for st, _ in sets]
    compose([imgs for _, imgs in sets], [st or 0 for st, _ in sets], [1.0] * len(tests),
            t("Best checkpoint: step {n}", n=best["step"]) + " — " + prompts[0][0], seed, sheet,
            cell_scores, col_labels=col_labels, row_labels=row_labels)

    data = {"final_step": final_step, "mode": mode, "best": best["step"],
            "file": os.path.basename(copied) if copied else None, "sheet": os.path.basename(sheet),
            "base": base, "checkpoints": rows_m, "prompts": [p[0] for p in prompts[:2]],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = os.path.join(grid.dir, RESULT + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(grid.dir, RESULT))
    return data
