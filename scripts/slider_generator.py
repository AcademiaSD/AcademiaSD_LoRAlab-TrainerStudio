# -*- coding: utf-8 -*-
"""
slider_generator.py — Generador de datasets sintéticos para LoRAs Slider
Synthetic dataset generator for Slider LoRAs

Escribe un dataset grupo_posición (docs/SLIDERS.md) con un modelo de edición: una imagen base por
grupo (generada, o la grupo_0 del usuario) y una edición hacia cada posición pedida (-100, -50, 50,
100), más anchors (escenas sin el sujeto, en estilos variados). Sirve para cualquier entrenador: el
dataset no depende del modelo con el que se genera.

Writes a group_position dataset (docs/SLIDERS.md) with an edit model: one base image per group
(generated, or the user's group_0) and one edit towards each requested position, plus anchors.

Lee settings/slider_gen_settings.json (lo escribe el servidor). Si se detiene, al relanzarlo sigue
donde lo dejó: solo genera las imágenes que faltan. Si se borra la base de un grupo generado, se
rehace el grupo entero (sus ediciones partían de la base anterior).

Basado en experiments/slider/gen_pairs.py y gen_anchors.py.
"""
import gc
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
import importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import slider_core

CONFIG_PATH = "settings/slider_gen_settings.json"
POSITIONS = ("-100", "-50", "50", "100")
DEFAULTS = {
    "dataset_path": "./dataset",
    "engine": "klein9b",
    "subject": "",
    "neutral": "",
    "edits": {},
    "people": 24,
    "vary_people": True,
    "anchors": 12,
    "base": "generate",   # "generate": las crea el modelo | "mine": las grupo_0 del dataset
    "size": 768,
    "seed": 1000,
}

# Variedad de las imágenes base: el slider tiene que aprender el concepto, no una cara, un sitio o una luz.
PEOPLE = [
    "a young East Asian woman with long straight black hair", "an elderly white man with a grey beard",
    "a middle-aged Black woman with short curly hair", "a teenage boy with freckles and red hair",
    "a South Asian man in his thirties with a short beard", "an old Japanese woman with white hair in a bun",
    "a blonde woman in her twenties with wavy hair", "a Latino man in his forties with glasses",
    "a young Black man with short dreadlocks", "a middle-aged woman with a brown bob haircut",
    "a bald man in his sixties", "a young Middle Eastern woman wearing a headscarf",
    "a little girl with pigtails", "a man in his fifties with salt and pepper hair",
    "an Indigenous woman with long braided hair", "a young man with a buzz cut and earrings",
    "a Korean man in his twenties with dyed blue hair", "an elderly Black man with a white moustache",
    "a woman in her thirties with freckles and a red ponytail", "a chubby man in his forties with curly brown hair",
    "a young woman with an undercut and nose piercing", "an old woman with curly grey hair and glasses",
    "a boy around ten years old with messy brown hair", "a Scandinavian man with long blond hair",
]
PLACES = ["in a park", "in a modern office", "on a city street", "in a kitchen", "against a plain grey wall",
          "in a library", "at the beach", "in a cafe", "in a living room", "in a garden"]
LIGHTS = ["soft daylight", "warm golden hour light", "studio lighting", "overcast light", "window light"]
KEEP = ("Change only that. Keep everything else exactly the same: the same identity, face, hair, clothes, "
        "pose, background, lighting, colors and framing.")
# Anchors en estilos variados: a fuerzas altas el slider no debe arrastrar el aspecto del dataset.
SCENES = [
    "a quiet beach with gentle waves", "a cozy living room with a sofa and bookshelves",
    "a modern office with desks and large windows", "a red bicycle leaning against a brick wall",
    "a kitchen counter with fruit and a coffee machine", "a park path with autumn trees",
    "a library aisle full of books", "a cafe table with a cup of coffee and a croissant",
    "a city street with parked cars", "a mountain lake", "a desk with a laptop, a lamp and a plant",
    "a garden with flowers and a wooden fence", "an old wooden door in a stone wall", "a vintage car on a country road",
    "a bowl of soup on a table", "a snowy forest",
]
STYLES = ["Realistic photo", "Black and white photo", "Vintage sepia photo", "35mm film photo",
          "Studio product photo", "Digital illustration", "Watercolor painting", "Oil painting"]


def log(en, es=None):
    print(f"{en} / {es}" if es else en, flush=True)


def fit_size(w, h, side):
    """Tamaño con el área de side x side y la proporción de la imagen, en múltiplos de 16."""
    ar = w / h
    bh = math.sqrt(side * side / ar)
    return max(16, round(ar * bh / 16) * 16), max(16, round(bh / 16) * 16)


def save_atomic(img, path):
    # Si se detiene a mitad, no queda un PNG cortado que luego se tome por terminado.
    tmp = path + ".tmp"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def get_hf_token():
    try:
        with open("settings/HF_token.json", "r", encoding="utf-8") as f:
            return json.load(f).get("token", "").strip() or None
    except Exception:
        return None


class KleinEngine:
    """FLUX.2 Klein 9B NF4 + LoRA turbo de kalle07 (4 pasos, sin CFG), como el prototipo."""
    name = "FLUX.2 Klein 9B"
    model_dir = "FLUX.2-Klein-9B_NF4"
    repo_id = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"

    def ensure(self):
        if (os.path.exists(os.path.join(self.model_dir, "transformer", "index.json"))
                and os.path.exists(os.path.join(self.model_dir, "text_encoder", "config.json"))):
            return
        log(f"Downloading {self.repo_id} (~8.8 GB)...", f"Descargando {self.repo_id} (~8,8 GB)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.repo_id, local_dir=self.model_dir, token=get_hf_token(), max_workers=2)

    def encode(self, prompts):
        """{prompt: embedding} en CPU. El text encoder se libera antes de cargar el transformer."""
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import AutoTokenizer, Qwen3Model
        log("Loading Text Encoder (Qwen3-8B NF4)...", "Cargando Text Encoder (Qwen3-8B NF4)...")
        te = Qwen3Model.from_pretrained(os.path.join(self.model_dir, "text_encoder"), dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(self.model_dir, subfolder="tokenizer")
        out = {}
        with torch.inference_mode():
            for p in prompts:
                out[p] = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, [p], dtype=torch.bfloat16, device="cuda").cpu()
        del te, tok
        gc.collect()
        torch.cuda.empty_cache()
        return out

    def load(self):
        import torch
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        # El cargador NF4 y el LoRA turbo son los del trainer. Se importa desde una carpeta temporal:
        # al importarse lee ajustes y crea su carpeta de salida en la carpeta actual.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                spec = importlib.util.spec_from_file_location("klein_trainer", os.path.join(SCRIPTS_DIR, "2_train_lora_klein9b.py"))
                kt = importlib.util.module_from_spec(spec)
                saved_argv, sys.argv = sys.argv, [sys.argv[0]]
                spec.loader.exec_module(kt)
                sys.argv = saved_argv
            finally:
                os.chdir(cwd)
        kt.TURBO_LORA_STRENGTH = 1.0
        log("Loading FLUX.2 Klein 9B (NF4) + Turbo LoRA...", "Cargando FLUX.2 Klein 9B (NF4) + Turbo LoRA...")
        transformer = kt.load_nf4_transformer(self.model_dir).to("cuda")
        # apply_turbo_lora busca las capas como en el modelo PEFT del trainer (base_model.model.*).
        wrap = torch.nn.Module()
        wrap.base_model = torch.nn.Module()
        wrap.base_model.model = transformer
        kt.apply_turbo_lora(wrap)
        self.pipe = Flux2KleinPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_dir, subfolder="scheduler"),
            vae=AutoencoderKLFlux2.from_pretrained(self.model_dir, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda"),
            text_encoder=None, tokenizer=None, transformer=transformer)

    def _run(self, emb, size, seed, image=None):
        import torch
        w, h = size
        kwargs = {"image": image} if image is not None else {}
        return self.pipe(prompt_embeds=emb.to("cuda"), height=h, width=w, num_inference_steps=4, guidance_scale=1.0,
                         generator=torch.Generator("cuda").manual_seed(seed), **kwargs).images[0]

    def generate(self, emb, size, seed):
        return self._run(emb, size, seed)

    def edit(self, image, emb, size, seed):
        return self._run(emb, size, seed, image)


ENGINES = {"klein9b": KleinEngine}


def plan(cfg):
    """
    Lo que hay que generar: (grupos, posiciones, base_prompts, anchors) y la lista de trabajos
    [(tipo, nombre_de_salida, datos)] de lo que aún no existe en el dataset.
    """
    ds = cfg["dataset_path"]
    existing = {}
    for f in os.listdir(ds):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            existing[stem] = os.path.join(ds, f)

    positions = [p for p in POSITIONS if str(cfg["edits"].get(p, "")).strip()]
    rng = random.Random(cfg["seed"])
    groups = []  # (grupo, ruta_base, prompt_base o None)
    if cfg["base"] == "mine":
        for stem, path in sorted(existing.items()):
            parsed = slider_core.parse_name(stem)
            if parsed and parsed[1] == 0:
                groups.append((parsed[0], path, None))
    else:
        for i in range(int(cfg["people"])):
            who = f" of {PEOPLE[i % len(PEOPLE)]}" if cfg["vary_people"] else ""
            prompt = f"{cfg['subject'].strip().rstrip('.')}{who}, {rng.choice(PLACES)}, {rng.choice(LIGHTS)}"
            if cfg["neutral"].strip():
                prompt += f", {cfg['neutral'].strip().rstrip('.')}"
            groups.append((f"g{i:02d}", os.path.join(ds, f"g{i:02d}_0.png"), prompt + "."))

    jobs = []
    for gi, (group, base_path, base_prompt) in enumerate(groups):
        new_base = base_prompt is not None and f"{group}_0" not in existing
        if new_base:
            jobs.append(("base", base_path, {"prompt": base_prompt, "seed": cfg["seed"] + gi}))
        for p in positions:
            out = os.path.join(ds, f"{group}_{p}.png")
            # Una base nueva invalida las ediciones que partían de la anterior.
            if new_base or f"{group}_{p}" not in existing:
                jobs.append(("edit", out, {"base": base_path, "prompt": f"{cfg['edits'][p].strip().rstrip('.')}. {KEEP}",
                                           "seed": cfg["seed"] + 1000 + gi * 10 + POSITIONS.index(p)}))
    for j in range(int(cfg["anchors"])):
        if f"anc{j:02d}_anc" not in existing:
            prompt = f"{STYLES[j % len(STYLES)]} of {SCENES[j % len(SCENES)]}, with no people."
            jobs.append(("anchor", os.path.join(ds, f"anc{j:02d}_anc.png"), {"prompt": prompt, "seed": cfg["seed"] + 5000 + j}))
    total = len(groups) * (len(positions) + (1 if cfg["base"] != "mine" else 0)) + int(cfg["anchors"])
    return groups, positions, jobs, total


def main():
    if not os.path.exists(CONFIG_PATH):
        log(f"[!] {CONFIG_PATH} not found", f"No se encontró {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = {**DEFAULTS, **json.load(f)}
    if cfg["engine"] not in ENGINES:
        log(f"[!] Unknown engine '{cfg['engine']}'", f"Motor desconocido '{cfg['engine']}'")
        sys.exit(1)
    os.makedirs(cfg["dataset_path"], exist_ok=True)

    groups, positions, jobs, total = plan(cfg)
    log(f"Slider dataset: {cfg['dataset_path']}", f"Dataset del slider: {cfg['dataset_path']}")
    log(f"  Engine / Motor: {ENGINES[cfg['engine']].name} | groups / grupos: {len(groups)} | positions / posiciones: "
        f"0, {', '.join(positions)} | anchors: {cfg['anchors']} | {cfg['size']} px")
    if not positions:
        log("[!] Write at least one edit (-100, -50, +50 or +100)", "Escribe al menos una edición (-100, -50, +50 o +100)")
        sys.exit(1)
    if not groups:
        if cfg["base"] == "mine":
            log("[!] Base images = mine, but the dataset has no group_0 images", "Imágenes base = mías, pero el dataset no tiene imágenes grupo_0")
        else:
            log("[!] People = 0", "Personas = 0")
        sys.exit(1)
    if cfg["base"] != "mine" and not cfg["subject"].strip():
        log("[!] Write the subject of the images", "Escribe el tema de las imágenes")
        sys.exit(1)
    done = total - len(jobs)
    if done:
        log(f"  {done} images already in the dataset, skipped", f"{done} imágenes ya en el dataset, se saltan")
    print(f"[SliderGen] {done}/{total} - -", flush=True)
    if not jobs:
        log("\n✓ The synthetic dataset is complete", "El dataset sintético está completo")
        return

    # Detener (Stop) = Ctrl+C / Ctrl+Break: se termina la imagen en curso sin dejar ficheros a medias.
    def stop(sig, frame):
        raise KeyboardInterrupt
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    try:
        from PIL import Image
        engine = ENGINES[cfg["engine"]]()
        engine.ensure()
        embeds = engine.encode(sorted({d["prompt"] for _, _, d in jobs}))
        engine.load()
        side, t0, made = int(cfg["size"]), time.time(), 0
        for kind, out, d in jobs:
            name = os.path.splitext(os.path.basename(out))[0]
            if kind == "edit":
                src = Image.open(d["base"]).convert("RGB")
                img = engine.edit(src, embeds[d["prompt"]], fit_size(*src.size, side), d["seed"])
            else:
                img = engine.generate(embeds[d["prompt"]], (side, side), d["seed"])
            save_atomic(img, out)
            done, made = done + 1, made + 1
            eta = (time.time() - t0) / made * (len(jobs) - made)
            group = slider_core.caption_stem(name)
            # El servidor y la GUI leen esta línea: hechas/total, imagen y grupo, segundos restantes.
            print(f"[SliderGen] {done}/{total} {name} {group} eta={int(eta)}", flush=True)
    except KeyboardInterrupt:
        log("\n[!] Stopped: run it again to continue where it left off.",
            "Detenido: vuelve a lanzarlo para seguir donde lo dejó.")
        return
    log("\n✓ Synthetic dataset finished! Review it and delete the pairs that went wrong.",
        "¡Dataset sintético terminado! Revísalo y borra los pares que hayan salido mal.")


if __name__ == "__main__":
    main()
