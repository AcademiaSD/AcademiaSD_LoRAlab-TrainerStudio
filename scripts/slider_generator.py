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
    "vary_people": True,  # edad, sexo y origen distintos en cada base
    "vary_hair": True,    # peinado distinto en cada base (desactivar en sliders de pelo)
    "variations": [],     # posturas / ángulos / encuadres del sujeto: cada base toma uno
    "chain": False,       # +100 se edita desde +50 (y -100 desde -50): la deformación se acumula
    "anchors": 12,
    "base": "generate",   # "generate": las crea el modelo | "mine": las grupo_0 del dataset
    "size": 768,
    "seed": 1000,
}

# Variedad de las imágenes base: el slider tiene que aprender el concepto, no una cara, un sitio o una luz.
# Personas y pelo van aparte: en un slider de pelo, el pelo de la base lo fija la posición 0, no esta lista.
PEOPLE = [
    "a young East Asian woman", "an elderly white man", "a middle-aged Black woman", "a teenage boy with freckles",
    "a South Asian man in his thirties", "an old Japanese woman", "a woman in her twenties", "a Latino man in his forties with glasses",
    "a young Black man", "a middle-aged white woman", "a man in his sixties", "a young Middle Eastern woman",
    "a little girl", "a man in his fifties", "an Indigenous woman", "a young man with earrings",
    "a Korean man in his twenties", "an elderly Black man", "a woman in her thirties with freckles", "a chubby man in his forties",
    "a young woman with a nose piercing", "an old woman with glasses", "a boy around ten years old", "a Scandinavian man",
]
HAIRS = [
    "long straight black hair", "short curly hair", "red hair in a ponytail", "white hair in a bun", "wavy blonde hair",
    "short dreadlocks", "a brown bob haircut", "salt and pepper hair", "long braided hair", "a buzz cut",
    "dyed blue hair", "messy brown hair", "long blond hair", "curly grey hair", "a shaved head",
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


def fit_size(w, h, side, multiple=16):
    """Tamaño con el área de side x side y la proporción de la imagen, en múltiplos de multiple."""
    ar = w / h
    bh = math.sqrt(side * side / ar)
    return max(multiple, round(ar * bh / multiple) * multiple), max(multiple, round(bh / multiple) * multiple)


def fit_crop(img, size):
    """Escala hasta cubrir size y recorta al centro (como fit() del pre-caché)."""
    w, h = size
    ratio = max(w / img.width, h / img.height)
    img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)))
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def waves(jobs):
    """
    Oleadas por dependencia: bases y anchors, después las ediciones que parten de una imagen ya
    existente o de la oleada anterior. Solo hacen falta si los embeddings dependen de la imagen.
    """
    level, out = {}, []
    for job in jobs:
        kind, path, d = job
        level[path] = level.get(d["base"], 0) + 1 if kind == "edit" else 0
        while len(out) <= level[path]:
            out.append([])
        out[level[path]].append(job)
    return [w for w in out if w]


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


def import_script(name, filename):
    """
    Importa un trainer o pre-caché del repo para reutilizar su cargador NF4 y su LoRA turbo. Se
    importa desde una carpeta temporal: al importarse lee ajustes y crea carpetas en la actual.
    """
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS_DIR, filename))
            module = importlib.util.module_from_spec(spec)
            saved_argv, sys.argv = sys.argv, [sys.argv[0]]
            spec.loader.exec_module(module)
            sys.argv = saved_argv
        finally:
            os.chdir(cwd)
    return module


def free_vram():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class KleinEngine:
    """FLUX.2 Klein 9B NF4 + LoRA turbo de kalle07 (4 pasos, sin CFG), como el prototipo."""
    name = "FLUX.2 Klein 9B"
    model_dir = "FLUX.2-Klein-9B_NF4"
    repo_id = "AcademiaSD/FLUX.2-Klein-9B-NF4-for-LoRA-Training"
    multiple = 16
    # El text encoder de Klein solo lee el texto: todo se codifica de una vez, antes de generar.
    image_in_prompt = False

    def ensure(self):
        if (os.path.exists(os.path.join(self.model_dir, "transformer", "index.json"))
                and os.path.exists(os.path.join(self.model_dir, "text_encoder", "config.json"))):
            return
        log(f"Downloading {self.repo_id} (~8.8 GB)...", f"Descargando {self.repo_id} (~8,8 GB)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.repo_id, local_dir=self.model_dir, token=get_hf_token(), max_workers=2)

    def encode(self, keys, side):
        """{(prompt, imagen): embedding} en CPU. El text encoder se libera antes de cargar el transformer."""
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import AutoTokenizer, Qwen3Model
        log("Loading Text Encoder (Qwen3-8B NF4)...", "Cargando Text Encoder (Qwen3-8B NF4)...")
        te = Qwen3Model.from_pretrained(os.path.join(self.model_dir, "text_encoder"), dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(self.model_dir, subfolder="tokenizer")
        out = {}
        with torch.inference_mode():
            for key in keys:
                out[key] = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, [key[0]], dtype=torch.bfloat16, device="cuda").cpu()
        del te, tok
        free_vram()
        return out

    def load(self):
        import torch
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        kt = import_script("klein_trainer", "2_train_lora_klein9b.py")
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

    def unload(self):
        self.pipe = None
        free_vram()

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


class QwenImage21Engine:
    """
    Qwen-Image 2.1 NF4 + LoRA turbo de Viggle (4 pasos, sin CFG), con el cargador, el muestreo y el
    LoRA turbo de su trainer. En edición la imagen de partida entra en el text encoder (Qwen3-VL-8B)
    junto a la instrucción, como el nodo TextEncodeQwenImage21 de ComfyUI: los embeddings dependen
    de la imagen, así que se generan por oleadas (main) y el text encoder nunca coincide en VRAM con
    el transformer.
    """
    name = "Qwen-Image 2.1"
    model_dir = "Qwen-Image21-NF4"
    repo_id = "AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training"
    captioner_te = os.path.join("Captioner-Qwen3-VL-8B", "text_encoder_NF4")
    multiple = 32
    image_in_prompt = True

    def text_encoder_dir(self):
        """El text encoder ya descargado: el elegido en el pre-caché de Qwen, otra variante o el del captioner."""
        try:
            with open("settings/pre_cache_settings_qwenimage21.json", "r", encoding="utf-8") as f:
                chosen = {"BF16_offload": "BF16"}.get(json.load(f).get("text_encoder", ""), None)
        except Exception:
            chosen = None
        for variant in ([chosen] if chosen else []) + ["NF4", "INT8", "BF16"]:
            path = os.path.join(self.model_dir, f"text_encoder_{variant}")
            if os.path.exists(os.path.join(path, "config.json")):
                return path
        if os.path.exists(os.path.join(self.captioner_te, "config.json")):
            return self.captioner_te
        return None

    def ensure(self):
        have_model = (os.path.exists(os.path.join(self.model_dir, "transformer", "index.json"))
                      and os.path.exists(os.path.join(self.model_dir, "model_index.json")))
        te = self.text_encoder_dir()
        if have_model and te:
            return
        # Solo el text encoder NF4 (~5 GB), y ninguno si ya hay uno (el del captioner sirve).
        ignore = [f"text_encoder_{v}/*" for v in ("BF16", "INT8")] + (["text_encoder_NF4/*"] if te else [])
        log(f"Downloading {self.repo_id}...", f"Descargando {self.repo_id}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.repo_id, local_dir=self.model_dir, token=get_hf_token(), max_workers=2,
                          ignore_patterns=ignore)

    def encode(self, keys, side):
        """{(prompt, imagen o None): (embeds, mask, imgmask)} en CPU."""
        import torch
        from diffusers import DiffusionPipeline
        from PIL import Image
        from transformers import Qwen3VLForConditionalGeneration
        path = self.text_encoder_dir()
        log(f"Loading Text Encoder (Qwen3-VL-8B, {os.path.basename(path)})...",
            f"Cargando Text Encoder (Qwen3-VL-8B, {os.path.basename(path)})...")
        if path.endswith("BF16"):
            # BF16 no cabe en 8-12 GB: lo que no entra en la VRAM libre se ejecuta desde la RAM.
            free_bytes, _ = torch.cuda.mem_get_info()
            te = Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map="auto",
                max_memory={0: max(free_bytes - int(1.5 * 1024**3), 0), "cpu": "512GiB"})
        else:
            te = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        pipe = DiffusionPipeline.from_pretrained(self.model_dir, transformer=None, vae=None, text_encoder=te, dtype=torch.bfloat16)
        out = {}
        with torch.inference_mode():
            for prompt, image_path in keys:
                image = None
                if image_path:
                    src = Image.open(image_path)
                    image = [fit_crop(src.convert("RGBA"), fit_size(*src.size, side, self.multiple))]
                embeds, mask, imgmask = pipe.encode_prompt(prompt=prompt, image=image, device="cuda")
                if mask is None:
                    mask = torch.ones(embeds.shape[:2], dtype=torch.bool)
                out[(prompt, image_path)] = (embeds.cpu(), mask.bool().cpu(), None if imgmask is None else imgmask.bool().cpu())
        del pipe, te
        free_vram()
        return out

    def load(self):
        import torch
        from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
        qt = import_script("qwen_trainer", "2_train_lora_qwen_image21.py")
        qt.PREVIEW_STEPS, qt.PREVIEW_CFG, qt.TURBO_LORA_STRENGTH = 4, 1.0, 1.0
        if not os.path.exists(qt.TURBO_LORA_PATH):
            raise FileNotFoundError(f"Turbo LoRA not found / No se encuentra el Turbo LoRA: {qt.TURBO_LORA_PATH}")
        log("Loading Qwen-Image 2.1 (NF4) + Turbo LoRA...", "Cargando Qwen-Image 2.1 (NF4) + Turbo LoRA...")
        self.qt = qt
        self.transformer = qt.load_nf4_transformer(self.model_dir).to("cuda")
        wrap = torch.nn.Module()
        wrap.base_model = torch.nn.Module()
        wrap.base_model.model = self.transformer
        qt.apply_turbo_lora(wrap)
        # El Turbo LoRA se entrenó sin shift_terminal (como en las previews del trainer).
        base = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_dir, subfolder="scheduler")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base.config, shift_terminal=None)
        self.vae = AutoencoderKLQwenImage21.from_pretrained(self.model_dir, subfolder="vae", dtype=torch.bfloat16).to("cuda")
        z = self.vae.config.z_dim
        self.mean = torch.tensor(self.vae.config.latents_mean, device="cuda").view(1, z, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device="cuda").view(1, z, 1, 1, 1)

    def unload(self):
        self.transformer = self.vae = self.qt = None
        free_vram()

    def _run(self, emb, size, seed, image=None):
        import torch
        import torchvision.transforms.functional as F_vision
        from PIL import Image
        embeds, mask, imgmask = emb
        w, h = size
        H, W = h // 16, w // 16
        sample = {"emb": embeds, "msk": mask}
        with torch.inference_mode():
            if image is not None:
                # Latente de la imagen de partida, como encode_latents del pre-caché (el VAE lee RGBA).
                x = F_vision.pil_to_tensor(fit_crop(image.convert("RGBA"), size)).unsqueeze(0).unsqueeze(2)
                x = (x.float() / 127.5 - 1.0).to("cuda", dtype=self.vae.dtype)
                z = self.vae.encode(x).latent_dist.mode().float()
                sample["ctrl"] = ((z - self.mean) / self.std)[:, :, 0].to(torch.bfloat16)
                if imgmask is not None:
                    sample["imgmask"] = imgmask
            g = torch.Generator(device="cuda").manual_seed(seed)
            latents = self.qt.denoise(self.transformer, self.scheduler, sample, H, W, g)
            lat = self.qt.unpack_latents(latents, H, W).to(self.vae.dtype).unsqueeze(2)
            img = self.vae.decode(lat * self.std.to(lat.dtype) + self.mean.to(lat.dtype), return_dict=False)[0][:, :3, 0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(img)

    def generate(self, emb, size, seed):
        return self._run(emb, size, seed)

    def edit(self, image, emb, size, seed):
        return self._run(emb, size, seed, image)


ENGINES = {"klein9b": KleinEngine, "qwenimage21": QwenImage21Engine}


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
    variations = cfg["variations"]
    if isinstance(variations, str):
        variations = variations.splitlines()
    variations = [v.strip().rstrip(".") for v in variations if str(v).strip()]
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
            # Paso 7 en la lista de peinados: que persona y pelo no vayan siempre juntos.
            hair = f", with {HAIRS[i * 7 % len(HAIRS)]}" if cfg["vary_hair"] else ""
            pose = f", {variations[i % len(variations)]}" if variations else ""
            prompt = f"{cfg['subject'].strip().rstrip('.')}{who}{pose}{hair}, {rng.choice(PLACES)}, {rng.choice(LIGHTS)}"
            if cfg["neutral"].strip():
                prompt += f", {cfg['neutral'].strip().rstrip('.')}"
            groups.append((f"g{i:02d}", os.path.join(ds, f"g{i:02d}_0.png"), prompt + "."))

    # Las ±50 antes que las ±100: encadenadas, cada extremo se edita desde la intermedia de su lado.
    order = [p for p in ("-50", "-100", "50", "100") if p in positions]
    jobs = []
    for gi, (group, base_path, base_prompt) in enumerate(groups):
        new_base = base_prompt is not None and f"{group}_0" not in existing
        if new_base:
            jobs.append(("base", base_path, {"prompt": base_prompt, "seed": cfg["seed"] + gi}))
        remade = set()
        for p in order:
            prev = {"-100": "-50", "100": "50"}.get(p) if cfg["chain"] else None
            prev = prev if prev in positions else None
            src = existing.get(f"{group}_{prev}", os.path.join(ds, f"{group}_{prev}.png")) if prev else base_path
            out = os.path.join(ds, f"{group}_{p}.png")
            # Una imagen de partida nueva invalida las ediciones que partían de la anterior.
            if new_base or prev in remade or f"{group}_{p}" not in existing:
                remade.add(p)
                jobs.append(("edit", out, {"base": src, "prompt": f"{cfg['edits'][p].strip().rstrip('.')}. {KEEP}",
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
        f"0, {', '.join(positions)} | anchors: {cfg['anchors']} | {cfg['size']} px"
        + (" | chained edits / ediciones encadenadas" if cfg["chain"] else ""))
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
        side, t0, made = int(cfg["size"]), time.time(), 0
        m = engine.multiple
        steps = waves(jobs) if engine.image_in_prompt else [jobs]
        for wi, wave in enumerate(steps, 1):
            if len(steps) > 1:
                log(f"\n── Stage {wi}/{len(steps)}: {len(wave)} images ──", f"Etapa {wi}/{len(steps)}: {len(wave)} imágenes")
            keys = {(d["prompt"], d["base"] if kind == "edit" and engine.image_in_prompt else None) for kind, _, d in wave}
            embeds = engine.encode(sorted(keys, key=lambda k: (k[0], k[1] or "")), side)
            engine.load()
            for kind, out, d in wave:
                name = os.path.splitext(os.path.basename(out))[0]
                if kind == "edit":
                    src = Image.open(d["base"]).convert("RGB")
                    key = (d["prompt"], d["base"] if engine.image_in_prompt else None)
                    img = engine.edit(src, embeds[key], fit_size(*src.size, side, m), d["seed"])
                else:
                    img = engine.generate(embeds[(d["prompt"], None)], (side // m * m, side // m * m), d["seed"])
                save_atomic(img, out)
                done, made = done + 1, made + 1
                eta = (time.time() - t0) / made * (len(jobs) - made)
                group = slider_core.caption_stem(name)
                # El servidor y la GUI leen esta línea: hechas/total, imagen y grupo, segundos restantes.
                print(f"[SliderGen] {done}/{total} {name} {group} eta={int(eta)}", flush=True)
            engine.unload()
    except KeyboardInterrupt:
        log("\n[!] Stopped: run it again to continue where it left off.",
            "Detenido: vuelve a lanzarlo para seguir donde lo dejó.")
        return
    log("\n✓ Synthetic dataset finished! Review it and delete the pairs that went wrong.",
        "¡Dataset sintético terminado! Revísalo y borra los pares que hayan salido mal.")


if __name__ == "__main__":
    main()
