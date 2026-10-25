# -*- coding: utf-8 -*-
"""
1_pre_cache_yue2.py — Pre-caché de latentes de audio + prefijos de texto para YuE2-3B
Pre-cache of audio latents + text prefixes for YuE2-3B

YuE2 nunca ve píxeles: cada muestra del dataset es un clip de AUDIO (escrito por
0_prepare_yue2.py: 48 kHz estéreo, PCM16, <base>__cNN.wav) más su letra en un
.txt del mismo nombre. Aquí se cachean las dos cosas que el entrenamiento lee
una y otra vez:

1. LATENTES: YuE2VAE.encode() en FP32 (el VAE del release lo exige). Cada clip
   se recorta a múltiplos de 1920 muestras (1 frame de latente = 1 código
   semántico = 40 ms) y se guarda [1, 64, T] sin normalizar: el pipeline oficial
   (yue2_infer) consume el latente del encoder tal cual. El trainer lo transpone
   al formato tiempo-mayor [T, 64] cuando construye el estado del NAR.
2. PREFIJOS DE TEXTO: tokenización Qwen BPE (qwen.tiktoken) vendida en
   scripts/yue2/, el MISMO compositor del wheel yue2_infer-0.1.5:
   [EOD] + encode("{instr}\n[Tags]\n{estilo}\n[Lyrics]\n{letra}\n").
   El andamio ABC ([ABC_START]/[ABC_END]/[MUSIC_START]) NO se hornea aquí:
   depende del modo del training y de cómo mezcle condición/objetivo, y el
   trainer (Fase 4) lo compone con las piezas guardadas.

Salida en ./cached_data_yue2_{proyecto}:
  <name>_latent.pt   tensor FP32 [1, 64, T]        (audio del clip)
  <name>_prompt.pt   dict(base_ids, abc_ids, neg_ids, lyrics_len)
  _neg_prefix.pt     (solo cot=off) prefijo negativo simbólico global
  _meta.json         versión de formato + cot + estilo + protocolo
  _custom_prompt.pt  (server --prompt-only) ids del prompt de preview

La GPU SÓLO entra en la fase de latentes (encoder del VAE). Aviso expreso al
usuario: esta es la fase que la pide; al acabar se libera.
The GPU is only touched while encoding latents, and freed afterwards.

Lee settings/pre_cache_settings_yue2.json: dataset_path, project_name,
style_prompt, cot = off | melody | full, force_gpu.
"""
import os
import gc
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"torch\.nn\.utils\.weight_norm is deprecated in favor of torch\.nn\.utils\.parametrizations\.weight_norm\.",
    category=FutureWarning,
)
import json

import numpy as np
from i18n import t
from project_layout import resolve_cache_dir, resolve_model_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

STUDIO_ROOT = os.path.dirname(SCRIPTS_DIR)          # .../LoRAlab-TrainerStudio
# Los modelos viven en Models/ desde la migracion del layout.
# The models live in Models/ since the layout migration.
NF4_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_NF4", STUDIO_ROOT))
RAW_DIR = str(resolve_model_dir(None, "Models/YuE2-3B_Raw", STUDIO_ROOT))
RAW_REPO_ID = "m-a-p/YuE2-3B"
RAW_VAE_DIR = str(resolve_model_dir(None, "Models/YuE2-Vae", STUDIO_ROOT))
RAW_VAE_REPO_ID = "m-a-p/YuE2-Vae"
NF4_REPO_ID = "AcademiaSD/YuE2-3B-NF4-for-LoRA-Training"
TRAIN_CONFIG_PATH = os.path.join(STUDIO_ROOT, "settings", "train_settings_yue2.json")
try:
    with open(TRAIN_CONFIG_PATH, "r", encoding="utf-8") as _f:
        _precision_cfg = json.load(_f)
except (OSError, ValueError):
    _precision_cfg = {}
MODEL_PRECISION = "nf4" if str(_precision_cfg.get("precision", "bf16")).lower() == "nf4" else "bf16"
if MODEL_PRECISION == "nf4":
    MODEL_DIR = NF4_DIR
    VAE_DIR = os.path.join(NF4_DIR, "vae")
else:
    MODEL_DIR = RAW_DIR
    VAE_DIR = RAW_VAE_DIR

SR = 48000               # YuE2-VAE: 48 kHz estéreo
DOWNSAMPLING_RATIO = 1920
AUDIO_CHANNELS = 2
LATENT_DIM = 64
CACHE_FORMAT_VERSION = 1

DEFAULTS = {
    "dataset_path": "./dataset",
    "cache_dir": "./Cache_Data/cached_data_yue2",
    "project_name": "",
    "style_prompt": "",
    "cot": "off",            # off | melody | full
    "force_gpu": True,       # el encoder del VAE corre en CUDA
    "regularizer": True,      # descarga el minted pack (mezcla para el AR)
}

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".wma", ".aiff", ".aif", ".mp2")
CONFIG_PATH = "settings/pre_cache_settings_yue2.json"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def ensure_yue2_model():
    """Descarga el backbone y el VAE según la precisión configurada para entrenar."""
    global TOKENIZER
    from huggingface_hub import snapshot_download

    token_cfg = read_json(os.path.join(STUDIO_ROOT, "settings", "HF_token.json"))
    token = str(token_cfg.get("token", "")).strip() or None
    if MODEL_PRECISION == "nf4":
        required = (
            os.path.join(NF4_DIR, "model", "index.json"),
            os.path.join(NF4_DIR, "modeling_yue2.py"),
            os.path.join(NF4_DIR, "qwen.tiktoken"),
            os.path.join(NF4_DIR, "vae", "modeling_vae.py"),
        )
        if not all(os.path.isfile(path) for path in required):
            print(f"[DOWNLOAD] YuE2 NF4 desde Hugging Face: {NF4_REPO_ID}", flush=True)
            snapshot_download(repo_id=NF4_REPO_ID, local_dir=NF4_DIR, token=token)
        else:
            print(f"[OK] Reutilizando YuE2 NF4 local: {NF4_DIR}", flush=True)
        root = NF4_DIR
    else:
        required = (
            os.path.join(RAW_DIR, "config.json"),
            os.path.join(RAW_DIR, "modeling_yue2.py"),
            os.path.join(RAW_DIR, "qwen.tiktoken"),
            os.path.join(RAW_DIR, "yue2_generation_config.json"),
            os.path.join(RAW_VAE_DIR, "config.json"),
            os.path.join(RAW_VAE_DIR, "modeling_vae.py"),
        )
        weights = (os.path.join(RAW_DIR, "model.safetensors"),
                   os.path.join(RAW_DIR, "model.safetensors.index.json"))
        if not all(os.path.isfile(path) for path in required) or not any(os.path.isfile(path) for path in weights):
            print(f"[DOWNLOAD] YuE2 BF16 RAW desde Hugging Face: {RAW_REPO_ID}", flush=True)
            snapshot_download(repo_id=RAW_REPO_ID, local_dir=RAW_DIR, token=token)
        else:
            print(f"[OK] Reutilizando YuE2 RAW local: {RAW_DIR}", flush=True)
        vae_weights = (os.path.join(RAW_VAE_DIR, "model.safetensors"),
                       os.path.join(RAW_VAE_DIR, "pytorch_model.bin"))
        if not all(os.path.isfile(path) for path in required[-2:]) or not any(os.path.isfile(path) for path in vae_weights):
            print(f"[DOWNLOAD] YuE2 VAE desde Hugging Face: {RAW_VAE_REPO_ID}", flush=True)
            snapshot_download(repo_id=RAW_VAE_REPO_ID, local_dir=RAW_VAE_DIR, token=token)
        else:
            print(f"[OK] Reutilizando YuE2 VAE local: {RAW_VAE_DIR}", flush=True)
        root = RAW_DIR
        if not any(os.path.isfile(path) for path in weights):
            raise RuntimeError("Descarga YuE2 RAW incompleta; faltan los pesos model.safetensors")

    missing = [os.path.relpath(path, root) for path in required if not os.path.isfile(path)]
    if missing:
        raise RuntimeError("Descarga YuE2 incompleta; faltan: " + ", ".join(missing))
    if TOKENIZER is None:
        TOKENIZER = YuE2TextTokenizer.from_pretrained(MODEL_DIR)
    print(f"[OK] YuE2 {MODEL_PRECISION.upper()} disponible en {MODEL_DIR}", flush=True)

    # El regularizador del AR (canciones minted de la comunidad): con un set pequeno
    # el AR puede colapsar la gramatica de tokens.
    global REGULARIZER_PATH
    REGULARIZER_PATH = ""
    if REGULARIZER_ENABLED:
        from yue2 import regularizer as yue2_regularizer
        path = yue2_regularizer.pack_path(STUDIO_ROOT)
        if not os.path.isfile(path):
            print(f"[DOWNLOAD] YuE2 regularizer (minted pack) desde Hugging Face: "
                  f"{yue2_regularizer.PACK_REPO}", flush=True)
            path = yue2_regularizer.download_pack(STUDIO_ROOT, token=token)
        else:
            print(f"[OK] Reutilizando YuE2 regularizer local: {path}", flush=True)
        REGULARIZER_PATH = path


cfg = dict(DEFAULTS)
if os.path.exists(CONFIG_PATH):
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})

DATASET_PATH = str(cfg["dataset_path"])
PROJECT_NAME = str(cfg["project_name"]).strip()
STYLE_PROMPT = str(cfg["style_prompt"]).strip()
COT = str(cfg["cot"]).strip().lower()
FORCE_GPU = bool(cfg.get("force_gpu", True))
REGULARIZER_ENABLED = bool(cfg.get("regularizer", True))
REGULARIZER_PATH = ""
if COT not in ("off", "melody", "full"):
    print("[!] " + t("Unknown cot '{c}': must be off, melody or full. Using off.", c=COT))
    COT = "off"
CACHE_DIR = str(resolve_cache_dir(
    f"./Cache_Data/cached_data_yue2_{PROJECT_NAME}" if PROJECT_NAME else cfg["cache_dir"],
    DEFAULTS["cache_dir"]))

print(f"  {t('Dataset Path'):<22}: {os.path.abspath(DATASET_PATH)}")
print(f"  {t('Cache Dir'):<22}: {os.path.abspath(CACHE_DIR)}")
print(f"  {t('Project Name'):<22}: {PROJECT_NAME if PROJECT_NAME else t('(Default)')}")
print(f"  {t('Style prompt'):<22}: {STYLE_PROMPT if STYLE_PROMPT else t('(Empty)')}")
print(f"  {t('Mode (cot)'):<22}: {COT}")

# ============================================================================
# PROTOCOLO + TOKENIZADOR (CPU, vendorizados en scripts/yue2)
# ============================================================================
from yue2.protocol import INSTRUCTIONS, SongRequest, negative_prefix, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer

TOKENIZER = None


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def which_ffmpeg():
    import shutil as sh
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for cand in (os.environ.get("LORALAB_FFMPEG", ""), exe):
        if cand and os.path.isfile(cand):
            return cand
    return sh.which("ffmpeg")


def leer_pcm(path, sr=SR):
    """Fichero de audio -> [2, N] float32 a `sr` con ffmpeg. La misma fuente de
    verdad que 0_prepare_yue2.py: si ffmpeg lo abre así, así lo codifica el VAE."""
    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(t("ffmpeg not found in PATH. Install it or set LORALAB_FFMPEG."))
    import subprocess
    flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le",
                           "-ac", str(AUDIO_CHANNELS), "-ar", str(sr), "-"],
                          capture_output=True, timeout=3600, creationflags=flag)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + proc.stderr.decode("utf-8", "replace")[-400:])
    a = np.frombuffer(proc.stdout, dtype=np.float32)
    return np.ascontiguousarray(a[: (a.size // AUDIO_CHANNELS) * AUDIO_CHANNELS]
                                .reshape(-1, AUDIO_CHANNELS).T)


def guardar_atomico(path, write):
    """Escritura atómica (tmp + os.replace): el trainer puede estar leyendo la
    caché mientras el usuario relanza el pre-caché."""
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def ids64(values):
    """IDs de token como tensor torch int64 (nunca numpy dentro de un .pt para
    que el trainer pueda cargar todo con torch.load(weights_only=True))."""
    if isinstance(values, torch.Tensor):
        return values.to(torch.int64).cpu()
    return torch.from_numpy(np.asarray(values, dtype=np.int64))


def find_samples():
    """[(nombre_base, ruta)]: todo audio del primer nivel del dataset es una
    muestra (los clips __cNN incluidos, que ES lo que buena parte del dataset
    contendrá tras 0_prepare). _fuentes/ y otros subdirectorios quedan fuera."""
    if not os.path.isdir(DATASET_PATH):
        return []
    files = sorted(f for f in os.listdir(DATASET_PATH)
                   if f.lower().endswith(AUDIO_EXTS)
                   and os.path.isfile(os.path.join(DATASET_PATH, f)))
    return [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f)) for f in files]


def read_letra(name):
    """Letra del clip ({name}.txt). Sin .txt = instrumental: la sección [Lyrics]
    queda vacía, exactamente como una petición de YuE2 sin voz."""
    path = os.path.join(DATASET_PATH, name + ".txt")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_plan(name):
    """Plan ABC opcional ({name}.abc), solo con cot != off, como budget externo
    de YuE2 (request.abc). Cot=off lo ignora."""
    if COT == "off":
        return None
    path = os.path.join(DATASET_PATH, name + ".abc")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text or None


def cache_hit(name, src):
    """Latente ya escrito y más nuevo que el clip."""
    lat_path = os.path.join(CACHE_DIR, name + "_latent.pt")
    return (os.path.isfile(lat_path)
            and os.path.getmtime(lat_path) > os.path.getmtime(src))


def encode_preview_prompt(cache_dir, prompt, device):
    """Codifica SOLO el prompt custom de previews (lo lanza el server con
    save-precache). Lo que queda guardado son token IDs; el trainer compondrá
    el estilo y la letra de preview con este material en la Fase 4."""
    ensure_yue2_model()
    where = device.upper().replace("CUDA", "GPU")
    print("[Custom Prompt] " + t("Tokenizing preview prompt on {device}: '{prompt}'",
                                 device=where, prompt=prompt[:60]), flush=True)
    ids = TOKENIZER.encode(prompt)
    os.makedirs(cache_dir, exist_ok=True)
    guardar_atomico(os.path.join(cache_dir, "_custom_prompt.pt"),
                    lambda f: torch.save(ids64(ids), f))
    with open(os.path.join(cache_dir, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    print("[Custom Prompt] " + t("Ready."), flush=True)


def pre_cache_texto(samples):
    """FASE 1 — prefijos de texto (CPU). Tokenizar es instantáneo: reescribe
    todo (es el mecanismo de refresco cuando cambia cot/estilo)."""
    neg_path = os.path.join(CACHE_DIR, "_neg_prefix.pt")
    if COT == "off":
        # Negativo simbólico global: solo instrucción sin [Tags]/[Lyrics].
        # Con cot != off cada clip lleva SU plan en el negativo (mismo ABC),
        # así que ahí el negativo va dentro de <name>_prompt.pt.
        neg = negative_prefix(
            SongRequest(style=STYLE_PROMPT, lyrics="", cot="off", id="cache"), TOKENIZER)
        guardar_atomico(neg_path,
                        lambda f: torch.save(ids64(neg), f))
        print("  " + t("Negative prefix (cot=off): instruction only, {n} ids", n=len(neg)))
    elif os.path.isfile(neg_path):
        os.remove(neg_path)

    for idx, (name, _) in enumerate(samples, 1):
        letra = read_letra(name)
        plan = read_plan(name)
        if plan and COT == "off":
            plan = None  # un .abc suelto no entra con cot=off; el trainer avisara
            print("    " + t("{name}: .abc ignored while cot is off", name=name))

        positive = token_prefixes(SongRequest(style=STYLE_PROMPT, lyrics=letra, cot=COT,
                                              id="cache"), TOKENIZER)
        abc_ids = TOKENIZER.encode(plan) if plan is not None else None
        if plan is not None:
            positive = token_prefixes(SongRequest(style=STYLE_PROMPT, lyrics=letra, cot=COT,
                                                  id="cache"), TOKENIZER, abc_ids)
        neg_ids = None
        if plan is not None:
            neg_ids = negative_prefix(SongRequest(style=STYLE_PROMPT, lyrics=letra, cot=COT,
                                                  abc=plan, id="cache"), TOKENIZER, abc_ids)

        entry = {"base_ids": ids64(positive),
                 "abc_ids": None if abc_ids is None else ids64(abc_ids),
                 "neg_ids": None if neg_ids is None else ids64(neg_ids),
                 "lyrics_len": len(letra)}
        guardar_atomico(os.path.join(CACHE_DIR, name + "_prompt.pt"),
                        lambda f: torch.save(entry, f))
        print(f"[{idx}/{len(samples)}] {t('Text')}: {name} | "
              + t("{n} ids", n=len(positive)) + " | "
              + t("lyrics {c} chars", c=len(letra)))


def pre_cache_latentes(jobs):
    """FASE 2 — encoder del VAE en GPU (el header de carga lo imprime main).
    FP32 estricto, autocast desactivado (la implementación del release lo
    impone y el LoRA no debe heredar autocast del entorno)."""
    sys.path.insert(0, VAE_DIR)
    from modeling_vae import YuE2VAE

    vae = YuE2VAE.from_pretrained(VAE_DIR, device="cuda", local_files_only=True)
    saltados = 0
    for idx, (name, ruta) in enumerate(jobs, 1):
        if cache_hit(name, ruta):
            saltados += 1
            continue
        pcm = leer_pcm(ruta, SR)              # [2, N] float32 CPU
        n_recortado = pcm.shape[1] % DOWNSAMPLING_RATIO
        n = pcm.shape[1] - n_recortado        # recorte a frames completos
        pcm = pcm[:, :n]
        if n < DOWNSAMPLING_RATIO:
            print(f"[{idx}/{len(jobs)}] {name} " + t("skipped: shorter than one latent frame "
                                                     "(<{ms} ms)",
                                                     ms=int(DOWNSAMPLING_RATIO / SR * 1000)))
            continue
        audio = torch.from_numpy(np.ascontiguousarray(pcm[None]))  # [1, 2, N]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            latent = vae.encode(audio.to(device=device, dtype=torch.float32))  # [1,64,T]
        if (latent.dtype != torch.float32 or latent.shape[1] != LATENT_DIM
                or not torch.isfinite(latent).all()):
            del audio, latent
            print(f"[{idx}/{len(jobs)}] {name} [!] " + t("bad latent (dtype/channels/finiteness); "
                                                          "file NOT written"))
            continue
        guardar_atomico(os.path.join(CACHE_DIR, name + "_latent.pt"),
                        lambda f: torch.save(latent.cpu(), f))
        del audio, latent
        if idx % 8 == 0 or idx == len(jobs):
            free_vram()
        nota = (t("trimmed {m} samples to a multiple of 1920", m=n_recortado)
                if n_recortado else t("frame-aligned"))
        print(f"[{idx}/{len(jobs)}] {t('Audio')}: {name} | T={n // DOWNSAMPLING_RATIO} frames "
              f"({n / SR:.2f}s) | {nota}")
    if saltados:
        print("  " + t("{n} latents already cached, skipped.", n=saltados))


def pre_cache_autotest(vae):
    """Roundtrip del PRIMER clip: encode -> decode en GPU y compara reproduce
    el PCM (dB de error relativo RMS). Es la prueba de que la geometría de
    frames (1920, 64 canales) y el recorte casan."""
    samples = find_samples()
    for name, ruta in samples:
        if not cache_hit(name, ruta):
            break
    else:
        return
    pcm = leer_pcm(ruta, SR)
    n = pcm.shape[1] - pcm.shape[1] % DOWNSAMPLING_RATIO
    pcm = pcm[:, :n]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.from_numpy(np.ascontiguousarray(pcm[None])).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        z = vae.encode(audio)
        recon = vae.decode(z)
    ref = audio[0][:, :recon.shape[-1]]   # el decoder emite 1920*T - 64 muestras
    err = float(torch.sqrt(((ref - recon[0]) ** 2).mean())
                / (torch.sqrt((ref ** 2).mean()) + 1e-12))
    print("    [AUTOTEST] " + t("roundtrip RMS error: {err:.2f}% of signal RMS ({s:.2f}s clip)",
                                err=err * 100, s=n / SR))
    if err > 0.5:
        print("    [AUTOTEST] [!] " + t("roundtrip error above 50%%: check dtype/releases"))
    del audio, z, recon
    free_vram()


def pre_cache_codigos(samples):
    """FASE 3 — códigos semánticos REALES de cada clip (MERT-v2 + head de la comunidad, ver
    yue2/real_codes.py): 1 código por frame de latente. El AR aprende de ellos y el NAR los ve como
    contexto, igual que en ComfyUI. Se normaliza por canción (todos los __cNN de un mismo tema)."""
    from yue2.real_codes import RealCodes

    songs = {}
    for name, ruta in samples:
        lat = os.path.join(CACHE_DIR, name + "_latent.pt")
        if os.path.isfile(lat):
            songs.setdefault(name.rsplit("__c", 1)[0], []).append((name, ruta, lat))
    pending = {song: clips for song, clips in songs.items()
               if any(not os.path.isfile(codes_file(n)) or os.path.getmtime(codes_file(n)) < os.path.getmtime(lat)
                      for n, _, lat in clips)}
    if not pending:
        print("  " + t("{n} code files already cached, skipped.", n=sum(len(c) for c in songs.values())))
        return
    print("\n" + t("Loading {name}...", name="MERT-v2-FullSong + YuE2 real-audio tokenizer head"))
    token_cfg = read_json(os.path.join(STUDIO_ROOT, "settings", "HF_token.json"))
    rc = RealCodes(STUDIO_ROOT, device="cuda" if torch.cuda.is_available() else "cpu",
                   token=str(token_cfg.get("token", "")).strip() or None)
    for idx, (song, clips) in enumerate(pending.items(), 1):
        frames = [torch.load(lat, weights_only=True).shape[-1] for _, _, lat in clips]
        codes = rc.song([(ruta, n) for (_, ruta, _), n in zip(clips, frames)])
        for (name, _, _), c in zip(clips, codes):
            guardar_atomico(codes_file(name), lambda f, c=c: torch.save(c, f))
        print(f"[{idx}/{len(pending)}] {t('Codes')}: {song} | {len(clips)} clips | {sum(frames)} frames", flush=True)
    del rc
    free_vram()


def codes_file(name):
    return os.path.join(CACHE_DIR, f"_realcodes_{name}.pt")


def escribir_meta():
    meta = {"cache_format_version": CACHE_FORMAT_VERSION, "protocol": "yue2-native-v1",
            "cot": COT, "style_prompt": STYLE_PROMPT, "sample_rate": SR,
            "downsampling_ratio": DOWNSAMPLING_RATIO, "latent_dim": LATENT_DIM,
            "instruction": INSTRUCTIONS[COT],
            "regularizer_path": REGULARIZER_PATH}
    with open(os.path.join(CACHE_DIR, "_meta.tmp"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(os.path.join(CACHE_DIR, "_meta.tmp"), os.path.join(CACHE_DIR, "_meta.json"))


def main():
    if not os.path.isdir(DATASET_PATH):
        print("[!] " + t("Dataset folder does not exist: {path}", path=DATASET_PATH))
        sys.exit(1)
    if FORCE_GPU and not torch.cuda.is_available():
        print("[!] " + t("CUDA is not available and force_gpu is on: this phase needs the GPU."))
        sys.exit(1)
    samples = find_samples()
    if not samples:
        print("[!] " + t("No audio clips found in '{path}'. Run 0_prepare_yue2.py first.",
                         path=DATASET_PATH))
        sys.exit(1)
    print(f"  {t('Samples'):<22}: {len(samples)}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    ensure_yue2_model()

    # Obsoletos: el trainer carga TODO lo que haya en caché; lo que ya no está
    # en el dataset no puede seguir influyendo.
    expected = {f"{name}_{suf}.pt" for name, _ in samples for suf in ("latent", "prompt")}
    stale = [f for f in os.listdir(CACHE_DIR)
             if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print("  " + t("Removed {n} stale cache files", n=len(stale)))

    # ── FASE 1: TEXTO (CPU) ──────────────────────────────────────────────────
    pre_cache_texto(samples)
    escribir_meta()

    # ── FASE 2: VAE encoder en GPU ──────────────────────────────────────────
    # Los latentes ya existentes y MÁS NUEVOS que su clip no se recalculan: un
    # relanzamiento por cambio de estilo/letra solo cuesta la fase de texto.
    print("\n" + t("Loading {name}...", name="YuE2-Vae (encoder, FP32)"))
    sys.path.insert(0, VAE_DIR)
    from modeling_vae import YuE2VAE

    vae = YuE2VAE.from_pretrained(VAE_DIR, device="cuda" if torch.cuda.is_available() else "cpu",
                                  local_files_only=True)
    pre_cache_autotest(vae)
    pre_cache_latentes(samples)
    del vae
    free_vram()

    # ── FASE 3: códigos reales (MERT-v2 + head) en GPU ──────────────────────
    pre_cache_codigos(samples)

    print("\n✓ " + t("Pre-caching finished! VRAM freed."))
    print(t("Next: run the trainer (2_train_lora_yue2.py) — Fase 4 of the plan."))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt (texto que pega el server)> <cuda|cpu>
        encode_preview_prompt(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
