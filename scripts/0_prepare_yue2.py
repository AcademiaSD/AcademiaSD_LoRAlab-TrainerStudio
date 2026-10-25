# -*- coding: utf-8 -*-
"""
0_prepare_yue2.py — Preparación del dataset de AUDIO para el entrenador YuE2-3B
Audio dataset preparation for the YuE2-3B trainer

YuE2 no usa imágenes ni captions de imagen: su condición es el prompt de estilo
(+ letra y, si hay, plan ABC). Este script hace el equivalente del auto-caption
de los demás entrenadores sobre audio:

1. Descodifica cada fichero de audio a 48 kHz estéreo (lo que pide YuE2-Vae),
   con ffmpeg en PATH (wav, flac, mp3, m4a, ogg, opus...).
2. Opcional: separa la voz con Mel-Band RoFormer (vendorizado en
   scripts/melband) si los ficheros no vienen limpios. Va ANTES del troceado,
   igual que en MiniMax-H3: una pasada por fichero, no por clip, para que los
   artefactos se comporten igual de una toma a otra.
3. Trocea en clips de duración controlada y NO solapados. El corte se ajusta a
   la zona más silenciosa cercana (snap_seconds 0 para cortes exactos).
4. Escribe cada clip como `<nombre>__cNN.wav` (y con stems, `__voz__cNN` /
   `__inst__cNN`). Las fuentes se guardan en `dataset/_fuentes/`: subcarpeta
   que Clear Dataset respeta y que la pre-caché no escanea (solo mira el
   primer nivel, os.listdir).

La letra NO se reparte entre los clips: sin alineación, un reparto automático
inventaría cuándo canta cada verso. Cada clip empieza sin .txt y la letra se
asigna a mano en el Dataset Manager (el mismo editor que usan las imágenes),
o con ASR en fases posteriores.

Todo corre en CPU. La GPU solo entra con stems != 'off' y stems_device='cuda'.
Everything runs on CPU. The GPU is only touched with stems != 'off' and
stems_device='cuda'.

Lee caption_settings_yue2.json (clip_seconds, stems, overwrite...); la ruta del
dataset sale de pre_cache_settings_yue2.json. Igual que los captioners.
"""
import os
import gc
import re
import sys
import json
import time
import shutil
import wave

import numpy as np
from i18n import t

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# YuE2-Vae: 48 kHz estéreo, downsampling 1920 -> 25 frames de latente por segundo.
SR = 48000
CHANNELS = 2

DEFAULTS = {
    "dataset_path": "./dataset",
    "clip_seconds": 30.0,
    "min_clip_seconds": 10.0,
    # Ajuste del corte a la zona más silenciosa: se busca dentro de
    # +-[snap_seconds] alrededor del límite teórico. 0 = cortes exactos.
    "snap_seconds": 1.5,
    # Si el clip suena por encima de ~-1 dBFS se le baja el pico. Solo escala
    # hacia abajo: no se sube volumen ni se toca el tono.
    "peak_normalize": True,
    # off | vocal | instrumental | both
    "stems": "off",
    "stems_device": "cpu",
    "stems_folder": "MelBandRoFormer",
    # False = no toca las fuentes que ya tienen sus clips escritos.
    "overwrite": False,
    # True deja las fuentes en dataset/_fuentes/ (la letra vive junto al
    # fichero original); False las borra una vez escritos los clips.
    "keep_source": True,
}

# Un subconjunto de lo que ffmpeg abre; el servidor enlaza con la lista corta
# (DATASET_EXTS de server_yue2.py).
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".wma", ".aiff", ".aif", ".mp2", ".mpga")
CONFIG_PATH = "settings/caption_settings_yue2.json"
PRECACHE_CONFIG = "settings/pre_cache_settings_yue2.json"

STEMS_MODES = ("off", "vocal", "instrumental", "both")

# Nombres que este script EScribe: <base>__cNN.wav, y con stems
# <base>__voz__cNN.wav / <base>__inst__cNN.wav. Nunca son fuentes: si
# volvieran al escaneo, una segunda pasada los retrocearía (bug verificado
# con el test de idempotencia).
CLIP_NAME_RE = re.compile(r"__(?:voz|inst)?(?:__)?c\d{2}\.wav$")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_config():
    cfg = dict(DEFAULTS)
    precache = read_json(PRECACHE_CONFIG)
    for key in ("dataset_path",):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})
    return cfg


def which_ffmpeg():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for cand in (os.environ.get("LORALAB_FFMPEG", ""), exe):
        if cand and os.path.isfile(cand):
            return cand
    return shutil.which("ffmpeg")


def _cmd(cmd, input_bytes=None):
    import subprocess
    creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=3600,
                          creationflags=creation_flag)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + proc.stderr.decode("utf-8", "replace")[-400:])
    return proc.stdout


def leer_pcm(path, ffmpeg, sr=SR):
    """Fichero de audio -> [2, N] float32 a `sr`. / File -> [2, N] float32 at `sr`."""
    raw = _cmd([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le",
                "-ac", "2", "-ar", str(sr), "-"])
    a = np.frombuffer(raw, dtype=np.float32)
    return np.ascontiguousarray(a[: (a.size // 2) * 2].reshape(-1, 2).T)


def resample_48k(pcm, sr_in, ffmpeg):
    """[2, N] a `sr_in` -> [2, N] a 48 kHz, con ffmpeg (sin librerías nuevas)."""
    if int(sr_in) == SR:
        return pcm
    datos = np.ascontiguousarray(pcm.T.astype(np.float32)).tobytes()
    salida = _cmd([ffmpeg, "-v", "error", "-f", "f32le", "-ar", str(sr_in), "-ac", "2", "-i", "-",
                   "-f", "f32le", "-ar", str(SR), "-ac", "2", "-"], input_bytes=datos)
    a = np.frombuffer(salida, dtype=np.float32)
    return np.ascontiguousarray(a[: (a.size // 2) * 2].reshape(-1, 2).T)


def escribir_wav(pcm, path):
    """[2, N] float32 -> WAV PCM16 48 kHz estéreo. El VAE codifica en FP32, pero
    float32 en disco gasta 11.5 MB por clip de 30 s: 16 bits es la mitad y su
    ruido de cuantización (~96 dB) no entra en el LoRA."""
    mono = np.ascontiguousarray(np.clip(pcm, -1.0, 1.0).T)
    samples = (mono * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def cortes_silenciosos(pcm, clip_seconds, min_clip_seconds, snap_seconds):
    """[(inicio, fin)] en muestras. Cortes cada `clip_seconds`, con el límite
    llevado al punto de menos energía dentro de +-[snap_seconds] (suma de los
    dos canales, ventana de 50 ms / hop 25 ms). Los dos canales comparten corte,
    y el troceado nunca invierte: los límites son monótonos."""
    _, n = pcm.shape
    clip_samples = int(max(1.0, clip_seconds) * SR)
    min_samples = int(max(0.5, min_clip_seconds) * SR)
    if n < min_samples:
        return []

    hop = int(0.025 * SR)
    win = hop * 2
    mono2 = (pcm.astype(np.float64) ** 2).sum(axis=0)
    cs = np.concatenate(([0.0], np.cumsum(mono2)))
    idx = np.arange(0, max(n - win + 1, 1), hop)
    energia = (cs[idx + win] - cs[idx]) / win          # una entrada por hop
    tope = float(energia.max())
    if tope <= 0:
        energia[:] = 1.0
    else:
        energia /= tope

    snap = int(max(0.0, snap_seconds) * SR)
    limites = [0]
    objetivo = clip_samples
    # hi <= lo significa que la ventana de búsqueda quedó fuera del resto del
    # fichero: se corta en hi (lo más tarde válido). Con snap > min_samples un
    # corte podría quedar detrás del límite anterior: max() garantiza progreso.
    while objetivo < n - min_samples:
        lo = max(limites[-1] + min_samples, objetivo - snap)
        hi = min(objetivo + snap, n - min_samples)
        if hi <= lo:
            limite = hi
        else:
            rel = int(energia[lo // hop: hi // hop + 1].argmin())
            limite = (lo // hop + rel) * hop
        limite = max(int(limite), limites[-1] + min_samples)
        limites.append(limite)
        objetivo = limite + clip_samples
    limites.append(n)

    cortes = [(limites[i], limites[i + 1]) for i in range(len(limites) - 1)]
    return [(a, b) for (a, b) in cortes if b - a >= min_samples]


def trocear_fichero(base, pcm, cfg, dataset, done, total, started):
    """Divide UN [2, N] en clips y los escribe. `base` ya lleva el tag de stem.
    Devuelve los nombres de clip escritos."""
    cortes = cortes_silenciosos(
        pcm, float(cfg["clip_seconds"]), float(cfg["min_clip_seconds"]),
        float(cfg["snap_seconds"]))
    if not cortes:
        print(f"[{done}/{total}] {base}.wav " +
              t("skipped: shorter than {s}s", s=cfg["min_clip_seconds"]), flush=True)
        return []

    escritos = []
    for i, (a, b) in enumerate(cortes, 1):
        trozo = pcm[:, a:b]
        if cfg.get("peak_normalize", True):
            pico = float(np.abs(trozo).max())
            if pico > 0.999:
                trozo = trozo * (0.999 / pico)
        salida = "{}__c{:02d}.wav".format(base, i)
        escribir_wav(trozo, os.path.join(dataset, salida))
        escritos.append(salida)

    durs = " | ".join("c{:02d}:{:.1f}s".format(k + 1, (cortes[k][1] - cortes[k][0]) / SR)
                      for k in range(len(cortes)))
    eta = (time.time() - started) / max(done, 1) * max(total - done, 0)
    print(f"[{done}/{total}] {base}.wav | {durs} | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
    return escritos


def sufijos_stems(modo):
    fichas = []
    if modo in ("vocal", "both"):
        fichas.append("voz")
    if modo in ("instrumental", "both"):
        fichas.append("inst")
    return fichas


def stem_codename(nombre, ficha):
    base = os.path.splitext(nombre)[0]
    return f"{base}__{ficha}" if ficha else base


def cuda_disponible():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def separar_stems(ruta, ffmpeg, cfg):
    """{ficha: [2, N] float32 a 48 kHz}. Descodifica a 44.1 kHz (la tasa con la
    que se entrenó el separador), separa una sola vez y remuestrea cada stem a
    48 kHz. El modelo solo se carga aquí (no en cada trozo). Con stems activos
    nunca se devuelve el mix ('')."""
    from melband import separator as sep

    faltan = sep.dependencias_que_faltan()
    if faltan:
        raise RuntimeError(t("Missing packages: {names}. Install them with: pip install {names}",
                             names=", ".join(faltan)))

    device = str(cfg.get("stems_device", "cpu")).strip()
    if device.startswith("cuda") and not cuda_disponible():
        print("[!] " + t("stems_device is '{d}' but CUDA is not available: falling back to CPU.", d=device), flush=True)
        device = "cpu"
    if device.startswith("cuda"):
        print("[GPU] " + t("Separation on GPU ({d}). Set stems_device='cpu' if training needs all the VRAM.", d=device), flush=True)

    modelo = sep.cargar(cfg["stems_folder"], device)
    if modelo is None:
        raise RuntimeError(t("Could not load the separation model"))

    print("[MELBAND] " + t("Separating voice: one pass on {d}...", d=device), flush=True)
    t0 = time.time()
    fuente44 = sep.leer_pcm(ruta, ffmpeg, sr=sep.SR)
    voz44 = sep.separar_voz(modelo, fuente44, device=device)
    resultado = {}
    for ficha in sufijos_stems(cfg["stems"]):
        if ficha == "voz":
            resultado[ficha] = resample_48k(np.ascontiguousarray(voz44), sep.SR, ffmpeg)
        else:
            instrumento44 = np.ascontiguousarray(fuente44 - voz44)
            resultado[ficha] = resample_48k(instrumento44, sep.SR, ffmpeg)
    del modelo, fuente44, voz44
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[MELBAND] " + t("Separated in {s:.1f}s", s=time.time() - t0), flush=True)
    return resultado


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]

    if not os.path.isdir(dataset):
        print("[!] " + t("Dataset folder does not exist: {path}", path=dataset))
        return 1

    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        print("[!] " + t("ffmpeg not found in PATH (MiniMax-H3 needs it too). Install it or set LORALAB_FFMPEG."))
        return 1

    modo = str(cfg.get("stems", "off")).strip().lower()
    if modo not in STEMS_MODES:
        print("[!] " + t("Unknown stems mode '{m}': must be one of {modes}",
                         m=modo, modes=", ".join(STEMS_MODES)))
        return 1
    fichas_pedidas = sufijos_stems(modo)

    fuentes = sorted(f for f in os.listdir(dataset)
                     if f.lower().endswith(AUDIO_EXTS)
                     and os.path.isfile(os.path.join(dataset, f))
                     and not CLIP_NAME_RE.search(f.lower()))

    # Idempotencia: una fuente está lista si ya están TODOS los ficheros que
    # va a escribir (con stems off: __cNN; con stems: __voz__cNN / __inst__cNN).
    presentes = os.listdir(dataset)
    def cubierta(nombre):
        if cfg["overwrite"]:
            return False
        for ficha in ("", *fichas_pedidas):
            prefijo = stem_codename(nombre, ficha)
            if not any(f.startswith(prefijo + "__c") for f in presentes):
                return False
        return True

    pendientes = [n for n in fuentes if not cubierta(n)]

    print(f"  {'Dataset':<22}: {os.path.abspath(dataset)}")
    print(f"  {t('Files'):<22}: {len(fuentes)}, {t('{n} to prepare', n=len(pendientes))}")
    print(f"  {t('Clip length'):<22}: {cfg['clip_seconds']}s (min {cfg['min_clip_seconds']}s, snap ±{cfg['snap_seconds']}s)")
    print(f"  {'Stems':<22}: {modo} ({cfg['stems_device']})")
    print(f"  {'Overwrite':<22}: {t('yes') if cfg['overwrite'] else t('only missing')}")

    if not pendientes:
        print(t("Nothing to do: every source already has its clips."))
        return 0

    if modo != "off":
        print("[MELBAND] " + t("Separation runs BEFORE splitting (one pass per source, like MiniMax-H3)."), flush=True)

    fuentes_dir = os.path.join(dataset, "_fuentes")
    os.makedirs(fuentes_dir, exist_ok=True)
    manifiesto_path = os.path.join(fuentes_dir, "preparado.json")
    manifiesto = read_json(manifiesto_path)
    if not isinstance(manifiesto.get("fuentes"), list):
        manifiesto = {"fuentes": []}

    started = time.time()
    hechos, total_clips, fallos = 0, 0, []

    for i, nombre in enumerate(pendientes, 1):
        base = os.path.splitext(nombre)[0]
        try:
            if modo == "off":
                datos = {"": leer_pcm(os.path.join(dataset, nombre), ffmpeg, sr=SR)}
            else:
                datos = separar_stems(os.path.join(dataset, nombre), ffmpeg, cfg)

            clips_todos = []
            for ficha, pcm in datos.items():
                if pcm is None or pcm.ndim != 2 or pcm.shape[1] < SR // 4:
                    print(f"[{i}/{len(pendientes)}] " + t("{name}{tag}: no usable audio data.",
                                                          name=base, tag=f"__{ficha}" if ficha else ""), flush=True)
                    continue
                clips = trocear_fichero(stem_codename(nombre, ficha), pcm, cfg, dataset,
                                        i, len(pendientes), started)
                clips_todos.extend(clips)

            # La fuente y su letra (si la hay) van juntas a _fuentes/. Una
            # fuente sin clips usables se queda donde está: el mensaje de
            # skip se repite en cada pasada y es un aviso, no un silencio.
            letra = os.path.join(dataset, base + ".txt")
            if clips_todos:
                if cfg.get("keep_source", True):
                    if os.path.isfile(os.path.join(dataset, nombre)):
                        shutil.move(os.path.join(dataset, nombre), os.path.join(fuentes_dir, nombre))
                    if os.path.isfile(letra):
                        shutil.move(letra, os.path.join(fuentes_dir, base + ".txt"))
                        print("    " + t("Lyrics of {name} moved to _fuentes/: assign them per clip in the Dataset Manager.",
                                         name=base), flush=True)
                elif os.path.isfile(os.path.join(dataset, nombre)):
                    os.remove(os.path.join(dataset, nombre))

            manifiesto["fuentes"].append({
                "fichero": nombre, "clips": clips_todos, "stems": modo,
                "clips_segundos": float(cfg["clip_seconds"]),
            })
            with open(manifiesto_path, "w", encoding="utf-8") as f:
                json.dump(manifiesto, f, indent=2, ensure_ascii=False)
            hecho = bool(clips_todos)
            total_clips += len(clips_todos)
            if hecho:
                hechos += 1
            else:
                fallos.append(nombre)
        except Exception as e:
            fallos.append(nombre)
            print(f"[{i}/{len(pendientes)}] {nombre} {t('FAILED')}: {e}", flush=True)

    total_t = time.time() - started
    print("\n✓ " + t("{n} clips from {m} file(s) in {time}.", n=total_clips, m=hechos,
                    time=f"{int(total_t // 60)}m{int(total_t % 60):02d}s"))
    if fallos:
        print("[!] " + t("{n} not usable or failed: {names}", n=len(fallos), names=', '.join(fallos[:10])))
    print(t("Next: assign lyrics per clip in the Dataset Manager (leave empty for instrumental), then Pre-Cache."))
    return 0 if hechos else 1


if __name__ == "__main__":
    sys.exit(main())
