# -*- coding: utf-8 -*-
r"""
0b_transcribe_yue2.py - Transcribe cada clip con Whisper y escribe la letra REAL en su .txt
0b_transcribe_yue2.py - Transcribes each clip with Whisper and writes the ACTUAL lyrics into its .txt

POR QUE / WHY
-------------
El reparto manual de un fichero de letras (p.ej. `Letras/Agila.txt`) entre clips
troceados por silencios NO cuadra: los cortes no coinciden con los versos y el
mismo texto se repite en cada parte de la cancion. Whisper transcribe el AUDIO
de cada clip, asi que la letra que condiciona la etapa AR es la que de verdad
suena en ese trozo.
Distributing one lyrics file (e.g. `Letras/Agila.txt`) across silence-sliced
clips does NOT align: cuts rarely match verses and the same text repeats in
every part of the song. Whisper transcribes each clip's AUDIO, so the lyrics
that condition the AR stage are what really plays in that take.

USAGE (desde el studio root / from the studio root):
  .\venv\Scripts\python.exe scripts\0b_transcribe_yue2.py
  .\venv\Scripts\python.exe scripts\0b_transcribe_yue2.py --overwrite
  .\venv\Scripts\python.exe scripts\0b_transcribe_yue2.py --model openai/whisper-large-v3-turbo

Lee de / Reads:
  settings/pre_cache_settings_yue2.json  -> "dataset_path" (la carpeta de clips)
  settings/caption_settings_yue2.json    -> "whisper_model", "whisper_language", "overwrite"
  (los flags de CLI ganan / CLI flags win)

Notas / Notes:
  * Clips vocales (`__cNN`, con letra) -> .txt con la letra trascrita.
  * Clips instrumentales delimitados (`__inst__cNN`) -> .txt vacio, sin pasar
    por Whisper (no hay voz que transcribir).
  * Si un clip suena instrumental (o Whisper alucina repitiendo frases) el
    texto se descarta y el .txt queda VACIO: lo puedes corregir a mano en el
    Dataset Manager, exactamente igual que cualquier caption.
  * Los .txt que ya tienen texto NO se tocan salvo --overwrite (o el checkbox
    Overwrite de la GUI): lo escrito a mano se respeta.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import zlib

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

# YuE2-Vae: 48 kHz estereo. Whisper consume 16 kHz mono: lo decodifica ffmpeg, como en el Pre-Cache
# (torchaudio.load necesita torchcodec desde torchaudio 2.9).
# YuE2-Vae: 48 kHz stereo. Whisper eats 16 kHz mono, decoded by ffmpeg as in the Pre-Cache
# (torchaudio.load needs torchcodec since torchaudio 2.9).
SR_SOURCE = 48000
SR_WHISPER = 16000

CONFIG_PATH = "settings/caption_settings_yue2.json"
PRECACHE_CONFIG = "settings/pre_cache_settings_yue2.json"

CLIP_NAME_RE = re.compile(r"__(?:voz|inst)?(?:__)?c\d{2}\.wav$")

# Sufijo instrumental: no lleva letra, jamas.
# Instrumental suffix: carries no lyrics, ever.
INST_RE = re.compile(r"__inst(?:__)?c\d{2}\.wav$")

# Basura tipica de Whisper con musica/subtitulos (ES/EN). Se quita entera.
# Typical Whisper music/subtitles hallucinations. Removed wholesale.
BOILERPLATE = (
    "subtítulos por la comunidad de amara.org",
    "subtítulos por amara.org",
    "amara.org",
    "amara.org community",
    "¡hola a todos!",
    "¡hola a todos! bien venide!",
    "hola a todos",
    "gracias por ver el vídeo",
    "gracias por ver el video",
    "¡suscríbete!",
    "¡sigueme en tiktok!",
    "sígueme en tiktok",
    "suscríbete y dale a la campana",
    "[música]",
    "[musica]",
    "[aplausos]",
    "[risas]",
    "[silencio]",
    "[music]",
    "[applause]",
    "[laughter]",
    "[silence]",
    "谢谢观看",
    "字幕由amara.org社区提供",
    "ハートそのままに",
    "♪",
    "·",
)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_txt(path, text):
    """UTF-8 sin BOM (PowerShell Set-Content escribe BOM; aqui no)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_whisper_mono_16k(path):
    """Fichero de audio -> numpy float32 mono @16 kHz con ffmpeg. Clip plano o de stems."""
    import shutil
    import subprocess
    ffmpeg = next((c for c in (os.environ.get("LORALAB_FFMPEG", ""), "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                   if c and os.path.isfile(c)), None) or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(t("ffmpeg not found in PATH. Install it or set LORALAB_FFMPEG."))
    flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR_WHISPER), "-"],
                          capture_output=True, timeout=600, creationflags=flag)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + proc.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def clean_transcript(text):
    """
    Quita boilerplate de subtitulos y decide si el texto es letra o basura.
    Devuelve el texto final ("" = instrumental / sin letra usable).
    Removes subtitle boilerplate and decides lyrics vs garbage. Empty string
    means instrumental / unusable lyrics.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low in BOILERPLATE or low.replace("♪", "").strip() in BOILERPLATE:
            continue
        # Fila hecha solo de musica/simbolos: fuera.
        if all(ch in "♪♫♩♬·.- " for ch in line):
            continue
        lines.append(line)
    if not lines:
        return ""
    # Wh seldom repite la MISMA frase en bucle: se de-duplica en consecucion.
    dedup = []
    for line in lines:
        if not dedup or dedup[-1] != line:
            dedup.append(line)
    body = " ".join(dedup)
    # Ratio de compresion zlib (como el detector de bucles de openai-whisper):
    # texto muy repetido comprime brutal -> descartado como alucinacion.
    raw = body.encode("utf-8")
    ratio = len(raw) / max(1, len(zlib.compress(raw)))
    if ratio > 2.4:
        return ""
    return body


def main():
    parser = argparse.ArgumentParser(description="Transcribe dataset clips with Whisper (YuE2 lyrics).")
    parser.add_argument("--model", default=None, help="HF id del modelo Whisper (default: caption settings o large-v3).")
    parser.add_argument("--language", default=None, help="Idioma de transcripcion (default: 'spanish').")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--overwrite", action="store_true", help="Reescribe .txt que ya tengan texto.")
    args = parser.parse_args()

    caption_cfg = read_json(CONFIG_PATH)
    precache_cfg = read_json(PRECACHE_CONFIG)

    dataset_path = precache_cfg.get("dataset_path", "./dataset") or "./dataset"
    dataset_dir = os.path.abspath(dataset_path)
    if not os.path.isdir(dataset_dir):
        print(t("Dataset folder not found: {p}", p=dataset_dir))
        return 1

    model_id = args.model or caption_cfg.get("whisper_model") or "openai/whisper-large-v3"
    lang = args.language or caption_cfg.get("whisper_language") or "spanish"
    # GUI: checkbox Overwrite; CLI: --overwrite. Cualquiera de los dos basta,
    # pero con 'overwrite' a False en settings hay que pasar el flag.
    overwrite = args.overwrite or bool(caption_cfg.get("overwrite", False))

    clips = sorted(
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(".wav") and CLIP_NAME_RE.search(f)
    )
    if not clips:
        print(t("No YuE2 clips found in {p}", p=dataset_dir))
        return 1

    print(t("Transcribing {n} clips with {m} ({lang}), overwrite={o}", n=len(clips), m=model_id, lang=lang, o=overwrite))
    print(f"dataset: {dataset_dir}")

    # Se retrasa el import pesado para que --help arranque al instante.
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(t("Loading {m} on {d} ({dt})...", m=model_id, d=device, dt=dtype))

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    n_done = n_empty = n_kept = n_fail = 0
    t0 = time.time()

    for idx, fname in enumerate(clips, start=1):
        wav_path = os.path.join(dataset_dir, fname)
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(dataset_dir, base + ".txt")

        prefix = f"[{idx}/{len(clips)}] {fname}: "
        if INST_RE.search(fname):
            # Stems instrumental: nace vacio, sin gasto de GPU.
            # Instrumental stem: born empty, no GPU spent.
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 and not overwrite:
                print(prefix + t("kept instrument empty (txt exists)"))
                n_kept += 1
            else:
                write_txt(txt_path, "")
                print(prefix + t("instrumental -> empty txt"))
                n_empty += 1
            continue

        have_text = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0
        if have_text and not overwrite:
            print(prefix + t("txt exists, kept (no overwrite)"))
            n_kept += 1
            continue

        try:
            audio = load_whisper_mono_16k(wav_path)
            inputs = processor(audio, sampling_rate=SR_WHISPER, return_tensors="pt")
            feats = inputs.input_features.to(device=device, dtype=dtype)
            with torch.inference_mode():
                # language/task en kwargs: transformers rellena los ids de
                # decodificador por nosotros (evita el lio del forced ids
                # clasico). Si la version corta el kwarg, caemos al modo viejo.
                # language/task kwargs: transformers fills decoder ids itself
                # (avoids the old forced_ids mess). Falls back if unsupported.
                try:
                    seq = model.generate(feats, language=lang, task="transcribe")
                except (TypeError, ValueError):
                    forced = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                    seq = model.generate(feats, forced_decoder_ids=forced)
            raw = processor.batch_decode(seq, skip_special_tokens=True)[0]
            letra = clean_transcript(raw)
            write_txt(txt_path, letra)
            n_ch = len(letra)
            if letra:
                preview = (letra[:58] + "...") if n_ch > 58 else letra
                print(prefix + t("transcribed ({n} chars): {s}", n=n_ch, s=preview))
                n_done += 1
            else:
                print(prefix + t("empty lyrics (instrumental or loop) -> empty txt"))
                n_empty += 1
        except Exception as exc:
            print(prefix + t("ERROR: {e}", e=exc))
            n_fail += 1

        if idx % 10 == 0 or idx == len(clips):
            rate = (time.time() - t0) / idx
            print(t("progress {n}/{total} - {r:.1f} s/clip", n=idx, total=len(clips), r=rate))

    print(t("Done: {d} transcribed, {e} empty, {k} kept, {f} errors - {s:.1f} s total",
            d=n_done, e=n_empty, k=n_kept, f=n_fail, s=time.time() - t0))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
