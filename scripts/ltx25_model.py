# -*- coding: utf-8 -*-
"""
ltx25_model.py — piezas comunes del entrenador LTX-2.5 / shared pieces of the LTX-2.5 trainer.

* load_nf4_transformer(): construye LTX2VideoTransformer3DModel vacío y lo rellena desde Models/LTX25-NF4
  (Linear de los bloques en NF4, el resto de Linear en BF16, others.safetensors). Mismo formato que LTX23-NF4.
* BlockSwap, plan_resident_blocks, nf4_on_gpu...: ahora en block_swap.py (los comparte con FLUX.2 Klein 9B);
  se importan aquí para que el código de LTX 2.5 siga igual.
"""
import json
import os

import torch
from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from safetensors.torch import load_file

from block_swap import (  # noqa: F401  (piezas comunes, antes definidas aquí)
    mmap_tensor, to_device_except_nf4, nf4_on_gpu, patch_peft_dora_init, block_nf4_bytes, BlockSwap,
    VRAM_PROFILES_GB, plan_resident_blocks,
)

COMPUTE_DTYPE = torch.bfloat16


# ----------------------------------------------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------------------------------------------
def _parent(root, name):
    parts = name.split(".")
    mod = root
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]


class CastLinear(torch.nn.Linear):
    """Linear BF16 que se comporta como Linear4bit: calcula en el dtype del peso y devuelve el de la entrada."""

    def forward(self, x):
        y = torch.nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)
        return y.to(x.dtype)


def load_nf4_transformer(model_dir, device="cuda", compute_dtype=COMPUTE_DTYPE, blocks_on_cpu=False, log=print):
    """
    Devuelve el transformer listo. Con `blocks_on_cpu=True` el dato NF4 empaquetado de los bloques se queda en RAM
    (su quant_state, bias y todo lo demás van a `device`): así cabe en 8 GB, y BlockSwap sube después los residentes.
    Nunca hay que llamar a .to("cuda") sobre el modelo entero después: subiría todos los bloques. Usar to_device_except_nf4().
    """
    from diffusers import LTX2VideoTransformer3DModel

    with open(os.path.join(model_dir, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(model_dir))
    weights_dir = os.path.join(model_dir, "weights")

    for name, info in index.get("quantized", {}).items():
        path = os.path.join(weights_dir, info["file"])
        on_cpu = blocks_on_cpu and name.startswith("transformer_blocks.")
        with safe_open(path, framework="pt", device="cpu") as f:
            data = None if on_cpu else f.get_tensor("weight")   # los bloques se mapean, sin leerlos aquí
            bias = f.get_tensor("bias") if info.get("bias") else None
            qs = {k[len("quant_state."):]: f.get_tensor(k) for k in f.keys() if k.startswith("quant_state.")}
        with torch.device("meta"):
            layer = Linear4bit(int(info["in_features"]), int(info["out_features"]), bias=bool(info.get("bias")),
                               quant_type="nf4", compute_dtype=compute_dtype)
        layer._nf4_file = path   # origen del dato NF4: BlockSwap.restore_resident lo puede releer de aquí
        if on_cpu:
            # El dato empaquetado de los bloques no se copia a RAM: se mapea desde su fichero. BlockSwap decide
            # después si cada bloque va a la GPU, a RAM fijada (rápido) o se queda mapeado (disco).
            data = mmap_tensor(path, "weight")
        layer.weight = Params4bit.from_prequantized(data=data, quantized_stats=qs, requires_grad=False,
                                                    device="cpu" if on_cpu else device, module=layer)
        if on_cpu:
            layer.weight.quant_state.to(device)   # absmax/code en GPU; el dato empaquetado, mapeado
            layer._mmap_home = layer.weight.data
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias.to(device, dtype=compute_dtype), requires_grad=False)
        parent, child = _parent(model, name)
        setattr(parent, child, layer)

    for name, info in index.get("unquantized", {}).items():
        path = os.path.join(weights_dir, info["file"])
        with safe_open(path, framework="pt", device="cpu") as f:
            w = f.get_tensor("weight")
            b = f.get_tensor("bias") if info.get("bias") else None
        with torch.device("meta"):
            layer = CastLinear(int(info["in_features"]), int(info["out_features"]), bias=b is not None)
        layer.weight = torch.nn.Parameter(w.to(device, dtype=compute_dtype), requires_grad=False)
        if b is not None:
            layer.bias = torch.nn.Parameter(b.to(device, dtype=compute_dtype), requires_grad=False)
        parent, child = _parent(model, name)
        setattr(parent, child, layer)

    others = load_file(os.path.join(model_dir, "others.safetensors"), device=str(device))
    model.load_state_dict(others, strict=False, assign=True)

    # Buffers que no vienen en el checkpoint (rope, etc.): se recrean en CPU por from_config -> a device.
    missing = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError("NF4 model incomplete / modelo NF4 incompleto: {}".format(missing[:5]))
    for n, b in list(model.named_buffers()):
        if b.device.type != torch.device(device).type:
            parent, child = _parent(model, n)
            parent.register_buffer(child, b.to(device), persistent=child in getattr(parent, "_buffers", {}))

    model.requires_grad_(False)
    model.eval()
    nq = len(index.get("quantized", {}))
    log("NF4 layers / capas NF4: {} · BF16 Linear: {}".format(nq, len(index.get("unquantized", {}))))
    return model


# ----------------------------------------------------------------------------------------------------------------
# Gemma 4 (text encoder) para 8 GB
# ----------------------------------------------------------------------------------------------------------------
def load_text_encoder_lean(path, device="cuda", compute_dtype=COMPUTE_DTYPE, log=print):
    """
    Gemma 4 NF4 con solo las capas de texto en la GPU (~5.6 GB en vez de ~7.7):
      * la tabla de embeddings (262144 x 3840, 2 GB en BF16, NF4 no la comprime) se queda en RAM: la búsqueda se
        hace en CPU y solo el resultado [tokens x 3840] sube a la GPU;
      * lm_head (copia atada de esa tabla) no hace falta para sacar estados ocultos: se sustituye por Identity;
      * las torres de visión y audio no se usan y se quedan en CPU.
    Se carga en CPU y se sube a mano porque el dispatch de accelerate vuelve a subir la tabla a la GPU.
    """
    import transformers

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())   # bitsandbytes necesita el índice
    cfg = transformers.AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    model = cls.from_pretrained(path, device_map={"": "cpu"}, torch_dtype=compute_dtype)
    lm = model.model.language_model
    emb = lm.embed_tokens

    # Cada capa NF4 sube con Params4bit.to (mueve también su quant_state). Si transformers la dejó como nn.Linear
    # con un Params4bit dentro (pasa al cargar en CPU), se envuelve en Linear4bit para que haga el matmul NF4.
    moved = 0
    for name, mod in list(lm.named_modules()):
        for cname, child in list(mod.named_children()):
            w = getattr(child, "weight", None)
            if not isinstance(w, Params4bit):
                continue
            if isinstance(child, Linear4bit):
                child.weight = w.to(device)
                if child.bias is not None:
                    child.bias.data = child.bias.data.to(device)
            else:
                out_f, in_f = w.quant_state.shape
                with torch.device("meta"):
                    new = Linear4bit(in_f, out_f, bias=child.bias is not None, quant_type="nf4",
                                     compute_dtype=compute_dtype)
                new.weight = w.to(device)
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.to(device), requires_grad=False)
                setattr(mod, cname, new)
            moved += 1
    for cname, child in lm.named_children():
        if child is emb:
            continue
        for mod in child.modules():
            for k, p in list(mod._parameters.items()):
                if p is not None and not isinstance(p, Params4bit) and p.device.type != "cuda":
                    p.data = p.data.to(device)
            for k, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[k] = b.to(device)
    for k, b in list(lm._buffers.items()):
        if b is not None:
            lm._buffers[k] = b.to(device)

    orig = emb.forward
    emb.forward = lambda ids, *a, **kw: orig(ids.to("cpu"), *a, **kw).to(device)
    model.lm_head = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    torch.cuda.empty_cache()
    log("[TE] Gemma 4 NF4: {} NF4 layers in VRAM, embeddings in RAM ({:.2f} GB)".format(
        moved, torch.cuda.memory_allocated() / 1e9))
    return model


# ----------------------------------------------------------------------------------------------------------------
# Vídeo (ffmpeg). LTX: 24 fps, el VAE comprime 8x en el tiempo -> clips de 8n+1 fotogramas (1, 9, 17, ..., 121).
# ----------------------------------------------------------------------------------------------------------------
import json as _json
import math as _math
import shutil as _shutil
import subprocess as _subprocess

import numpy as _np

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
FPS = 24.0
AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4   # 25: sample_rate / hop / compresión temporal del VAE de audio


def ffbin(name):
    return _shutil.which(name) or _shutil.which(name + ".exe")


def video_info(path):
    """{width, height, fps, duration, has_audio} o None si ffprobe no lo lee."""
    probe = ffbin("ffprobe")
    if not probe:
        raise RuntimeError("ffmpeg/ffprobe not found in PATH / no están en el PATH")
    try:
        out = _subprocess.run([probe, "-v", "error", "-show_entries",
                               "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json",
                               str(path)], capture_output=True, text=True, timeout=60).stdout
        data = _json.loads(out)
        v = next(s for s in data["streams"] if s.get("codec_type") == "video")
        num, _, den = v["r_frame_rate"].partition("/")
        return {"width": int(v["width"]), "height": int(v["height"]), "fps": float(num) / float(den or 1),
                "duration": float(data["format"]["duration"]),
                "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"])}
    except Exception:
        return None


def audio_duration(path):
    """Duración en segundos de un fichero de audio (ffprobe), o None."""
    probe = ffbin("ffprobe")
    if not probe:
        return None
    try:
        out = _subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                               str(path)], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def valid_frames(n):
    """El mayor 8k+1 que no pasa de n."""
    return max(1, (int(n) - 1) // 8 * 8 + 1)


def clip_frames(duration, max_frames, fps=FPS):
    """Fotogramas que se entrenan de un clip: los que caben a `fps`, en la rejilla 8k+1, sin pasar de max_frames."""
    return valid_frames(min(int(max_frames), max(1, _math.floor(duration * fps + 1e-6))))


def latent_frames(frames):
    return (int(frames) - 1) // 8 + 1


def audio_latent_length(frames, fps=FPS):
    """Longitud del latente de audio para `frames` fotogramas: la cuenta de LTX2Pipeline."""
    return max(1, round(frames / fps * AUDIO_LATENTS_PER_SECOND))


def read_video(path, frames, width, height, fps=FPS):
    """Los `frames` primeros fotogramas a `fps`, escalados hasta cubrir width x height y recortados al centro."""
    ffmpeg = ffbin("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH / no está en el PATH")
    vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height}")
    raw = _subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", str(frames),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=600,
                          check=True).stdout
    got = len(raw) // (width * height * 3)
    if got < frames:
        raise RuntimeError("Only {} of {} frames could be read / solo se leyeron {} de {}".format(got, frames, got, frames))
    return _np.frombuffer(raw[:frames * width * height * 3], dtype=_np.uint8).reshape(frames, height, width, 3)


def write_video(frames, path, fps=FPS, audio=None, sample_rate=None):
    """[F, H, W, 3] uint8 -> mp4 H.264 (+ AAC si `audio` [canales, muestras] float). Sin ffmpeg, .webp animado."""
    import os
    import tempfile
    ffmpeg = ffbin("ffmpeg")
    if ffmpeg:
        h, w = frames.shape[1:3]
        tmp = path + ".tmp.mp4"
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-"]
        wav = None
        if audio is not None and sample_rate:
            import wave
            a = _np.clip(_np.asarray(audio, dtype=_np.float32), -1, 1)
            if a.ndim == 1:
                a = a[None]
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(wav, "wb") as wf:
                wf.setnchannels(a.shape[0])
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes((a.T * 32767).astype(_np.int16).tobytes())
            # apad + -shortest: el audio se rellena con silencio y el fichero acaba con el vídeo (sin perder fotogramas).
            cmd += ["-i", wav, "-af", "apad", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", tmp]
        try:
            _subprocess.run(cmd, input=frames.tobytes(), check=True, timeout=600)
        finally:
            if wav:
                try:
                    os.remove(wav)
                except OSError:
                    pass
        os.replace(tmp, path)
        return path
    from PIL import Image
    path = os.path.splitext(path)[0] + ".webp"
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, quality=90)
    return path


# ----------------------------------------------------------------------------------------------------------------
# Audio. Preprocesado de ltx-core (AudioProcessor.waveform_to_mel) y empaquetado de LTX2Pipeline:
#   onda 16 kHz, 2 canales (mono duplicado) -> mel de MAGNITUD: n_fft=win=1024, hop=160, 64 bandas slaney 0-8 kHz,
#   center + reflect, log(clamp(1e-5)) -> [B, 2, T, 64] -> VAE de audio -> media (no muestra) [B, 8, T/4, 16]
#   -> empaquetado [B, T/4, 128] -> (x - latents_mean) / latents_std.   25 latentes por segundo.
# ----------------------------------------------------------------------------------------------------------------
AUDIO_SR = 16000


def read_audio(path, start=0.0, duration=None, sample_rate=AUDIO_SR):
    """Pista de audio de un fichero (vídeo o audio) -> float32 [2, N] a `sample_rate`, o None si no tiene audio."""
    ffmpeg = ffbin("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH / no está en el PATH")
    cmd = [ffmpeg, "-v", "error", "-ss", "{:.4f}".format(start), "-i", str(path)]
    if duration is not None:
        cmd += ["-t", "{:.4f}".format(duration)]
    cmd += ["-vn", "-ac", "2", "-ar", str(int(sample_rate)), "-f", "f32le", "-"]
    raw = _subprocess.run(cmd, capture_output=True, timeout=600).stdout
    if not raw:
        return None
    a = _np.frombuffer(raw, dtype=_np.float32)
    a = a[: len(a) // 2 * 2].reshape(-1, 2).T.copy()
    if duration is not None:
        n = int(round(duration * sample_rate))
        if a.shape[1] < n:   # como ltx-trainer: relleno con silencio al final si la pista es más corta
            a = _np.pad(a, ((0, 0), (0, n - a.shape[1])))
        a = a[:, :n]
    return a


_MEL = {}


def waveform_to_mel(wave, device="cuda"):
    """[2, N] o [B, 2, N] float -> log-mel [B, 2, T, 64] (como ltx-core)."""
    import torchaudio
    key = str(device)
    if key not in _MEL:
        _MEL[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_SR, n_fft=1024, win_length=1024, hop_length=160, f_min=0.0, f_max=AUDIO_SR / 2.0,
            n_mels=64, window_fn=torch.hann_window, center=True, pad_mode="reflect", power=1.0,
            mel_scale="slaney", norm="slaney").to(device)
    w = torch.as_tensor(wave, dtype=torch.float32, device=device)
    if w.ndim == 2:
        w = w.unsqueeze(0)
    mel = _MEL[key](w)                                   # [B, 2, 64, T]
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.permute(0, 1, 3, 2).contiguous()          # [B, 2, T, 64]


def encode_audio(audio_vae, wave, device="cuda"):
    """Onda [2, N] a 16 kHz -> latente normalizado en el formato del caché del trainer: [1, 128, T]."""
    dtype = next(audio_vae.parameters()).dtype
    mel = waveform_to_mel(wave, device).to(dtype)
    with torch.no_grad():
        post = audio_vae.encode(mel).latent_dist
    z = post.mode() if hasattr(post, "mode") else post.mean       # [1, 8, T, 16]
    packed = z.transpose(1, 2).flatten(2, 3).float()              # [1, T, 128]
    mean = audio_vae.latents_mean.to(packed.device, torch.float32)
    std = audio_vae.latents_std.to(packed.device, torch.float32)
    packed = (packed - mean) / std
    return packed.transpose(1, 2).contiguous().to(torch.bfloat16).cpu()   # [1, 128, T]


def decode_audio(audio_vae, vocoder, latent):
    """Latente normalizado [1, T, 128] (tokens) o [1, 128, T] -> onda float32 [canales, N] y su sample rate."""
    dtype = next(audio_vae.parameters()).dtype
    x = latent.float()
    if x.shape[-1] != 128:
        x = x.transpose(1, 2)
    mean = audio_vae.latents_mean.to(x.device, torch.float32)
    std = audio_vae.latents_std.to(x.device, torch.float32)
    x = x * std + mean                                            # [1, T, 128]
    x = x.unflatten(2, (-1, 16)).transpose(1, 2)                  # [1, 8, T, 16]
    with torch.no_grad():
        mel = audio_vae.decode(x.to(dtype), return_dict=False)[0]
        wav = vocoder(mel.to(next(vocoder.parameters()).dtype))
    sr = int(getattr(vocoder.config, "output_sampling_rate", AUDIO_SR) or AUDIO_SR)
    return wav[0].float().cpu().numpy(), sr


# ----------------------------------------------------------------------------------------------------------------
# Descarga del repo NF4 con progreso visible en la consola web
# ----------------------------------------------------------------------------------------------------------------
def _dir_bytes(path):
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total


def download_repo(repo_id, local_dir, token=None, log=print, every=2.0, allow_patterns=None):
    """
    snapshot_download con una línea de progreso propia (GB descargados, velocidad, tiempo restante) que se
    reescribe con "\r" en la consola web. No depende de las barras de tqdm / hf_xet, que según la versión de
    huggingface_hub salen o no cuando la salida no es una terminal.
    """
    import threading
    import time
    from huggingface_hub import HfApi, snapshot_download

    total = 0
    try:
        info = HfApi(token=token).model_info(repo_id, files_metadata=True)
        import fnmatch
        total = sum((s.size or 0) for s in info.siblings
                    if not allow_patterns or any(fnmatch.fnmatch(s.rfilename, p) for p in allow_patterns))
    except Exception:
        pass
    os.makedirs(local_dir, exist_ok=True)
    start_bytes = _dir_bytes(local_dir)
    stop = threading.Event()

    def watch():
        t0 = time.time()
        while not stop.wait(every):
            now = _dir_bytes(local_dir)
            done = max(0, now - start_bytes)
            el = max(1e-6, time.time() - t0)
            speed = done / el
            line = "[HF] {:.2f} GB".format(now / 1e9)
            if total:
                pct = min(100.0, 100.0 * now / total)
                left = max(0, total - now)
                eta = left / speed if speed > 0 else 0
                line += " / {:.2f} GB ({:.0f}%) | {:.1f} MB/s | ETA {:d}m{:02d}s".format(
                    total / 1e9, pct, speed / 1e6, int(eta // 60), int(eta % 60))
            else:
                line += " | {:.1f} MB/s".format(speed / 1e6)
            print("\r" + line, end="", flush=True)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token, max_workers=4,
                          allow_patterns=allow_patterns)
    finally:
        stop.set()
        th.join(timeout=every + 1)
        got = _dir_bytes(local_dir)
        got = min(got, total) if total else got   # sin contar los metadatos de .cache
        print("\r[HF] {:.2f} GB".format(got / 1e9) + (" / {:.2f} GB".format(total / 1e9) if total else ""), flush=True)
