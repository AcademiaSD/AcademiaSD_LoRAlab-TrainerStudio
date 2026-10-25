# -*- coding: utf-8 -*-
"""
real_codes.py — Códigos semánticos de YuE2 a partir de audio REAL
Semantic YuE2 codes from REAL audio

m-a-p no ha publicado el tokenizer de audio a códigos de YuE2 (sus códigos salen de MERT-v2). Aquí se
usa el de la comunidad: MERT-v2-FullSong capa 20 (25 Hz, normalizada por canción) -> head de
Mothersuperior (transformer de 8 capas) -> uno de los 32768 códigos por frame de 40 ms, el mismo
paso que el latente del VAE. Con estos códigos el AR aprende de tus canciones y el NAR se entrena
viendo texto + códigos, como genera ComfyUI.

Créditos / credits:
  - Head y su arquitectura (clase Tok de scripts/train_v2.py): Mothersuperior,
    https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 (CC BY-NC 4.0)
  - MERT-v2-FullSong: m-a-p, https://huggingface.co/m-a-p/MERT-v2-FullSong (CC BY-NC 4.0)
"""
import os
import shutil
import subprocess
import sys

# El layout del estudio / the studio layout
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from project_layout import resolve_model_dir

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MERT_REPO, MERT_REV = "m-a-p/MERT-v2-FullSong", "7001f7657276238629e61c7746ce70735e0604c0"
HEAD_REPO, HEAD_REV = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4", "e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5"
# v5_30k: el único head entrenado solo contra el decodificador BASE (los "joint" se ajustaron a su propio
# LoRA de decoder), así que da los códigos que el modelo base produciría para ese audio.
HEAD_FILE = "tokenizer_head_v5_30k.bf16.safetensors"
MERT_SR, FPS, LAYER, CHUNK_S = 24000, 25, 20, 30
VOCAB, WIN, D, L, H = 32768, 512, 512, 8, 8


class Tok(nn.Module):
    # Arquitectura del head de Mothersuperior (train_v2.py); el dropout no actúa en eval.
    def __init__(self, din=1024):
        super().__init__()
        self.inp = nn.Linear(din, D)
        self.pos = nn.Parameter(torch.empty(1, WIN, D))
        layer = nn.TransformerEncoderLayer(D, H, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))


def ensure_models(base_dir, token=None):
    """Baja (una vez) MERT-v2-FullSong (~1.3 GB) y el head (~86 MB) en revisiones fijas."""
    from huggingface_hub import hf_hub_download, snapshot_download
    mert_dir = str(resolve_model_dir(None, "Models/MERT-v2-FullSong", base_dir))
    if not os.path.isfile(os.path.join(mert_dir, "model.safetensors")):
        snapshot_download(MERT_REPO, revision=MERT_REV, local_dir=mert_dir, token=token,
                          allow_patterns=["*.json", "*.py", "model.safetensors", "LICENSE", "THIRD_PARTY_NOTICES.md"])
    head_dir = str(resolve_model_dir(None, "Models/YuE2-RealAudio-Head", base_dir))
    if not os.path.isfile(os.path.join(head_dir, HEAD_FILE)):
        hf_hub_download(HEAD_REPO, HEAD_FILE, revision=HEAD_REV, local_dir=head_dir, token=token)
    return mert_dir, os.path.join(head_dir, HEAD_FILE)


def _mono24(path, ffmpeg):
    flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(MERT_SR), "-"],
                          capture_output=True, timeout=3600, creationflags=flag)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + proc.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


class RealCodes:
    def __init__(self, base_dir, device="cuda", token=None):
        from safetensors.torch import load_file
        from transformers import AutoFeatureExtractor, AutoModel
        mert_dir, head_path = ensure_models(base_dir, token)
        self.device = device
        self.ffmpeg = os.environ.get("LORALAB_FFMPEG") or shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH. Install it or set LORALAB_FFMPEG.")
        self.proc = AutoFeatureExtractor.from_pretrained(mert_dir, trust_remote_code=True)
        self.mert = AutoModel.from_pretrained(mert_dir, trust_remote_code=True).to(device).eval()
        self.head = Tok().to(device).eval()
        self.head.load_state_dict({k: v.float() for k, v in load_file(head_path).items()})

    def _features(self, mono, frames):
        """[N] mono 24 kHz -> MERT capa 20 interpolada a `frames` (25 Hz) [frames, 1024] float32."""
        chunk = MERT_SR * CHUNK_S
        parts = [mono[s:s + chunk] for s in range(0, len(mono), chunk)]
        parts = [p for p in parts if len(p) >= MERT_SR // 4]
        feats = []
        with torch.inference_mode():
            for part in parts:
                inp = {k: v.to(self.device) for k, v in self.proc([part], sampling_rate=MERT_SR, return_tensors="pt").items()}
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    feats.append(self.mert(**inp, output_hidden_states=True).hidden_states[LAYER][0].float())
        h = torch.cat(feats, 0)
        return F.interpolate(h.T[None], size=frames, mode="linear", align_corners=False)[0].T

    @torch.inference_mode()
    def song(self, clips):
        """clips = [(ruta_wav, frames_del_latente)] de UNA canción, en orden. Normaliza con las
        estadísticas de toda la canción (como el head se entrenó) y devuelve una lista de códigos
        int64 por clip, con len == frames (1 código por frame de latente)."""
        feats = [self._features(_mono24(path, self.ffmpeg), frames) for path, frames in clips]
        allf = torch.cat(feats, 0)
        mean, std = allf.mean(0), allf.std(0) + 1e-5
        out = []
        for f in feats:
            x = ((f - mean) / std)[None]
            codes = []
            for s in range(0, x.shape[1], WIN):
                codes.append(self.head(x[:, s:s + WIN]).argmax(-1)[0])
            out.append(torch.cat(codes).cpu().to(torch.int64))
        return out
