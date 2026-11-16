# -*- coding: utf-8 -*-
"""
Descarga de LTX-2.5 (Diffusers) para convertirlo a NF4 / Download LTX-2.5 (Diffusers) to convert it to NF4.

El repo oficial tiene gate: acepta los términos en https://huggingface.co/Lightricks/LTX-2.5-Diffusers con la
misma cuenta del token (settings/HF_token.json o `hf auth login`).
The official repo is gated: accept the terms there with the account of your token.

Solo se baja lo que usa el entrenador (~70 GB):
  transformer_full/  DiT completo (SFT / dev), el que se entrena. El destilado (transformer/) NO se baja.
  text_encoder/      Gemma 4 12B afinado para LTX (BF16, se cuantiza a NF4 después)
  connectors/        connectors de texto de 2.5 (no valen los de 2.3)
  vae/ audio_vae/ vocoder/ scheduler/ tokenizer/ duration_head/ model_index.json
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "Lightricks/LTX-2.5-Diffusers"
DEST = "./Models/LTX25-Raw"  # la carpeta que lee 5_conversor_LTX25_NF4.py

ALLOW = [
    "model_index.json",
    "README.md",
    "transformer_full/*",
    "text_encoder/*",
    "tokenizer/*",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors",  # el fichero único; los shards duplican lo mismo
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "scheduler/*",
    "duration_head/*",
]


def hf_token():
    path = os.path.join("settings", "HF_token.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            tok = (json.load(f).get("token") or "").strip()
            return tok or None
    except Exception:
        return None


def main():
    os.makedirs(DEST, exist_ok=True)
    print("=" * 60)
    print(" LTX-2.5 download / descarga  ->", os.path.abspath(DEST))
    print(" ~70 GB. transformer_full + Gemma 4 + connectors + VAEs")
    print("=" * 60)
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=DEST, allow_patterns=ALLOW,
                          token=hf_token(), max_workers=4)
    except Exception as exc:
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            print("\n[!] Gated repo. Accept the terms at https://huggingface.co/" + REPO_ID +
                  " with the account of your HF token, then run this again.")
            print("[!] Repo con gate. Acepta los términos en https://huggingface.co/" + REPO_ID +
                  " con la cuenta de tu token de HF y vuelve a ejecutar.")
            sys.exit(1)
        raise
    print("\nDownload complete / Descarga completada.")


if __name__ == "__main__":
    main()
