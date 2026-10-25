# -*- coding: utf-8 -*-
"""Prueba temporal: t2i con sigma inicial < 1 (evitar t_model=0 exacto)."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from PIL import Image  # noqa: E402

model = load_nf4_transformer("Ming-Image_NF4").to("cuda").eval()
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16).to("cuda")
emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)
tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"


def run(start, steps=12, hw=(96, 96), seed=42):
    h, w = hw
    g = torch.Generator("cuda").manual_seed(seed)
    lat = torch.randn((1, 16, h, w), generator=g, device="cuda", dtype=torch.float32)
    sigmas = torch.linspace(start, 0.0, steps + 1, device="cuda").tolist()
    with torch.no_grad():
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            sigma = torch.tensor([s], device="cuda")
            out = model(x=list(lat.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=1.0 - sigma,
                        cap_feats=[emb.squeeze(0)], cap_feats_2=[emb2.squeeze(0)], return_dict=False)[0]
            pred = torch.stack([o[:, 0].float() for o in out], dim=0)
            lat = lat + (s_next - s) * (-pred)
        img = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
    arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    p = os.path.join(tmp, f"t2i_start{int(start*100)}.png")
    Image.fromarray(arr).save(p)
    print("guardado", p, flush=True)


for st in (0.99, 0.95, 0.9, 0.8):
    run(st)
