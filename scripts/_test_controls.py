# -*- coding: utf-8 -*-
"""Controles del sampler: contenido inicial vs ruido."""
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
lat0 = torch.load("cached_data_mingimage/image_01_after_latent.pt", weights_only=True)
tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"


def run(name, lat, start, steps):
    sig = torch.linspace(start, 0.0, steps + 1).tolist()
    with torch.no_grad():
        for i in range(steps):
            s, s_next = sig[i], sig[i + 1]
            t = torch.tensor([1.0 - s], device="cuda")
            out = model(x=list(lat.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=t,
                        cap_feats=[emb.squeeze(0)], cap_feats_2=[emb2.squeeze(0)], return_dict=False)[0]
            pred = torch.stack([o[:, 0].float() for o in out], dim=0)
            lat = lat + (s_next - s) * (-pred)
        img = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
    arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    p = os.path.join(tmp, f"ctrl_{name}.png")
    Image.fromarray(arr).save(p)
    print("guardado", p, flush=True)


g = torch.Generator("cuda").manual_seed(42)
noise = torch.randn(lat0.shape, generator=g, device="cuda", dtype=torch.float32)

run("clean_as_noise", lat0.to("cuda").clone(), 1.0, 12)
run("noise_std01", noise * 0.1, 1.0, 12)
run("zeros", torch.zeros_like(lat0).to("cuda"), 1.0, 12)
run("img2img50", 0.5 * lat0.to("cuda") + 0.5 * noise, 0.5, 8)
run("img2img90", 0.1 * lat0.to("cuda") + 0.9 * noise, 0.9, 8)
