# -*- coding: utf-8 -*-
"""Diagnóstico: estadísticas de latentes y prueba img2img desde un latente cacheado."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from PIL import Image  # noqa: E402

lat0 = torch.load("cached_data_mingimage/image_01_after_latent.pt", weights_only=True)
print("cached latent", tuple(lat0.shape), "mean", round(float(lat0.mean()), 4),
      "std", round(float(lat0.std()), 4), "min", round(float(lat0.min()), 3),
      "max", round(float(lat0.max()), 3), flush=True)

model = load_nf4_transformer("Ming-Image_NF4").to("cuda").eval()
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16).to("cuda")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)


def predict(lat, sigma):
    out = model(x=list(lat.unsqueeze(2).unbind(0)), t=1.0 - sigma,
                cap_feats=[emb.squeeze(0)], cap_feats_2=[emb2.squeeze(0)], return_dict=False)[0]
    return torch.stack([o[:, 0].float() for o in out], dim=0)


tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"

# 1) decodificar el latente limpio cacheado (control del VAE)
with torch.no_grad():
    img = vae.decode((lat0.to("cuda") / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
Image.fromarray(arr).save(os.path.join(tmp, "control_vae.png"))
print("control_vae guardado", flush=True)

# 2) img2img: ruido parcial sobre el latente limpio y denoise
g = torch.Generator("cuda").manual_seed(42)
noise = torch.randn(lat0.shape, generator=g, device="cuda", dtype=torch.float32)
for start in (0.5, 0.85):
    sigmas = torch.tensor([start, 0.65, 0.45, 0.25, 0.1, 0.0], device="cuda")
    sched.set_timesteps(sigmas=sigmas.tolist(), device="cuda")
    lat = ((1 - start) * lat0.to("cuda") + start * noise).float()
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            sigma = (t / 1000).expand(1)
            out = predict(lat.to(torch.bfloat16), sigma)
            lat = sched.step(-out, t, lat, return_dict=False)[0]
        img = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
    arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    p = os.path.join(tmp, f"img2img_{int(start*100)}.png")
    Image.fromarray(arr).save(p)
    print("guardado", p, flush=True)
