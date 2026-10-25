# -*- coding: utf-8 -*-
"""Prueba temporal: convención de t (1-sigma vs sigma)."""
import importlib.util
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

spec = importlib.util.spec_from_file_location("tr", os.path.join(_here, "2_train_lora_mingimage.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

from mingimage_model import load_nf4_transformer  # noqa: E402
from PIL import Image  # noqa: E402

model = load_nf4_transformer("Ming-Image_NF4").to("cuda").eval()
sched = tr.FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
vae = tr.VaeHolder.get().to("cuda")

emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)

tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"
for mode in ("flip_t", "flip_t_pos"):
    g = torch.Generator("cuda").manual_seed(42)
    latents = torch.randn((1, 16, 128, 128), generator=g, device="cuda", dtype=torch.float32)
    sched.set_timesteps(12, device="cuda")
    with torch.no_grad():
        for t in sched.timesteps:
            sigma = (t / 1000).expand(1)
            tm = sigma if mode.startswith("flip_t") else 1.0 - sigma
            out = model(x=list(latents.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=tm,
                        cap_feats=[emb.squeeze(0)], cap_feats_2=[emb2.squeeze(0)], return_dict=False)[0]
            out = torch.stack([o[:, 0].float() for o in out], dim=0)
            sign = 1.0 if mode.endswith("pos") else -1.0
            latents = sched.step(sign * out, t, latents, return_dict=False)[0]
        img = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
    arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    p = os.path.join(tmp, f"{mode}.png")
    Image.fromarray(arr).save(p)
    print("guardado", p, flush=True)
