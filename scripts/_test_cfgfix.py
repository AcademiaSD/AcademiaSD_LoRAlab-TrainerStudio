# -*- coding: utf-8 -*-
"""Prueba temporal: CFG a escala 1.0 (pos + 1*(pos-neg)) a 768."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from PIL import Image  # noqa: E402


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


model = load_nf4_transformer("Ming-Image_NF4").to("cuda").eval()
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16).to("cuda")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True
tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"


def run(tag, emb, emb2, scale=1.0, steps=12, hw=(96, 96), seed=42):
    h, w = hw
    sched.sigma_min = 0.0
    mu = calculate_shift((h // 2) * (w // 2), sched.config.get("base_image_seq_len", 256),
                         sched.config.get("max_image_seq_len", 4096),
                         sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
    sched.set_timesteps(steps, device="cuda", mu=mu)
    g = torch.Generator("cuda").manual_seed(seed)
    lat = torch.randn((1, 16, h, w), generator=g, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for t in sched.timesteps:
            sigma = (t / 1000).expand(1)
            tm = (1.0 - sigma).to(torch.bfloat16)
            pos = model(x=list(lat.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=tm,
                        cap_feats=[emb], cap_feats_2=[emb2], return_dict=False)[0]
            pos = torch.stack([o[:, 0].float() for o in pos], dim=0)
            if scale > 0:
                neg = model(x=list(lat.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=tm,
                            cap_feats=[torch.zeros_like(emb)], cap_feats_2=[torch.zeros_like(emb2)],
                            return_dict=False)[0]
                neg = torch.stack([o[:, 0].float() for o in neg], dim=0)
                pred = pos + scale * (pos - neg)
            else:
                pred = pos
            lat = sched.step(-pred, t, lat, return_dict=False)[0]
        img = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
    arr = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    p = os.path.join(tmp, f"cfgfix_{tag}.png")
    Image.fromarray(arr).save(p)
    print("guardado", p, flush=True)


v_emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True).squeeze(0).cuda().bfloat16()
v_emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True).squeeze(0).cuda().bfloat16()
s_emb = torch.load(os.path.join(tmp, "space_emb.pt"), weights_only=True).cuda().bfloat16()
s_emb2 = torch.load(os.path.join(tmp, "space_emb2.pt"), weights_only=True).cuda().bfloat16()

run("vendor_cfg0", v_emb, v_emb2, scale=0.0)
run("vendor_cfg1", v_emb, v_emb2, scale=1.0)
run("space_cfg1", s_emb, s_emb2, scale=1.0)
