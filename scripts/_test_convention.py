# -*- coding: utf-8 -*-
"""Determina la convencion de t y la salida del modelo por correlacion."""
import os
import sys

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from mingimage_model import load_nf4_transformer  # noqa: E402

model = load_nf4_transformer("Ming-Image_NF4").to("cuda").eval()
emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)
x0 = torch.load("cached_data_mingimage/image_01_after_latent.pt", weights_only=True).to("cuda").float()

g = torch.Generator("cuda").manual_seed(42)
noise = torch.randn(x0.shape, generator=g, device="cuda", dtype=torch.float32)

targets = {
    "x0": x0,
    "noise": noise,
    "noise-x0": noise - x0,
    "x0-noise": x0 - noise,
}


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


with torch.no_grad():
    for sigma in (0.2, 0.5, 0.8):
        xt = (1 - sigma) * x0 + sigma * noise
        for tname, tval in (("1-s", 1.0 - sigma), ("s", sigma)):
            t = torch.tensor([tval], device="cuda", dtype=torch.bfloat16)
            out = model(x=list(xt.to(torch.bfloat16).unsqueeze(2).unbind(0)), t=t,
                        cap_feats=[emb.squeeze(0)], cap_feats_2=[emb2.squeeze(0)], return_dict=False)[0]
            pred = torch.stack([o[:, 0].float() for o in out], dim=0)
            sims = " ".join(f"{k}:{cos(pred, v):+.3f}" for k, v in targets.items())
            print(f"sigma={sigma:.1f} t={tname:4s} | {sims}", flush=True)
