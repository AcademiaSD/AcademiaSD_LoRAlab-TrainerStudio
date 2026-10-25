# -*- coding: utf-8 -*-
"""Prueba temporal: preview con el DiT BF16 original y los embeddings cacheados."""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location("tr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_train_lora_mingimage.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

from mingimage_model import DiffusionTransformer  # noqa: E402
from PIL import Image  # noqa: E402

torch.manual_seed(42)

print("Cargando DiT BF16...", flush=True)
model = DiffusionTransformer.from_pretrained(
    "Ming-Image_Raw", subfolder="transformer", torch_dtype=torch.bfloat16
).to("cuda").eval()
print("padding:", model.alignment_padding_mode, "| multf:", model.multi_frame_output, flush=True)

emb = torch.load("cached_data_mingimage/_custom_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/_custom_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)
print("emb", tuple(emb.shape), float(emb.abs().max()), "| emb2", tuple(emb2.shape), float(emb2.abs().max()), flush=True)

sched = tr.FlowMatchEulerDiscreteScheduler.from_pretrained(tr.MODEL_ID, subfolder="scheduler")
with torch.no_grad():
    lat = tr.denoise(model, sched, emb, emb2, 96, 96, torch.Generator("cuda").manual_seed(42))
    vae = tr.VaeHolder.get().to("cuda")
    img = vae.decode((lat / vae.config.scaling_factor + vae.config.shift_factor).to(vae.dtype).unsqueeze(2), return_dict=False)[0]
img = ((img[0, :, 0].float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
out = os.path.join(os.environ.get("TEMP", "."), "preview_bf16.png")
Image.fromarray(img).save(out)
print("guardado:", out, flush=True)
