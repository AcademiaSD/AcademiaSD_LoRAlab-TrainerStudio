# -*- coding: utf-8 -*-
"""Prueba temporal: pipeline oficial de Ming con DiT BF16 y embeddings cacheados."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from mingimage_model import DiffusionTransformer  # noqa: E402
from diffusion.generator import ConditionedTransformer  # noqa: E402
from diffusion.pipeline import ImageGenerationPipeline  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from PIL import Image  # noqa: E402

print("Cargando DiT BF16...", flush=True)
dit = DiffusionTransformer.from_pretrained(
    "Ming-Image_Raw", subfolder="transformer", torch_dtype=torch.bfloat16
)
cond = ConditionedTransformer(dit, vision_dim=1152, use_identity_mlp=True, text_encoder_norm=False)
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True  # igual que generator.py:214

pipe = ImageGenerationPipeline(
    vae=vae, transformer=cond, text_encoder=None, tokenizer=None, scheduler=sched
).to("cuda")

emb = torch.load("cached_data_mingimage/_custom_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/_custom_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)

with torch.no_grad():
    out = pipe(
        prompt_embeds=[emb.squeeze(0)],
        negative_prompt_embeds=[torch.zeros_like(emb.squeeze(0))],
        prompt_embeds_2=[emb2.squeeze(0)],
        negative_prompt_embeds_2=[torch.zeros_like(emb2.squeeze(0))],
        guidance_scale=1.0,
        generator=torch.manual_seed(42),
        num_inference_steps=12,
        height=768,
        width=768,
        max_sequence_length=512,
        device="cuda",
        sample_mode="sample",
        num_frames_per_prompt=1,
    )

img = out.images[0]
path = os.path.join(os.environ.get("TEMP", "."), "preview_oficial.png")
img.save(path)
print("guardado:", path, flush=True)
