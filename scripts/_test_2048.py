# -*- coding: utf-8 -*-
"""Prueba temporal: 2048 con DiT NF4 (bucket oficial de text-to-image)."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusion.generator import ConditionedTransformer  # noqa: E402
from diffusion.pipeline import ImageGenerationPipeline  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

dit = load_nf4_transformer("Ming-Image_NF4")
cond = ConditionedTransformer(dit, vision_dim=2560, use_identity_mlp=True, text_encoder_norm=False)
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16)
try:
    vae.enable_tiling()
    print("vae tiling ON", flush=True)
except Exception as exc:
    print("vae tiling FAIL", exc, flush=True)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True
pipe = ImageGenerationPipeline(vae=vae, transformer=cond, text_encoder=None, tokenizer=None, scheduler=sched).to("cuda")

emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True)

tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"
with torch.no_grad():
    out = pipe(
        prompt_embeds=[emb.squeeze(0).to("cuda", torch.bfloat16)],
        negative_prompt_embeds=[torch.zeros_like(emb.squeeze(0)).to("cuda", torch.bfloat16)],
        prompt_embeds_2=[emb2.squeeze(0).to("cuda", torch.bfloat16)],
        negative_prompt_embeds_2=[torch.zeros_like(emb2.squeeze(0)).to("cuda", torch.bfloat16)],
        guidance_scale=1.0, generator=torch.manual_seed(42), num_inference_steps=12,
        height=2048, width=2048, max_sequence_length=512, device="cuda",
        sample_mode="argmax", num_frames_per_prompt=1,
    )
p = os.path.join(tmp, "res2048.png")
out.images[0].save(p)
print("guardado", p, flush=True)
