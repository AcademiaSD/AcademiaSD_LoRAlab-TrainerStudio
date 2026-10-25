# -*- coding: utf-8 -*-
"""Prueba temporal: variantes de condicionamiento con el pipeline oficial."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from mingimage_model import DiffusionTransformer  # noqa: E402
from diffusion.generator import ConditionedTransformer  # noqa: E402
from diffusion.pipeline import ImageGenerationPipeline  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

print("Cargando DiT BF16...", flush=True)
dit = DiffusionTransformer.from_pretrained("Ming-Image_Raw", subfolder="transformer", torch_dtype=torch.bfloat16)
cond = ConditionedTransformer(dit, vision_dim=1152, use_identity_mlp=True, text_encoder_norm=False)
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True
pipe = ImageGenerationPipeline(vae=vae, transformer=cond, text_encoder=None, tokenizer=None, scheduler=sched).to("cuda")

emb = torch.load("cached_data_mingimage/_custom_embed.pt", weights_only=True).to("cuda", torch.bfloat16)
emb2 = torch.load("cached_data_mingimage/_custom_embed2.pt", weights_only=True).to("cuda", torch.bfloat16)
neg = torch.load("cached_data_mingimage/_neg_embed.pt", weights_only=True)
print("emb norm/token:", float(emb.float().norm(dim=-1).mean()), "| neg absmax:", float(neg.abs().max()), flush=True)

tmp = os.environ.get("TEMP", ".")
c1 = emb.squeeze(0)
c2 = emb2.squeeze(0)

variants = {
    "both": dict(prompt_embeds=[c1], prompt_embeds_2=[c2]),
    "cap_only": dict(prompt_embeds=[c1], prompt_embeds_2=None),
    "vlm_only": dict(prompt_embeds=None, prompt_embeds_2=[c2]),
    "norm_cap": dict(prompt_embeds=[F.normalize(c1.float(), dim=-1).to(torch.bfloat16) * 1000], prompt_embeds_2=[c2]),
    "cap_div1000": dict(prompt_embeds=[c1 / 1000], prompt_embeds_2=[c2]),
    "vlm_offset": dict(prompt_embeds=[c1], prompt_embeds_2=[c2 * 0.0]),
}

for name, kw in variants.items():
    with torch.no_grad():
        out = pipe(
            negative_prompt_embeds=[torch.zeros_like(c1)] if kw["prompt_embeds"] is not None else None,
            negative_prompt_embeds_2=[torch.zeros_like(c2)] if kw.get("prompt_embeds_2") is not None else None,
            guidance_scale=1.0,
            generator=torch.manual_seed(42),
            num_inference_steps=12,
            height=512,
            width=512,
            max_sequence_length=512,
            device="cuda",
            sample_mode="sample",
            num_frames_per_prompt=1,
            **kw,
        )
    p = os.path.join(tmp, f"var_{name}.png")
    out.images[0].save(p)
    print("guardado", name, flush=True)
