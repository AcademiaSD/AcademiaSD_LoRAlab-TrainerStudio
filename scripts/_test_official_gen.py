# -*- coding: utf-8 -*-
"""Prueba temporal: sampler oficial (ImageGenerator) y sample_mode argmax."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"
emb = torch.load("cached_data_mingimage/image_01_after_embed.pt", weights_only=True)
emb2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True)

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusion.generator import ConditionedTransformer  # noqa: E402
from diffusion.pipeline import ImageGenerationPipeline  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

dit = load_nf4_transformer("Ming-Image_NF4")
cond = ConditionedTransformer(dit, vision_dim=2560, use_identity_mlp=True, text_encoder_norm=False)
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True
pipe = ImageGenerationPipeline(vae=vae, transformer=cond, text_encoder=None, tokenizer=None, scheduler=sched).to("cuda")

with torch.no_grad():
    out = pipe(
        prompt_embeds=[emb.squeeze(0).to("cuda", torch.bfloat16)],
        negative_prompt_embeds=[torch.zeros_like(emb.squeeze(0)).to("cuda", torch.bfloat16)],
        prompt_embeds_2=[emb2.squeeze(0).to("cuda", torch.bfloat16)],
        negative_prompt_embeds_2=[torch.zeros_like(emb2.squeeze(0)).to("cuda", torch.bfloat16)],
        guidance_scale=1.0, generator=torch.manual_seed(42), num_inference_steps=12,
        height=1024, width=1024, max_sequence_length=512, device="cuda",
        sample_mode="argmax", num_frames_per_prompt=1,
    )
p = os.path.join(tmp, "argmax_1024.png")
out.images[0].save(p)
print("guardado", p, flush=True)
del pipe, cond, dit, vae
torch.cuda.empty_cache()

from inference_profile import load_checkpoint_capabilities  # noqa: E402
from diffusion.generator import ImageGenerator  # noqa: E402

profile = load_checkpoint_capabilities("Ming-Image_Raw")
print("profile:", profile, flush=True)
gen = ImageGenerator(
    model_path="Ming-Image_Raw", scheduler_path="Ming-Image_Raw", vision_dim=2560,
    mlp_state_dict={}, torch_dtype=torch.bfloat16, device="cuda",
    use_identity_mlp=True, text_encoder_norm=False, inference_profile=profile,
)
with torch.no_grad():
    imgs = gen.sample(encoder_hidden_states=emb, directvlm_hidden_states=emb2,
                      steps=12, cfg=1.0, seed=42, height=1024, width=1024)
p = os.path.join(tmp, "official_gen_1024.png")
imgs[0].save(p)
print("guardado", p, flush=True)
