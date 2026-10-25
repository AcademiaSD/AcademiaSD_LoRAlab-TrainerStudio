# -*- coding: utf-8 -*-
"""Prueba temporal: condicion con image_gen_highres=1024 para 2 prompts + pipeline oficial."""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration  # noqa: E402
from processing_bailingmm2 import load_bailingmm2_processor  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402

tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"
prompts = {
    "tosti01": open("Dataset_tosti_edit/image_01_after.txt", encoding="utf-8").read(),
    "cabin": open(os.path.join(tmp, "official_prompt.json"), encoding="utf-8").read(),
}

processor = load_bailingmm2_processor("Ming-Image_NF4/mllm")
model = BailingMM2NativeForConditionalGeneration.from_pretrained(
    "Ming-Image_NF4",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    load_image_gen=True,
    load_image_gen_diffusion=False,
    image_gen_device="cuda",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map={"": "cuda:0"},
)
device = next(model.parameters()).device

embs = {}
for name, prompt in prompts.items():
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, _ = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt",
        image_gen_highres=1024, image_gen_aspect_ratio=None, image_gen_input_channels=4,
    )
    inputs = inputs.to(device)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    with torch.no_grad():
        emb, _n, emb2, _n2 = model.generate(
            **inputs, image_gen=True, image_gen_task="text-to-image",
            image_gen_only_extract_hidden_states=True,
        )
    embs[name] = (emb.cpu(), emb2.cpu())
    print("encode", name, tuple(emb.shape), tuple(emb2.shape), flush=True)

del model
torch.cuda.empty_cache()

from mingimage_model import load_nf4_transformer  # noqa: E402
from diffusion.generator import ConditionedTransformer  # noqa: E402
from diffusion.pipeline import ImageGenerationPipeline  # noqa: E402
from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

dit = load_nf4_transformer("Ming-Image_NF4")
cond = ConditionedTransformer(dit, vision_dim=1152, use_identity_mlp=True, text_encoder_norm=False)
vae = AutoencoderKLQwenImage.from_pretrained("Ming-Image_NF4", subfolder="vae", dtype=torch.bfloat16)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Ming-Image_NF4", subfolder="scheduler")
sched.config["use_dynamic_shifting"] = True
pipe = ImageGenerationPipeline(vae=vae, transformer=cond, text_encoder=None, tokenizer=None, scheduler=sched).to("cuda")

for name, (emb, emb2) in embs.items():
    with torch.no_grad():
        out = pipe(
            prompt_embeds=[emb.squeeze(0).to("cuda", torch.bfloat16)],
            negative_prompt_embeds=[torch.zeros_like(emb.squeeze(0)).to("cuda", torch.bfloat16)],
            prompt_embeds_2=[emb2.squeeze(0).to("cuda", torch.bfloat16)],
            negative_prompt_embeds_2=[torch.zeros_like(emb2.squeeze(0)).to("cuda", torch.bfloat16)],
            guidance_scale=1.0, generator=torch.manual_seed(42), num_inference_steps=12,
            height=1024, width=1024, max_sequence_length=512, device="cuda",
            sample_mode="sample", num_frames_per_prompt=1,
        )
    p = os.path.join(tmp, f"fix_{name}.png")
    out.images[0].save(p)
    print("guardado", p, flush=True)
