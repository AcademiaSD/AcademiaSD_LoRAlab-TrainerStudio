# -*- coding: utf-8 -*-
"""Prueba temporal: codificacion estilo Space (sin plantilla de chat)."""
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
prompt = open("Dataset_tosti_edit/image_01_after.txt", encoding="utf-8").read()

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
tok = processor.tokenizer
cfg = model.config.llm_config
n = 16 * 16
suffix = "<image>" + "<imagePatch>" * n + "</image>"
input_ids = tok(prompt + suffix, return_tensors="pt")["input_ids"].to(device)
expected = [cfg.image_start_token] + [cfg.image_patch_token] * n + [cfg.image_end_token]
print("sufijo OK:", input_ids[0, -(n + 2):].tolist() == expected, flush=True)

embeds = model.model.get_input_embeddings()(input_ids).clone()
qt = model.query_tokens_dict["16x16"].to(device=device, dtype=embeds.dtype)
embeds[0, -(n + 1):-1] = qt
direct_end = input_ids.shape[1] - (n + 2)

with torch.no_grad():
    out = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=None,
        past_key_values=None,
        inputs_embeds=embeds,
        image_grid_thw=None,
        use_cache=False,
        image_mask=input_ids == cfg.image_patch_token,
        audio_mask=None,
        output_hidden_states=True,
    )
    states = out.hidden_states
    query = states[-1][:, -(n + 1):-1].contiguous()
    direct = torch.cat([states[l][:, :direct_end] for l in model.selected_hidden_states_layers], dim=-1)
    mask = torch.zeros((1, 1, n, n), device=device, dtype=torch.bfloat16)
    query = model.connector(inputs_embeds=model.proj_in(query), attention_mask=mask,
                            output_hidden_states=True).hidden_states[-1]
    query = model.proj_out(query)
    direct = model.proj_directvlm(direct)
print("query", tuple(query.shape), "direct", tuple(direct.shape), flush=True)

emb = query[0].detach().cpu()
emb2 = direct[0].detach().cpu()
torch.save(emb, os.path.join(tmp, "space_emb.pt"))
torch.save(emb2, os.path.join(tmp, "space_emb2.pt"))
del model
torch.cuda.empty_cache()

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

for scale in (1.0, 1.01):
    with torch.no_grad():
        o = pipe(
            prompt_embeds=[emb.to("cuda", torch.bfloat16)],
            negative_prompt_embeds=[torch.zeros_like(emb).to("cuda", torch.bfloat16)],
            prompt_embeds_2=[emb2.to("cuda", torch.bfloat16)],
            negative_prompt_embeds_2=[torch.zeros_like(emb2).to("cuda", torch.bfloat16)],
            guidance_scale=scale, generator=torch.manual_seed(42), num_inference_steps=12,
            height=1024, width=1024, max_sequence_length=512, device="cuda",
            sample_mode="argmax", num_frames_per_prompt=1,
        )
    p = os.path.join(tmp, f"space_style_{scale}.png")
    o.images[0].save(p)
    print("guardado", p, flush=True)
