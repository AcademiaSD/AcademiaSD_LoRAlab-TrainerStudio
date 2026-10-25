# -*- coding: utf-8 -*-
"""Prueba temporal: conector con mascara full_attention (estilo Space)."""
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
    "Ming-Image_NF4", dtype=torch.bfloat16, attn_implementation="eager",
    load_image_gen=True, load_image_gen_diffusion=False, image_gen_device="cuda",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map={"": "cuda:0"},
)
device = next(model.parameters()).device
tok = processor.tokenizer
cfg = model.config.llm_config
n = 256
suffix = "<image>" + "<imagePatch>" * n + "</image>"
input_ids = tok(prompt + suffix, return_tensors="pt")["input_ids"].to(device)
embeds = model.model.get_input_embeddings()(input_ids).clone()
embeds[0, -(n + 1):-1] = model.query_tokens_dict["16x16"].to(device=device, dtype=embeds.dtype)
direct_end = input_ids.shape[1] - (n + 2)

with torch.no_grad():
    out = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                      position_ids=None, past_key_values=None, inputs_embeds=embeds,
                      image_grid_thw=None, use_cache=False,
                      image_mask=input_ids == cfg.image_patch_token, audio_mask=None,
                      output_hidden_states=True)
    states = out.hidden_states
    query = states[-1][:, -(n + 1):-1].contiguous()
    direct = torch.cat([states[l][:, :direct_end] for l in model.selected_hidden_states_layers], dim=-1)
    q_in = model.proj_in(query)
    mask = torch.zeros((1, 1, n, n), device=device, dtype=torch.bfloat16)
    old = torch.load(os.path.join(tmp, "space_emb.pt"), weights_only=True)
    for name, am in (("dict", {"full_attention": mask}), ("tensor", mask)):
        try:
            q = model.connector(inputs_embeds=q_in, attention_mask=am, output_hidden_states=True).hidden_states[-1]
            q = model.proj_out(q)
            cos = torch.nn.functional.cosine_similarity(q[0].flatten(), old.flatten(), dim=0)
            print(f"connector {name}: OK cos_vs_old={float(cos):.6f}", flush=True)
            if name == "dict":
                torch.save(q[0].detach().cpu(), os.path.join(tmp, "space_emb_full.pt"))
        except Exception as exc:
            print(f"connector {name}: FAIL {type(exc).__name__}: {exc}", flush=True)
    direct = model.proj_directvlm(direct)
    torch.save(direct[0].detach().cpu(), os.path.join(tmp, "space_emb2_full.pt"))
    print("direct", tuple(direct.shape), flush=True)

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
emb = torch.load(os.path.join(tmp, "space_emb_full.pt"), weights_only=True)
emb2 = torch.load(os.path.join(tmp, "space_emb2_full.pt"), weights_only=True)
with torch.no_grad():
    o = pipe(
        prompt_embeds=[emb.to("cuda", torch.bfloat16)],
        negative_prompt_embeds=[torch.zeros_like(emb).to("cuda", torch.bfloat16)],
        prompt_embeds_2=[emb2.to("cuda", torch.bfloat16)],
        negative_prompt_embeds_2=[torch.zeros_like(emb2).to("cuda", torch.bfloat16)],
        guidance_scale=1.0, generator=torch.manual_seed(42), num_inference_steps=12,
        height=1024, width=1024, max_sequence_length=512, device="cuda",
        sample_mode="argmax", num_frames_per_prompt=1,
    )
o.images[0].save(os.path.join(tmp, "space_fullattn.png"))
print("guardado space_fullattn.png", flush=True)
