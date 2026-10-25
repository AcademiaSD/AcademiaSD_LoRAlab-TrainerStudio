# -*- coding: utf-8 -*-
"""Prueba temporal: conector con position_ids en CUDA; causal vs bidireccional."""
import os
import sys

import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "mingimage_vendor"))

from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

tmp = r"C:\Users\AcademiaSD\AppData\Local\Temp\opencode"
q_in = torch.load(os.path.join(tmp, "space_q_in.pt"), weights_only=True).to("cuda", torch.bfloat16)
mlp = load_file("Ming-Image_NF4/mlp/model.safetensors")

connector = AutoModelForCausalLM.from_pretrained("Ming-Image_NF4", subfolder="connector", dtype=torch.bfloat16).to("cuda").eval()
proj_out = nn.Linear(1536, 2560).to("cuda", torch.bfloat16)
proj_out.load_state_dict({"weight": mlp["proj_out.weight"], "bias": mlp["proj_out.bias"]})
n = 256
pos = torch.arange(n, device="cuda").unsqueeze(0)


def call(**kw):
    with torch.no_grad():
        q = connector(inputs_embeds=q_in, position_ids=pos, output_hidden_states=True, **kw).hidden_states[-1]
    return proj_out(q)


def stats(tag, q):
    print(tag, "nan", int(torch.isnan(q).sum()), "std", round(float(q.std()), 3),
          "max", round(float(q.abs().max()), 3), flush=True)


for layer in connector.model.layers:
    layer.self_attn.is_causal = True
q_causal = call()
stats("causal", q_causal)
for layer in connector.model.layers:
    layer.self_attn.is_causal = False
q_bi = call()
stats("bidir_nomask", q_bi)
cos = float(torch.nn.functional.cosine_similarity(q_causal[0].flatten(), q_bi[0].flatten(), dim=0))
print("cos causal vs bidir:", round(cos, 6), flush=True)
try:
    q_dict = call(attention_mask={"full_attention": torch.zeros(1, 1, n, n, device="cuda", dtype=torch.bfloat16)})
    stats("dict_full", q_dict)
    print("cos bidir vs dict:", round(float(torch.nn.functional.cosine_similarity(q_bi[0].flatten(), q_dict[0].flatten(), dim=0)), 6), flush=True)
except Exception as exc:
    print("dict FAIL", type(exc).__name__, str(exc)[:160], flush=True)
    q_dict = q_bi
torch.save(q_bi[0].detach().cpu().to(torch.bfloat16), os.path.join(tmp, "space_emb_bidir.pt"))
torch.save(q_dict[0].detach().cpu().to(torch.bfloat16), os.path.join(tmp, "space_emb_dict.pt"))
