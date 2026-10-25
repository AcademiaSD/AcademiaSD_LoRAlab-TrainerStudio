# -*- coding: utf-8 -*-
"""Prueba temporal: conector aislado con varias mascaras (reusa q_in guardado)."""
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
direct_raw = torch.load(os.path.join(tmp, "space_direct_raw.pt"), weights_only=True).to("cuda", torch.bfloat16)
mlp = load_file("Ming-Image_NF4/mlp/model.safetensors")

connector = AutoModelForCausalLM.from_pretrained("Ming-Image_NF4", subfolder="connector", dtype=torch.bfloat16).to("cuda")
for layer in connector.model.layers:
    layer.self_attn.is_causal = False
connector.eval()
proj_out = nn.Linear(1536, 2560).to("cuda", torch.bfloat16)
proj_out.load_state_dict({"weight": mlp["proj_out.weight"], "bias": mlp["proj_out.bias"]})
proj_direct = nn.Sequential(nn.RMSNorm(6144, eps=1e-5), nn.Linear(6144, 3840)).to("cuda", torch.bfloat16)
proj_direct.load_state_dict({"0.weight": mlp["proj_directvlm.0.weight"],
                             "1.weight": mlp["proj_directvlm.1.weight"],
                             "1.bias": mlp["proj_directvlm.1.bias"]})
n = 256
old = torch.load(os.path.join(tmp, "space_emb.pt"), weights_only=True).flatten()
with torch.no_grad():
    for name, am in (("ones4d_bf16", torch.ones(1, 1, n, n, device="cuda", dtype=torch.bfloat16)),
                     ("ones2d", torch.ones(1, n, device="cuda", dtype=torch.long)),
                     ("zeros4d_bf16", torch.zeros(1, 1, n, n, device="cuda", dtype=torch.bfloat16))):
        try:
            q = connector(inputs_embeds=q_in, attention_mask=am, output_hidden_states=True).hidden_states[-1]
            q = proj_out(q)
            cos = float(torch.nn.functional.cosine_similarity(q[0].flatten(), old, dim=0))
            print(name, "nan", int(torch.isnan(q).sum()), "std", round(float(q.std()), 3),
                  "max", round(float(q.abs().max()), 3), "cos_vs_zeros", round(cos, 6), flush=True)
            torch.save(q[0].detach().cpu().to(torch.bfloat16), os.path.join(tmp, f"space_emb_{name}.pt"))
        except Exception as exc:
            print(name, "FAIL", type(exc).__name__, str(exc)[:160], flush=True)
    d = proj_direct(direct_raw)
    print("direct", tuple(d.shape), "std", round(float(d.std()), 3), "max", round(float(d.abs().max()), 3), flush=True)
    torch.save(d.cpu().to(torch.bfloat16), os.path.join(tmp, "space_direct2.pt"))
    print("vendor emb2 check:", flush=True)
    v2 = torch.load("cached_data_mingimage/image_01_after_embed2.pt", weights_only=True)
    print("vendor emb2 std", round(float(v2.std()), 3), "max", round(float(v2.abs().max()), 3), flush=True)
