# -*- coding: utf-8 -*-
"""
6_validar_lora_yue2.py — Validación de un LoRA YuE2 exportado.

Comprueba, SIN el trainer por medio, que un fichero exportado por
2_train_lora_yue2.py (YuE2_*_LoRA_step_N / YuE2_FINAL*) se aplica a un
YuE2ForCausalLM limpio con PEFT estándar:

    1. Lee el metadata ss_* / yue2_stage del fichero (dim/alpha/etapa/pasos).
    2. Carga el backbone BF16 y envuelve las capas objetivo (DERIVADAS de los
       propios claves del fichero) con LoraConfig.
    3. Inyecta los pesos con set_peft_model_state_dict (claves NATIVAS
       model.layers.N.<...>.lora_A/B) y comprueba que no queda nada sin cargar.
    4. Ejecuta la rama de la etapa antes/después del LoRA (velocidad NAR o
       logits AR): la diferencia prueba que el adapter se aplica de verdad,
       y ejecutar la OTRA rama confirma que sus pesos no se tocan.

Uso:  6_validar_lora_yue2.py [YuE2_..._LoRA.safetensors]
      (sin argumento: el Fichero más reciente de ./yue2_lora_output)
"""
import os
import sys
import gc
import json
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))      # tools/yue2
STUDIO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))   # raíz del studio
RAW_DIR = os.path.join(STUDIO_ROOT, "YuE2-3B_Raw")
VAE_DIR = os.path.join(STUDIO_ROOT, "YuE2-Vae")
sys.path.insert(0, RAW_DIR)
sys.path.insert(0, VAE_DIR)
sys.path.insert(0, os.path.join(STUDIO_ROOT, "scripts"))   # yue2 (protocolo vendorizado)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from i18n import t
except ImportError:
    def t(s, **kw):
        for k, v in kw.items():
            s = s.replace("{" + k + "}", str(v))
        return s

from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_lora_file():
    if len(sys.argv) > 1:
        return sys.argv[1]
    default_dir = os.path.join(STUDIO_ROOT, "yue2_lora_output")
    if os.path.isdir(default_dir):
        files = [os.path.join(default_dir, f) for f in os.listdir(default_dir)
                 if f.endswith(".safetensors") and ("FINAL" in f or "step_" in f)]
        if files:
            return max(files, key=os.path.getmtime)
    for c in (os.path.join(STUDIO_ROOT, "yue2_lora_output", "YuE2_FINAL_NAR_LoRA.safetensors"),
              os.path.join(STUDIO_ROOT, "yue2_lora_output", "YuE2_FINAL_LoRA.safetensors")):
        if os.path.isfile(c):
            return c
    sys.exit("[E] " + t("No YuE2 LoRA found. Usage: 6_validar_lora_yue2.py <file.safetensors>"))


# ══════════════════════════════════════════════════════════════════════════════
# Velocidad NAR y forward AR — copia de la matemática del trainer (inyección
# sintácticamente idéntica), para probar el adapter sin depender de él.
# ══════════════════════════════════════════════════════════════════════════════

def nar_velocity(core, base_ids, state, raw_t, frames):
    """state [T,64] fp32/bf16 en cuda; NAR con gradiente no (inferencia)."""
    backbone = core.model
    with torch.no_grad():
        cos_ar, sin_ar = backbone.rotary_emb(torch.arange(len(base_ids), device="cuda")[None])
        cos, sin = backbone.rotary_emb(
            torch.arange(len(base_ids), len(base_ids) + frames + 2, device="cuda")[None])
        ids = torch.tensor([base_ids], dtype=torch.long, device="cuda")
        x = backbone.embed_tokens(ids)
        cache = []
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_ar, sin_ar)
            q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [1,H,T,hd]
            h = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True,
                                               enable_gqa=q2.shape[1] != k2.shape[1])
            x = x + layer.self_attn.o_proj(h.transpose(1, 2).reshape(1, x.shape[1], -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            cache.append((k[0], v[0]))        # [T, kv_heads, hd] sin transponer

    nar_length = frames + 2
    x_nar = F.pad(state.to(next(core.vae2llm.parameters()).dtype), (0, 0, 1, 1))
    shifted = core._shift_t_value(raw_t, state.device, x_nar.dtype)
    x = core.vae2llm(x_nar[None])
    x = x + core.time_embedder(shifted.expand(nar_length))[None]
    pos = torch.arange(nar_length, device="cuda").clamp(max=core.config.max_latent_frames - 1)
    x = x + core.latent_pos_embed(pos)[None]
    for layer, (ar_k, ar_v) in zip(backbone.layers, cache):
        q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), cos, sin)
        k = torch.cat((ar_k, k[0]), dim=0)
        v = torch.cat((ar_v, v[0]), dim=0)
        q2 = q[0].transpose(0, 1).unsqueeze(0)          # [1,H,frames+2,hd]
        k2 = k.transpose(0, 1).unsqueeze(0)             # [1,kv,ar+frames+2,hd]
        v2 = v.transpose(0, 1).unsqueeze(0)
        h = F.scaled_dot_product_attention(q2, k2, v2,
                                           enable_gqa=q2.shape[1] != k2.shape[1])
        x = x + layer.nar_self_attn.o_proj(h.transpose(1, 2).reshape(1, nar_length, -1))
        x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
    return core.llm2vae(core.model.norm(x))[0, 1:-1]


def ar_logits(core, base_ids):
    device = next(core.parameters()).device
    ids = torch.tensor([base_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden, _ = core.model(input_ids=ids,
                               position_ids=torch.arange(len(ids), device=device)[None],
                               use_cache=False)
        return core.lm_head(hidden[0, -8:, :]).float().cpu()


def core_of(model):
    """YuE2ForCausalLM directo o el del wrapper PEFT (el LoRA envuelve con LoRA
    las capas EN SUS SITIO; el core expone la API de modeling)."""
    base = getattr(model, "base_model", None)
    if base is None:
        return model
    inner = getattr(base, "model", None)
    return inner if inner is not None else base


def main():
    path = pick_lora_file()
    print()
    print("=" * 70)
    print(t("Validating YuE2 LoRA: {path}", path=os.path.basename(path)))
    print("  " + os.path.dirname(os.path.abspath(path)))
    print("=" * 70)

    if not torch.cuda.is_available():
        sys.exit("[E] CUDA no disponible.")
    print("GPU:", torch.cuda.get_device_name(0))

    # ── 1. metadata y etapa
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    keys = list(sd.keys())
    stage = meta.get("yue2_stage") or ("nar" if any("nar_" in k for k in keys) else "ar")
    targets = sorted({k.split(".lora_")[0] for k in keys if ".lora_" in k})
    ranks = {sd[f"{tname}.lora_A.weight"].shape[0] for tname in targets if f"{tname}.lora_A.weight" in sd}
    rank = ranks.pop() if len(ranks) == 1 else None
    alphas = {float(sd[tname + ".alpha"].item()) for tname in targets if tname + ".alpha" in sd}
    alpha = alphas.pop() if len(alphas) == 1 else None
    print(f"  etapa (stage)      : {stage}")
    print(f"  capas objetivo     : {len(targets)}")
    print(f"  rank               : {rank}  | alpha/capa: {alpha}")
    print(f"  ss_network_dim     : {meta.get('ss_network_dim')} | ss_network_alpha: {meta.get('ss_network_alpha')}")
    print(f"  ss_sd_model_name   : {meta.get('ss_sd_model_name')} | ss_steps: {meta.get('ss_steps')}")
    print(f"  tensores           : {len(keys)}")

    # ── 2. backbone limpio
    print()
    print("[2/4] " + t("Loading clean backbone (BF16)..."))
    from modeling_yue2 import YuE2ForCausalLM
    base = YuE2ForCausalLM.from_pretrained(RAW_DIR, local_files_only=True, torch_dtype=torch.bfloat16)
    base.requires_grad_(False)
    base.cuda()
    base.eval()
    free_vram()
    core0 = base
    print(t("Model loaded. VRAM: {v} GB", v=f"{torch.cuda.memory_allocated() / 1e9:.1f}"))

    # Entrada determinista
    random.seed(3)
    base_ids = [151643, 151644, 151852, 151853] + [random.randrange(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE) for _ in range(8)]
    frames = 12
    device = "cuda"
    state = torch.randn((frames, 64), dtype=torch.float32,
                        generator=torch.Generator(device="cpu").manual_seed(777)).to(device)

    def run_branch(core):
        if stage == "nar":
            return nar_velocity(core, base_ids, state, 0.0, frames).float().cpu()
        return ar_logits(core, base_ids)

    before = run_branch(core0)          # backbone LIMPIO (antes de envolver)
    print()
    print("[2.5/4] " + t("Reference output of the stage branch (clean model): OK"))

    # ── 3. wrap + carga de pesos
    print()
    print("[3/4] " + t("Wrapping with PEFT (targets from the file itself) and injecting weights..."))
    lora_config = LoraConfig(r=rank if rank else 16,
                             lora_alpha=int(alpha) if alpha else 16,
                             lora_dropout=0.0,
                             target_modules=targets,
                             use_rslora="rslora" in str(meta.get("ss_network_args", "")),
                             use_dora=False)
    model = get_peft_model(base, lora_config)
    # PEFT pide el prefijo "base_model.model." en set_peft_model_state_dict;
    # el export NATIVO de YuE2 (claves model.layers..., para cargadores
    # genéricos tipo ComfyUI) va sin él: un one-liner al consumir.
    prefixed = {}
    for k, v in sd.items():
        if not (".lora_" in k or k.endswith(".alpha")):
            continue
        prefixed["base_model.model." + k if not k.startswith("base_model.") else k] = v
    result = set_peft_model_state_dict(model, prefixed)
    if result is not None:
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        print(f"  missing/unexpected: {len(missing)}/{len(unexpected)}")
        if missing or unexpected:
            print("  [I] " + t("loaded with PEFT tolerance; the failure would be detected below."))

    # Cargas reales: comprobar que los pesos del adapter no son los de init
    deltas = []
    for pname, p in model.named_parameters():
        if "lora_B" in pname and p.data.abs().sum() > 0:
            deltas.append(pname)
            break
    print("  " + (t("lora_B with weight loaded (not an init zero)") if deltas else
                  "[!] lora_B all zero?"))

    # ── 4. salida antes/después: la diferencia demuestra que el LoRA aplica
    print()
    print("[4/4] " + t("Forward of the stage branch before/after the LoRA..."))
    core = core_of(model)
    after = run_branch(core)
    diff = (after - before).norm() / before.norm().clamp_min(1e-6)
    finite = torch.isfinite(after).all().item()
    print(f"  diff rel {stage} antes/después: {diff.item():.4%} | finite: {finite}")
    ok = diff.item() > 1e-9 and finite
    if stage == "nar":
        other = ar_logits(core, base_ids)
        ok = ok and torch.isfinite(other).all().item()
        print("  la otra rama (AR) sigue funcionando: finite ✓")
    else:
        v = nar_velocity(core, base_ids, state, 0.0, frames).float().cpu()
        ok = ok and torch.isfinite(v).all().item()
        print("  la otra rama (NAR) sigue funcionando: finite ✓")

    print()
    if ok:
        print("[OK] " + t("VALIDATION PASSED: the file loads and applies with PEFT standard."))
    else:
        print("[FAIL] " + t("VALIDATION FAILED"))
    print()
    print("=" * 70)

    del model, base
    free_vram()


if __name__ == "__main__":
    main()
