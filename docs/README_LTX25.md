# LTX-2.5 trainer — technical notes

> **Technical notes** for the LTX-2.5 trainer of **AcademiaSD LoRAlab Trainer Studio**. Installation, updating and everyday use are in the [main README](../README.md).

> **Status: Phase 1 (image LoRAs).** Video, video + audio and IC-LoRA edit modes are planned; see [PLAN_LTX25.md](PLAN_LTX25.md).

## Which model

LTX-2.5 is LTX-2.3 with a retrained transformer and a new text encoder. Measured by comparing the official checkpoints tensor by tensor:

| Part | 2.3 → 2.5 |
| :--- | :--- |
| Transformer (22B, 48 blocks) | Same shapes and layers. The video feed-forward loses its bias (`ff_bias=false`) and a keyframe embedding is added. Weights moved 11–31 % (cosine 0.96–0.99): a continued training, not a new model. |
| Text connectors | Retrained from scratch (cosine 0.00 against 2.3). |
| Text encoder | **Gemma 4 12B** fine-tuned for LTX (Gemma 3 in 2.3). |
| Video VAE, audio VAE, vocoder | **Bit-identical** to 2.3. |
| LoRA key names | Identical to 2.3 (`diffusion_model.transformer_blocks.N.attn1.to_q.lora_A.weight`...). |

That is why the LTX-2.3 trainer does **not** train LTX-2.5 (its LoRAs usually load in 2.5, but they are trained on the 2.3 weights and text encoder).

The trainer uses `transformer_full/` of `Lightricks/LTX-2.5-Diffusers` (the full / SFT model). The repository's `scheduler/` is configured for the distilled model; the converter enables `use_dynamic_shifting` and `shift_terminal=0.1`, as the model card says for the full model.

## The model in NF4

`Models/LTX25-NF4/`, same layout as `LTX23-NF4/`:

- `weights/`: one file per Linear. The 1,632 Linear of the 48 blocks in NF4 (double quantization): **9.6 GB**, **0.19 GB per block**. The 28 Linear outside the blocks (input/output projections, AdaLN, timestep embeddings) stay in **BF16** (0.85 GB): quantization error there disturbs the final modulation (lesson from MiniMax-H3).
- `text_encoder_NF4/`: Gemma 4 in NF4 (7.2 GB). After the connectors, its embeddings match BF16 at cosine ≥ 0.998 (video) / ≥ 0.997 (audio).
- `connectors/` (6.3 GB, BF16), `vae/`, `audio_vae/`, `vocoder/`, `scheduler/`, `tokenizer/`: copied.

`tools/ltx25/` builds it from the official repo (gated: accept the terms with the account of your HF token): `4_descarga_LTX25.py` downloads only what is needed (~66 GB), `5_conversor_LTX25_NF4.py` converts streaming one Linear at a time (RAM ≈ one tensor, VRAM ≈ one layer; 79 s for the transformer on an RTX 5080), `--validate-te` compares Gemma 4 NF4 against BF16.

## 8 GB of VRAM

Everything is measured with the **VRAM Profile** set to 8 GB on a 16 GB card. The profile is a hard cap on PyTorch's allocator at **card − 1 GB** (7 GiB), the memory PyTorch does not see on a real card (CUDA context, cuBLAS, Windows desktop), so a run that fits the profile fits a real 8 GB card.

| Stage | Peak | How |
| :--- | ---: | :--- |
| Pre-cache | **6.0 GB** (9.5 before) | Two passes: all captions with Gemma 4, which is then freed, then all latents with the VAE. Gemma 4 keeps only its text layers in VRAM (5.6 GB): the 2 GB embedding table (262,144 × 3,840, not compressible by NF4) stays in RAM and only the looked-up rows go to the GPU; `lm_head` (a tied copy of it) is replaced by an identity; vision and audio towers stay on the CPU. Outputs are bit-identical to the fully loaded model. |
| Training, images 768² | **6.65 GB** | Block swap (below): 17–18 of 48 blocks resident. |
| Preview | **6.84 GiB** | The resident blocks go to RAM while the VAE decodes (as Wan 2.2 does with its expert) and come back after. |

### Block swap (`scripts/ltx25_model.py`)

The MiniMax-H3 approach, simplified for a model whose blocks are 40 % smaller:

- The model is built empty and filled from the NF4 files; the **packed NF4 data of every block stays in RAM** (pinned) from the start, its `quant_state`, biases, norms and tables go to the GPU. Nothing is ever loaded to the GPU only to be moved out.
- `BlockSwap` uploads the first *N* blocks for good; the rest are uploaded just in time and released afterwards. Releasing never copies back (frozen weights: the RAM copy is always valid; H3's `nf4_cpu_home`).
- **Prefetch**: while block *i* runs, block *i+1* (forward) or *i−1* (recompute in backward) is uploaded on a side CUDA stream.
- **Reentrant** gradient checkpointing (bitsandbytes keeps its weight in `ctx`; the non-reentrant one would pin every NF4 weight in VRAM). With it, every parked block is uploaded twice per step.
- Checked: loss and every LoRA gradient are **identical** (difference 0.0) with all blocks parked or all resident.
- **Resident Blocks** `-1` lets the plan decide: what is already in VRAM is measured, the step's activations are estimated from the cached token count.

Measured (RTX 5080, 768² images, rank 32, batch 1):

| Profile | Resident blocks | Peak | s/it |
| :--- | ---: | ---: | ---: |
| 8 GB | 18 / 48 | 6.65 GB | ~6.8 |
| Auto (16 GB) | 48 / 48 | 12.2 GB | ~6.0 |

The prefetch hides most of the swap: 30 parked blocks cost ~0.8 s/it.

## Training details

- **`sigma` in the timestep scale.** LTX-2.3/2.5 feed `sigma` to `prompt_adaln` (the text modulation), a timestep embedding: `LTX2Pipeline` passes `sigma=timestep` (0–1000) and `use_cross_timestep=True`. The trainer does the same. Passing it in 0–1 (as the LTX-2.3 trainer did) makes the model believe the input is almost noise-free: the previews came out deformed even without a LoRA.
- **Timestep sampling** as Lightricks' `ltx-trainer` (`timestep_sampling: shifted_logit_normal`): `mu` linear in the token count (0.95 at 1,024 tokens, 2.05 at 4,096), `sigmoid(N(0,1) + mu)` stretched between its 0.5 % and 99.9 % percentiles, plus 10 % uniform samples. Median sigma: 0.64 for a 768² image, 0.81 for a 4,096-token clip. `uniform` keeps the LTX-2.3 behaviour.
- Flow matching: `x_t = (1 − σ)·x0 + σ·noise`, target `noise − x0`.
- LoRA on the attention layers of the video branch (as LTX-2.3); exported with the official key names.

## Not done yet

- DoRA with block swap (DoRA needs the dequantized weight of every layer; it refuses to start while blocks are parked).
- Video, video + audio and IC-LoRA modes.
- Validation of trained LoRAs in ComfyUI.
