---
license: cc-by-nc-4.0
license_name: bria-fibo
license_link: https://creativecommons.org/licenses/by-nc/4.0/deed.en
base_model: briaai/Fibo-2.0-dev
base_model_relation: quantized
pipeline_tag: text-to-image
library_name: diffusers
language:
- en
tags:
- fibo-2
- fibo
- bria
- nf4
- bitsandbytes
- lora-training
- text-to-image
- image-editing
- academiasd
---

# Fibo 2.0 NF4 for LoRA Training

> [!IMPORTANT]
> **Not an official BRIA release, and not made for inference.**
>
> This repository holds an **NF4 conversion of [Fibo 2.0 Dev](https://huggingface.co/briaai/Fibo-2.0-dev)**, made so that the Fibo 2.0 trainer of **[AcademiaSD LoRAlab Trainer Studio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)** can download it and train LoRAs on consumer NVIDIA GPUs (from 8 GB of VRAM).
>
> The weights are the property of **BRIA AI** and are redistributed here **only to make LoRA training possible on a single consumer GPU**. The transformer is stored in the trainer's own pre-quantized cache format, which **diffusers and ComfyUI do not load**: this repository is for **training, not for generating images**.
>
> **Academic and personal use only.** Fibo 2.0 is published under **CC BY-NC 4.0**: **non-commercial use only**. You may not use the model, the LoRAs trained with it, or any derivative for commercial purposes, for a product or service, or for anything that brings you money. Commercial use of Fibo 2.0 requires a **separate agreement with BRIA** (see the [model page](https://huggingface.co/briaai/Fibo-2.0-dev)). Read the terms of the original model before you use it: they are the ones that apply here too.

You do not need to download this repository by hand: the trainer downloads only the parts it uses on first use.

## Contents

| Folder | Content | Size |
| :--- | :--- | ---: |
| `transformer/` | Fibo 2.0 Dev transformer (8.1 B): the attention, MLP and cross attention of the 44 blocks and of the 2 refiner blocks in NF4 (bitsandbytes, pre-quantized), 342 layers; the perceptor that turns the caption into gist tokens, the adaLN modulation, the embedders and the output layer in BF16, 73 layers | 5.2 GB |
| `text_encoder_NF4/` | Qwen3-VL-4B text encoder (the model's own) in NF4 | 2.9 GB |
| `text_encoder_INT8/` | The same text encoder in INT8 (bitsandbytes LLM.int8) | 4.8 GB |
| `text_encoder_BF16/` | The same text encoder in BF16, unchanged | 8.9 GB |
| `vae/` | AutoencoderKLFlux2 (the FLUX.2 VAE, 32 latent channels) | 0.3 GB |
| `processor/`, `scheduler/`, `fonts/` | Unchanged from the original model; the fonts are the ones Fibo 2.0 was trained with, to draw the text of the captions | 0.01 GB |
| **Total** | with the three text encoders | **22.1 GB** (original: 24.3 GB) |

With only the NF4 text encoder (the default of the trainer) the download is **8.4 GB**. The pre-cache pulls just the variant it is going to use and skips the other two.

## Format

- `transformer/` is **not** a standard diffusers checkpoint. It is a pre-quantized cache: one `.safetensors` file per layer in `weights/` (the NF4 weight plus its packed bitsandbytes `QuantState`, or BF16 for the excluded layers), `others.safetensors` with the norms, the modulations and the perceptor, and `index.json`. The trainer builds an empty `BriaFibo2Transformer2DModel` from `config.json` and fills it from this cache, so the BF16 transformer is never downloaded or loaded. `metadata.json` lists the layers kept in BF16.
- `text_encoder_NF4/`, `text_encoder_INT8/` and `text_encoder_BF16/` load with `transformers`: `Qwen3VLForConditionalGeneration.from_pretrained(".../text_encoder_NF4")`.
- `vae/`, `processor/` and `scheduler/` are unchanged from the original model.
- The converter that produced all of this is `tools/fibo2/5_conversor_Fibo2_NF4.py` in the Trainer Studio repository.

**NF4 is meant for training.** The quantization of the transformer costs a little accuracy in the weights, and a pre-quantized cache cannot be used to generate images at all. For inference, use BRIA's original model.

## LoRAs trained with it

The trainer exports LoRAs in the format ComfyUI's loader takes (`diffusion_model.<layer>.lora_A / lora_B` plus `alpha`).

ComfyUI runs its own repack of Fibo 2.0, where the attention is a single `qkv` matrix, the MLP a single `w13` and the cross attention a `to_kv`, so the layers that fuse are rewritten as one LoRA of several times the rank: the change that ComfyUI applies is exactly the same as the one of the diffusers LoRA.

Measured on an RTX 5080 at 512×512, rank 8: ~8.7 GB of VRAM and ~2 s per step.

## License and ownership

**The weights belong to BRIA AI.** This repository is a redistribution of [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev) in a different storage precision, for **LoRA training only**.

- The model is published by BRIA AI under **CC BY-NC 4.0** ([licence](https://creativecommons.org/licenses/by-nc/4.0/deed.en)): **non-commercial use only**.
- **Academic research and personal use only.** No commercial use, no products, no services, no paid work, and nothing derived from it that is used commercially.
- The LoRAs you train with it inherit those terms: they are derivatives of BRIA's weights.
- For commercial use, contact BRIA: <https://platform.bria.ai/licensing/pricing?model=fibo-2.0>.
- The original model's page, its gated access form and its terms are the ones that apply: <https://huggingface.co/briaai/Fibo-2.0-dev>.

The code of the trainer is MIT licensed; the weights are not.

## Links

- Model: [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev) (BRIA AI)
- Trainer: [AcademiaSD LoRAlab Trainer Studio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)
- ▶ YouTube: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 💬 Discord: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ Ko-Fi: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)
