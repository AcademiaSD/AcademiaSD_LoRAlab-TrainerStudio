---
license: other
license_name: ltx-2.x-community-license-agreement
license_link: LICENSE
base_model: Lightricks/LTX-2.5
base_model_relation: quantized
pipeline_tag: text-to-video
tags:
- ltx-2.5
- nf4
- bitsandbytes
- lora-training
---

# LTX-2.5 — NF4 for LoRA training

NF4 (4-bit) version of [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (full / SFT transformer) and its Gemma 4 text encoder, used by [AcademiaSD LoRAlab Trainer Studio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio) to train LTX-2.5 LoRAs (images, video, video + audio and IC-LoRA edit) on consumer GPUs, down to 8 GB of VRAM. The trainer downloads it automatically; it is not meant for inference.

## Contents

| Folder / file | What it is | Size |
| :--- | :--- | ---: |
| `weights/`, `index.json`, `config.json`, `others.safetensors`, `metadata.json` | Transformer from `transformer_full/` of `Lightricks/LTX-2.5-Diffusers`, one file per Linear layer. The 1,632 Linear layers of the 48 blocks in NF4 (bitsandbytes, double quantization, blocksize 64); the 28 Linear layers outside the blocks in BF16; norms, modulation tables and other tensors unchanged | 9.8 GB |
| `text_encoder_NF4/` | Gemma 4 12B text encoder fine-tuned for LTX-2.5, quantized to NF4 with bitsandbytes | 7.2 GB |
| `connectors/`, `vae/`, `audio_vae/`, `vocoder/`, `tokenizer/`, `duration_head/` | Unchanged copies from `Lightricks/LTX-2.5-Diffusers` | 7.8 GB |
| `scheduler/` | From `Lightricks/LTX-2.5-Diffusers`, **modified** (see below) | — |
| `model_index.json` | From `Lightricks/LTX-2.5-Diffusers`, **modified** (see below) | — |

## Modifications

As required by Section 3.3 of the LTX-2.x Community License Agreement, these files were changed by AcademiaSD:

- **Transformer** (`weights/`, `index.json`, `others.safetensors`): quantized to NF4 and split into one file per layer.
- **Text encoder** (`text_encoder_NF4/`): quantized to NF4.
- **`scheduler/scheduler_config.json`**: `use_dynamic_shifting` set to `true` and `shift_terminal` to `0.1`, the settings the model card gives for the full transformer (the repository's scheduler is configured for the distilled one).
- **`model_index.json`**: the components not shipped here (`prompt_enhancer`, `processor`, `diffusion_decoder`, latent upsamplers) set to null.

The conversion scripts are in `tools/ltx25/` of the Trainer Studio repository.

## License

LTX-2.5 and these weights, which are a Derivative of LTX-2.x, are distributed under the **LTX-2.x Community License Agreement** (license date August 11, 2026), included in this repository as [`LICENSE`](LICENSE) and published by Lightricks at [github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x). By downloading or using these files you agree to its terms.

- The use restrictions of **Section 4 and Attachment A** apply to these weights and to any LoRA or other Derivative trained with them, and must be passed on to anyone you distribute them to.
- Entities with annual revenues of **USD 10,000,000 or more** need a commercial license from Lightricks for commercial use (Section 2).
- Transferring a Derivative (fine-tuned weights, LoRA adapters) to a third party does not grant them commercial rights (Section 3.5).

LTX-2.5 was created by [Lightricks](https://www.lightricks.com/). This repository is not affiliated with or endorsed by Lightricks.
