# Wan 2.2 T2V-A14B — technical notes

Trainer: `scripts/0_caption_wan22.py`, `1_pre_cache_wan22.py`, `2_train_lora_wan22.py`, `server_wan22.py`, with the shared pieces in `wan22_model.py`. Interface: `GUI/trainer_ui_wan22.html`.

## Which Wan 2.2 and why

The [Wan 2.2 diffusers collection](https://huggingface.co/collections/Wan-AI/wan22-diffusers) has six models:

| Model | What it is | Here |
| :--- | :--- | :--- |
| **T2V-A14B** | Text-to-video, two 14B experts (MoE by denoising step), 480p / 720p, 81 frames at 16 fps | ✅ trained |
| I2V-A14B | The same MoE, conditioned on a first frame | not yet (same code plus the image conditioning) |
| TI2V-5B | A dense 5B with the high-compression Wan 2.2 VAE, 720p at 24 fps | not trained: very few people use its LoRAs |
| Animate-14B, Animate-2-14B (and Distilled) | Character animation driven by pose / video | a different kind of training |

T2V-A14B is the one the community uses most (ComfyUI loads its LoRAs in high / low noise pairs).

## The two experts

`transformer` (diffusers name) is the **high noise** expert and `transformer_2` the **low noise** one. `model_index.json` sets `boundary_ratio = 0.875`: steps with `t >= 875` go to the high noise expert.

- **Low noise** (σ < 0.875): detail, faces, textures, style. Enough for characters, objects and styles. Default.
- **High noise** (σ ≥ 0.875): layout and motion. For new motions, choreographies or camera moves.
- **Both**: one LoRA for each in the same run. Each Grad Accum group belongs to one expert, picked with the share of noise levels it gets under the training distribution (about 30 % high with shift 3). The waiting expert stays in RAM (~7.6 GB); swapping them on the GPU takes about 4 s (measured on an RTX 5080), so a Grad Accum of 2–4 makes it rarer.

Timesteps: the same shifted distribution as the other trainers (`train_shift` 3.0), cut to the expert's range by rejection sampling, so the shape inside the range is kept.

## The model in NF4

The NF4 model lives in `AcademiaSD/Wan22-T2V-A14B-NF4-for-LoRA-Training`: the same folder the trainer builds below, uploaded as it is. The trainer reads its file list once and downloads from it what it has: the base files and each expert's `*_nf4/` folder (its `index.json` last, since that file marks the expert as ready). Whatever that repo does not have (or when it cannot be reached), the trainer makes on your PC from the original:

1. Downloads `vae/`, `scheduler/`, `tokenizer/`, the configs and, for the pre-cache, the UMT5-XXL text encoder (~12 GB) from `Wan-AI/Wan2.2-T2V-A14B-Diffusers`.
2. For each expert it trains, downloads its 12 FP32 shards **one at a time** (~4.9 GB each), quantizes the Linear layers of the 40 blocks to NF4 on the GPU and deletes the shard. The disk never holds more than one FP32 shard; if it stops, it continues from the next shard.

Result in `Models/Wan22-T2V-A14B-NF4/transformer_nf4/` (high) and `transformer_2_nf4/` (low): about 7.6 GB per expert instead of 57 GB. Block Linear weights in NF4 (`QuantState` packed, blocksize 64, compressed statistics: the same as `Params4bit.cuda()`); the input/output, timestep and text layers in BF16; norms, rope and modulation tables in FP32 (diffusers' `_keep_in_fp32_modules`).

To fill the NF4 repo: convert both experts (a *Both* run stopped after its first step does it) and upload `Models/Wan22-T2V-A14B-NF4` without its `.cache/` folder (`hf upload <repo> Models/Wan22-T2V-A14B-NF4 . --exclude ".cache/*"`).

The expert that is not trained is only downloaded if you convert it (train it once, or Both): previews use it when it is there.

## Pre-cache

- **Text:** UMT5-XXL in BF16 on the GPU (~11 GB; with less than 14 GB of VRAM it is quantized to NF4 when loading). Same as `WanPipeline`: up to 512 tokens, cleaned with `prompt_clean`; only the real tokens are stored and the trainer pads with zeros to 512. The negative prompt for the previews is Wan's default one.
- **Latents:** the Wan 2.1 VAE (16 channels, 8× in space, 4× in time) normalized with its per-channel mean and std. An image is a 1-frame video.
- **Videos:** read with ffmpeg at **16 fps** from their start, scaled to cover the bucket and center-cropped. The frame count is the largest **4k+1** that fits the clip and Max Frames (1, 5, 9 … 33 … 81). Buckets are per latent shape (frames × height × width), so clips of different length never share a batch.

## Training

- Each expert is built empty from its config and filled from the NF4 parts: the BF16 model is never in memory.
- PEFT LoRA (or LoKr) on the block layers, FP32 LoRA and AdamW states; the forward runs in BF16 autocast, so PEFT does not copy every layer's input to FP32 for the LoRA.
- **Gradient checkpointing in two halves per block** (self-attention; text attention + FFN) instead of one. diffusers keeps Wan's residuals in FP32, so recomputing a whole block in the backward holds about 45 times the size of the activations; half a block is what lets video fit.
- **Activation Offload** (on by default): the checkpoint is reentrant, so what it keeps (the input of each half) goes through `save_for_backward`, and `torch.autograd.graph.save_on_cpu` sends it to pinned RAM instead of VRAM.
- The rope buffers are cast from float64 (diffusers' default on CUDA) to float32, as diffusers already does on MPS: it halves the rope temporaries of every block.
- `expandable_segments` in the CUDA allocator: images and clips of different sizes fragmented the VRAM.
- **No silent slowdown:** on Windows, when the VRAM is full the NVIDIA driver spills into shared system memory instead of failing, and a step takes 30 times longer without any message (a 33-frame clip at 480p on 16 GB: 544 s instead of ~43 s). The trainer caps PyTorch's allocator to the free VRAM, so a sample that does not fit fails at once; that bucket is left out of the run with a message saying which size it was.
- Flow matching: `x_t = (1 − σ)·x0 + σ·noise`, target `noise − x0`, model time `σ·1000`.

### Measured (RTX 5080 16 GB, low noise expert, rank 16, Activation Offload on, 480p)

| Sample | Peak VRAM | Time per step |
| :--- | ---: | ---: |
| Image 832×480 | 10.0 GB | ~3 s |
| Clip 17 frames | 12.0 GB | ~18.5 s |
| Clip 33 frames | 14.3 GB (15.6 GB reserved) | ~43 s — does not fit next to a desktop on a 16 GB card |
| Clip 33 frames, before the two-half checkpoint | 16.6 GB | 544 s (driver spilling into system RAM) |

The expert itself takes 8.1 GB on the GPU (7.2 GB of NF4 weights plus the LoRA and its optimizer). With *Both*, each expert swap adds about 4 s.

## Previews

UniPC with the repository's scheduler config, like `WanPipeline`: steps with `t >= 875` on the high noise expert (CFG High, default 4) and the rest on the low noise one (CFG Low, default 3). The trained expert carries its LoRA; the other one, if converted, runs without LoRA. Only one expert is on the GPU at a time. 1 frame gives a `.png`; more frames, an `.mp4` at 16 fps (an animated `.webp` without ffmpeg).

## Exported LoRAs

`Wan22_HIGH_*` and `Wan22_LOW_*`, with the original Wan key names (`diffusion_model.blocks.N.self_attn.q.lora_A.weight`, `.alpha` per layer). ComfyUI, Wan2GP and diffusers (`WanLoraLoaderMixin`) load them. Send to Models and Download add `_high_noise` / `_low_noise` to the name, read from the LoRA's metadata (`wan_expert`). In ComfyUI, each LoRA goes on its own model: the high noise one on the high noise model, the low noise one on the low noise model.
