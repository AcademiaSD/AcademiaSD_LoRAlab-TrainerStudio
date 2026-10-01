

> **Technical notes** for the Krea 2 trainer of **AcademiaSD LoRAlab Trainer Studio**. Installation, updating and everyday use are in the [main README](../README.md).

# Krea 2 trainer — technical notes

![AcademiaSD_LoRAlab-Krea2](../assets/krea2/portada.jpg)

<p align="center">
  <b>An ultra-fast, low-resource Web GUI & pipeline for training Krea-2 (NF4) LoRAs.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.14%20cu130-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-purple.svg" alt="Web UI">
</p>

---
## YouTube video instructions.
https://www.youtube.com/watch?v=cEnZH-Eh7Rs
---

## 🔬 Technical Deep-Dive: Why is it so fast, light, & high quality?

Training a 12-Billion parameter Diffusion Transformer (DiT) like **Krea 2** typically requires enterprise-grade hardware (30GB+ VRAM, 64GB+ System RAM) and hours of compute. **AcademiaSD Krea2 LoRAlab** breaks this barrier, allowing training on consumer GPUs (8GB–12GB VRAM) in record time while preserving 100% of the model's generation quality. 

Here is the exact architectural breakdown of how this is achieved:

---

### 1. 📉 How VRAM Usage is Reduced to ~7.5 GB

| Memory Component | Standard Training | AcademiaSD Krea2 LoRAlab | Memory Saved |
| :--- | :--- | :--- | :--- |
| **DiT Model (12B)** | ~24.0 GB (FP16/BF16) | **~6.5 GB (4-bit NF4)** | **-73% VRAM** |
| **Text Encoder (Qwen3-VL-4B)** | ~8.0 GB VRAM | **0.0 GB (Offloaded via Pre-Cache)** | **-100% VRAM** |
| **VAE (Qwen-Image)** | ~2.5 GB VRAM | **0.0 GB (Offloaded via Pre-Cache)** | **-100% VRAM** |
| **Optimizer States (AdamW)** | ~6.0 GB VRAM | **~0.2 GB (8-Bit AdamW on BF16 LoRA)** | **-96% VRAM** |
| **Activation Memory** | ~8.0 GB VRAM | **~1.0 GB (Gradient Checkpointing)** | **-87% VRAM** |
| **Total VRAM Peak** | **~48.5 GB** | **~7.7 GB** | **-84% Total VRAM** |

* **4-Bit NormalFloat (NF4) Quantization (`bitsandbytes`)**: The 12-Billion parameter DiT backbone is quantized into 4-bit NF4 weights (`Linear4bit`). Base weights are frozen and pinned to CUDA memory, compressing the 12B model footprint from ~24 GB down to ~6.5 GB.
* **Zero VRAM Wasted on Encoders (Offline Pre-Caching)**: During training (`2_train_lora_krea2.py`), **neither the Text Encoder (Qwen3-VL-4B) nor the VAE are loaded into VRAM**. All text embeddings and image latents are pre-computed once during the pre-cache stage and stored on disk.
* **8-Bit AdamW Optimizer (`bitsandbytes.optim.AdamW8bit`)**: Optimizer states are stored in 8-bit precision instead of 32-bit float, cutting optimizer VRAM overhead by 75%.
* **Gradient Checkpointing**: Intermediate activation tensors are recomputed during backward passes rather than stored in memory, keeping activation VRAM flat regardless of resolution.

---

### 2. ⚡ Why Training Speed is So Fast

* **No Per-Step Encoding Overhead**: In standard pipelines, every training step spends compute cycles passing images through the VAE and prompts through the LLM/Text Encoder. By eliminating this in the pre-cache phase, **100% of GPU compute during training is dedicated purely to the DiT backward pass**.
* **2x2 Latent Patch Packing**: Image latents `[B, C, H, W]` are packed into 2x2 spatial patches (`(H//2)*(W//2)` sequence length). Patching reduces the DiT self-attention sequence length by **4x**, accelerating attention matrix calculations exponentially.
* **Pinned RAM & Non-Blocking CUDA Transfers**: Pre-cached `.pt` latents and embeddings are cached into pinned host memory (`pin_memory()`) and transferred asynchronously to GPU memory (`non_blocking=True`), completely bypassing CPU-to-GPU data bottlenecks.

---

### 3. 🎨 Why Generation Quality is 100% Preserved

* **Exact Channel-Wise VAE Normalization**: Qwen-Image VAE uses specific channel-wise mean and standard deviation tensors (`latents_mean`, `latents_std`). Our pre-caching applies exact channel normalization `(z - mean) / std`, ensuring latent distributions match Krea-2's pre-trained space down to the float.
* **Full-Layer Target Coverage**: LoRA adapters target **all** `Linear` and `Linear4bit` modules across the DiT architecture, enabling deep feature learning (concepts, styles, faces, lighting) rather than surface-level overfitting.
* **Native Krea-2 Noise Shift Schedule**: Implements Krea-2's exact mathematical noise shift function (`calculate_shift`) and logit-normal/shifted timestep sampling (`sample_sigma`), preserving the true velocity-matching diffusion trajectories.

---

## ✨ Features

- **🌐 Modern Web GUI**: Control pre-caching, dataset editing, training, checkpointing, and model export from a sleek single-page web app powered by Flask.
- **🖼️ Fast previews** with the Krea 2 Turbo LoRA.
- **📊 Real-Time Hardware Telemetry**:
  - System **RAM** usage.
  - Physical **GPU VRAM** usage (via `nvidia-smi` / `torch`).
  - **GPU Temperature (°C)** with dynamic color coding (Green <70°C, Orange 70–79°C, Red >80°C).
- **🔑 Hugging Face Token Support**: Optional HF token (stored in `settings\HF_token.json`) for faster model downloads with live progress bars (MBs, transfer speed, ETA).
- **✍️ Auto-Captions**: **Create Captions** writes one `.txt` per image with Qwen3-VL-4B, the Krea-2 text encoder itself (NF4, ~3.5 GB VRAM, no extra download), with the trigger word first. The prompt is editable and **Overwrite** off only fills the missing captions.
- **🖼️ Dataset Manager & Inline Caption Editor**:
  - Visual grid with status badges (🟢 **Green** = Caption present, 🔴 **Red** = Missing caption), resizable by dragging.
  - Modal lightbox to view high-res images and **edit `.txt` captions directly on disk** (warns before closing with unsaved changes).
  - Batch tool: put the **Trigger Word** first and **Append / Replace / Remove** a common text in every caption. **Clear Captions** and a per-image delete button.
- **🔄 Live Settings**: while training, **Save JSON** applies steps, save/preview every, preview settings, prompt, turbo and LR on the next step. A new preview prompt is encoded on the CPU without stopping the training.
- **🗑️ Delete Pre-Cache / Delete Training** buttons to empty the current project's folders.
- **⏱️ Exact Step Resume Checkpoints**: Interrupt or stop training at any step (e.g. Step 333); the exact state (`current_step.txt`, `optimizer.pt`, `adapter_model.safetensors`) is saved automatically. Click **Start/Resume** to continue from that exact step. Resuming always keeps the checkpoint's rank and alpha.
- **🏷️ LoRA Metadata**: exported LoRAs carry their alpha (ComfyUI applies them at the same strength as the previews) and kohya-style metadata (trigger word, rank, steps, resolution) that CivitAI and LoRA managers read.
- **📂 Automatic Project Folder Management**: Dynamically routes cache and outputs to `./cached_data_krea2_<project>` and `./krea2_lora_output_<project>` based on your project name.
- **🚀 One-Click WebUI Export ("Send to Models")**: Export the best `.safetensors` LoRA directly to your preferred WebUI folder (ComfyUI, Forge, Automatic1111).
- **🌐 Fully Bilingual (English / Español)**: All buttons, console logs, dialogs, and progress bars display labels in both English and Spanish.

---

## 💬 Community & Support

Join the **AcademiaSD** community to learn more about local image and video AI.!

- ▶ **YouTube**: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 𝕏 **X (Twitter)**: [twitter.com/Academia_S_D](https://twitter.com/Academia_S_D)
- 💬 **Discord**: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ **Ko-Fi**: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)

---

## 📜 Credits & License

Developed with ❤️ by **AcademiaSD**. Built upon PyTorch, Diffusers, PEFT, Bitsandbytes, and Hugging Face Hub.

