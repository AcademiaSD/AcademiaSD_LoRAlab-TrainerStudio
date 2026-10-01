# LTX-2.3 trainer — technical notes

> **Technical notes** for the LTX-2.3 trainer of **AcademiaSD LoRAlab Trainer Studio**. Installation, updating and everyday use are in the [main README](../README.md).

> **LTX-2.5:** this trainer is also compatible with LTX-2.5. / Este entrenador también es compatible con LTX-2.5.


![AcademiaSD_LoRAlab-LTX23](../assets/ltx23/portada.jpg)

<p align="center">
  <b>An ultra-fast, low-resource Web GUI & pipeline for training LTX 2.3 LoRAs.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.14%20cu130-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-purple.svg" alt="Web UI">
</p>

---
## YouTube video instructions.
https://www.youtube.com/watch?v=52Kyr7GZ6Y0
---

## 🔬 Technical Deep-Dive: Why is it so fast, light, & high quality?

Training a 22-Billion parameter Diffusion Transformer (DiT) like **LTX-2.3** typically requires enterprise-grade hardware (30GB+ VRAM, 64GB+ System RAM) and hours of compute. **AcademiaSD LoRAlab LTX 2.3** breaks this barrier, allowing training on consumer GPUs (12GB-16GB VRAM) in record time while preserving 100% of the model's generation quality. 

Here is the exact architectural breakdown of how this is achieved:

---

### 1. 📉 VRAM Usage is Reduced to ~12 GB

* **4-Bit NormalFloat (NF4) Quantization (`bitsandbytes`)**: The 22-Billion parameter DiT backbone is stored as frozen 4-bit NF4 weights (`Linear4bit`): ~38 GB in BF16 → ~9.8 GB. The model is built empty and filled directly from the NF4 weights, so the BF16 transformer is never downloaded or loaded. The Gemma 3 text encoder used by the pre-cache is also pre-quantized (FP32 ~49 GB → NF4 ~7.8 GB).
* **Zero VRAM Wasted on Encoders (Offline Pre-Caching)**: During training (`2_train_lora_ltx23.py`), **neither the Text Encoder nor the VAE are loaded into VRAM**. All text embeddings and image latents are pre-computed once during the pre-cache stage and stored on disk.
* **Paged 8-Bit AdamW Optimizer (`bitsandbytes.optim.PagedAdamW8bit`)**: Optimizer states are stored in 8-bit precision instead of 32-bit float and can page out of VRAM under pressure.
* **Gradient Checkpointing**: Intermediate activation tensors are recomputed during backward passes rather than stored in memory, keeping activation VRAM flat regardless of resolution.

---

### 2. ⚡ Why Training Speed is So Fast

* **No Per-Step Encoding Overhead**: In standard pipelines, every training step spends compute cycles passing images through the VAE and prompts through the LLM/Text Encoder. By eliminating this in the pre-cache phase, **100% of GPU compute during training is dedicated purely to the DiT backward pass**.
* **Compact latents**: the LTX video VAE already compresses 32× in each spatial dimension, so even 768×768 images become short token sequences, and the cached latents are copied to the GPU with non-blocking transfers.

---

### 3. 🎨 Why Generation Quality is 100% Preserved

* **Exact Channel-Wise VAE Normalization**: VAE uses specific channel-wise mean and standard deviation tensors (`latents_mean`, `latents_std`). Our pre-caching applies exact channel normalization `(z - mean) / std`, ensuring latent distributions match LTX-2.3s pre-trained space down to the float.
* **Full-Layer Target Coverage**: LoRA adapters target the `Linear` / `Linear4bit` layers of the DiT blocks, enabling deep feature learning (concepts, styles, faces, lighting) rather than surface-level overfitting.
* **Native LTX 2.3 Noise Shift Schedule**: the timestep shift depends on the sequence length exactly as in the LTX scheduler (`base_shift` / `max_shift` from its config), preserving the true velocity-matching diffusion trajectories.

---

## ✨ Features

- **🌐 Modern Web GUI**: Control pre-caching, dataset editing, training, checkpointing, and model export from a sleek single-page web app powered by Flask.
- **📊 Real-Time Hardware Telemetry**:
  - System **RAM** usage.
  - Physical **GPU VRAM** usage (via `nvidia-smi` / `torch`).
  - **GPU Temperature (°C)** with dynamic color coding (Green <70°C, Orange 70–79°C, Red >80°C).
- **🔑 Hugging Face Token Support**: Optional HF token (stored in `settings\HF_token.json`) for faster model downloads with live progress bars (MBs, transfer speed, ETA).
- **✍️ Auto-Captions**: **Create Captions** writes one `.txt` per image with Qwen3-VL-8B (NF4, ~5.5 GB VRAM), the same captioner as the Qwen-Image 2.1 LoRAlab, with the trigger word first. The first time it downloads ~5 GB to `Captioner-Qwen3-VL-8B/`. The prompt is editable and **Overwrite** off only fills the missing captions.
- **🖼️ Dataset Manager & Inline Caption Editor**:
  - Visual grid with status badges (🟢 **Green** = Caption present, 🔴 **Red** = Missing caption), resizable by dragging.
  - Modal lightbox to view high-res images and **edit `.txt` captions directly on disk** (warns before closing with unsaved changes).
  - Batch tool: put the **Trigger Word** first and **Append / Replace / Remove** a common text in every caption. **Clear Captions** and a per-image delete button.
- **🔄 Live Settings**: while training, **Save JSON** applies steps, save/preview every, preview settings and LR on the next step.
- **🗑️ Delete Pre-Cache / Delete Training** buttons to empty the current project's folders.
- **⏱️ Exact Step Resume Checkpoints**: Interrupt or stop training at any step (e.g. Step 333); the exact state (`current_step.txt`, `optimizer.pt`, `adapter_model.safetensors`) is saved automatically. Click **Start/Resume** to continue from that exact step. Resuming always keeps the checkpoint's rank and alpha.
- **🏷️ LoRA Metadata**: exported LoRAs carry kohya-style metadata (trigger word, rank, steps, resolution) that CivitAI and LoRA managers read.
- **📂 Automatic Project Folder Management**: Dynamically routes cache and outputs to `./cached_data_LTX23_<project>` and `./LTX23_lora_output_<project>` based on your project name.
- **🚀 One-Click WebUI Export ("Send to Models")**: Export the best `.safetensors` LoRA directly to your preferred WebUI folder (ComfyUI, ForgeNeo, etc.).
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


