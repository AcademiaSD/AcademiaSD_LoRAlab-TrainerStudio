# AcademiaSD LoRAlab Trainer

All the AcademiaSD LoRAlabs in one folder, with **one shared Python environment** and **one launcher**.
Todos los LoRAlab de AcademiaSD en una sola carpeta, con **un solo entorno de Python** y **un solo lanzador**.

| LoRAlab | Trains / Entrena | Guide / Guía |
| :--- | :--- | :--- |
| LTX-2.3 (also LTX-2.5 / también LTX-2.5) | Video LoRAs / LoRAs de vídeo | [README_LTX23.md](docs/README_LTX23.md) |
| Krea 2 | Image LoRAs / LoRAs de imagen | [README_Krea2.md](docs/README_Krea2.md) |
| MiniMax-H3 | Video and audio LoRAs / LoRAs de vídeo y audio | [README_MiniMaxH3.md](docs/README_MiniMaxH3.md) |
| Qwen-Image 2.1 | Image and edit LoRAs / LoRAs de imagen y edición | [README_QwenImage21.md](docs/README_QwenImage21.md) |

---

## 📦 Installation / Instalación

1. Clone the repository / Clona el repositorio:
   ```bash
   git clone https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Trainer.git
   ```
2. Double-click `Install_LoRAlab.bat`. It creates one `venv` for every LoRAlab (Python 3.13 + PyTorch CUDA 13.0 + Diffusers from GitHub).
   Doble clic en `Install_LoRAlab.bat`. Crea un único `venv` para todos los LoRAlab.
3. (Optional / Opcional) `Install_Triton&SageAtten220.bat` for SageAttention 2.2.
4. MiniMax-H3 reads video with `ffmpeg`: it must be in the `PATH`. / MiniMax-H3 lee vídeo con `ffmpeg`: tiene que estar en el `PATH`.

Each model is downloaded from Hugging Face the first time you use its LoRAlab. / Cada modelo se descarga de Hugging Face la primera vez que usas su LoRAlab.

## 🚀 Usage / Uso

- `Start_LoRAlab.bat` opens the launcher at `http://127.0.0.1:4990`. Click a trainer to open it; its window and its web interface (`http://127.0.0.1:5000`) open on their own.
  `Start_LoRAlab.bat` abre el lanzador. Pulsa un entrenador para abrirlo; se abren su ventana y su interfaz web.
- Only one trainer can be open at a time (they all use port 5000). / Solo puede haber un entrenador abierto a la vez (todos usan el puerto 5000).
- Each trainer can also be opened directly with its `code\Run_LoRAlab-<Model>.bat`. / También se abre cada uno directamente con su `code\Run_LoRAlab-<Modelo>.bat`.
- `Update_LoRAlab.bat` updates everything from GitHub. Your models, projects and settings are kept. / Actualiza todo desde GitHub; tus modelos, proyectos y ajustes se conservan.

## ➕ Adding a LoRAlab / Añadir un LoRAlab

Add an entry to `GUI/launcher.json` (`id`, `name`, `description`, `image`, `run`). The launcher places the cards by itself: up to 3 in one row, then two rows with the extra one on top (4 → 2+2, 5 → 3+2, 6 → 3+3). `image_scale` sets the cover size (0.2 = 20 %) and `rows` can force the number of rows.
Añade una entrada a `GUI/launcher.json`. El lanzador coloca las tarjetas solo; `image_scale` fija el tamaño de la portada y `rows` fuerza el número de filas.

## 📁 Structure / Estructura

```text
AcademiaSD_LoRAlab-Trainer/
├── Start_LoRAlab.bat                # Launcher / Lanzador
├── Update_LoRAlab.bat               # Updater / Actualizador
├── Install_LoRAlab.bat, Install_Triton&SageAtten220.bat
├── README.md, LICENSE
├── code/                            # Run_LoRAlab-<Model>.bat
├── scripts/                         # launcher.py, server_<model>.py, 0_caption / 1_pre_cache / 2_train_lora, refmod.py, melband/
├── GUI/                             # launcher.html, launcher.json, trainer_ui_<model>.html
├── docs/                            # README_<Model>.md
├── assets/<model>/                  # Covers and logos / Portadas y logos
├── tools/<model>/                   # NF4 converters / Conversores NF4
├── Example_Dataset/                 # MiniMax-H3
└── settings/                        # Your settings and HF token (created on first use) / Tus ajustes y token de HF (se crea al usarlo)
```

---

## 💬 Community & Support

- ▶ **YouTube**: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 𝕏 **X (Twitter)**: [twitter.com/Academia_S_D](https://twitter.com/Academia_S_D)
- 💬 **Discord**: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ **Ko-Fi**: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)

Developed with ❤️ by **AcademiaSD**. MIT License.
