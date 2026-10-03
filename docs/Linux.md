# Linux support — Soporte Linux

> [EN] This project runs on Linux besides Windows. Every `.bat` has a same-named
> `.sh` equivalent, and `scripts/launcher.py` picks the right one per platform.
> [ES] El proyecto funciona en Linux además de en Windows. Cada `.bat` tiene su
> equivalente `.sh` con el mismo nombre, y `scripts/launcher.py` elige el que toca
> según la plataforma.

## [EN] Requirements / [ES] Requisitos

- Linux 64-bit (tested on Nobara/Fedora 44, should work on Ubuntu/Debian/Arch).
- NVIDIA GPU RTX 20xx / GTX 16xx or newer, driver 580+, 8 GB VRAM (4 GB for Anima in NF4, 12 GB for LTX-2.3).
- Git, Python 3.13 (`Install_LoRAlab-TrainerStudio.sh` tries to install it via
  `apt`/`dnf`/`pacman`, sudo asked), CUDA Toolkit with `nvcc` (only needed to
  build SageAttention), and `ffmpeg` (needed for MiniMax-H3 video clips).

## [EN] Install / [ES] Instalación

```bash
git clone https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git
cd AcademiaSD_LoRAlab-TrainerStudio
chmod +x *.sh code/*.sh
./Install_LoRAlab-TrainerStudio.sh   # one shared venv: PyTorch cu130, Diffusers, Transformers, PEFT, bitsandbytes, Flask...
./Install_Triton_SageAtten220.sh     # optional: Triton from PyPI + SageAttention 2.2.0 built from source
```

## [EN] Use / [ES] Uso

```bash
./Start_LoRAlab-TrainerStudio.sh     # launcher at http://127.0.0.1:4990
./Update_LoRAlab-TrainerStudio.sh    # git pull equivalent (keeps models, datasets, settings)
```

Clicking a trainer card opens its `code/Run_LoRAlab-*.sh` in a new terminal
(`konsole` → `gnome-terminal` → `xfce4-terminal` → `xterm`); closing that window
stops the trainer, same as `start` does with the `.bat` on Windows. Over SSH
(no graphical terminal) the trainer runs detached and logs to
`settings/trainer_console.log`. Trainers can also be started directly:

```bash
./code/Run_LoRAlab-Krea2.sh
./code/Run_LoRAlab-LTX23.sh
./code/Run_LoRAlab-MiniMaxH3.sh
./code/Run_LoRAlab-QwenImage21.sh
./code/Run_LoRAlab-ZImage.sh
./code/Run_LoRAlab-Anima.sh
```

## [EN] Notes / [ES] Notas

- SageAttention 2.2.0 is **not** on PyPI (only 1.x there) and its Windows `.whl`
  files don't work on Linux, so it is built from source
  (`thu-ml/SageAttention@v2.2.0`). The build needs `nvcc` + `gcc` + `ninja` and
  takes several minutes; torch 2.14 requires `-std=c++20`, which the script
  passes via `CXX/NVCC_APPEND_FLAGS`. If the build fails, the PyPI 1.x version
  is installed instead, with an explicit warning that it is **not** 2.2.0.
- `GUI/launcher.json` is shared between platforms: on Linux the launcher maps
  each `Run_LoRAlab-*.bat` entry to its `.sh`. No Windows file was modified in
  behavior; all Python changes are `os.name == "nt"` branches.
- Out of scope: `tools/*.bat` model converters (advanced use, Windows-only for now).
- If a trainer stops with an error, its terminal stays open until you press Enter, like the
  `pause` at the end of the `.bat` files.
- **Browse** buttons open the system folder dialog through `tkinter`; without it (e.g. no
  `python3.13-tk` package) they open the web interface's own folder browser instead.
- Edit the `.sh` files with LF line endings (`.gitattributes` enforces it on checkout).
