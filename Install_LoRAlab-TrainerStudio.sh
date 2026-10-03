#!/usr/bin/env bash
# ========================================================
# INSTALADOR ACADEMIASD LORALAB TRAINER STUDIO (Linux)
# ACADEMIASD LORALAB TRAINER STUDIO INSTALLER (Linux)
# Un solo entorno para / One environment for: LTX-2.3, Krea 2, MiniMax-H3, Qwen-Image 2.1, Z-Image, Anima
# Entorno Python 3.13 + PyTorch CUDA / Python 3.13 + PyTorch CUDA Env
# Compatible con GPUs NVIDIA modernas / Compatible with modern NVIDIA GPUs
# Equivalente Linux de / Linux equivalent of: Install_LoRAlab-TrainerStudio.bat
# ========================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
VENV_DIR="${BASE_DIR}venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
PYTHON_EXE=""

echo "========================================================"
echo "  INSTALADOR ACADEMIASD LORALAB TRAINER STUDIO / INSTALLER"
echo "  Un solo entorno para / One environment for: LTX-2.3, Krea 2, MiniMax-H3, Qwen-Image 2.1, Z-Image, Anima"
echo "  Entorno Python 3.13 + PyTorch CUDA / Python 3.13 + PyTorch CUDA Env"
echo "========================================================"
echo
echo "Carpeta del instalador / Installer folder:"
echo "${BASE_DIR}"
echo

# --------------------------------------------------------
# [1/9] Git
# --------------------------------------------------------
echo "[1/9] Comprobando Git / Checking Git..."
if command -v git >/dev/null 2>&1; then
    echo "[OK] Git disponible / Git available."
    git --version
else
    echo "[AVISO/WARNING] Git no esta instalado / Git is not installed."
    echo "El entrenador funciona igual, pero para actualizar instalalo con tu gestor de paquetes:"
    echo "The trainer works anyway, but to update it install it with your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y git"
    echo "  Fedora:        sudo dnf install -y git"
    echo "  Arch:          sudo pacman -S --noconfirm git"
    echo
fi

# --------------------------------------------------------
# [2/9] Python 3.13
# --------------------------------------------------------
echo
echo "[2/9] Comprobando Python 3.13 / Checking Python 3.13..."

find_python313() {
    for cand in python3.13 python3 /usr/bin/python3.13 /usr/local/bin/python3.13; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)" >/dev/null 2>&1; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}

if PYTHON_EXE="$(find_python313)"; then
    echo "[OK] Python 3.13 detectado / Python 3.13 detected: ${PYTHON_EXE}"
    "${PYTHON_EXE}" --version
else
    echo "[INFO] Python 3.13 no esta disponible / Python 3.13 is not available."
    echo
    if command -v apt-get >/dev/null 2>&1; then
        echo "Intentando instalar Python 3.13 via apt (se pedira sudo)..."
        echo "Trying to install Python 3.13 via apt (sudo will be asked)..."
        if ! sudo apt-get update; then
            echo "[ERROR] sudo fallo o fue cancelado. Ejecuta manualmente:"
            echo "[ERROR] sudo failed or was cancelled. Run manually:"
            echo "  sudo apt update && sudo apt install -y python3.13 python3.13-venv python3.13-dev"
            echo "y vuelve a lanzar este script / then re-run this script."
            exit 1
        fi
        if ! sudo apt-get install -y python3.13 python3.13-venv python3.13-dev python3-pip 2>/dev/null \
        && ! sudo apt-get install -y python3.13 python3.13-venv python3-pip; then
            echo "Anadiendo PPA deadsnakes e intentando de nuevo..."
            echo "Adding deadsnakes PPA and retrying..."
            sudo apt-get install -y software-properties-common \
            && sudo add-apt-repository -y ppa:deadsnakes/ppa \
            && sudo apt-get update \
            && sudo apt-get install -y python3.13 python3.13-venv python3.13-dev python3-pip || {
                echo "[ERROR] No se pudo instalar Python 3.13 via apt."
                echo "[ERROR] Could not install Python 3.13 via apt."
                echo "Instalalo manualmente: https://www.python.org/downloads/"
                exit 1
            }
        fi
    elif command -v dnf >/dev/null 2>&1; then
        echo "Intentando instalar Python 3.13 via dnf (se pedira sudo)..."
        echo "Trying to install Python 3.13 via dnf (sudo will be asked)..."
        if ! sudo dnf install -y python3.13 python3.13-devel git; then
            echo "[ERROR] sudo fallo o fue cancelado. Ejecuta manualmente en tu terminal:"
            echo "[ERROR] sudo failed or was cancelled. Run manually in your terminal:"
            echo "  sudo dnf install -y python3.13 python3.13-devel git"
            echo "y vuelve a lanzar este script / then re-run this script:"
            echo "  ./Install_LoRAlab-TrainerStudio.sh"
            exit 1
        fi
    elif command -v pacman >/dev/null 2>&1; then
        echo "Intentando instalar Python 3.13 via pacman (se pedira sudo)..."
        if ! sudo pacman -S --noconfirm python git; then
            echo "[ERROR] sudo fallo o fue cancelado. Ejecuta manualmente:"
            echo "  sudo pacman -S --noconfirm python git"
            exit 1
        fi
    else
        echo "[ERROR] Instala Python 3.13.x manualmente: https://www.python.org/downloads/"
        echo "[ERROR] Install Python 3.13.x manually: https://www.python.org/downloads/"
        exit 1
    fi

    if PYTHON_EXE="$(find_python313)"; then
        echo "[OK] Python 3.13 instalado / Python 3.13 installed: ${PYTHON_EXE}"
    else
        # Ultimo recurso: python3 aunque no sea 3.13 -> error claro
        if command -v python3 >/dev/null 2>&1; then
            echo "Python encontrado / Python found: $(python3 --version 2>&1)"
        fi
        echo "[ERROR] No se encontro Python 3.13 tras la instalacion / Python 3.13 not found after install."
        echo "Se requiere Python 3.13.x / Python 3.13.x is required."
        exit 1
    fi
fi

echo
echo "========================================================"
echo "  PYTHON DETECTADO / PYTHON DETECTED"
echo "======================================================="
echo "Ejecutable / Executable: ${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)" || {
    echo "[ERROR] La version de Python no es compatible / Python version is not compatible."
    echo "Se requiere Python 3.13.x / Python 3.13.x is required."
    exit 1
}
echo "[OK] Python 3.13 compatible detectado / Compatible Python 3.13 detected."

# --------------------------------------------------------
# [3/9] venv
# --------------------------------------------------------
echo
echo "[3/9] Comprobando modulo venv... / Checking venv module..."
"${PYTHON_EXE}" -m venv --help >/dev/null 2>&1 || {
    echo "[ERROR] El modulo venv no esta disponible / venv module is not available."
    echo "Ubuntu/Debian: sudo apt install -y python3.13-venv"
    exit 1
}
echo "[OK] Modulo venv disponible / venv module available."

# --------------------------------------------------------
# [4/9] NVIDIA
# --------------------------------------------------------
echo
echo "[4/9] Comprobando compatibilidad NVIDIA... / Checking NVIDIA compatibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
else
    echo "nvidia-smi no encontrado: la deteccion de GPU se hara tras instalar PyTorch."
    echo "nvidia-smi not found: GPU detection will occur after PyTorch installation."
fi
echo "[OK] Continuando con la instalacion / Continuing installation."

# --------------------------------------------------------
# [5/9] venv limpio / clean venv
# --------------------------------------------------------
echo
echo "[5/9] Preparando entorno virtual limpio... / Preparing clean virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "Entorno virtual anterior detectado. Eliminando... / Previous virtual environment detected. Removing..."
    rm -rf "${VENV_DIR}"
    if [ -d "${VENV_DIR}" ]; then
        echo "[ERROR] No se pudo eliminar el venv anterior / Could not remove previous venv."
        echo "Cierra cualquier programa que use venv/bin/python y reintenta."
        exit 1
    fi
fi
echo "Creando nuevo entorno virtual... / Creating new virtual environment..."
"${PYTHON_EXE}" -m venv "${VENV_DIR}" || {
    echo "[ERROR] No se pudo crear el entorno virtual / Could not create virtual environment."
    exit 1
}

# --------------------------------------------------------
# [6/9] activar venv (usar ruta directa, no 'source' obligatorio)
# --------------------------------------------------------
echo
echo "[6/9] Activando entorno virtual... / Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate" || {
    echo "[ERROR] No se pudo activar el entorno virtual / Could not activate virtual environment."
    exit 1
}
VENV_PYTHON="${VENV_DIR}/bin/python"
echo "Python del entorno virtual / Virtual environment Python: ${VENV_PYTHON}"
"${VENV_PYTHON}" --version

# --------------------------------------------------------
# [7/9] pip, setuptools, wheel
# --------------------------------------------------------
echo
echo "[7/9] Actualizando pip, setuptools y wheel... / Upgrading pip, setuptools, and wheel..."
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel || {
    echo "[ERROR] No se pudieron actualizar las herramientas / Could not upgrade Python tools."
    exit 1
}

# --------------------------------------------------------
# [8/9] PyTorch + dependencias / dependencies
# --------------------------------------------------------
echo
echo "[8/9] Instalando PyTorch con soporte CUDA... / Installing PyTorch with CUDA support..."
echo
echo "========================================================"
echo "  IMPORTANTE / IMPORTANT"
echo "========================================================"
echo "Se instalara PyTorch con CUDA 13.0 / PyTorch with CUDA 13.0 will be installed."
echo "Para GPUs NVIDIA modernas (incluyendo RTX 50xx / Blackwell)."
echo "For modern NVIDIA GPUs (including RTX 50xx / Blackwell)."
echo "La instalacion puede tardar minutos / Installation may take minutes."
echo "========================================================"
echo
"${VENV_PYTHON}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 || {
    echo "[ERROR] No se pudo instalar PyTorch / Could not install PyTorch."
    exit 1
}

echo
echo "Instalando Diffusers desde GitHub (Qwen-Image 2.1 y MiniMax-H3 aun no estan en una version publicada)..."
echo "Installing Diffusers from GitHub (Qwen-Image 2.1 and MiniMax-H3 are not in a released version yet)..."
"${VENV_PYTHON}" -m pip install https://github.com/huggingface/diffusers/archive/refs/heads/main.zip || {
    echo "[ERROR] Error instalando Diffusers / Error installing Diffusers."
    exit 1
}

echo
echo "Instalando Transformers 5.17+, PEFT, Accelerate, Safetensors y Hugging Face Hub..."
echo "Installing Transformers 5.17+, PEFT, Accelerate, Safetensors, and Hugging Face Hub..."
"${VENV_PYTHON}" -m pip install "transformers>=5.17" peft accelerate safetensors huggingface_hub || {
    echo "[ERROR] Error instalando dependencias / Error installing dependencies."
    exit 1
}

echo
echo "Instalando BitsAndBytes y utilidades... / Installing BitsAndBytes and utilities..."
"${VENV_PYTHON}" -m pip install bitsandbytes sentencepiece protobuf pyarrow einops rotary_embedding_torch flask psutil || {
    echo "[ERROR] Error instalando BitsAndBytes o utilidades / Error installing BitsAndBytes or utilities."
    exit 1
}

# tkinter: los botones Browse lo usan para el dialogo de carpetas del sistema. Sin el se usa el
# explorador de carpetas de la propia web. / tkinter: the Browse buttons use it for the system folder
# dialog. Without it the web interface's own folder browser is used.
if ! "${VENV_PYTHON}" -c "import tkinter" >/dev/null 2>&1; then
    echo
    echo "[AVISO] tkinter no disponible: Browse usara el explorador de carpetas de la web."
    echo "[WARNING] tkinter not available: Browse will use the web folder browser."
    echo "  Ubuntu/Debian: sudo apt install -y python3.13-tk"
    echo "  Fedora:        sudo dnf install -y python3.13-tkinter"
    echo "  Arch:          sudo pacman -S --noconfirm tk"
fi

# ffmpeg (necesario para clips de video de MiniMax-H3 / needed for MiniMax-H3 video clips)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo
    echo "[AVISO] ffmpeg no detectado. Los clips de video de MiniMax-H3 lo necesitan."
    echo "[WARNING] ffmpeg not detected. MiniMax-H3 video clips need it."
    echo "  Ubuntu/Debian: sudo apt install -y ffmpeg"
    echo "  Fedora:        sudo dnf install -y ffmpeg"
    echo "  Arch:          sudo pacman -S --noconfirm ffmpeg"
fi

# --------------------------------------------------------
# [9/9] Verificacion / Verification
# --------------------------------------------------------
echo
echo "[9/9] Verificando instalacion completa... / Verifying installation..."
echo
echo "========================================================"
echo "PyTorch"
echo "========================================================"
"${VENV_PYTHON}" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA compilada:', torch.version.cuda); print('CUDA disponible:', torch.cuda.is_available()); print('GPUs detectadas:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')" || {
    echo "[ERROR] PyTorch no se inicializo correctamente / PyTorch failed to initialize."
    exit 1
}

echo
echo "========================================================"
echo "Diffusers"
echo "========================================================"
"${VENV_PYTHON}" -c "import diffusers; print('Diffusers:', diffusers.__version__); from diffusers import LTX2VideoTransformer3DModel; print('LTX-2.3: OK'); from diffusers import Krea2Transformer2DModel, AutoencoderKLQwenImage; print('Krea 2: OK'); from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler; from diffusers.models.transformers import transformer_minimax_h3; print('MiniMax-H3: OK'); from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21; print('Qwen-Image 2.1: OK'); from diffusers import ZImagePipeline, ZImageTransformer2DModel; print('Z-Image: OK'); from diffusers import AnimaModularPipeline, AnimaTextConditioner, CosmosTransformer3DModel; print('Anima: OK')" || {
    echo "[ERROR] Diffusers no incluye todos los modelos / Diffusers does not include every model."
    exit 1
}
echo "[OK] Diffusers cargado correctamente / Diffusers loaded successfully."

echo
echo "========================================================"
echo "Hugging Face Hub"
echo "========================================================"
"${VENV_PYTHON}" -c "import huggingface_hub; print('Hugging Face Hub:', huggingface_hub.__version__)" || {
    echo "[ERROR] Hugging Face Hub no esta instalado correctamente / Hugging Face Hub is not installed properly."
    exit 1
}
echo "[OK] Hugging Face Hub cargado correctamente / Hub loaded successfully."

echo
echo "========================================================"
echo "Flask / psutil"
echo "========================================================"
"${VENV_PYTHON}" -c "import flask, psutil; from importlib.metadata import version; print('Flask:', version('flask'), '| psutil:', psutil.__version__)" || {
    echo "[ERROR] Flask o psutil no estan instalados / Flask or psutil is not installed."
    exit 1
}
echo "[OK] Flask y psutil cargados correctamente / Flask and psutil loaded successfully."

echo
echo "========================================================"
echo "Transformers"
echo "========================================================"
"${VENV_PYTHON}" -c "import transformers; from packaging.version import Version; print('Transformers:', transformers.__version__); from transformers import Qwen3VLForConditionalGeneration, Gemma3ForConditionalGeneration; raise SystemExit(0 if Version(transformers.__version__) >= Version('5.17') else 1)" || {
    echo "[ERROR] Se requiere Transformers 5.17+ con Qwen3-VL y Gemma 3 / Transformers 5.17+ with Qwen3-VL and Gemma 3 is required."
    exit 1
}
echo "[OK] Transformers cargado correctamente / Transformers loaded successfully."

echo
echo "========================================================"
echo "PEFT"
echo "========================================================"
"${VENV_PYTHON}" -c "import peft; print('PEFT:', peft.__version__)" || {
    echo "[ERROR] PEFT no esta instalado correctamente / PEFT is not installed properly."
    exit 1
}
echo "[OK] PEFT cargado correctamente / PEFT loaded successfully."

echo
echo "========================================================"
echo "Accelerate"
echo "========================================================"
"${VENV_PYTHON}" -c "import accelerate; print('Accelerate:', accelerate.__version__)" || {
    echo "[ERROR] Accelerate no esta instalado correctamente / Accelerate is not installed properly."
    exit 1
}
echo "[OK] Accelerate cargado correctamente / Accelerate loaded successfully."

echo
echo "========================================================"
echo "BitsAndBytes"
echo "========================================================"
"${VENV_PYTHON}" -c "import bitsandbytes as bnb; print('BitsAndBytes:', bnb.__version__)" || {
    echo "[ERROR] BitsAndBytes no se inicializo / BitsAndBytes failed to initialize."
    echo "Es obligatorio para los modelos NF4 / It is required for the NF4 models."
    exit 1
}
echo "[OK] BitsAndBytes cargado correctamente / BitsAndBytes loaded successfully."

echo
echo "========================================================"
echo "  RESULTADO DE LA COMPROBACION / CHECK RESULT"
echo "========================================================"
echo
if "${VENV_PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "[OK] PyTorch detecta correctamente la GPU NVIDIA / PyTorch correctly detects NVIDIA GPU."
else
    echo "[ADVERTENCIA/WARNING] PyTorch NO detecta una GPU CUDA / PyTorch does NOT detect a CUDA GPU."
    echo "Posibles causas / Possible causes:"
    echo "- Driver NVIDIA antiguo / Outdated NVIDIA driver (se requiere 580+ / 580+ required)."
    echo "- Instalacion incorrecta de PyTorch / Incorrect PyTorch installation."
    echo "- Problema de GPU / GPU driver issue."
    echo "Ejecuta / Run: nvidia-smi"
fi

echo
echo "========================================================"
echo "  INSTALACION COMPLETADA / INSTALLATION COMPLETED"
echo "========================================================"
echo
echo "El entorno virtual \"venv\" ha sido creado / Virtual environment \"venv\" created."
echo "Python utilizado / Python used: ${PYTHON_EXE}"
echo "Python del entorno virtual / Virtual env Python: ${VENV_PYTHON}"
"${VENV_PYTHON}" --version
echo "GPU detectada por PyTorch / GPU detected by PyTorch:"
"${VENV_PYTHON}" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA / NOT DETECTED')"
echo "Version CUDA de PyTorch / PyTorch CUDA version:"
"${VENV_PYTHON}" -c "import torch; print(torch.version.cuda)"
echo "Version Diffusers / Diffusers version:"
"${VENV_PYTHON}" -c "import diffusers; print(diffusers.__version__)"
echo "Version Transformers / Transformers version:"
"${VENV_PYTHON}" -c "import transformers; print(transformers.__version__)"
echo "Version Hugging Face Hub / Hugging Face Hub version:"
"${VENV_PYTHON}" -c "import huggingface_hub; print(huggingface_hub.__version__)"
echo
echo "========================================================"
echo
echo "El entorno esta listo. Abre los entrenadores con ./Start_LoRAlab-TrainerStudio.sh"
echo "Environment is ready. Open the trainers with ./Start_LoRAlab-TrainerStudio.sh"
