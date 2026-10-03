#!/usr/bin/env bash
# Triton and SageAttention Installer by Academia SD (Linux)
# Equivalente Linux de / Linux equivalent of: Install_Triton&SageAtten220.bat
# En Linux no existen los .whl de Windows: Triton viene de PyPI y
# SageAttention 2.2.0 se compila desde el codigo fuente (hay nvcc + gcc).
# On Linux the Windows .whl files do not exist: Triton comes from PyPI and
# SageAttention 2.2.0 is built from source (needs nvcc + gcc).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_EXE="venv/bin/python"
SAGE_TAG="v2.2.0"
SAGE_REPO="https://github.com/thu-ml/SageAttention.git"

echo "======================================================="
echo "  Triton & SageAttention Installer"
echo "  Created by: Academia SD (Linux)"
echo "======================================================="
echo

# Check venv
if [ ! -x "${PYTHON_EXE}" ]; then
    echo "[ERROR] ${PYTHON_EXE} not found."
    echo "Run ./Install_LoRAlab-TrainerStudio.sh first."
    exit 1
fi

echo "======================================================="
echo "   1. INSTALLING TRITON"
echo "======================================================="
"${PYTHON_EXE}" -m pip install -U triton || {
    echo "[WARNING] No se pudo instalar Triton via PyPI / Could not install Triton via PyPI."
    echo "Normalmente Triton ya viene con PyTorch. Se continua igualmente."
    echo "Usually Triton already ships with PyTorch. Continuing anyway."
}

echo
echo "======================================================="
echo "   2. ANALYZING ENVIRONMENT FOR SAGEATTENTION 2.2.0"
echo "======================================================="

if ! "${PYTHON_EXE}" - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is not installed. Cannot continue.")
    sys.exit(1)

py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
t_ver = torch.__version__
c_ver = torch.version.cuda or "None"
print("[INFO] Detected Environment:")
print(f"       - Python: {py_ver}")
print(f"       - PyTorch: {t_ver}")
print(f"       - CUDA: {c_ver}\n")
PYEOF
then
    echo "[ERROR] PyTorch no esta instalado en el venv / PyTorch is not installed in the venv."
    exit 1
fi

# La version 2.2.0 NO esta en PyPI (alli solo llega a 1.x): se compila desde GitHub.
# Version 2.2.0 is NOT on PyPI (only 1.x there): it is built from GitHub source.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc no encontrado: sin CUDA Toolkit no se puede compilar SageAttention 2.2.0."
    echo "[ERROR] nvcc not found: without the CUDA Toolkit SageAttention 2.2.0 cannot be built."
    echo "  Ubuntu/Debian: sudo apt install -y nvidia-cuda-toolkit build-essential ninja-build"
    echo "  Fedora/Nobara: sudo dnf install -y cuda-nvcc gcc-c++ ninja-build"
    echo "  Arch:          sudo pacman -S --noconfirm cuda gcc ninja"
    exit 1
fi
if ! command -v gcc >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    echo "[ERROR] Faltan herramientas de compilacion (gcc/ninja)."
    echo "[ERROR] Missing build tools (gcc/ninja)."
    echo "  Ubuntu/Debian: sudo apt install -y build-essential ninja-build"
    echo "  Fedora/Nobara: sudo dnf install -y gcc-c++ ninja-build"
    echo "  Arch:          sudo pacman -S --noconfirm base-devel ninja"
    exit 1
fi

# Preferir el toolkit completo de /usr/local (suele ser mas nuevo y completo
# que el nvcc suelto de /usr/bin). / Prefer the full toolkit under /usr/local.
if [ -x "/usr/local/cuda/bin/nvcc" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[INFO] Usando CUDA_HOME=${CUDA_HOME} ($(nvcc --version | tail -1))"
fi

# La compilacion necesita las cabeceras de cuSPARSE.
# The build needs the cuSPARSE headers.
if ! ls "${CUDA_HOME:-/usr/local/cuda}"/targets/*/include/cusparse.h >/dev/null 2>&1 \
&& ! ls /usr/include/cusparse.h /usr/local/cuda*/targets/*/include/cusparse.h >/dev/null 2>&1; then
    echo "[ERROR] Faltan las cabeceras de cuSPARSE (cusparse.h)."
    echo "[ERROR] Missing cuSPARSE headers (cusparse.h)."
    echo "  Fedora/Nobara: sudo dnf install -y libcusparse-devel-13-4"
    echo "  Ubuntu/Debian (repo NVIDIA): sudo apt install -y libcusparse-dev"
    echo "Despues vuelve a lanzar este script / Then re-run this script."
    exit 1
fi

echo "[INFO] Compilando SageAttention ${SAGE_TAG} desde el codigo fuente..."
echo "[INFO] Building SageAttention ${SAGE_TAG} from source..."
echo "(esto tarda varios minutos / this takes several minutes)"
# --no-build-isolation: el setup.py importa torch, que solo existe dentro del venv.
# --no-build-isolation: setup.py imports torch, which only exists inside the venv.
# CXX/NVCC_APPEND_FLAGS: torch 2.14 exige C++20 y el setup.py fija -std=c++17;
# al ir al final de los flags, el ultimo -std= es el que manda en GCC/NVCC.
# CXX/NVCC_APPEND_FLAGS: torch 2.14 requires C++20 while setup.py pins -std=c++17;
# appended last, so it wins in GCC/NVCC.
if CXX_APPEND_FLAGS="-std=c++20" NVCC_APPEND_FLAGS="-std=c++20" \
    "${PYTHON_EXE}" -m pip install --upgrade --no-build-isolation "git+${SAGE_REPO}@${SAGE_TAG}"; then
    echo "[OK] SageAttention ${SAGE_TAG} instalado / installed."
else
    echo "[WARNING] No se pudo compilar SageAttention ${SAGE_TAG}."
    echo "[WARNING] Could not build SageAttention ${SAGE_TAG}."
    echo "Instalando la ultima version precompilada de PyPI (1.x, NO es la 2.2.0)..."
    echo "Installing the latest prebuilt version from PyPI (1.x, NOT 2.2.0)..."
    "${PYTHON_EXE}" -m pip install -U sageattention || {
        echo "[ERROR] No se pudo instalar SageAttention / Could not install SageAttention."
        exit 1
    }
fi

echo
echo "[INFO] Verificando importacion / Verifying import..."
"${PYTHON_EXE}" -c "import sageattention; print('SageAttention:', getattr(sageattention, '__version__', 'OK'))" || {
    echo "[WARNING] SageAttention se instalo pero no se puede importar (posible falta de kernel CUDA para tu GPU)."
    echo "[WARNING] SageAttention installed but cannot be imported (possibly no CUDA kernel for your GPU)."
}

echo
echo "======================================================="
echo "   PROCESS COMPLETED"
echo "======================================================="
