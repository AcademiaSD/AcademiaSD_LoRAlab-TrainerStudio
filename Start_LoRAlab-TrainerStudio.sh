#!/usr/bin/env bash
# AcademiaSD - LoRAlab Trainer Studio Launcher (Linux)
# Equivalente Linux de / Linux equivalent of: Start_LoRAlab-TrainerStudio.bat
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) / your own environment's python
PYTHON_EXE="${LORALAB_PYTHON:-./venv/bin/python}"

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "[ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:"
    echo "  ${PYTHON_EXE}"
    echo
    echo "Ejecuta primero ./Install_LoRAlab-TrainerStudio.sh / Run ./Install_LoRAlab-TrainerStudio.sh first."
    exit 1
fi

"${PYTHON_EXE}" "./scripts/launcher.py"
