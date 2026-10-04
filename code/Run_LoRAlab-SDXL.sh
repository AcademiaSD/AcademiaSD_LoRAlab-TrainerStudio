#!/usr/bin/env bash
# AcademiaSD LoRAlab Trainer Studio - SDXL (Linux)
# Equivalente Linux de / Linux equivalent of: code/Run_LoRAlab-SDXL.bat
set -euo pipefail

# Como el "pause" del .bat: la terminal no se cierra sin dejar leer el error o el final.
# Like the .bat "pause": the terminal does not close before the error or the end can be read.
pause_on_exit() { if [ -t 0 ]; then read -r -p "Pulsa Enter para cerrar / Press Enter to close..." _ || true; fi; }
trap pause_on_exit EXIT

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_DIR="$(pwd)/"
# LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) / your own environment's python
PYTHON_EXE="${LORALAB_PYTHON:-${BASE_DIR}venv/bin/python}"

echo
echo "================================================================"
echo "       ACADEMIASD LORALAB TRAINER STUDIO - SDXL"
echo "================================================================"
echo

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "[ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:"
    echo "  ${PYTHON_EXE}"
    echo
    echo "Ejecuta primero ./Install_LoRAlab-TrainerStudio.sh / Run ./Install_LoRAlab-TrainerStudio.sh first."
    exit 1
fi

if [ ! -f "${BASE_DIR}scripts/server_sdxl.py" ]; then
    echo "[ERROR] No existe / Missing: scripts/server_sdxl.py"
    exit 1
fi

if [ ! -f "${BASE_DIR}GUI/trainer_ui_sdxl.html" ]; then
    echo "[ERROR] No existe / Missing: GUI/trainer_ui_sdxl.html"
    exit 1
fi

echo "Iniciando servidor web / Starting web server..."
echo
echo "Pulsa Ctrl+C para detener el servidor / Press Ctrl+C to stop the server."
echo
"${PYTHON_EXE}" "${BASE_DIR}scripts/server_sdxl.py"

echo
echo "================================================================"
echo "El servidor ha finalizado / The server has stopped."
echo "================================================================"
