#!/usr/bin/env bash
# AcademiaSD LoRAlab Trainer Studio - LTX-2.3 (Linux)
# Equivalente Linux de / Linux equivalent of: code/Run_LoRAlab-LTX23.bat
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_DIR="$(pwd)/"
PYTHON_EXE="${BASE_DIR}venv/bin/python"

echo
echo "================================================================"
echo "       ACADEMIASD LORALAB TRAINER STUDIO - LTX-2.3"
echo "================================================================"
echo

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "[ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:"
    echo "  ${PYTHON_EXE}"
    echo
    echo "Ejecuta primero ./Install_LoRAlab-TrainerStudio.sh / Run ./Install_LoRAlab-TrainerStudio.sh first."
    exit 1
fi

if [ ! -f "${BASE_DIR}scripts/server_ltx23.py" ]; then
    echo "[ERROR] No existe / Missing: scripts/server_ltx23.py"
    exit 1
fi

if [ ! -f "${BASE_DIR}GUI/trainer_ui_ltx23.html" ]; then
    echo "[ERROR] No existe / Missing: GUI/trainer_ui_ltx23.html"
    exit 1
fi

echo "Iniciando servidor web / Starting web server..."
echo
echo "    http://127.0.0.1:5000"
echo
echo "Pulsa Ctrl+C para detener el servidor / Press Ctrl+C to stop the server."
echo
"${PYTHON_EXE}" "${BASE_DIR}scripts/server_ltx23.py"

echo
echo "================================================================"
echo "El servidor ha finalizado / The server has stopped."
echo "================================================================"
