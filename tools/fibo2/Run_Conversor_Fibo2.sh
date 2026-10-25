#!/usr/bin/env bash
# AcademiaSD LoRAlab - Conversor Fibo 2.0 NF4 (Linux)
# Equivalente Linux de / Linux equivalent of: tools/fibo2/Run_Conversor_Fibo2.bat
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BASE_DIR="$(pwd)/"
# LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) / your own environment's python
PYTHON_EXE="${LORALAB_PYTHON:-${BASE_DIR}venv/bin/python}"

echo
echo "================================================================"
echo "       ACADEMIASD LORALAB - CONVERSOR FIBO 2.0 NF4"
echo "================================================================"
echo
echo "  1. Coloca la carpeta del modelo sin cuantizar en:"
echo "     ${BASE_DIR}Fibo2_Dev-Raw"
echo "  2. Este script escribe:"
echo "     ${BASE_DIR}Fibo2_Dev-NF4"
echo

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "[ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:"
    echo "  ${PYTHON_EXE}"
    exit 1
fi

"${PYTHON_EXE}" "${BASE_DIR}tools/fibo2/5_conversor_Fibo2_NF4.py" "$@"

echo
echo "================================================================"
echo "Conversor finalizado / Converter finished."
echo "================================================================"
