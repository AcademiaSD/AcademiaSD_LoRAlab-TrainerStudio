#!/usr/bin/env bash
# AcademiaSD - LoRAlab Trainer Studio Updater (Linux)
# Equivalente Linux de / Linux equivalent of: Update_LoRAlab-TrainerStudio.bat
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "================================================================"
echo "  ACADEMIASD - LORALAB TRAINER STUDIO UPDATER"
echo "  [EN] Repository Update Utility"
echo "  [ES] Utilidad de Actualizacion del Repositorio"
echo "================================================================"
echo

# 1. Check Git
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] Git is not installed or not available in PATH."
    echo "[ERROR] Git no esta instalado o no esta disponible en el PATH."
    echo
    echo "[EN] Please install Git:"
    echo "[ES] Por favor instala Git:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y git"
    echo "  Fedora:        sudo dnf install -y git"
    echo "  Arch:          sudo pacman -S --noconfirm git"
    echo "[EN] Or run ./Install_LoRAlab-TrainerStudio.sh, which tries to install it."
    echo "[ES] O ejecuta ./Install_LoRAlab-TrainerStudio.sh, que intenta instalarlo."
    exit 1
fi

# 2. Check .git
if [ ! -d ".git" ]; then
    echo "[INFO] Initializing Git repository..."
    echo "[INFO] Inicializando repositorio Git..."
    echo
    git init
    git remote add origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git 2>/dev/null || true
fi

# 3. Ensure remote URL
git remote set-url origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git 2>/dev/null || true

echo "[EN] Syncing latest updates from GitHub..."
echo "[ES] Sincronizando ultimas actualizaciones desde GitHub..."
echo

# 4. Rama a actualizar: la de esta carpeta (main, dev...); main si no hay ninguna, master como ultimo recurso
#    Branch to update: this folder's branch (main, dev...); main if there is none, master as a last resort
BRANCH="$(git symbolic-ref --short -q HEAD 2>/dev/null || true)"
BRANCH="${BRANCH:-main}"
if ! git fetch origin "${BRANCH}" 2>/dev/null; then
    if git fetch origin main 2>/dev/null; then
        BRANCH="main"
    elif git fetch origin master 2>/dev/null; then
        BRANCH="master"
    else
        echo
        echo "[ERROR] Failed to update repository from GitHub (fetch failed)."
        echo "[ERROR] No se pudo actualizar el repositorio desde GitHub (fallo fetch)."
        echo "Revisa tu conexion a internet / Check your internet connection."
        exit 1
    fi
fi
echo "[EN] Branch: ${BRANCH}"
echo "[ES] Rama: ${BRANCH}"
echo

# 5. Force reset to remote branch
git checkout -f "${BRANCH}"
git reset --hard "origin/${BRANCH}" || {
    echo
    echo "[ERROR] Failed to update repository from GitHub (reset failed)."
    echo "[ERROR] No se pudo actualizar el repositorio desde GitHub (fallo reset)."
    exit 1
}

echo
echo "================================================================"
echo "[OK] Repository updated successfully! / Repositorio actualizado!"
echo "================================================================"
echo
echo "[EN] Update process completed."
echo "[ES] Proceso de actualizacion completado."
echo
echo "[EN] Your models, datasets, projects, LoRAs and settings were kept."
echo "[ES] Tus modelos, datasets, proyectos, LoRAs y ajustes se han conservado."

# 6. Librerias nuevas o actualizadas de requirements.txt / New or updated libraries from requirements.txt
PY_EXE="${LORALAB_PYTHON:-./venv/bin/python}"
echo
echo "Actualizando librerias de Python... / Updating Python libraries..."
if [ -x "${PY_EXE}" ]; then
    "${PY_EXE}" -m pip install -r requirements.txt || echo "[AVISO] No se pudieron actualizar las librerias / [WARNING] Could not update the libraries."
else
    echo "[AVISO] No hay entorno: ejecuta ./Install_LoRAlab-TrainerStudio.sh / [WARNING] No environment: run ./Install_LoRAlab-TrainerStudio.sh"
fi
