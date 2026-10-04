#!/usr/bin/env bash
# Arranque del contenedor: clona o actualiza el repo en /workspace (volumen que se conserva) y pasa el
# control a docker/start.sh del propio repo, que se actualiza con el código.
#   LORALAB_REF  rama o etiqueta a usar (main por defecto); fija una versión si una actualización falla.
set -euo pipefail

REPO_URL="${LORALAB_REPO:-https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git}"
REF="${LORALAB_REF:-main}"
DIR="${LORALAB_DIR:-/workspace/AcademiaSD_LoRAlab-TrainerStudio}"

if [ -d "${DIR}/.git" ]; then
    echo "Updating / Actualizando ${DIR} (${REF})..."
    # Modelos, datasets, proyectos y ajustes no están en git: reset --hard los conserva, como Update_LoRAlab.
    if git -C "${DIR}" fetch origin "${REF}"; then
        git -C "${DIR}" reset --hard FETCH_HEAD
    else
        echo "[WARNING] Could not update, using the local copy / No se pudo actualizar, se usa la copia local."
    fi
else
    echo "Cloning / Clonando ${REPO_URL} -> ${DIR}..."
    git clone "${REPO_URL}" "${DIR}"
    git -C "${DIR}" checkout "${REF}"
fi

exec bash "${DIR}/docker/start.sh"
