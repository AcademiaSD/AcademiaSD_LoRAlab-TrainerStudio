#!/usr/bin/env bash
# Arranque en RunPod / Docker (lo lanza bootstrap.sh tras actualizar el repo).
#   LORALAB_USER, LORALAB_PASSWORD  login del navegador. Sin contraseña se genera una y se muestra en el log.
#   HF_TOKEN                         token de Hugging Face (opcional), lo leen las descargas directamente.
# El lanzador y los entrenadores escuchan en toda la red con login; en RunPod la URL pública del
# entrenador sale de RUNPOD_POD_ID (https://<pod>-5000.proxy.runpod.net).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${LORALAB_PYTHON:-/opt/venv/bin/python}"

echo "Checking libraries / Comprobando librerias (requirements.txt)..."
"${PY}" -m pip install -r requirements.txt || echo "[WARNING] Could not update the libraries / No se pudieron actualizar las librerias."

"${PY}" - <<'PYEOF'
import os, secrets, sys
sys.path.insert(0, "scripts")
import remote_access

cfg = remote_access.load()
password = os.environ.get("LORALAB_PASSWORD", "")
if not password and not cfg["password_hash"]:
    password = secrets.token_urlsafe(12)
    print("\n" + "=" * 64)
    print(f"  Generated password / Contraseña generada: {password}")
    print("  Set LORALAB_PASSWORD in the template to choose your own / Pon LORALAB_PASSWORD en la plantilla para elegir la tuya.")
    print("=" * 64 + "\n")
pod = os.environ.get("RUNPOD_POD_ID", "")
url = lambda port: f"https://{pod}-{port}.proxy.runpod.net" if pod else ""
cfg = remote_access.save(True, cfg["launcher_port"], cfg["trainer_port"], os.environ.get("LORALAB_USER", "loralab"),
                         password, False, url(cfg["trainer_port"]))
print(f"  User / Usuario: {cfg['user']}")
if pod:
    print(f"  Open / Abre: {url(cfg['launcher_port'])}")
PYEOF

exec "${PY}" scripts/launcher.py
