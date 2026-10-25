# -*- coding: utf-8 -*-
"""
sync_pinned.py — Reinstala los paquetes de requirements.txt fijados por URL ("nombre @ url") cuando la
instalación viene de otra URL. Lo llaman el actualizador y el arranque de Docker después de
"pip install -r requirements.txt".

pip compara solo la versión: diffusers de dos commits distintos de main se llama igual
(0.41.0.dev0), así que da el paquete por instalado aunque requirements.txt apunte a otro commit.
Aquí se compara la URL de origen (direct_url.json) y, si no coincide, se reinstala solo ese paquete
(--no-deps: sus dependencias ya las ha resuelto el paso anterior).

Reinstalls the requirements.txt packages pinned by URL when the installed copy came from another URL:
pip only compares versions, and two diffusers main commits share the same one.
"""
import json
import os
import subprocess
import sys
from importlib import metadata

REQUIREMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")


def installed_url(name):
    try:
        text = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(text).get("url") if text else None


def main():
    with open(REQUIREMENTS, "r", encoding="utf-8") as f:
        pinned = [line.split("#")[0].strip() for line in f if " @ " in line.split("#")[0]]
    code = 0
    for req in pinned:
        name, url = (part.strip() for part in req.split(" @ ", 1))
        if installed_url(name) == url:
            continue
        print(f"{name}: reinstalando desde / reinstalling from {url}", flush=True)
        code |= subprocess.call([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", req])
    return code


if __name__ == "__main__":
    sys.exit(main())
