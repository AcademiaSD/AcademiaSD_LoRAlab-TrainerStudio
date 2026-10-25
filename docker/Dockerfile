# AcademiaSD LoRAlab Trainer Studio — imagen para RunPod (y cualquier Docker con GPU NVIDIA).
# Solo lleva el entorno: Python 3.13, PyTorch CUDA 13.0 y las librerías de requirements.txt. El código
# se clona o actualiza en /workspace en cada arranque (bootstrap.sh), así que las actualizaciones normales
# no necesitan una imagen nueva. La GPU necesita driver 580 o superior (CUDA 13).
FROM nvidia/cuda:13.0.1-base-ubuntu24.04
LABEL org.opencontainers.image.source=https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    LORALAB_PYTHON=/opt/venv/bin/python

# build-essential: Triton compila un módulo C la primera vez que lanza un kernel (torch 2.14 lo usa en
# operaciones normales, y SageAttention también); la imagen base de CUDA no trae compilador.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ffmpeg ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# uv solo descarga Python 3.13; las librerías se instalan con pip, igual que los instaladores.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv --seed --python 3.13 /opt/venv

COPY requirements.txt /opt/requirements.txt
RUN /opt/venv/bin/python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
 && /opt/venv/bin/python -m pip install --no-cache-dir -r /opt/requirements.txt

COPY docker/bootstrap.sh /usr/local/bin/loralab-bootstrap
RUN chmod +x /usr/local/bin/loralab-bootstrap

WORKDIR /workspace
EXPOSE 4990 5000
CMD ["loralab-bootstrap"]
