# RealAC pipeline image for an NVIDIA DGX (A100/H100, driver >= 525).
# Build:  docker build -t realac:latest dgx/
# Run:    docker run --gpus all --shm-size=32g -v $PWD:/work -w /work -it realac:latest bash
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    PYVISTA_OFF_SCREEN=true MPLBACKEND=Agg

# python 3.10 (Ubuntu 22.04 default) + virtual X display + Mesa software OpenGL for VTK
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python-is-python3 git ca-certificates \
        xvfb libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libxrender1 libxext6 libsm6 libx11-6 \
        libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# same torch build as the development laptop (2.11.0 + CUDA 12.8)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

WORKDIR /work
CMD ["bash"]
