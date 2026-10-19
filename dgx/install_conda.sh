#!/usr/bin/env bash
# Bare-metal alternative to the Dockerfile (conda/miniforge already on the DGX).
# Usage: bash dgx/install_conda.sh [env-name]
# System packages still needed (ask the admin if missing): xvfb libgl1-mesa-dri libglx-mesa0 libegl1
set -euo pipefail
ENV_NAME="${1:-realac}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda create -y -n "$ENV_NAME" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$HERE/requirements.txt"
python - <<'EOF'
import torch, pyvista, vtk, transformers, diffusers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| pyvista", pyvista.__version__, "| vtk", vtk.vtkVersion.GetVTKVersion())
EOF
echo "env '$ENV_NAME' ready. If 'nvidia-smi' reports a driver older than 525, reinstall torch with --index-url https://download.pytorch.org/whl/cu126"
