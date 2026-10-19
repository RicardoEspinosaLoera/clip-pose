#!/usr/bin/env bash
# Prepare an EXISTING container (e.g. root@...:/workspace/clip-pose) instead of building dgx/Dockerfile.
#   bash dgx/setup_container.sh                      # venv .venv/ with the exact pinned stack (torch 2.11 cu128, ~1.5 GB download)
#   USE_SYSTEM_TORCH=1 bash dgx/setup_container.sh   # keep the container's own torch (NGC image); install the rest on top
# Afterwards every dgx/*.sh script picks .venv/bin/python automatically (dgx/env.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== system packages: Xvfb + Mesa (head-less VTK), procps"
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null
    apt-get install -y -qq --no-install-recommends xvfb libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libxrender1 libxext6 libsm6 \
        libx11-6 libglib2.0-0 libgomp1 procps >/dev/null
else
    echo "   no apt-get here: install xvfb + mesa some other way (or use the vtk-osmesa wheel, README section 7)"
fi

echo "== python environment"
if [ -n "${USE_SYSTEM_TORCH:-}" ]; then
    PY="$(command -v python3)"
    $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this torch'; print('   keeping system torch', torch.__version__)"
    $PY -m pip install -q -r "$ROOT/dgx/requirements.txt"
else
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || python3 -m venv "$ROOT/.venv"
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    $PY -m pip install -q -r "$ROOT/dgx/requirements.txt"
fi

echo "== checks"
$PY - <<'EOF'
import torch, numpy, cv2, pyvista, vtk, scipy, transformers, diffusers
print(f"   torch {torch.__version__} | cuda {torch.cuda.is_available()} | gpus {torch.cuda.device_count()} | numpy {numpy.__version__} | "
      f"opencv {cv2.__version__} | pyvista {pyvista.__version__} | vtk {vtk.vtkVersion.GetVTKVersion()} | transformers {transformers.__version__}")
EOF
export PY
# shellcheck disable=SC1091
source "$ROOT/dgx/env.sh"
$PY - <<'EOF'
import pyvista as pv
pl = pv.Plotter(off_screen=True, window_size=[320, 240])
pl.add_mesh(pv.Sphere(), pbr=True, metallic=0.8, roughness=0.3)
img = pl.screenshot(return_img=True)
pl.close()
print("   head-less VTK render OK", img.shape)
EOF
shm=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
cpus=$(nproc)
echo "   /dev/shm ${shm:-?} MB (DataLoader workers want >= 8 GB; if smaller: restart the container with --ipc=host, or train with WORKERS=0)"
echo "   cpus $cpus, gpus $NGPU  ->  suggested: SHARDS=$(( cpus / 8 > 1 ? cpus / 8 : 1 )) LP_NUM_THREADS=8 for 10_generate.sh"
echo "   models offline: $([ -d "$HF_HOME/hub/models--IDEA-Research--grounding-dino-tiny" ] && echo GroundingDINO ok || echo 'GroundingDINO MISSING in models/hf')"
echo "setup done. next: bash dgx/00_smoke.sh"
