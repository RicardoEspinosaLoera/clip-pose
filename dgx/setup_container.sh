#!/usr/bin/env bash
# Prepare an EXISTING container (e.g. root@...:/workspace/clip-pose) instead of building dgx/Dockerfile.
#   bash dgx/setup_container.sh                      # if the image already has torch+CUDA (NGC) it is kept and the rest is installed on top;
#                                                    # otherwise a venv .venv/ gets torch 2.11 cu128 (or the newest build available) + requirements
#   USE_SYSTEM_TORCH=1 / FORCE_VENV=1                # force one of the two routes
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
SYS_PY="$(command -v python3)"
echo "   python $($SYS_PY --version 2>&1 | cut -d' ' -f2) | $(uname -m) | pip $($SYS_PY -m pip --version 2>/dev/null | cut -d' ' -f2)"
has_sys_torch=0
$SYS_PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && has_sys_torch=1
if [ -z "${USE_SYSTEM_TORCH:-}" ] && [ -z "${FORCE_VENV:-}" ] && [ "$has_sys_torch" = 1 ]; then
    echo "   the container already has torch with CUDA ($($SYS_PY -c 'import torch; print(torch.__version__)')): using it (FORCE_VENV=1 builds .venv instead)"
    USE_SYSTEM_TORCH=1
fi
install_reqs() {                           # pinned versions first; unpinned fallback for exotic python versions
    $1 -m pip install -q -r "$ROOT/dgx/requirements.txt" && return 0
    echo "   pinned requirements failed for this python; installing the same packages unpinned"
    sed -e 's/==.*//' -e '/^#/d' -e '/^$/d' "$ROOT/dgx/requirements.txt" | xargs $1 -m pip install -q
}
http_code() { command -v curl >/dev/null 2>&1 && curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" || echo "unreachable"; }
if [ -n "${USE_SYSTEM_TORCH:-}" ]; then
    PY="$SYS_PY"
    $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this torch'; print('   keeping system torch', torch.__version__)"
    install_reqs "$PY"
else
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || $SYS_PY -m venv "$ROOT/.venv"
    $PY -m pip install -q --upgrade pip
    install_torch() {
        $PY -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 && return 0
        echo "   torch 2.11.0 cu128 not available for this python/platform; trying the newest cu128 build"
        $PY -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 && return 0
        echo "   cu128 index unusable; trying PyPI (its Linux wheels bundle CUDA 12.x)"
        $PY -m pip install -q torch torchvision && return 0
        return 1
    }
    if ! install_torch; then
        echo "   could not install torch in .venv. Diagnostics:"
        echo "     python $($PY --version 2>&1), $(uname -m)"
        echo "     download.pytorch.org/whl/cu128 -> HTTP $(http_code https://download.pytorch.org/whl/cu128/)"
        echo "     pypi.org/simple/torch          -> HTTP $(http_code https://pypi.org/simple/torch/)"
        echo "   unreachable = no internet/proxy in the container (set HTTPS_PROXY, or use an image that ships torch: nvcr.io/nvidia/pytorch, then USE_SYSTEM_TORCH=1)"
        echo "   HTTP 200 but no wheel = unsupported python version or architecture for torch; use a python 3.10-3.13 image"
        exit 1
    fi
    install_reqs "$PY"
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
