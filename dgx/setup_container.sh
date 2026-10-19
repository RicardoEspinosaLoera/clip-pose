#!/usr/bin/env bash
# Prepare an EXISTING container (e.g. root@...:/workspace/clip-pose) instead of building dgx/Dockerfile.
#   bash dgx/setup_container.sh
#     - image already has torch >= 2.3 with CUDA and python >= 3.9 (NGC)  -> kept, the rest is installed on top
#     - python >= 3.9 without a usable torch                              -> venv .venv/ + torch + requirements
#     - python < 3.9 (old images)                                         -> Miniforge in /opt/miniforge, env "realac" (python 3.10) + torch + requirements
#     torch build follows the HOST DRIVER: >= 525 -> torch 2.11 cu128; 450-524 -> torch 2.6.0 cu118 (last cu118 build; CUDA 11 minor-version compatibility)
#   USE_SYSTEM_TORCH=1 / FORCE_VENV=1 / MINIFORGE=1  # force one route
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
# torch build by host driver: CUDA 12.x wheels need driver >= 525; older data-centre drivers (>= 450) run CUDA 11.8 wheels
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$DRV" ] && [ "$DRV" -lt 525 ]; then
    [ "$DRV" -ge 450 ] || { echo "   driver $DRV is too old for any supported torch build (needs >= 450)"; exit 1; }
    TORCH_SPEC="torch==2.6.0 torchvision==0.21.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu118"; TORCH_LOOSE="torch==2.6.* torchvision"   # 2.6.0 = last cu118 build
else
    TORCH_SPEC="torch==2.11.0 torchvision==0.26.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"; TORCH_LOOSE="torch torchvision"
fi
echo "   driver ${DRV:-?} -> $TORCH_SPEC from $TORCH_INDEX"
py_ok() { "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; }
torch_ok() { "$1" -c "import torch, sys; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if torch.cuda.is_available() and v >= (2, 3) else 1)" 2>/dev/null; }
if [ -z "${USE_SYSTEM_TORCH:-}" ] && [ -z "${FORCE_VENV:-}" ] && [ -z "${MINIFORGE:-}" ] && py_ok "$SYS_PY" && torch_ok "$SYS_PY"; then
    echo "   the container already has torch >= 2.3 with CUDA ($($SYS_PY -c 'import torch; print(torch.__version__)')): using it (FORCE_VENV=1 builds .venv instead)"
    USE_SYSTEM_TORCH=1
fi
if [ -z "${USE_SYSTEM_TORCH:-}" ] && [ -z "${MINIFORGE:-}" ] && ! py_ok "$SYS_PY"; then
    echo "   python < 3.9: installing Miniforge with python 3.10 (MINIFORGE=1 route)"
    MINIFORGE=1
fi
install_reqs() {                           # pinned versions first; unpinned fallback for exotic python versions
    "$1" -m pip install -q -r "$ROOT/dgx/requirements.txt" && return 0
    echo "   pinned requirements failed for this python; installing the same packages unpinned"
    sed -e 's/==.*//' -e '/^#/d' -e '/^$/d' "$ROOT/dgx/requirements.txt" | xargs "$1" -m pip install -q
}
http_code() { command -v curl >/dev/null 2>&1 && curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" || echo "unreachable"; }
install_torch() {                          # $1 = python
    # shellcheck disable=SC2086
    "$1" -m pip install -q ${2:-} $TORCH_SPEC --index-url "$TORCH_INDEX" && return 0
    echo "   $TORCH_SPEC not available for this python/platform; trying the newest build on that index"
    # shellcheck disable=SC2086
    "$1" -m pip install -q $TORCH_LOOSE --index-url "$TORCH_INDEX" && return 0
    echo "   could not install torch. Diagnostics:"
    echo "     python $("$1" --version 2>&1), $(uname -m), driver ${DRV:-?}"
    echo "     $TORCH_INDEX -> HTTP $(http_code "$TORCH_INDEX/")"
    echo "     pypi.org/simple/torch -> HTTP $(http_code https://pypi.org/simple/torch/)"
    echo "   unreachable = no internet/proxy in the container (set HTTPS_PROXY); HTTP 200 but no wheel = unsupported python version or architecture"
    return 1
}
if [ -n "${USE_SYSTEM_TORCH:-}" ]; then
    PY="$SYS_PY"
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this torch'; print('   keeping system torch', torch.__version__)"
    install_reqs "$PY"
elif [ -n "${MINIFORGE:-}" ]; then
    MF="${MF:-/opt/miniforge}"
    if [ ! -x "$MF/bin/conda" ]; then
        curl -L -s -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash /tmp/miniforge.sh -b -p "$MF" >/dev/null
    fi
    [ -x "$MF/envs/realac/bin/python" ] || "$MF/bin/conda" create -y -q -n realac python=3.10 >/dev/null
    PY="$MF/envs/realac/bin/python"
    echo "   using $PY ($("$PY" --version 2>&1))"
    "$PY" -m pip install -q --upgrade pip
    install_torch "$PY" || exit 1
    install_reqs "$PY"
else
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || "$SYS_PY" -m venv "$ROOT/.venv"
    "$PY" -m pip install -q --upgrade pip
    install_torch "$PY" || exit 1
    install_reqs "$PY"
fi
torch_ok "$PY" || { echo "   torch was replaced by a dependency; restoring $TORCH_SPEC"; install_torch "$PY" --force-reinstall || exit 1; }

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
