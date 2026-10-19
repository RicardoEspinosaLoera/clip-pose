#!/usr/bin/env bash
# Common environment for every dgx/*.sh script: paths, model caches, head-less rendering, GPU count.
# Every variable can be overridden from the shell (e.g. DATA_OUT=/raid/out_v3 bash dgx/10_generate.sh).

export ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # bundle root (contains GT_Generator/, pose_train/)
if [ -z "${PY:-}" ]; then                                                      # venv made by setup_container.sh, else system python
    if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python || command -v python3)"; fi
fi
export PY
export DATA_OUT="${DATA_OUT:-$ROOT/GT_Generator/out_v3}"                     # the 5000-sample training set
export RUNS="${RUNS:-$ROOT/pose_train/runs}"                                 # training outputs
export LOGS="${LOGS:-$ROOT/logs}"
export ANNOTATOR="${ANNOTATOR:-ricardo}"                                     # real_eval/annotations/<ANNOTATOR>/

# model caches shipped in the bundle (models/hf = GroundingDINO; models/torch = ImageNet ResNets)
export HF_HOME="${HF_HOME:-$ROOT/models/hf}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/models/torch}"
if [ -z "${HF_HUB_OFFLINE:-}" ]; then                     # bundled cache present -> offline; else download into models/hf (needs internet)
    if [ -d "$HF_HOME/hub/models--IDEA-Research--grounding-dino-tiny" ]; then HF_HUB_OFFLINE=1; else HF_HUB_OFFLINE=0; fi
fi
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

# head-less rendering: VTK draws through a virtual X display with Mesa (llvmpipe, CPU)
export PYVISTA_OFF_SCREEN=true MPLBACKEND=Agg PYTHONUNBUFFERED=1
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"            # llvmpipe threads per render process
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1; then
        if ! { command -v pgrep >/dev/null 2>&1 && pgrep -f "Xvfb :99" >/dev/null 2>&1; }; then
            Xvfb :99 -screen 0 1920x1080x24 +extension GLX >/dev/null 2>&1 &
            sleep 2
        fi
        export DISPLAY=:99
    else
        echo "[env] WARNING: Xvfb not found; the python scripts will try pyvista.start_xvfb() (needs xvfb installed)" >&2
    fi
fi

NGPU_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
export NGPU="${NGPU:-${NGPU_DETECTED:-1}}"
[ "$NGPU" -ge 1 ] || export NGPU=1
mkdir -p "$RUNS" "$LOGS"
