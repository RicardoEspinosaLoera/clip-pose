#!/usr/bin/env bash
# Option A: fine-tune on the annotated real frames with video-grouped 5-fold cross-validation.
# Compares init synthetic (M1 checkpoint) vs ImageNet, at 25 / 50 / 100 % of the real training frames,
# 3 seeds each, plus the synthetic zero-shot reference. Runs spread over GPUs; table in $RUNS_FT/results_finetune.md
#   ANNOTATOR=ricardo bash dgx/60_finetune_real.sh            (later: ANNOTATOR=consensus)
#   SYN_CKPT=pose_train/runs/e0_m1_s0/best.pt ITERS=2000 SEEDS="0 1 2" FRACS="0.25 0.5 1.0" GPUS="5 6 7" NOWAIT=1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ANN_DIR="${ANN_DIR:-$ROOT/real_eval/annotations/$ANNOTATOR}"
RUNS_FT="${RUNS_FT:-$ROOT/pose_train/runs_ft_$ANNOTATOR}"
ITERS="${ITERS:-2000}"; SEEDS="${SEEDS:-0 1 2}"; FRACS="${FRACS:-0.25 0.5 1.0}"
if [ -z "${SYN_CKPT:-}" ]; then
    SYN_CKPT="$RUNS/e0_m1_s0_amodal/best.pt"; [ -f "$SYN_CKPT" ] || SYN_CKPT="$RUNS/e0_m1_s0/best.pt"
fi
[ -f "$SYN_CKPT" ] || { echo "synthetic checkpoint not found: $SYN_CKPT"; exit 1; }
n_ann=$(ls "$ANN_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ' || true)
[ "$n_ann" -ge 20 ] || { echo "too few annotations in $ANN_DIR ($n_ann)"; exit 1; }
read -r -a GPU_LIST <<< "${GPUS:-$(seq -s ' ' 0 $((NGPU - 1)))}"
mkdir -p "$RUNS_FT"
echo "annotations $ANN_DIR ($n_ann) | synthetic ckpt $SYN_CKPT | iters $ITERS | seeds $SEEDS | fracs $FRACS | gpus ${GPU_LIST[*]}"
cd "$ROOT/pose_train"
JOBS=("syn_zeroshot|--init synthetic --ckpt $SYN_CKPT --iters 0 --seed 0")
for s in $SEEDS; do
    for f in $FRACS; do
        p=$(awk -v f="$f" 'BEGIN{printf "%d", f*100+0.5}')
        JOBS+=("syn_f${p}_s$s|--init synthetic --ckpt $SYN_CKPT --frac $f --iters $ITERS --seed $s")
        JOBS+=("imn_f${p}_s$s|--init imagenet --frac $f --iters $ITERS --seed $s")
    done
done
i=0; declare -a PIDS=() NAMES=()
for spec in "${JOBS[@]}"; do
    name="${spec%%|*}"; args="${spec#*|}"
    if [ -f "$RUNS_FT/$name/summary.json" ] && [ -z "${FORCE:-}" ]; then echo "  skip $name"; continue; fi
    gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}; i=$((i + 1))
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -W ignore finetune_real.py --ann "$ANN_DIR" $args --workers 2 --out "$RUNS_FT/$name" \
        > "$LOGS/ft_${ANNOTATOR}_$name.log" 2>&1 &
    PIDS+=("$!"); NAMES+=("$name")
    echo "  launched $name on gpu $gpu"
done
[ "$i" -gt 0 ] || { echo "nothing to launch"; }
if [ -z "${NOWAIT:-}" ]; then
    fail=0
    for j in "${!PIDS[@]}"; do wait "${PIDS[$j]}" || { echo "  FAILED ${NAMES[$j]} -> $LOGS/ft_${ANNOTATOR}_${NAMES[$j]}.log"; fail=$((fail + 1)); }; done
    $PY collect_finetune.py --runs "$RUNS_FT" --out "$RUNS_FT/results_finetune.md"
    $PY collect_finetune.py --runs "$RUNS_FT" --out "$RUNS_FT/results_finetune_held.md" --group held
    [ "$fail" -eq 0 ] || exit 1
else
    echo "running in background; when done: $PY pose_train/collect_finetune.py --runs $RUNS_FT --out $RUNS_FT/results_finetune.md"
fi
