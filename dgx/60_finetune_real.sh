#!/usr/bin/env bash
# Option A: fine-tune on the annotated real frames with video-grouped 5-fold cross-validation.
# Compares init synthetic (M1 checkpoint) vs ImageNet, at 25 / 50 / 100 % of the real training frames,
# 3 seeds each, plus the synthetic zero-shot reference. Runs spread over GPUs; table in $RUNS_FT/results_finetune.md
#   ANNOTATOR=ricardo bash dgx/60_finetune_real.sh            (later: ANNOTATOR=consensus)
#   ANN_DIR=$PWD/real_eval_v2/annotations/ricardo ANNOTATOR=v2_ricardo bash dgx/60_finetune_real.sh   (matched set)
#   SYN_CKPT=pose_train/runs/e0_m1_s0/best.pt ITERS=2000 SEEDS="0 1 2" FRACS="0.25 0.5 1.0" GPUS="5 6 7" NOWAIT=1
# Jobs are queued: at most PER_GPU (default 2) fine-tunes run on one GPU at a time. 29 Oct 2026: launching all
# 13 jobs at once ran the GPUs out of memory (cuDNN HOST_ALLOCATION_FAILED; with driver 465 the OOM path of
# torch 2.6 then dies on "undefined symbol nvmlDeviceGetNvLinkRemoteDeviceType" instead of a clean OOM).
# NOWAIT=1 runs the whole queue in the background (driver log: $LOGS/ft_<ANNOTATOR>_queue.log).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if [ -n "${NOWAIT:-}" ] && [ -z "${_FT_QUEUE:-}" ]; then
    _FT_QUEUE=1 NOWAIT= nohup bash "${BASH_SOURCE[0]}" > "$LOGS/ft_${ANNOTATOR}_queue.log" 2>&1 &
    echo "queue running in background (pid $!): tail -f $LOGS/ft_${ANNOTATOR}_queue.log"
    exit 0
fi
ANN_DIR="${ANN_DIR:-$ROOT/real_eval/annotations/$ANNOTATOR}"
FRAMES="${FRAMES:-$(dirname "$(dirname "$ANN_DIR")")/frames}"          # frames of the same set as the annotations
RUNS_FT="${RUNS_FT:-$ROOT/pose_train/runs_ft_$ANNOTATOR}"
ITERS="${ITERS:-2000}"; SEEDS="${SEEDS:-0 1 2}"; FRACS="${FRACS:-0.25 0.5 1.0}"
if [ -z "${SYN_CKPT:-}" ]; then
    SYN_CKPT="$RUNS/e0_m1_s0_amodal/best.pt"; [ -f "$SYN_CKPT" ] || SYN_CKPT="$RUNS/e0_m1_s0/best.pt"
fi
[ -f "$SYN_CKPT" ] || { echo "synthetic checkpoint not found: $SYN_CKPT"; exit 1; }
SYN_CKPT="$(cd "$(dirname "$SYN_CKPT")" && pwd)/$(basename "$SYN_CKPT")"   # absolute: the jobs run from pose_train/
n_ann=$(ls "$ANN_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ' || true)
[ "$n_ann" -ge 20 ] || { echo "too few annotations in $ANN_DIR ($n_ann)"; exit 1; }
read -r -a GPU_LIST <<< "${GPUS:-$(seq -s ' ' 0 $((NGPU - 1)))}"
mkdir -p "$RUNS_FT"
echo "frames $FRAMES | annotations $ANN_DIR ($n_ann) | synthetic ckpt $SYN_CKPT | iters $ITERS | seeds $SEEDS | fracs $FRACS | gpus ${GPU_LIST[*]}"
cd "$ROOT/pose_train"
JOBS=("syn_zeroshot|--init synthetic --ckpt $SYN_CKPT --iters 0 --seed 0")
for s in $SEEDS; do
    for f in $FRACS; do
        p=$(awk -v f="$f" 'BEGIN{printf "%d", f*100+0.5}')
        JOBS+=("syn_f${p}_s$s|--init synthetic --ckpt $SYN_CKPT --frac $f --iters $ITERS --seed $s")
        JOBS+=("imn_f${p}_s$s|--init imagenet --frac $f --iters $ITERS --seed $s")
    done
done
PER_GPU="${PER_GPU:-2}"
NSLOT=$(( ${#GPU_LIST[@]} * PER_GPU ))
declare -a QUEUE=()
for spec in "${JOBS[@]}"; do
    name="${spec%%|*}"
    if [ -f "$RUNS_FT/$name/summary.json" ] && [ -z "${FORCE:-}" ]; then echo "  skip $name"; continue; fi
    QUEUE+=("$spec")
done
echo "${#QUEUE[@]} jobs, $NSLOT slots (${#GPU_LIST[@]} gpus x $PER_GPU)"
FAILS="$RUNS_FT/.failed_$$"; : > "$FAILS"
declare -a SLOT_PIDS=()
for k in $(seq 0 $((NSLOT - 1))); do
    gpu=${GPU_LIST[$((k % ${#GPU_LIST[@]}))]}
    (   # slot k: its share of the queue, one job after the other
        for q in $(seq "$k" "$NSLOT" $(( ${#QUEUE[@]} - 1 ))); do
            spec="${QUEUE[$q]}"; name="${spec%%|*}"; args="${spec#*|}"
            echo "  start $name on gpu $gpu"
            # shellcheck disable=SC2086
            if CUDA_VISIBLE_DEVICES=$gpu $PY -W ignore finetune_real.py --frames "$FRAMES" --ann "$ANN_DIR" $args --workers 2                     --out "$RUNS_FT/$name" > "$LOGS/ft_${ANNOTATOR}_$name.log" 2>&1; then
                echo "  done  $name"
            else
                echo "  FAILED $name -> $LOGS/ft_${ANNOTATOR}_$name.log"; echo "$name" >> "$FAILS"
            fi
        done
    ) &
    SLOT_PIDS+=("$!")
done
for p in "${SLOT_PIDS[@]}"; do wait "$p" || true; done
fail=$(wc -l < "$FAILS" | tr -d ' '); rm -f "$FAILS"
$PY collect_finetune.py --runs "$RUNS_FT" --out "$RUNS_FT/results_finetune.md"
$PY collect_finetune.py --runs "$RUNS_FT" --out "$RUNS_FT/results_finetune_held.md" --group held
echo "queue finished: $fail failed; table $RUNS_FT/results_finetune.md"
[ "$fail" -eq 0 ] || exit 1
