#!/usr/bin/env bash
# Launch the sprint training runs (protocol section 7, reduced plan), one process per run, round-robin
# over the GPUs, all in parallel. Logs in $LOGS/train_<run>.log, checkpoints in $RUNS/<run>/{last,best}.pt.
#   bash dgx/30_train.sh                 # E0 (M1, M2), E1 (1250, 2500), E2 A0 (flat) [+ A1/A2/A3/A5 if 50_variants.sh made their sets]
#   ONLY='e0_' bash dgx/30_train.sh      # regex filter on run names
#   SEEDS='1 2' bash dgx/30_train.sh     # extra E0 seeds (journal version)
#   ITERS=30000 BATCH=16 WORKERS=8 FORCE=1 NOWAIT=1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ITERS="${ITERS:-30000}"; BATCH="${BATCH:-16}"; WORKERS="${WORKERS:-8}"
COMMON="--iters $ITERS --batch $BATCH --workers $WORKERS --val-every 2000 --val-batches 25"

RUNS_LIST=(
    "e0_m1_s0|--model m1 --seed 0"
    "e0_m2_s0|--model m2 --seed 0"
    "e1_m1_1250|--model m1 --seed 0 --train-list $DATA_OUT/splits/train_1250.txt"
    "e1_m1_2500|--model m1 --seed 0 --train-list $DATA_OUT/splits/train_2500.txt"
    "e2_a0_flat|--model m1 --seed 0 --image-suffix _flat"
)
for s in ${SEEDS:-}; do
    RUNS_LIST+=("e0_m1_s$s|--model m1 --seed $s" "e0_m2_s$s|--model m2 --seed $s")
done
for v in a1 a2 a3 a5; do                                  # appearance ladder sets made by 50_variants.sh (A4 = E0)
    [ -f "${DATA_OUT}_$v/splits/train.txt" ] && RUNS_LIST+=("e2_$v|--model m1 --seed 0 --data ${DATA_OUT}_$v")
done

cd "$ROOT/pose_train"
i=0
for spec in "${RUNS_LIST[@]}"; do
    name="${spec%%|*}"; args="${spec#*|}"
    if [ -n "${ONLY:-}" ] && ! echo "$name" | grep -Eq "$ONLY"; then continue; fi
    if [ -f "$RUNS/$name/best.pt" ] && [ -z "${FORCE:-}" ]; then echo "  skip $name (best.pt exists; FORCE=1 to redo)"; continue; fi
    gpu=$((i % NGPU)); i=$((i + 1))
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -W ignore train.py --data "$DATA_OUT" $args $COMMON --out "$RUNS/$name" \
        > "$LOGS/train_$name.log" 2>&1 &
    echo "  launched $name on gpu $gpu (pid $!): train.py --data $DATA_OUT $args $COMMON"
done
[ "$i" -gt 0 ] || { echo "nothing launched"; exit 0; }
echo "$i runs launched; progress: tail -f $LOGS/train_*.log"
[ -n "${NOWAIT:-}" ] && exit 0
wait
echo "training done; next: bash dgx/40_eval.sh"
