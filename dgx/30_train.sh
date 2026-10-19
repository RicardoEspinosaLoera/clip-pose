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
if [ ! -f "$DATA_OUT/splits/train.txt" ] || [ ! -f "$DATA_OUT/splits/val.txt" ]; then
    echo "no training data: $DATA_OUT/splits/train.txt missing. Run bash dgx/10_generate.sh and bash dgx/20_splits.sh first (or set DATA_OUT)."
    exit 1
fi
echo "data: $DATA_OUT ($(wc -l < "$DATA_OUT/splits/train.txt") train / $(wc -l < "$DATA_OUT/splits/val.txt") val images)"
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
declare -a PIDS=() NAMES=()
for spec in "${RUNS_LIST[@]}"; do
    name="${spec%%|*}"; args="${spec#*|}"
    if [ -n "${ONLY:-}" ] && ! echo "$name" | grep -Eq "$ONLY"; then continue; fi
    if [ -f "$RUNS/$name/best.pt" ] && [ -z "${FORCE:-}" ]; then echo "  skip $name (best.pt exists; FORCE=1 to redo)"; continue; fi
    gpu=$((i % NGPU)); i=$((i + 1))
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -W ignore train.py --data "$DATA_OUT" $args $COMMON --out "$RUNS/$name" \
        > "$LOGS/train_$name.log" 2>&1 &
    PIDS+=("$!"); NAMES+=("$name")
    echo "  launched $name on gpu $gpu (pid $!): train.py --data $DATA_OUT $args $COMMON"
done
[ "$i" -gt 0 ] || { echo "nothing launched"; exit 0; }
echo "$i runs launched; progress: tail -f $LOGS/train_*.log"
[ -n "${NOWAIT:-}" ] && exit 0
fail=0
for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then
        echo "  ok      ${NAMES[$j]}"
    else
        echo "  FAILED  ${NAMES[$j]}  -> tail -5 $LOGS/train_${NAMES[$j]}.log"; fail=$((fail + 1))
    fi
done
[ "$fail" -eq 0 ] && echo "training done; next: bash dgx/40_eval.sh" || { echo "$fail run(s) failed"; exit 1; }
