#!/usr/bin/env bash
# End-to-end smoke test (~10-20 min): 10 samples in 2 shards -> validation -> splits -> 60-iteration
# training -> synthetic evaluation. Prints the measured seconds per sample so 10_generate.sh can be sized.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SMOKE="$ROOT/GT_Generator/out_smoke"
rm -rf "$SMOKE"
cd "$ROOT/GT_Generator"
t0=$(date +%s)
for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$((k % NGPU)) OUT_DIR="$SMOKE" N_SAMPLES=5 START_IDX=$((k * 5)) STEM_WIDTH=5 SEED=$((1 + k)) KEEP_FLAT=1 \
        $PY -W ignore render_clip_poses_depth.py > "$LOGS/smoke_gen$k.log" 2>&1 &
done
wait
t1=$(date +%s)
n=$(ls "$SMOKE"/[0-9]*.json 2>/dev/null | wc -l)
echo "generation: $n samples in $((t1 - t0)) s with 2 shards  ->  $(( (t1 - t0) * 2 / (n > 0 ? n : 1) )) s/sample/shard"
[ "$n" -ge 8 ] || { echo "SMOKE FAILED: generation (see $LOGS/smoke_gen*.log)"; exit 1; }
$PY fix_pose_se3.py --dirs "$SMOKE" --dry-run | tail -3
$PY measure_pose_fidelity.py --dir "$SMOKE" --n-montage 10 | tail -4
$PY check_tool_clearance.py --dir "$SMOKE" --n-montage 10 | tail -4

cd "$ROOT/pose_train"
$PY make_splits.py --dir "$SMOKE" --seed 0
$PY -W ignore train.py --data "$SMOKE" --model m1 --iters 60 --batch 4 --workers 2 --warmup 10 --val-every 30 --val-batches 1 \
    --out "$RUNS/smoke_m1" | tail -3
$PY -W ignore eval_synth.py --data "$SMOKE" --split "$SMOKE/splits/test.txt" --ckpt "$RUNS/smoke_m1/last.pt" --model m1 --workers 2 \
    --out "$RUNS/smoke_m1/test" | tail -4
echo "SMOKE OK  (gpu: $NGPU, display: ${DISPLAY:-none})"
