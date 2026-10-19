#!/usr/bin/env bash
# Generate the training set with the final recipe, sharded over CPU cores (rendering) and GPUs
# (GroundingDINO tool avoidance), all shards writing disjoint stems into ONE directory, then validate.
#   N_TOTAL=5000 SHARDS=16 BASE_SEED=1000 DATA_OUT=... bash dgx/10_generate.sh
# Stems: 00001..0N000 (STEM_WIDTH=5). Shard k owns stems k*PER+1 .. (k+1)*PER and seed BASE_SEED+k.
# Re-running with the same arguments rewrites the same stems (same seeds -> same poses).
# Recipe knobs below are the generator defaults (final recipe, 2026-10-19); override them only for the
# journal ablations E3 (CLIP_FRAC_MAX=0.003 = old size cap) and E4 (USE_DINO_AVOID=0 BLADE_PROBE=0 = old filter),
# always with a different DATA_OUT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N_TOTAL="${N_TOTAL:-5000}"
SHARDS="${SHARDS:-16}"
BASE_SEED="${BASE_SEED:-1000}"
: "${KEEP_FLAT:=1}" "${APPEARANCE:=1}" "${APPEARANCE_PRESET:=anodized2}" "${OCCLUDE:=0}"
: "${TOOL_PROB:=0.40}" "${JAW_MIN_OPEN_TOOL:=8}" "${JAW_FREE_CLOSED:=1}"
: "${CLIP_FRAC_MIN:=0.006}" "${CLIP_FRAC_MAX:=0.015}" "${BBOX_RATIO_MAX:=5}"
: "${AVOID_OVERLAP_MAX:=0.0}" "${AVOID_MARGIN_PX:=25}" "${BLADE_PROBE:=1}" "${BLADE_PROBE_FRAC:=1.2}" "${BLADE_PROBE_HALFW:=6}"
: "${USE_DINO_AVOID:=1}" "${DINO_BOX_THRESH:=0.20}" "${DINO_DILATE:=25}" "${FOCUS_MIN_PCT:=0.30}"
export KEEP_FLAT APPEARANCE APPEARANCE_PRESET OCCLUDE TOOL_PROB JAW_MIN_OPEN_TOOL JAW_FREE_CLOSED CLIP_FRAC_MIN CLIP_FRAC_MAX BBOX_RATIO_MAX \
       AVOID_OVERLAP_MAX AVOID_MARGIN_PX BLADE_PROBE BLADE_PROBE_FRAC BLADE_PROBE_HALFW USE_DINO_AVOID DINO_BOX_THRESH DINO_DILATE FOCUS_MIN_PCT
PER=$(( (N_TOTAL + SHARDS - 1) / SHARDS ))
mkdir -p "$DATA_OUT"
cd "$ROOT/GT_Generator"
echo "generate | $N_TOTAL samples | $SHARDS shards x $PER | seeds $BASE_SEED.. | GPUs $NGPU | out $DATA_OUT"
env | grep -E "^(KEEP_FLAT|APPEARANCE|OCCLUDE|TOOL_PROB|JAW_|CLIP_FRAC|BBOX_RATIO|AVOID_|BLADE_|USE_DINO|DINO_|FOCUS_MIN)" | sort | tr '\n' ' '; echo
t0=$(date +%s)
for k in $(seq 0 $((SHARDS - 1))); do
    start=$((k * PER)); n=$PER
    [ $((start + n)) -gt "$N_TOTAL" ] && n=$((N_TOTAL - start))
    [ "$n" -le 0 ] && break
    CUDA_VISIBLE_DEVICES=$((k % NGPU)) OUT_DIR="$DATA_OUT" N_SAMPLES=$n START_IDX=$start STEM_WIDTH=5 SEED=$((BASE_SEED + k)) \
        nohup $PY -W ignore render_clip_poses_depth.py > "$LOGS/gen_shard$(printf %02d "$k").log" 2>&1 &
    echo "  shard $k: stems $((start + 1))..$((start + n)) seed $((BASE_SEED + k)) gpu $((k % NGPU)) pid $!"
done
wait
t1=$(date +%s)
n=$(ls "$DATA_OUT"/[0-9]*.json | wc -l)
echo "saved $n / $N_TOTAL samples in $(( (t1 - t0) / 60 )) min"
grep -h "WARN\|Traceback" "$LOGS"/gen_shard*.log | sort | uniq -c | head -20 || true

echo "--- validation (protocol section 2): GT reprojection, image-level pose fidelity, tool clearance, domain gap"
$PY fix_pose_se3.py --dirs "$DATA_OUT" --dry-run | tail -4
$PY measure_pose_fidelity.py --dir "$DATA_OUT" --n-montage 30 | tail -8
$PY check_tool_clearance.py --dir "$DATA_OUT" --n-montage 50 | tail -8
if [ -d "$ROOT/Inpaint_Realism/lora/clean_clips2" ]; then
    $PY domain_gap.py --dirs "$DATA_OUT" --n 300 | tee "$LOGS/domain_gap_$(basename "$DATA_OUT").log" | tail -8
else
    echo "domain gap skipped: Inpaint_Realism/lora/clean_clips2 (real clip crops) not present"
fi
echo "generation done -> $DATA_OUT  (next: bash dgx/20_splits.sh)"
