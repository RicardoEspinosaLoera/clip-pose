#!/usr/bin/env bash
# Regenerate stems of an Item.obj set that were overwritten by a run with another mesh (22 Oct 2026:
# a stale DATA_OUT=out_v3 in the shell made 10_generate.sh write L-Clip stems into out_v3 for a few
# minutes). Shards are deterministic per seed, so re-running shard k (seed BASE_SEED+k, START_IDX=k*PER)
# for its first n_k stems reproduces the original samples (up to GPU detector non-determinism).
#   bash dgx/restore_overwritten.sh logs/out_v3_overwritten.txt            # dry run: shows what it would do
#   RUN=1 bash dgx/restore_overwritten.sh logs/out_v3_overwritten.txt      # regenerate
# Assumes the original set used PER=500, BASE_SEED=1000, STEM_WIDTH=5 and the final recipe (10_generate.sh
# defaults + MAX_PLACE_ATTEMPTS=60); override with env vars if not. Always renders Item.obj.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LIST="${1:?usage: restore_overwritten.sh <list of overwritten json paths>}"
SET_DIR="${SET_DIR:-$(dirname "$(head -1 "$LIST")")}"
SET_DIR="$(cd "$SET_DIR" && pwd)"
PER="${PER:-500}"; BASE_SEED="${BASE_SEED:-1000}"
export CLIP_MESH="$ROOT/GT_Generator/Item.obj"
: "${KEEP_FLAT:=1}" "${APPEARANCE:=1}" "${APPEARANCE_PRESET:=anodized2}" "${OCCLUDE:=0}" "${MAX_PLACE_ATTEMPTS:=60}"
export KEEP_FLAT APPEARANCE APPEARANCE_PRESET OCCLUDE MAX_PLACE_ATTEMPTS
echo "restore | set $SET_DIR | mesh Item.obj | list $LIST ($(wc -l < "$LIST") stems)"
# stems per shard: stem s (1-based) belongs to shard (s-1)/PER; count contiguous stems from START_IDX+1
declare -A COUNT
while read -r p; do
    s=$(basename "$p" .json); s=$((10#$s))
    k=$(( (s - 1) / PER ))
    COUNT[$k]=$(( ${COUNT[$k]:-0} + 1 ))
done < "$LIST"
cd "$ROOT/GT_Generator"
for k in $(printf '%s\n' "${!COUNT[@]}" | sort -n); do
    n=${COUNT[$k]}; start=$((k * PER))
    echo "  shard $k: regenerate stems $((start + 1))..$((start + n)) (n=$n) seed $((BASE_SEED + k))"
    if [ -n "${RUN:-}" ]; then
        CUDA_VISIBLE_DEVICES=$((k % NGPU)) OUT_DIR="$SET_DIR" N_SAMPLES=$n START_IDX=$start STEM_WIDTH=5 SEED=$((BASE_SEED + k)) \
            nohup $PY -W ignore render_clip_poses_depth.py > "$LOGS/restore_shard$(printf %02d "$k").log" 2>&1 &
    fi
done
if [ -n "${RUN:-}" ]; then
    wait
    echo "regenerated; verify:  grep -L '\"mesh\": \"Item.obj\"' $SET_DIR/[0-9]*.json | wc -l   # stems without the Item.obj tag (legacy ones have no tag)"
    echo "and:  $PY GT_Generator/fix_pose_se3.py --dirs $SET_DIR --dry-run | tail -1"
else
    echo "dry run only; RUN=1 to regenerate"
fi
