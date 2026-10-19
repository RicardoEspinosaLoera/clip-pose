#!/usr/bin/env bash
# Train / val / test split by backplate (80/10/10) + nested learning-curve subsets of the train split.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
n=$(count_json "$DATA_OUT")
[ "$n" -ge 10 ] || { echo "no samples in $DATA_OUT ($n json): run bash dgx/10_generate.sh first"; exit 1; }
cd "$ROOT/pose_train"
$PY make_splits.py --dir "$DATA_OUT" --seed "${SPLIT_SEED:-0}"
$PY make_subsets.py --dir "$DATA_OUT" --sizes 1250 2500 --seed "${SPLIT_SEED:-0}"
ls -1 "$DATA_OUT/splits"
