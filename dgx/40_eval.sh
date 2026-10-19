#!/usr/bin/env bash
# Evaluate every finished run: synthetic test split (exact 6-DoF GT) and, when the annotated real frames
# exist (real_eval/annotations/$ANNOTATOR/), the real MACS frames (M1 only). Then build the results table.
#   bash dgx/40_eval.sh            ONLY='e0_' ANNOTATOR=ricardo FORCE=1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ANN_DIR="${ANN_DIR:-$ROOT/real_eval/annotations/$ANNOTATOR}"
cd "$ROOT/pose_train"
for run in "$RUNS"/*/; do
    run="${run%/}"; name="$(basename "$run")"
    [ -f "$run/best.pt" ] || continue
    if [ -n "${ONLY:-}" ] && ! echo "$name" | grep -Eq "$ONLY"; then continue; fi
    read -r model suffix data < <($PY -c "import json,sys; c=json.load(open(sys.argv[1]))['cfg'] if 'cfg' in json.load(open(sys.argv[1])) else json.load(open(sys.argv[1])); print(c['model'], c.get('image_suffix','') or '-', c['data'])" "$run/config.json")
    [ "$suffix" = "-" ] && suffix=""
    if [ ! -f "$run/test/summary.json" ] || [ -n "${FORCE:-}" ]; then
        echo "--- $name: synthetic test ($model, suffix '$suffix')"
        $PY -W ignore eval_synth.py --data "$data" --split "$data/splits/test.txt" --ckpt "$run/best.pt" --model "$model" \
            --image-suffix "$suffix" --workers 4 --out "$run/test" | tail -10
    fi
    if [ "$model" = "m1" ] && [ -d "$ANN_DIR" ] && { [ ! -f "$run/real/summary.json" ] || [ -n "${FORCE:-}" ]; }; then
        echo "--- $name: real frames ($ANN_DIR)"
        $PY -W ignore eval_real.py --ann "$ANN_DIR" --ckpt "$run/best.pt" --out "$run/real" | tail -8
    fi
done
$PY "$ROOT/dgx/collect_results.py" --runs "$RUNS" --out "$RUNS/results.md"
echo "results -> $RUNS/results.md (+ results.csv)"
