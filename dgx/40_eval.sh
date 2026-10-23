#!/usr/bin/env bash
# Evaluate every finished run: synthetic test split (exact 6-DoF GT) and, when the annotated real frames
# exist (real_eval/annotations/$ANNOTATOR/), the real MACS frames (M1 only). Then build the results table.
#   bash dgx/40_eval.sh            ONLY='e0_' ANNOTATOR=ricardo FORCE=1
#   ANN_DIR=$PWD/real_eval_v2/annotations/ricardo REAL_OUT=real_v2 SKIP_SYNTH=1 FORCE=1 bash dgx/40_eval.sh
#       -> real frames of the v2 set (frames taken from the same folder as the annotations), results in
#          <run>/real_v2/ and runs/results_real_v2.md; SKIP_SYNTH=1 leaves the synthetic test untouched.
# The clip mesh for each run is read from its training set ("mesh" tag of the first GT json; legacy = Item.obj).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ANN_DIR="${ANN_DIR:-$ROOT/real_eval/annotations/$ANNOTATOR}"
FRAMES="${FRAMES:-$(dirname "$(dirname "$ANN_DIR")")/frames}"          # frames of the same set as the annotations
REAL_OUT="${REAL_OUT:-real}"                                            # <run>/<REAL_OUT>/ for the real metrics
cd "$ROOT/pose_train"
for run in "$RUNS"/*/; do
    run="${run%/}"; name="$(basename "$run")"
    [ -f "$run/best.pt" ] || continue
    if [ -n "${ONLY:-}" ] && ! echo "$name" | grep -Eq "$ONLY"; then continue; fi
    read -r model suffix data < <($PY -c "import json,sys; c=json.load(open(sys.argv[1]))['cfg'] if 'cfg' in json.load(open(sys.argv[1])) else json.load(open(sys.argv[1])); print(c['model'], c.get('image_suffix','') or '-', c['data'])" "$run/config.json")
    [ "$suffix" = "-" ] && suffix=""
    mesh=$($PY -c "import json,glob,sys; fs=sorted(glob.glob(sys.argv[1]+'/[0-9]*.json')); print(json.load(open(fs[0]))['clip'].get('mesh','Item.obj') if fs else 'Item.obj')" "$data")
    export CLIP_MESH="$ROOT/GT_Generator/$mesh"                          # ClipModel/articulate must use the mesh the run was trained on
    if [ -z "${SKIP_SYNTH:-}" ] && { [ ! -f "$run/test/summary.json" ] || [ -n "${FORCE:-}" ]; }; then
        echo "--- $name: synthetic test ($model, suffix '$suffix', mesh $mesh)"
        $PY -W ignore eval_synth.py --data "$data" --split "$data/splits/test.txt" --ckpt "$run/best.pt" --model "$model" \
            --image-suffix "$suffix" --workers 4 --out "$run/test" | tail -10
    fi
    if [ "$model" = "m1" ] && [ -d "$ANN_DIR" ] && { [ ! -f "$run/$REAL_OUT/summary.json" ] || [ -n "${FORCE:-}" ]; }; then
        echo "--- $name: real frames ($ANN_DIR, frames $FRAMES, mesh $mesh) -> $REAL_OUT"
        $PY -W ignore eval_real.py --frames "$FRAMES" --ann "$ANN_DIR" --ckpt "$run/best.pt" --out "$run/$REAL_OUT" | tail -8
    fi
done
TABLE="$RUNS/results.md"; [ "$REAL_OUT" = "real" ] || TABLE="$RUNS/results_$REAL_OUT.md"
$PY "$ROOT/dgx/collect_results.py" --runs "$RUNS" --out "$TABLE" --real-sub "$REAL_OUT"
echo "results -> $TABLE (+ csv)"
