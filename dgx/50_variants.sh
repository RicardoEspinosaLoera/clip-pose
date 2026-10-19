#!/usr/bin/env bash
# Appearance ablations of protocol section 7 (E2) on the SAME poses, backplates and seeds as $DATA_OUT:
#   A0 flat render          -> no variant set: the generator stores NNNNN_flat.png (KEEP_FLAT=1); 30_train.sh uses --image-suffix _flat
#   A1 anodized + IBL       -> bash dgx/50_variants.sh a1     (restyle --preset a1_ibl    --no-applier --reuse-seed)
#   A2 + optics             -> bash dgx/50_variants.sh a2     (restyle --preset a2_optics --no-applier --reuse-seed)
#   A3 + video footprint    -> bash dgx/50_variants.sh a3     (restyle --preset anodized2 --no-applier --reuse-seed)
#   A4 + applier            -> = E0, the set itself
#   A5 + SDXL repaint       -> bash dgx/50_variants.sh a5     (needs the SDXL inpainting weights + MACS LoRA, see README section 7)
# Output: ${DATA_OUT}_a<k> with a copy of splits/; 30_train.sh picks them up automatically (runs e2_a<k>).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
what="${1:-a2}"
OUT="${DATA_OUT}_${what}"
restyle_variant() {                       # $1 = preset
    cd "$ROOT/GT_Generator"
    SHARDS="${SHARDS:-8}"
    n=$(count_json "$DATA_OUT"); per=$(( (n + SHARDS - 1) / SHARDS ))
    for k in $(seq 0 $((SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES="" $PY -W ignore restyle_appearance.py --src "$DATA_OUT" --out "$OUT" --preset "$1" --reuse-seed --no-applier \
            --start $((k * per)) --limit $per --montage "$LOGS/variant_${what}_$k.png" > "$LOGS/variant_${what}_$k.log" 2>&1 &
    done
    wait
    cp -r "$DATA_OUT/splits" "$OUT/"
    echo "$what ($1) -> $OUT ($(count_json "$OUT") samples)"
}
case "$what" in
a1) restyle_variant a1_ibl ;;
a2) restyle_variant a2_optics ;;
a3) restyle_variant anodized2 ;;
a5)
    mkdir -p "$OUT"
    $PY - "$DATA_OUT" "$OUT" <<'EOF'
import os, sys, glob, shutil
src, out = sys.argv[1], sys.argv[2]
keep = ("", "_bg", "_mask", "_mask_visible", "_mask_applier", "_flat")
for jp in sorted(glob.glob(os.path.join(src, "[0-9]*.json"))):
    stem = os.path.splitext(os.path.basename(jp))[0]
    shutil.copy2(jp, out)
    for suf in keep:
        p = os.path.join(src, f"{stem}{suf}.png")
        if os.path.exists(p) and not os.path.exists(os.path.join(out, os.path.basename(p))):
            shutil.copy2(p, out)
print("copied", len(glob.glob(os.path.join(out, "[0-9]*.json"))), "samples")
EOF
    cp -r "$DATA_OUT/splits" "$OUT/"
    cd "$ROOT/GT_Generator"
    $PY -W ignore inpaint_applier.py --src "$OUT" --strength 0.50 --lora-scale 0.4 --keep-clean --montage "$LOGS/variant_a5.jpg" 2>&1 | tail -5
    echo "a5 (SDXL repaint of the applier) -> $OUT" ;;
*)  echo "usage: $0 a1|a2|a3|a5"; exit 1 ;;
esac
