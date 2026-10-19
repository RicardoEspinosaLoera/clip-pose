"""
refine_backplate_bank.py — second-pass clean of Data/noclips_bank.

The harvest's fast heuristic rejected only ACHROMATIC bright metal, so blue-anodized
titanium clips (chromatic) leaked in. This pass re-scans the bank and QUARANTINES
(moves, non-destructive) any frame that actually contains a clip, using two
complementary detectors:

  * gdino      — GroundingDINO "aneurysm clip" + the titanium/blue metal gate
                 (semantic; catches silver AND blue clips regardless of colour).
  * blue-blob  — a COMPACT bright-blue blob gate (no model): the anodized clip is a
                 small, solid, bright-cyan structure, unlike the thin blue vessel
                 loops / sutures present in almost every frame (rejected by
                 thinness / low solidity / elongation).

Flagged frames move to  <dir>/_flagged_clip/ ; montages of flagged + a kept sample
are written for review. Nothing is deleted — inspect, then delete _flagged_clip if happy.

Usage:
  python refine_backplate_bank.py --dir ../../GT_Generator/Data/noclips_bank --detector both
  python refine_backplate_bank.py --dir <dir> --detector blue          # instant, no download
  python refine_backplate_bank.py --dir <dir> --dry-run                 # flag only, don't move
"""

import os
import sys
import glob
import shutil
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_clip_crops import detect_gdino, load_gdino, metal_gate   # noqa: E402


def blue_clip_blobs(bgr, min_area=0.0007, max_area=0.06, min_side=10,
                    max_aspect=4.5, min_solidity=0.45):
    """True if a COMPACT bright-blue blob (anodized clip) is present. Thin blue
    vessel loops / sutures fail the thinness / aspect / solidity tests."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((H >= 90) & (H <= 132)) & (S >= 95) & (V >= 85)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img_area = bgr.shape[0] * bgr.shape[1]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area * img_area or a > max_area * img_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < min_side:
            continue
        if max(w, h) / max(1.0, min(w, h)) > max_aspect:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and a / hull >= min_solidity:
            return True
    return False


def has_clip(bgr, args):
    if args.detector in ("blue", "both") and blue_clip_blobs(bgr):
        return True, "blue"
    if args.detector in ("gdino", "both"):
        cands = detect_gdino(bgr, args.text_prompt, args.box_threshold,
                             args.text_threshold, topk=4)
        for score, box in cands:
            if score < args.clip_score:
                continue
            # accept the box as a clip if it is titanium-metal (gate) OR bright-blue
            ok_metal, _ = metal_gate(bgr, box, min_metal=args.metal_min)
            x, y, w, h = box
            ok_blue = blue_clip_blobs(bgr[y:y + h, x:x + w]) if w > 4 and h > 4 else False
            if ok_metal or ok_blue:
                return True, "gdino"
    return False, ""


def montage(paths, cols, cell, title):
    paths = list(paths)
    if not paths:
        return np.full((60, cols * cell, 3), 32, np.uint8)
    rows = (len(paths) + cols - 1) // cols
    s = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        r, c = divmod(i, cols)
        s[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(im, (cell, cell))
    cv2.putText(s, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--detector", choices=["gdino", "blue", "both"], default="both")
    ap.add_argument("--clip-score", type=float, default=0.30)
    ap.add_argument("--text-prompt",
                    default="a metallic aneurysm clip. a titanium surgical clip closing a blood vessel.")
    ap.add_argument("--box-threshold", type=float, default=0.28)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--metal-min", type=float, default=0.08)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(HERE, args.dir)
    quar = os.path.join(d, "_flagged_clip")
    if not args.dry_run:
        os.makedirs(quar, exist_ok=True)

    if args.detector in ("gdino", "both"):
        print("Loading GroundingDINO (first run downloads ~700 MB)...")
        load_gdino()

    files = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    print(f"Scanning {len(files)} backplates in {d} (detector={args.detector})")

    flagged, kept, why = [], [], {"blue": 0, "gdino": 0}
    for i, fp in enumerate(files):
        bgr = cv2.imread(fp)
        if bgr is None:
            continue
        hit, tag = has_clip(bgr, args)
        if hit:
            flagged.append(fp); why[tag] += 1
            if not args.dry_run:
                shutil.move(fp, os.path.join(quar, os.path.basename(fp)))
        else:
            kept.append(fp)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  flagged {len(flagged)} (blue {why['blue']}, gdino {why['gdino']})")

    # review montages (read flagged from quarantine if moved)
    flag_show = flagged if args.dry_run else \
        sorted(glob.glob(os.path.join(quar, "*.png")))[:48]
    cv2.imwrite(os.path.join(d, "_flagged_montage.png"),
                montage(flag_show[:48], 8, 150, f"FLAGGED as clip ({len(flagged)})"))
    import random
    random.seed(0)
    cv2.imwrite(os.path.join(d, "_kept_sample.png"),
                montage(random.sample(kept, min(48, len(kept))), 8, 150, f"KEPT sample ({len(kept)})"))

    print(f"\n{'DRY-RUN: would flag' if args.dry_run else 'Quarantined'} "
          f"{len(flagged)} / {len(files)}  (blue {why['blue']}, gdino {why['gdino']}).  "
          f"Kept {len(kept)}.")
    print(f"Review: {d}/_flagged_montage.png  and  _kept_sample.png"
          + ("" if args.dry_run else f"\nFlagged moved to {quar} (delete it once verified)."))


if __name__ == "__main__":
    main()
