"""
eval_appearance.py — how close does a generated set look to real MACS clips?

For a GT output directory (render_clip_poses_depth.py, with or without the in-line
APPEARANCE pass) this writes a review montage and prints colour/size statistics of the
clip pixels against real MACS clip crops (Inpaint_Realism/lora/clean_clips2 + masks).

  montage  : <dir>/_eval_montage.png  — per sample: [flat render if NNN_flat.png exists |]
             generated zoom | random real crop
  metrics  : clip-pixel HSV Wasserstein-1 to real clip pixels (H, S, V), internal contrast
             (std of V), mean BGR, clip mask area %, jaw-angle and defocus distributions

Usage
  python eval_appearance.py --dir out_pilot_v2 --n 24
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restyle_appearance import zoom_cell, real_crop_cell, clip_pixels, real_clip_pixels, hsv_w1

HERE = os.path.dirname(os.path.abspath(__file__))


def vstd(px):
    return float(cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[..., 2].std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_pilot_v2")
    ap.add_argument("--n", type=int, default=24, help="samples in the montage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", type=int, default=224)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(HERE, a.dir)
    jsons = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
    if not jsons:
        print(f"no samples in {d}")
        return
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(jsons), size=min(a.n, len(jsons)), replace=False))

    pix, flat_pix, areas, jaws, sigmas, rows = [], [], [], [], [], []
    for i, jp in enumerate(jsons):
        meta = json.load(open(jp))
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = cv2.imread(os.path.join(d, meta["image"].get("file_rgb", f"{stem}.png")))
        mask = cv2.imread(os.path.join(d, meta["image"].get("file_mask", f"{stem}_mask.png")), 0)
        if img is None or mask is None:
            continue
        pix.append(clip_pixels(img, mask))
        areas.append((mask > 127).mean() * 100)
        jaws.append(float(meta["clip"].get("jaw_angle_deg", 0.0)))
        if "appearance" in meta:
            sigmas.append(float(meta["appearance"].get("defocus_sigma", np.nan)))
        flat_p = os.path.join(d, f"{stem}_flat.png")
        flat = cv2.imread(flat_p) if os.path.exists(flat_p) else None
        if flat is not None:
            flat_pix.append(clip_pixels(flat, mask))
        if i in pick:
            cells = []
            if flat is not None:
                cells.append(zoom_cell(flat, mask, a.cell, label=f"{stem} flat"))
            lab = f"{stem} {meta.get('appearance', {}).get('preset', 'render')}"
            cells.append(zoom_cell(img, mask, a.cell, label=lab))
            cells.append(real_crop_cell(rng, a.cell))
            rows.append(np.hstack(cells))

    per_row = 3 if rows and rows[0].shape[1] <= 2 * a.cell else 2
    while len(rows) % per_row:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[j:j + per_row]) for j in range(0, len(rows), per_row)])
    out_png = os.path.join(d, "_eval_montage.png")
    cv2.imwrite(out_png, grid)

    real_pix, _ = real_clip_pixels(seed=a.seed)
    P = np.concatenate(pix)
    w = hsv_w1(P, real_pix)
    print(f"{os.path.basename(os.path.normpath(d))}: n={len(pix)} samples")
    print("clip-pixel HSV W1 distance to REAL clip pixels (lower = closer):")
    if flat_pix:
        F = np.concatenate(flat_pix)
        wf = hsv_w1(F, real_pix)
        print(f"  flat render : H {wf[0]:6.1f}  S {wf[1]:6.1f}  V {wf[2]:6.1f}   mean BGR {F.mean(0).round(0)}  V-std {vstd(F):5.1f}")
    print(f"  generated   : H {w[0]:6.1f}  S {w[1]:6.1f}  V {w[2]:6.1f}   mean BGR {P.mean(0).round(0)}  V-std {vstd(P):5.1f}")
    print(f"  REAL        : mean BGR {real_pix.mean(0).round(0)}  V-std {vstd(real_pix):5.1f}")
    areas, jaws = np.array(areas), np.array(jaws)
    print(f"clip mask area %: median {np.median(areas):.2f}  min {areas.min():.2f}  max {areas.max():.2f}")
    print(f"jaw angle deg: closed {np.mean(jaws == 0) * 100:.0f}%  open median {np.median(jaws[jaws > 0]) if (jaws > 0).any() else 0:.1f}")
    if sigmas:
        s = np.array(sigmas)
        print(f"defocus sigma: median {np.nanmedian(s):.2f}  range [{np.nanmin(s):.2f}, {np.nanmax(s):.2f}]")
    print(f"montage -> {out_png}")


if __name__ == "__main__":
    main()
