"""
analyze_real_frames.py -- break the real-frame errors of one or more runs down by clip size, direction flips and video.
Reads <run>/real_v2/per_frame.csv (eval_real.py). For each run:
  * axis median directed vs undirected (min(a, 180-a)): how much of the axis error is a tip/back flip
  * by annotated clip length L (px): err/L, error in px, axis -- end-on or tiny clips make err/L explode
  * per video: err/L and axis medians (are failures spread or concentrated?)
  * 'collapse': predicted points all within 0.15 L of each other is not in the csv, so it is approximated by
    tip and back errors both ~0.5 L (prediction near the clip centre)
Usage: python analyze_real_frames.py runs/e0_m1_s0_lclip_aug runs/e0_m1_s0 [--sub real_v2]
"""
import os
import sys
import csv
import argparse
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
    return dict(L=np.array([f(r, "L_px") for r in rows]), ax=np.array([f(r, "axis_deg") for r in rows]),
                ef=np.array([f(r, "mean_frac") for r in rows]), tip=np.array([f(r, "tip_frac") for r in rows]),
                back=np.array([f(r, "back_frac") for r in rows]), vid=np.array([r["video"] for r in rows]),
                occ=np.array([f(r, "attr_occluded") for r in rows]))


def line(name, sel, d):
    if not sel.any():
        return f"  {name:16s}    0"
    und = np.minimum(d["ax"][sel], 180 - d["ax"][sel])
    return (f"  {name:16s} {sel.sum():4d}  err/L {np.nanmedian(d['ef'][sel]):5.2f}  err px {np.nanmedian(d['ef'][sel] * d['L'][sel]):6.0f}"
            f"  axis {np.nanmedian(d['ax'][sel]):5.1f}  undirected {np.nanmedian(und):5.1f}  axis<20 {np.mean(d['ax'][sel] < 20) * 100:4.0f}%"
            f"  flipped(>90) {np.mean(d['ax'][sel] > 90) * 100:4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sub", default="real_v2")
    a = ap.parse_args()
    for run in a.runs:
        p = os.path.join(run, a.sub, "per_frame.csv")
        if not os.path.exists(p):
            print(f"{run}: no {p}")
            continue
        d = load(p)
        print(f"== {os.path.basename(os.path.normpath(run))}  ({len(d['L'])} frames)")
        print(line("all", np.ones(len(d["L"]), bool), d))
        for lo, hi in ((0, 60), (60, 150), (150, 300), (300, 5000)):
            print(line(f"L {lo}-{hi} px", (d["L"] >= lo) & (d["L"] < hi), d))
        print(line("occluded", d["occ"] == 1, d))
        print(line("not occluded", d["occ"] == 0, d))
        print(line("L>=60 not occl", (d["L"] >= 60) & (d["occ"] == 0), d))
        print("  per video (n, err/L, axis):  " + "  ".join(
            f"{v[-2:]}:{(d['vid'] == v).sum()},{np.nanmedian(d['ef'][d['vid'] == v]):.2f},{np.nanmedian(d['ax'][d['vid'] == v]):.0f}"
            for v in sorted(set(d["vid"]))))


if __name__ == "__main__":
    main()
