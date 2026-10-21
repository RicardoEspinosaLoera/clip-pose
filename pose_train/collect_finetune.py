"""
collect_finetune.py — table for option A: mean +/- sd over seeds of the real-frame metrics of every
(init, frac) configuration found in --runs (outputs of finetune_real.py), plus the zero-shot reference.

Usage: python collect_finetune.py --runs runs_ft --out runs_ft/results_finetune.md
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np

KEYS = [("pck_0_1", "PCK@0.1L"), ("pck_0_2", "PCK@0.2L"), ("mean_frac_med", "err/L med"), ("axis_deg_med", "axis med (deg)"),
        ("axis_lt_10", "axis<10"), ("axis_lt_20", "axis<20")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="all", help="summary group to tabulate (all, held, not_held, ...)")
    a = ap.parse_args()
    cfgs = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        c = json.load(open(os.path.join(d, "config.json")))
        s = {m["name"]: m for m in json.load(open(os.path.join(d, "summary.json")))}
        if a.group not in s:
            continue
        label = (c["init"], "zero-shot" if c["iters"] == 0 else f"{int(round(100 * c['frac']))}%")
        cfgs[label].append(s[a.group])
    order = sorted(cfgs, key=lambda k: (k[0] != "synthetic", k[1] != "zero-shot", -int(k[1][:-1]) if k[1].endswith("%") else 0))
    lines = ["| init | real train frames | seeds | n | " + " | ".join(n for _, n in KEYS) + " |",
             "|---|---|---|---|" + "---|" * len(KEYS)]
    for k in order:
        ms = cfgs[k]
        cells = []
        for key, _ in KEYS:
            v = np.array([m[key] for m in ms], float)
            cells.append(f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else ""))
        lines.append(f"| {k[0]} | {k[1]} | {len(ms)} | {ms[0]['n']} | " + " | ".join(cells) + " |")
    txt = f"Real frames, group '{a.group}' (mean ± sd over seeds; video-grouped cross-validation)\n\n" + "\n".join(lines) + "\n"
    open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
