"""
make_splits.py — train / val / test splits BY BACKPLATE (scene), never by image.

Reads scene.backplate from every NNN.json in --dir, shuffles the backplate names with --seed,
assigns 80 / 10 / 10 % of the backplates to train / val / test and writes stem lists
(train.txt, val.txt, test.txt) plus splits.json (backplate lists + hashes) into --dir/splits/.
Samples whose JSON lacks scene.backplate (sets generated before 2026-10-19) go to train only.

Usage: python make_splits.py --dir ../GT_Generator/out_v3 [--seed 0] [--frac 0.8 0.1 0.1]
"""

import os
import json
import glob
import hashlib
import argparse
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--orphan-split", action="store_true",
                    help="samples without scene.backplate form one group each (legacy sets / smoke tests) instead of going to train")
    a = ap.parse_args()
    by_bp = defaultdict(list)
    orphan = []
    for jp in sorted(glob.glob(os.path.join(a.dir, "[0-9]*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        bp = json.load(open(jp)).get("scene", {}).get("backplate")
        if not bp and a.orphan_split:
            bp = f"stem:{stem}"
        (by_bp[bp] if bp else orphan).append(stem)
    bps = sorted(by_bp)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(bps)
    n = len(bps)
    n_tr, n_va = int(round(a.frac[0] * n)), int(round(a.frac[1] * n))
    groups = {"train": bps[:n_tr], "val": bps[n_tr:n_tr + n_va], "test": bps[n_tr + n_va:]}
    out = os.path.join(a.dir, "splits")
    os.makedirs(out, exist_ok=True)
    info = {"seed": a.seed, "frac": a.frac, "n_backplates": n, "orphans_in_train": len(orphan)}
    for name, bl in groups.items():
        stems = sorted(s for bp in bl for s in by_bp[bp]) + (sorted(orphan) if name == "train" else [])
        with open(os.path.join(out, f"{name}.txt"), "w") as f:
            f.write("\n".join(stems) + "\n")
        info[name] = {"backplates": sorted(bl), "n_images": len(stems),
                      "sha1": hashlib.sha1("\n".join(stems).encode()).hexdigest()[:12]}
        print(f"{name:5s}: {len(bl):4d} backplates, {len(stems):5d} images, sha1 {info[name]['sha1']}")
    json.dump(info, open(os.path.join(out, "splits.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
