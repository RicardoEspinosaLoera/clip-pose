"""
make_subsets.py — learning-curve subsets of the train split, sampled BY BACKPLATE so each subset
keeps the scene variety (protocol section 5).

Usage: python make_subsets.py --dir ../GT_Generator/out_v3 --sizes 1250 2500 [--seed 0]
Writes <dir>/splits/train_<N>.txt (nested: the 1250 set is inside the 2500 set).
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1250, 2500])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    split_dir = os.path.join(a.dir, "splits")
    train = [l.strip() for l in open(os.path.join(split_dir, "train.txt")) if l.strip()]
    by_bp = defaultdict(list)
    for stem in train:
        bp = json.load(open(os.path.join(a.dir, f"{stem}.json"))).get("scene", {}).get("backplate", "none")
        by_bp[bp].append(stem)
    rng = np.random.default_rng(a.seed)
    order = []                                   # round-robin over shuffled backplates -> nested subsets
    bps = sorted(by_bp)
    rng.shuffle(bps)
    lists = {bp: list(rng.permutation(by_bp[bp])) for bp in bps}
    while any(lists.values()):
        for bp in bps:
            if lists[bp]:
                order.append(lists[bp].pop())
    for n in sorted(a.sizes):
        sub = sorted(order[:n])
        with open(os.path.join(split_dir, f"train_{n}.txt"), "w") as f:
            f.write("\n".join(sub) + "\n")
        print(f"train_{n}.txt: {len(sub)} images from {len({json.load(open(os.path.join(a.dir, s + '.json'))).get('scene', {}).get('backplate') for s in sub})} backplates")


if __name__ == "__main__":
    main()
