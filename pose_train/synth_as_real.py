"""
synth_as_real.py — control for eval_real.py: writes synthetic test samples in the real-evaluation
format (frames + one annotation JSON per frame, box = mask bbox, tip = midpoint of the two projected
jaw tips, spring = projected spring keypoint, coil = projected coil centre, attributes from the GT).
Running eval_real.py on it must give small errors; if it does, a large error on the real frames is
a real-domain failure, not an evaluation bug.

Usage:
  python synth_as_real.py --data ../GT_Generator/out_v3 --split ../GT_Generator/out_v3/splits/test.txt --out ../real_eval_synthctl
  python eval_real.py --frames ../real_eval_synthctl/frames --ann ../real_eval_synthctl/annotations/gt --ckpt runs/e0_m1_s0/best.pt --out runs/e0_m1_s0/real_synthctl
"""

import os
import sys
import json
import shutil
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ClipModel, pose_from_json, project, read_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cm = ClipModel()
    ix = {n: i for i, n in enumerate(cm.names)}
    fdir, adir = os.path.join(a.out, "frames"), os.path.join(a.out, "annotations", "gt")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)
    stems = read_split(a.split)[:a.n]
    for stem in stems:
        meta = json.load(open(os.path.join(a.data, f"{stem}.json")))
        mask = cv2.imread(os.path.join(a.data, meta["image"].get("file_mask", f"{stem}_mask.png")), 0)
        ys, xs = np.nonzero(mask > 127)
        R, t, K = pose_from_json(meta)
        jaw = float(meta["clip"].get("jaw_angle_deg", 0.0))
        uv, _ = project(cm.keypoints3d(jaw), R, t, K)
        name = f"{stem}.png"
        shutil.copy2(os.path.join(a.data, name), os.path.join(fdir, name))
        rec = dict(file=name, src=stem, video="synthetic", width=int(mask.shape[1]), height=int(mask.shape[0]),
                   box=[float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                   points={"tip": ((uv[ix["tip_a"]] + uv[ix["tip_b"]]) / 2).tolist(),
                           "spring": uv[ix["spring"]].tolist(), "coil": uv[ix["coil_centre"]].tolist()},
                   attrs=dict(held=bool(meta["clip"].get("tool_held")), open=jaw > 0, occluded=False, blurry=False, multi=False),
                   annotator="gt")
        json.dump(rec, open(os.path.join(adir, name + ".json"), "w"), indent=1)
    print(f"{len(stems)} synthetic samples in real-eval format -> {a.out}")


if __name__ == "__main__":
    main()
