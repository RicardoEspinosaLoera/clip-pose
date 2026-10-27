"""
eval_real.py — evaluate M1 on the annotated real MACS frames (protocol sections 3 and 6).

Input: real_eval/frames/*.jpg + real_eval/annotations/<annotator>/<frame>.json from annotate_tool.py
(box, points tip / spring / coil, attrs held / open / occluded / blurry). The crop is taken from the
annotated box (clip localisation given); the model's 8 keypoints are mapped to the three annotated
points (tip = mean of tip_a/tip_b, spring, coil_centre).

Metrics per frame, normalised by the annotated clip length L = |tip - spring|:
  * point error for tip / spring / coil (px and fraction of L), PCK at 0.1 L and 0.2 L
  * axis error: angle between predicted and annotated back->tip directions (deg), its undirected version and
    the tip/back flip rate
Group "all" = main set (L >= 60 px); end-on / tiny clips are reported as "small" and "all_incl_small"
(see real_points.MIN_L_PX). Reported overall and per attribute. Writes per_frame.csv, summary.json and an overlay montage.

Usage: python eval_real.py --ann ../real_eval/annotations/ricardo --ckpt runs/e0_m1_s0/best.pt --out runs/e0_m1_s0/real
"""

import os
import sys
import csv
import json
import glob
import argparse

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ClipModel, decode_heatmaps, IMAGENET_MEAN, IMAGENET_STD
from model import KeypointNet
from real_points import (KEYS, SYNTH_MAP, norm_points, axis_key, crop_window, WINDOW_MARGIN, auto_window,
                         frame_metrics, summarize_real, SUMMARY_HEADER, summary_line, MIN_L_PX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "..", "real_eval", "frames"))
    ap.add_argument("--ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--margin", type=float, default=None, help="default: 1.75 for points, 1.5 for box / union")
    ap.add_argument("--window", choices=["auto", "points", "union", "box"], default="auto",
                    help="crop window source; auto = box for scheme-2 annotations, box U points for scheme 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-montage", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = ClipModel()
    names = cm.names
    model = KeypointNet(len(cm.idx), a.arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device)["model"])
    model.eval()
    rows, cells, n_excluded = [], [], 0
    for jp in sorted(glob.glob(os.path.join(a.ann, "*.json"))):
        ann = json.load(open(jp))
        if ann.get("excluded"):                  # annotator: no clip / not annotatable
            n_excluded += 1
            continue
        img = cv2.imread(os.path.join(a.frames, ann["file"]))
        if img is None:
            continue
        P = np.array([v if v is not None else [np.nan, np.nan] for v in norm_points(ann["points"]).values()], float)
        mode = auto_window(ann) if a.window == "auto" else a.window
        cx, cy, side = crop_window(ann["box"], P, mode)
        side = max(64.0, (a.margin or WINDOW_MARGIN[mode]) * side)
        s = a.crop / side
        A = np.array([[s, 0, a.crop / 2 - s * cx], [0, s, a.crop / 2 - s * cy]], np.float64)
        crop = cv2.warpAffine(img, A, (a.crop, a.crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        xt = (crop[..., ::-1].astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        xt = torch.from_numpy(np.ascontiguousarray(xt.transpose(2, 0, 1)))[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            hm = model(xt).float()[0].cpu().numpy()
        pts, conf = decode_heatmaps(hm)
        pc = pts * (a.crop / hm.shape[-1])
        po = (pc - A[:, 2][None, :]) / s                     # back to original pixels
        kp = {n: po[i] for i, n in enumerate(names)}
        pred = {k: np.mean([kp[n] for n in SYNTH_MAP[k]], 0) for k in KEYS}
        gt = {k: np.array(v, float) for k, v in norm_points(ann["points"]).items() if v is not None}   # back may be hidden
        row = dict(file=ann["file"], video=ann.get("video", ""), **{f"attr_{k}": int(v) for k, v in ann["attrs"].items()})
        row.update(frame_metrics(pred, gt))
        rows.append(row)
        if len(cells) < a.n_montage:
            vis = img.copy()
            for k, col in (("tip", (0, 0, 255)), ("head", (255, 128, 0)), ("back", (255, 0, 255))):
                if k not in gt:
                    continue
                cv2.circle(vis, tuple(int(v) for v in gt[k]), 6, col, 2)
                cv2.circle(vis, tuple(int(v) for v in pred[k]), 4, col, -1)
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            c = vis[y0:int(y0 + side), x0:int(x0 + side)]
            c = cv2.resize(c, (320, 320))
            cv2.rectangle(c, (0, 0), (320, 18), (0, 0, 0), -1)
            cv2.putText(c, f"{ann['file'][:14]} err {row['mean_frac']:.2f}L axis {row['axis_deg']:.0f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cells.append(c)
    print(f"{len(rows)} annotated frames, {n_excluded} excluded by the annotator")
    if not rows:
        print("no annotated frames found in", a.ann)
        return
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = summarize_real(rows)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"main set = frames with clip length L >= {MIN_L_PX:.0f} px ('all'); axis = directed, undir = line only, flip = tip/back swapped")
    print(SUMMARY_HEADER)
    for m in summary:
        print(summary_line(m))
    if cells:
        while len(cells) % 5:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(os.path.join(a.out, "_real_montage.jpg"), np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("->", a.out)


if __name__ == "__main__":
    main()
