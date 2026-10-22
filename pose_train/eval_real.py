"""
eval_real.py — evaluate M1 on the annotated real MACS frames (protocol sections 3 and 6).

Input: real_eval/frames/*.jpg + real_eval/annotations/<annotator>/<frame>.json from annotate_tool.py
(box, points tip / spring / coil, attrs held / open / occluded / blurry). The crop is taken from the
annotated box (clip localisation given); the model's 8 keypoints are mapped to the three annotated
points (tip = mean of tip_a/tip_b, spring, coil_centre).

Metrics per frame, normalised by the annotated clip length L = |tip - spring|:
  * point error for tip / spring / coil (px and fraction of L), PCK at 0.1 L and 0.2 L
  * axis error: angle between predicted and annotated spring->tip directions (deg)
Reported overall and per attribute. Writes per_frame.csv, summary.json and an overlay montage.

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
from real_points import KEYS, SYNTH_MAP, norm_points, axis_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "..", "real_eval", "frames"))
    ap.add_argument("--ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--margin", type=float, default=1.5)
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
        x, y, w, h = ann["box"]
        P = np.array([v for v in norm_points(ann["points"]).values() if v is not None], float)   # union with the points:
        x0, y0 = min(x, P[:, 0].min()), min(y, P[:, 1].min())                       # the box can miss the clip
        x, y, w, h = x0, y0, max(x + w, P[:, 0].max()) - x0, max(y + h, P[:, 1].max()) - y0
        cx, cy, side = x + w / 2, y + h / 2, max(64.0, a.margin * max(w, h))
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
        ak = axis_key(gt)
        L = float(np.linalg.norm(gt["tip"] - gt[ak])) + 1e-6
        row = dict(file=ann["file"], video=ann.get("video", ""), L_px=L, **{f"attr_{k}": int(v) for k, v in ann["attrs"].items()})
        for k in KEYS:
            if k not in gt:
                continue
            e = float(np.linalg.norm(pred[k] - gt[k]))
            row[f"{k}_px"], row[f"{k}_frac"] = e, e / L
        dp, dg = pred["tip"] - pred[ak], gt["tip"] - gt[ak]
        cosang = float(np.dot(dp, dg) / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-6))
        row["axis_deg"] = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        row["mean_frac"] = float(np.mean([row[f"{k}_frac"] for k in KEYS if f"{k}_frac" in row]))
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

    def summ(sel, name):
        v = np.array([r["mean_frac"] for r in sel])
        ax = np.array([r["axis_deg"] for r in sel])
        return dict(name=name, n=len(sel), mean_frac_med=float(np.median(v)), mean_frac_p90=float(np.percentile(v, 90)),
                    pck_0_1=float(np.mean(v < 0.1)), pck_0_2=float(np.mean(v < 0.2)),
                    tip_frac_med=float(np.median([r["tip_frac"] for r in sel])), head_frac_med=float(np.median([r["head_frac"] for r in sel if "head_frac" in r] or [np.nan])),
                    axis_deg_med=float(np.median(ax)), axis_lt_10=float(np.mean(ax < 10)))
    groups = [(rows, "all")]
    for k in ("held", "open", "occluded", "blurry", "multi"):
        groups += [([r for r in rows if r.get(f"attr_{k}")], k), ([r for r in rows if not r.get(f"attr_{k}")], f"not_{k}")]
    summary = [summ(sel, name) for sel, name in groups if sel]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{'group':12s} {'n':>4s} {'err/L med':>10s} {'p90':>6s} {'PCK.1':>6s} {'PCK.2':>6s} {'tip/L':>6s} {'head/L':>6s} {'axis':>6s} {'<10deg':>6s}")
    for m in summary:
        print(f"{m['name']:12s} {m['n']:4d} {m['mean_frac_med']:10.3f} {m['mean_frac_p90']:6.3f} {m['pck_0_1']:6.2f} {m['pck_0_2']:6.2f} {m['tip_frac_med']:6.3f} {m['head_frac_med']:6.3f} {m['axis_deg_med']:6.1f} {m['axis_lt_10']:6.2f}")
    if cells:
        while len(cells) % 5:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(os.path.join(a.out, "_real_montage.jpg"), np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("->", a.out)


if __name__ == "__main__":
    main()
