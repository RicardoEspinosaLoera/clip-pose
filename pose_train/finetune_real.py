"""
finetune_real.py — option A of the sprint: fine-tune a 3-keypoint heatmap model on the annotated real
MACS frames (tip, back end = "spring", coil centre) with video-grouped cross-validation, starting either
from the synthetic M1 checkpoint (8 synthetic keypoints mapped to the 3 real ones) or from ImageNet.

  init synthetic : backbone + deconvs from the synthetic checkpoint; head channels initialised from the
                   synthetic heads (tip = mean of tip_a/tip_b, spring, coil_centre) -> iters 0 = zero-shot
  init imagenet  : ImageNet ResNet-34 backbone, random deconvs and head (no synthetic data)

Folds: videos are assigned greedily to --folds folds (balanced frame counts); each fold is trained on the
other folds (optionally only --frac of their frames) and evaluated on its own frames, so every frame is
tested exactly once by a model that never saw its video. Augmentation: full rotation, flip, scale /
shift jitter, photometric (the real clip appears at any orientation).

Metrics per frame (original px, normalised by the annotated clip length L = |tip - back end|):
tip / back-end / coil error, mean error, PCK@0.1L and @0.2L, axis error (deg). Groups by attribute.

Usage:
  python finetune_real.py --ann ../real_eval/annotations/ricardo --init synthetic --ckpt runs/e0_m1_s0_amodal/best.pt \
      --frac 1.0 --seed 0 --out runs_ft/syn_f100_s0
  python finetune_real.py ... --iters 0 --out runs_ft/syn_zeroshot          (synthetic model, no real data)
"""

import os
import sys
import csv
import json
import math
import glob
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import decode_heatmaps, IMAGENET_MEAN, IMAGENET_STD, ClipPoseDataset
from model import KeypointNet, heatmap_loss

REAL = ("tip", "spring", "coil")
ATTRS = ("held", "open", "occluded", "blurry", "multi")


def clip_window(box, pts):
    """Clip region = union of the annotated box and the annotated points (the pre-loaded detector box can
    miss the clip or cut the jaws; the points are the reliable part of the annotation)."""
    x, y, w, h = box
    p = pts[~np.isnan(pts[:, 0])]
    x0, y0 = min(x, p[:, 0].min()), min(y, p[:, 1].min())
    x1, y1 = max(x + w, p[:, 0].max()), max(y + h, p[:, 1].max())
    return x0, y0, x1 - x0, y1 - y0


def load_items(frames_dir, ann_dir):
    items = []
    for jp in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        d = json.load(open(jp))
        if d.get("excluded"):
            continue
        img = cv2.imread(os.path.join(frames_dir, d["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pts = np.array([d["points"][k] if d["points"].get(k) is not None else [np.nan, np.nan] for k in REAL], np.float64)
        items.append(dict(file=d["file"], video=d.get("video") or d["file"][:6], img=img, box=np.array(d["box"], float),
                          pts=pts, attrs={k: bool(d.get("attrs", {}).get(k, False)) for k in ATTRS}))
    return items


def make_folds(items, n_folds):
    by_vid = defaultdict(list)
    for i, it in enumerate(items):
        by_vid[it["video"]].append(i)
    folds = [[] for _ in range(n_folds)]
    for v in sorted(by_vid, key=lambda v: (-len(by_vid[v]), v)):          # greedy, largest video first
        min(folds, key=len).extend(by_vid[v])
    return folds


class RealKP(Dataset):
    def __init__(self, items, idx, train, crop=512, heat=128, sigma=2.0, margin=1.5, seed=0):
        self.items, self.idx, self.train = items, list(idx), train
        self.crop, self.heat, self.sigma, self.margin = crop, heat, sigma, margin
        self.seed = seed

    def __len__(self):
        return len(self.idx)

    def affine(self, it, rng):
        x, y, w, h = clip_window(it["box"], it["pts"])
        cx, cy, side = x + w / 2, y + h / 2, self.margin * max(w, h)
        theta, flip = 0.0, False
        if self.train:
            side *= rng.uniform(0.85, 1.15)
            cx += rng.uniform(-0.10, 0.10) * side
            cy += rng.uniform(-0.10, 0.10) * side
            theta, flip = rng.uniform(-180, 180), rng.random() < 0.5
        side = max(64.0, side)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), theta, self.crop / side)
        M[:, 2] += (self.crop / 2 - cx, self.crop / 2 - cy)
        if flip:
            M = np.array([[-1, 0, self.crop], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
        return M

    def __getitem__(self, j):
        i = self.idx[j]
        it = self.items[i]
        rng = np.random.default_rng(None if self.train else self.seed * 100003 + i)
        M = self.affine(it, rng)
        crop = cv2.warpAffine(it["img"], M, (self.crop, self.crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        kp = it["pts"] @ M[:, :2].T + M[:, 2]
        vis = (~np.isnan(kp[:, 0])) & (kp[:, 0] >= 0) & (kp[:, 0] < self.crop) & (kp[:, 1] >= 0) & (kp[:, 1] < self.crop)
        if self.train:
            crop = ClipPoseDataset._photometric(crop, rng)
        hm = np.zeros((len(REAL), self.heat, self.heat), np.float32)
        r = self.heat / self.crop
        gy, gx = np.mgrid[0:self.heat, 0:self.heat].astype(np.float32)
        for k in range(len(REAL)):
            if vis[k]:
                hm[k] = np.exp(-((gx - kp[k, 0] * r) ** 2 + (gy - kp[k, 1] * r) ** 2) / (2 * self.sigma ** 2))
        x = (np.clip(crop, 0, 255)[..., ::-1].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return dict(image=torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), heatmaps=torch.from_numpy(hm),
                    vis=torch.from_numpy(vis.astype(np.float32)), M=torch.from_numpy(M.astype(np.float32)),
                    i=torch.tensor(i))


def build_model(init, ckpt, arch):
    if init == "imagenet":
        return KeypointNet(len(REAL), arch, pretrained=True)
    kp_json = os.path.join(HERE, "..", "GT_Generator", "keypoints.json")
    names = json.load(open(kp_json))["names"]
    m = KeypointNet(len(names), arch, pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    W, b = m.head.weight.data, m.head.bias.data
    ix = {n: i for i, n in enumerate(names)}
    rows = [(ix["tip_a"], ix["tip_b"]), (ix["spring"],), (ix["coil_centre"],)]
    head = nn.Conv2d(W.shape[1], len(REAL), 1)
    head.weight.data = torch.stack([W[list(r)].mean(0) for r in rows])
    head.bias.data = torch.stack([b[list(r)].mean() for r in rows])
    m.head = head
    return m


def train_fold(model, ds, a, device):
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=len(ds) > a.batch,
                    persistent_workers=a.workers > 0)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": hd, "lr": a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    it = 0
    while it < a.iters:
        for b in dl:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = heatmap_loss(model(b["image"].to(device)).float(), b["heatmaps"].to(device), b["vis"].to(device)) * 1000.0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 500 == 0:
                print(f"    it {it} loss {loss.item():.4f}", flush=True)
            if it >= a.iters:
                break


@torch.no_grad()
def predict(model, ds, device):
    model.eval()
    out = {}
    for b in DataLoader(ds, batch_size=16, shuffle=False, num_workers=0):
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            hm = model(b["image"].to(device)).float().cpu().numpy()
        for j in range(hm.shape[0]):
            pts, conf = decode_heatmaps(hm[j])
            pc = pts * (ds.crop / hm.shape[-1])
            Minv = cv2.invertAffineTransform(b["M"][j].numpy().astype(np.float64))
            out[int(b["i"][j])] = (pc @ Minv[:, :2].T + Minv[:, 2], conf)
    return out


def frame_row(it, pred, fold):
    gt = it["pts"]
    L = float(np.linalg.norm(gt[0] - gt[1])) + 1e-6
    row = dict(file=it["file"], video=it["video"], fold=fold, L_px=L, **{f"attr_{k}": int(v) for k, v in it["attrs"].items()})
    fr = []
    for k, name in enumerate(REAL):
        if np.isnan(gt[k, 0]):
            continue
        e = float(np.linalg.norm(pred[k] - gt[k]))
        row[f"{name}_px"], row[f"{name}_frac"] = e, e / L
        fr.append(e / L)
    row["mean_frac"] = float(np.mean(fr))
    dp, dg = pred[0] - pred[1], gt[0] - gt[1]
    c = float(np.dot(dp, dg) / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-6))
    row["axis_deg"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return row


def summarize(rows):
    def summ(sel, name):
        v = np.array([r["mean_frac"] for r in sel])
        ax = np.array([r["axis_deg"] for r in sel])
        return dict(name=name, n=len(sel), mean_frac_med=float(np.median(v)), mean_frac_p90=float(np.percentile(v, 90)),
                    pck_0_1=float(np.mean(v < 0.1)), pck_0_2=float(np.mean(v < 0.2)),
                    tip_frac_med=float(np.median([r["tip_frac"] for r in sel])), spring_frac_med=float(np.median([r["spring_frac"] for r in sel])),
                    axis_deg_med=float(np.median(ax)), axis_lt_10=float(np.mean(ax < 10)), axis_lt_20=float(np.mean(ax < 20)))
    groups = [(rows, "all")]
    for k in ATTRS:
        groups += [([r for r in rows if r.get(f"attr_{k}")], k), ([r for r in rows if not r.get(f"attr_{k}")], f"not_{k}")]
    return [summ(sel, name) for sel, name in groups if sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "..", "real_eval", "frames"))
    ap.add_argument("--ann", required=True)
    ap.add_argument("--init", choices=["synthetic", "imagenet"], required=True)
    ap.add_argument("--ckpt", default=None, help="synthetic M1 checkpoint (init synthetic)")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the training-fold frames used")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.init == "synthetic" and not a.ckpt:
        ap.error("--ckpt is required with --init synthetic")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_items(a.frames, a.ann)
    folds = make_folds(items, a.folds)
    print(f"{len(items)} frames, {len({it['video'] for it in items})} videos, folds {[len(f) for f in folds]} | init {a.init} "
          f"frac {a.frac} iters {a.iters} seed {a.seed} | {device}", flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for f, test_idx in enumerate(folds):
        train_idx = [i for g, fl in enumerate(folds) if g != f for i in fl]
        if a.frac < 1.0:
            train_idx = sorted(rng.choice(train_idx, size=max(1, int(round(a.frac * len(train_idx)))), replace=False).tolist())
        model = build_model(a.init, a.ckpt, a.arch).to(device)
        if a.iters > 0:
            print(f"  fold {f}: train {len(train_idx)} frames, test {len(test_idx)} ({sorted({items[i]['video'] for i in test_idx})})", flush=True)
            train_fold(model, RealKP(items, train_idx, True), a, device)
        preds = predict(model, RealKP(items, test_idx, False, seed=a.seed), device)
        rows += [frame_row(items[i], preds[i][0], f) for i in test_idx]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "per_frame.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(dict(vars(a), n_frames=len(items), folds=[[items[i]["file"] for i in fl] for fl in folds]),
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"{'group':12s} {'n':>4s} {'err/L':>6s} {'PCK.1':>6s} {'PCK.2':>6s} {'tip/L':>6s} {'spr/L':>6s} {'axis':>6s} {'<10':>5s} {'<20':>5s}")
    for m in summary:
        print(f"{m['name']:12s} {m['n']:4d} {m['mean_frac_med']:6.3f} {m['pck_0_1']:6.2f} {m['pck_0_2']:6.2f} {m['tip_frac_med']:6.3f} "
              f"{m['spring_frac_med']:6.3f} {m['axis_deg_med']:6.1f} {m['axis_lt_10']:5.2f} {m['axis_lt_20']:5.2f}")
    print("->", a.out)


if __name__ == "__main__":
    main()
