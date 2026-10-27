"""
finetune_real.py — option A of the sprint: fine-tune a 3-keypoint heatmap model on the annotated real
MACS frames (tip, clip head centre, back end; see real_points.py) with video-grouped cross-validation, starting either
from the synthetic M1 checkpoint (8 synthetic keypoints mapped to the 3 real ones) or from ImageNet.

  init synthetic : backbone + deconvs from the synthetic checkpoint; head channels initialised from the
                   synthetic heads (tip = mean of tip_a/tip_b, head = coil_centre, back = spring) -> iters 0 = zero-shot
  init imagenet  : ImageNet ResNet-34 backbone, random deconvs and head (no synthetic data)

Folds: videos are assigned greedily to --folds folds (balanced frame counts); each fold is trained on the
other folds (optionally only --frac of their frames) and evaluated on its own frames, so every frame is
tested exactly once by a model that never saw its video. Augmentation: full rotation, flip, scale /
shift jitter, photometric (the real clip appears at any orientation).

Metrics per frame (original px, normalised by the annotated clip length L = |tip - back end|):
tip / head / back-end error, mean error, PCK@0.1L and @0.2L, axis error (deg). Groups by attribute.

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

from real_points import (KEYS as REAL, SYNTH_MAP, norm_points, crop_window, WINDOW_MARGIN, auto_window,
                         frame_metrics, summarize_real, SUMMARY_HEADER, summary_line, MIN_L_PX)
ATTRS = ("held", "open", "occluded", "blurry", "multi")


def load_items(frames_dir, ann_dir):
    items = []
    for jp in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        d = json.load(open(jp))
        if d.get("excluded"):
            continue
        img = cv2.imread(os.path.join(frames_dir, d["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        P = norm_points(d["points"])
        pts = np.array([P[k] if P[k] is not None else [np.nan, np.nan] for k in REAL], np.float64)
        items.append(dict(file=d["file"], video=d.get("video") or d["file"][:6], img=img, box=np.array(d["box"], float),
                          pts=pts, window=auto_window(d), attrs={k: bool(d.get("attrs", {}).get(k, False)) for k in ATTRS}))
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
    def __init__(self, items, idx, train, crop=512, heat=128, sigma=2.0, margin=None, seed=0, window="auto"):
        self.window = window
        self.items, self.idx, self.train = items, list(idx), train
        self.crop, self.heat, self.sigma, self.margin = crop, heat, sigma, margin
        self.seed = seed

    def __len__(self):
        return len(self.idx)

    def affine(self, it, rng):
        mode = it["window"] if self.window == "auto" else self.window
        cx, cy, side = crop_window(it["box"], it["pts"], mode)
        side *= self.margin or WINDOW_MARGIN[mode]
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
    rows = [tuple(ix[n] for n in SYNTH_MAP[k]) for k in REAL]
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
    """Same metrics as eval_real.py (real_points.frame_metrics): axis/L = tip -> back end, fallback spring centre.
    (Before 27 Oct 2026 this used tip -> spring centre, so its numbers were not comparable with eval_real.)"""
    gt = {name: it["pts"][k] for k, name in enumerate(REAL) if not np.isnan(it["pts"][k, 0])}
    pd_ = {name: pred[k] for k, name in enumerate(REAL)}
    row = dict(file=it["file"], video=it["video"], fold=fold, **{f"attr_{k}": int(v) for k, v in it["attrs"].items()})
    row.update(frame_metrics(pd_, gt))
    return row


def summarize(rows):
    return summarize_real(rows, ATTRS)


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
    ap.add_argument("--window", choices=["auto", "points", "union", "box"], default="auto",
                    help="crop window source; auto = box for scheme-2 annotations, box U points for scheme 1")
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
            train_fold(model, RealKP(items, train_idx, True, window=a.window), a, device)
        preds = predict(model, RealKP(items, test_idx, False, seed=a.seed, window=a.window), device)
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
    print(f"main set = frames with clip length L >= {MIN_L_PX:.0f} px ('all')")
    print(SUMMARY_HEADER)
    for m in summary:
        print(summary_line(m))
    print("->", a.out)


if __name__ == "__main__":
    main()
