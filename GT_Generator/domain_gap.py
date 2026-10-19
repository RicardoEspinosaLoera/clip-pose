"""
domain_gap.py — a NUMBER for "how real do the synthetic clips look".

Trains a small real-vs-synthetic classifier on clip crops and reports its cross-validated
accuracy: 50% = indistinguishable, 100% = trivially separable. Every realism change should
push this DOWN. Two crop windows are evaluated so the gap can be attributed:
  * tight   (1.4 x clip bbox) : material / shading / edge statistics of the clip itself
  * context (2.5 x clip bbox) : clip + surroundings (occlusion, applier, shadow, blending)

Features: frozen ImageNet ResNet-18 (torchvision) global features, 512-d, L2-normalised.
Classifier: logistic regression (torch), 5-fold CV, class-balanced sampling.
Real crops: Inpaint_Realism/lora/clean_clips2 (+ masks). Synthetic: any GT output dir
(NNN.png + NNN_mask.png). Several --dirs can be compared in one run.

Usage
  python domain_gap.py --dirs out_pose100 out_pilot_v2 out_pilot_v3 --n 200
  python domain_gap.py --dirs out_pilot_v3 --save-crops out_pilot_v3/_gap_crops.png
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CROPS = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2"))
REAL_MASKS = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2_masks"))
SIZE = 160


def window_crop(img, mask, factor, size=SIZE):
    """Square crop centred on the mask bbox, side = factor x max(bw, bh), reflect-padded.
    Returns (crop, mask_crop) or None."""
    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return None
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = int(max(24, factor * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    pad = side
    imp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    mkp = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(round(cx - side / 2.0)) + pad, int(round(cy - side / 2.0)) + pad
    crop = cv2.resize(imp[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
    mcrop = cv2.resize(mkp[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_NEAREST)
    return crop, mcrop


def stats_features(crop, mcrop):
    """Shape-blind appearance statistics of the clip pixels and of the ring around them:
    colour histograms, brightness/saturation moments, highlight fraction, gradient and
    Laplacian energy, high-pass noise level, JPEG blockiness, chroma-vs-luma edge softness.
    A classifier on these measures the APPEARANCE gap without seeing the silhouette (the
    mesh is a different clip than MACS, so any shape-aware feature separates trivially)."""
    m = mcrop > 127
    ring = (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~m
    if m.sum() < 20 or ring.sum() < 20:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = []
    for reg in (m, ring):
        h, s, v = hsv[..., 0][reg], hsv[..., 1][reg], hsv[..., 2][reg]
        f += list(np.histogram(h, bins=8, range=(0, 180), density=True)[0])
        f += list(np.histogram(s, bins=6, range=(0, 256), density=True)[0])
        f += list(np.histogram(v, bins=6, range=(0, 256), density=True)[0])
        f += [v.mean() / 255, v.std() / 255, s.mean() / 255, (v > 204).mean(), (v < 51).mean()]
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        gm = np.hypot(gx, gy)[reg]
        f += list(np.histogram(np.log1p(gm), bins=6, range=(0, 7), density=True)[0])
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)[reg]
        hp = (g - cv2.GaussianBlur(g, (0, 0), 1.0))[reg]
        f += [np.log1p(np.abs(lap).mean()), np.log1p(1.4826 * np.median(np.abs(hp - np.median(hp))))]
    # JPEG blockiness over the crop: gradient across 8-px block boundaries vs inside blocks
    dx = np.abs(np.diff(g, axis=1))
    bnd = dx[:, 7::8].mean()
    inn = np.delete(dx, np.s_[7::8], axis=1).mean()
    f += [np.log((bnd + 1e-3) / (inn + 1e-3))]
    # chroma softness at clip edges: chroma gradient / luma gradient on the mask boundary
    edge = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    def gmag(ch):
        return np.hypot(cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1))
    lum_e, chr_e = gmag(ycc[..., 0])[edge].mean(), (gmag(ycc[..., 1]) + gmag(ycc[..., 2]))[edge].mean()
    f += [np.log((chr_e + 1e-3) / (lum_e + 1e-3))]
    return np.array(f, np.float32)


def load_real(n, factor, rng, native=240):
    """Real crops are 768 px upsamples of small frame regions; bring them back to a frame-like
    scale (native px) before windowing so their pixel statistics match the synthetic frames."""
    paths = sorted(glob.glob(os.path.join(REAL_CROPS, "*.png")))
    pick = rng.permutation(len(paths))
    out = []
    for i in pick:
        im = cv2.imread(paths[i])
        mk = cv2.imread(os.path.join(REAL_MASKS, os.path.basename(paths[i])), 0)
        if im is None or mk is None:
            continue
        im = cv2.resize(im, (native, native), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, (native, native), interpolation=cv2.INTER_NEAREST)
        c = window_crop(im, mk, factor)
        if c is not None:
            out.append(c)
        if len(out) >= n:
            break
    return out


def load_synth(d, n, factor, rng):
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "[0-9]*_mask.png")))
    stems = [s[:-5] for s in stems]                       # strip "_mask"
    pick = rng.permutation(len(stems))
    out = []
    for i in pick:
        im = cv2.imread(os.path.join(d, f"{stems[i]}.png"))
        mk = cv2.imread(os.path.join(d, f"{stems[i]}_mask.png"), 0)
        if im is None or mk is None:
            continue
        c = window_crop(im, mk, factor)
        if c is not None:
            out.append(c)
        if len(out) >= n:
            break
    return out


_NET = {}


def features(crops, device, kind="resnet"):
    """Feature matrix for a list of (crop, mask_crop) pairs."""
    if kind == "stats":
        return np.stack([stats_features(c, mk) for c, mk in crops])
    crops = [c for c, _ in crops]
    if "net" not in _NET:
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        _NET["net"] = net.eval().to(device)
    net = _NET["net"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            b = np.stack(crops[i:i + 64])[..., ::-1].astype(np.float32) / 255.0      # BGR->RGB
            x = torch.from_numpy(np.ascontiguousarray(b)).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            f = net(x)
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(feats).numpy()


STAT_NAMES = []
for _reg in ("clip", "ring"):
    STAT_NAMES += [f"{_reg}:hue{i}" for i in range(8)] + [f"{_reg}:sat{i}" for i in range(6)] + [f"{_reg}:val{i}" for i in range(6)]
    STAT_NAMES += [f"{_reg}:V_mean", f"{_reg}:V_std", f"{_reg}:S_mean", f"{_reg}:highlight_frac", f"{_reg}:dark_frac"]
    STAT_NAMES += [f"{_reg}:grad{i}" for i in range(6)] + [f"{_reg}:laplacian", f"{_reg}:noise"]
STAT_NAMES += ["jpeg_blockiness", "chroma_edge_softness"]


def kid(Xa, Xb, subsets=20, subset_size=100, seed=0):
    """Kernel Inception Distance style MMD^2 (cubic polynomial kernel) on standardized
    features: a CONTINUOUS gap measure that keeps ranking changes after a classifier
    saturates at 100%. 0 = same distribution."""
    rng = np.random.default_rng(seed)
    Z = np.concatenate([Xa, Xb])
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    d = A.shape[1]
    k = lambda P, Q: (P @ Q.T / d + 1.0) ** 3
    m = min(subset_size, len(A), len(B))
    vals = []
    for _ in range(subsets):
        a = A[rng.choice(len(A), m, replace=False)]
        b = B[rng.choice(len(B), m, replace=False)]
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        np.fill_diagonal(kaa, 0)
        np.fill_diagonal(kbb, 0)
        vals.append(kaa.sum() / (m * (m - 1)) + kbb.sum() / (m * (m - 1)) - 2 * kab.mean())
    return float(np.mean(vals))


def top_differences(Xr, Xs, names, k=6):
    """Largest standardized mean differences (Cohen's d) between real and synthetic stats."""
    d = (Xs.mean(0) - Xr.mean(0)) / (np.sqrt(0.5 * (Xr.var(0) + Xs.var(0))) + 1e-6)
    order = np.argsort(-np.abs(d))[:k]
    return [(names[i], float(d[i])) for i in order]


def logreg_cv(X, y, folds=5, seed=0, epochs=300, wd=1e-3):
    """5-fold CV logistic regression; returns (accuracy mean, std, mean P(real) of synthetic)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    accs, p_real_synth = [], []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = torch.tensor((X[tr] - mu) / sd), torch.tensor((X[te] - mu) / sd)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + wd * (w * w).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = torch.sigmoid(Xte @ w + b).numpy()
        accs.append(float(((p > 0.5) == (y[te] > 0.5)).mean()))
        p_real_synth.extend(p[y[te] == 0].tolist())
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(p_real_synth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["out_pilot_v2"])
    ap.add_argument("--n", type=int, default=200, help="crops per class (real is capped to the same n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-crops", default=None, help="write a sanity grid (real row / synthetic rows)")
    ap.add_argument("--features", default="stats,resnet",
                    help="comma list of: stats (shape-blind appearance statistics), resnet (whole crop)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    windows = {"tight": 1.4, "context": 2.5}
    kinds = [k for k in a.features.split(",") if k]
    real = {k: load_real(a.n, f, np.random.default_rng(a.seed)) for k, f in windows.items()}
    print(f"real crops: {len(real['tight'])} | device {device}")
    print("classifier accuracy (50% = indistinguishable, 100% = separable) and KID distance (0 = same distribution)")
    print(f"{'dir':24s} {'window':8s} {'n':>4s}  " + "  ".join(f"{k + ' acc / KID':>26s}" for k in kinds))
    grid_rows = [np.hstack([c for c, _ in real["tight"][:12]])]
    breakdown = []
    for d in a.dirs:
        dd = d if os.path.isabs(d) else os.path.join(HERE, d)
        for k, f in windows.items():
            synth = load_synth(dd, a.n, f, np.random.default_rng(a.seed))
            if len(synth) < 20:
                print(f"{os.path.basename(dd):24s} {k:8s} too few samples ({len(synth)})")
                continue
            nn_ = min(len(synth), len(real[k]))
            y = np.array([1] * nn_ + [0] * nn_, dtype=np.float32)
            cells = []
            for kind in kinds:
                X = features(real[k][:nn_] + synth[:nn_], device, kind)
                acc, sd, pr = logreg_cv(X, y, seed=a.seed)
                dist = kid(X[:nn_], X[nn_:], seed=a.seed)
                cells.append(f"{acc*100:5.1f}%  KID {dist:7.3f}")
                if kind == "stats" and k == "tight":
                    breakdown.append((os.path.basename(dd), top_differences(X[:nn_], X[nn_:], STAT_NAMES)))
            print(f"{os.path.basename(dd):24s} {k:8s} {nn_:4d}  " + "  ".join(f"{c:>26s}" for c in cells))
            if k == "tight":
                grid_rows.append(np.hstack([c for c, _ in synth[:12]]))
    for name, diffs in breakdown:
        print(f"largest appearance differences, {name} (tight window; d>0 = synthetic higher):")
        print("   " + ", ".join(f"{n} {v:+.1f}" for n, v in diffs))
    if a.save_crops:
        g = np.vstack(grid_rows)
        cv2.putText(g, "row 1 = REAL (tight window); next rows = --dirs in order", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(a.save_crops, g)
        print(f"crops grid -> {a.save_crops}")


if __name__ == "__main__":
    main()
