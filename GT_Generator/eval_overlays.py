"""
eval_overlays.py — measure the realism/diversity of a GT_Generator overlay set.

We cannot judge "more real" by eye across thousands of images, so this scores an
output directory (produced by render_clip_poses_depth.py) on three axes and writes
a contact sheet for eyeballing. Run it on the CURRENT dataset first to get a
baseline, then re-run after each realism phase and compare.

Axes
----
1. POSE spread   — histograms of camera tilt (deg off straight-down), in-plane roll,
                   model-up dot (how flat the clip lies), on-screen clip-area fraction,
                   and clip distance. A wider spread = more varied poses.
2. SCENE spread  — number of DISTINCT backplates actually used, estimated by
                   perceptual-hash clustering of the *_bg.png files. 30 source images
                   cluster to ~30 no matter how many samples you render.
3. REALISM proxy — torch-free colour-distribution distance (Wasserstein-1 per HSV
                   channel) between the overlays and a reference set of REAL MACS
                   frames. Lower = the overlay colour statistics match real footage.
                   Optional true FID via --fid if torch-fidelity is installed.

Usage
-----
  # baseline the current dataset (pose + scene stats + contact sheet)
  python eval_overlays.py --dir out_rand_depth

  # add the realism proxy vs real frames
  python eval_overlays.py --dir out_rand_depth --real <dir_of_real_MACS_frames> --n 400

  # add a true FID (needs: pip install torch-fidelity)
  python eval_overlays.py --dir out_rand_depth --real <dir> --fid
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

WORLD_UP = np.array([0.0, 0.0, 1.0])
MODEL_UP = np.array([1.0, 0.0, 0.0])


# ----------------------------- pose metrics -----------------------------
def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return scipyR.from_quat([x, y, z, w]).as_matrix()


def pose_metrics_from_json(jpath):
    """tilt, roll, model-up-dot, distance from one sample's GT json (+ mask area if present)."""
    with open(jpath) as f:
        d = json.load(f)
    cam = d["camera"]
    pos = np.array(cam["position"], float)
    foc = np.array(cam["focal_point"], float)
    vup = np.array(cam["view_up"], float)

    fwd = _norm(foc - pos)
    # tilt: angle of the view direction away from straight-down (-WORLD_UP)
    tilt = np.degrees(np.arccos(np.clip(np.dot(fwd, -WORLD_UP), -1, 1)))
    # roll: angle between the actual view-up and the zero-roll image-up about fwd
    right = _norm(np.cross(fwd, WORLD_UP))
    img_up = _norm(np.cross(right, fwd))
    roll = np.degrees(np.arccos(np.clip(np.dot(_norm(vup), img_up), -1, 1)))

    # model-up dot: how aligned the clip's own up axis is with world-up (1 = lies flat)
    q = d["clip"]["pose_world"]["quaternion_wxyz"]
    R = quat_wxyz_to_R(q)
    up_dot = float(np.dot(_norm(R @ MODEL_UP), WORLD_UP))

    dist = float(np.linalg.norm(d["clip"]["pose_se3"]["translation_m"]))

    area = np.nan
    mpath = os.path.join(os.path.dirname(jpath), d["image"].get("file_mask", ""))
    if os.path.isfile(mpath):
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            area = float((m > 127).mean())
    return dict(tilt=tilt, roll=roll, up_dot=up_dot, dist=dist, area=area)


def hist_line(name, vals, lo, hi, bins=10, width=40):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size == 0:
        return f"  {name:9s}: (no data)"
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    peak = max(h.max(), 1)
    rows = [f"  {name} [{vals.min():.2f}..{vals.max():.2f}] mean={vals.mean():.2f} std={vals.std():.2f} n={vals.size}"]
    for i in range(bins):
        bar = "#" * int(round(width * h[i] / peak))
        rows.append(f"    {edges[i]:7.2f}|{bar} {h[i]}")
    return "\n".join(rows)


# ----------------------------- scene diversity -----------------------------
def phash(bgr, hash_size=8):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)[:hash_size, :hash_size]
    med = np.median(dct[1:].flatten())
    bits = (dct > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def distinct_scene_count(bg_paths, thresh=6):
    """Greedy pHash clustering of the backplates -> # of visually distinct scenes."""
    clusters = []
    for p in bg_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        hcur = phash(im)
        if not any(hamming(hcur, c) <= thresh for c in clusters):
            clusters.append(hcur)
    return len(clusters)


# ----------------------------- realism proxy (torch-free) -----------------------------
def wasserstein1_1d(a, b, bins=64, rng=(0, 256)):
    """1-D Wasserstein-1 (earth-mover) distance between two value sets via CDF diff."""
    ha, _ = np.histogram(a, bins=bins, range=rng, density=True)
    hb, _ = np.histogram(b, bins=bins, range=rng, density=True)
    ca, cb = np.cumsum(ha), np.cumsum(hb)
    ca /= ca[-1] + 1e-9
    cb /= cb[-1] + 1e-9
    step = (rng[1] - rng[0]) / bins
    return float(np.sum(np.abs(ca - cb)) * step)


def hsv_samples(paths, n_px=200_000):
    """Pooled H,S,V pixel samples across a set of images (subsampled)."""
    Hs, Ss, Vs = [], [], []
    per = max(1, n_px // max(1, len(paths)))
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        idx = np.random.choice(hsv.shape[0], min(per, hsv.shape[0]), replace=False)
        s = hsv[idx]
        Hs.append(s[:, 0]); Ss.append(s[:, 1]); Vs.append(s[:, 2])
    if not Hs:
        return None
    return np.concatenate(Hs), np.concatenate(Ss), np.concatenate(Vs)


def colour_realism(overlay_paths, real_paths):
    o = hsv_samples(overlay_paths)
    r = hsv_samples(real_paths)
    if o is None or r is None:
        return None
    dH = wasserstein1_1d(o[0], r[0], bins=45, rng=(0, 180))   # hue is [0,180] in OpenCV
    dS = wasserstein1_1d(o[1], r[1], bins=64, rng=(0, 256))
    dV = wasserstein1_1d(o[2], r[2], bins=64, rng=(0, 256))
    return dict(hue=dH, sat=dS, val=dV, total=dH + dS + dV)


# ----------------------------- contact sheet -----------------------------
def contact_sheet(paths, out_path, cols=8, rows=6, cell=180, title=""):
    paths = list(paths)[: cols * rows]
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        r, c = divmod(i, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(im, (cell, cell))
    if title:
        cv2.putText(sheet, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(out_path, sheet)


# ----------------------------- FID (optional) -----------------------------
def try_fid(overlay_dir, real_dir):
    try:
        import torch_fidelity
    except ImportError:
        return None, "torch-fidelity not installed (pip install torch-fidelity)"
    m = torch_fidelity.calculate_metrics(
        input1=overlay_dir, input2=real_dir, cuda=False, fid=True, verbose=False)
    return float(m["frechet_inception_distance"]), None


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="overlay output dir (has NNN.json/NNN.png)")
    ap.add_argument("--real", default=None, help="dir of real MACS frames for the realism proxy")
    ap.add_argument("--n", type=int, default=500, help="max samples to score (subsample)")
    ap.add_argument("--fid", action="store_true", help="also compute true FID (needs torch-fidelity)")
    ap.add_argument("--out", default=None, help="report/contact-sheet prefix (default <dir>/_eval)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ddir = args.dir if os.path.isabs(args.dir) else os.path.join(here, args.dir)
    out_prefix = args.out or os.path.join(ddir, "_eval")

    jsons = sorted(glob.glob(os.path.join(ddir, "*.json")))
    jsons = [j for j in jsons if os.path.basename(j)[0].isdigit()]
    if args.n and len(jsons) > args.n:
        sel = np.linspace(0, len(jsons) - 1, args.n).astype(int)
        jsons = [jsons[i] for i in sel]
    print(f"Scoring {len(jsons)} samples from {ddir}\n")

    # --- pose spread ---
    mets = [pose_metrics_from_json(j) for j in jsons]
    print("=== POSE SPREAD ===")
    print(hist_line("tilt(deg)",  [m["tilt"] for m in mets], 0, 60))
    print(hist_line("roll(deg)",  [m["roll"] for m in mets], 0, 45))
    print(hist_line("up_dot",     [m["up_dot"] for m in mets], 0, 1))
    print(hist_line("area_frac",  [m["area"] for m in mets], 0, 0.3))
    ds = [m["dist"] for m in mets]
    print(hist_line("distance",   ds, min(ds), max(ds)))

    # --- scene diversity ---
    bgs = sorted(glob.glob(os.path.join(ddir, "*_bg.png")))
    if args.n and len(bgs) > args.n:
        sel = np.linspace(0, len(bgs) - 1, args.n).astype(int)
        bgs = [bgs[i] for i in sel]
    n_scene = distinct_scene_count(bgs)
    print("\n=== SCENE DIVERSITY ===")
    print(f"  backplates scored: {len(bgs)}   distinct scenes (pHash<=6): {n_scene}")
    # true source count if the generator recorded it (pHash over-counts when the
    # backplate is geometrically augmented, as in the old flat path).
    srcs = set()
    for j in jsons:
        try:
            with open(j) as f:
                s = json.load(f).get("image", {}).get("backplate_src")
            if s:
                srcs.add(s)
        except Exception:
            pass
    if srcs:
        print(f"  TRUE distinct source backplates used: {len(srcs)}")
    else:
        print("  (no backplate_src in JSON -> pHash count only; augmentation inflates it)")

    # --- realism proxy ---
    if args.real:
        rdir = args.real if os.path.isabs(args.real) else os.path.join(here, args.real)
        real_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            real_paths.extend(glob.glob(os.path.join(rdir, "**", ext), recursive=True))
        real_paths = real_paths[: max(args.n, 200)]
        overlays = [os.path.join(ddir, os.path.basename(j).replace(".json", ".png")) for j in jsons]
        overlays = [p for p in overlays if os.path.isfile(p)]
        cr = colour_realism(overlays, real_paths)
        print("\n=== REALISM PROXY (colour Wasserstein-1 vs real; lower=closer) ===")
        if cr:
            print(f"  hue={cr['hue']:.3f}  sat={cr['sat']:.3f}  val={cr['val']:.3f}  TOTAL={cr['total']:.3f}")
        contact_sheet(real_paths, out_prefix + "_real_grid.png", title="REAL")
        if args.fid:
            fid, err = try_fid(ddir, rdir)
            print(f"  FID: {fid:.2f}" if fid is not None else f"  FID: skipped ({err})")

    # --- contact sheet of overlays ---
    overlays = [os.path.join(ddir, os.path.basename(j).replace(".json", ".png")) for j in jsons]
    contact_sheet(overlays, out_prefix + "_grid.png", title=os.path.basename(ddir.rstrip("/\\")))
    print(f"\nContact sheet -> {out_prefix}_grid.png")


if __name__ == "__main__":
    main()
