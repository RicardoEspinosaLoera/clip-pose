"""
measure_appearance_stats.py — are the synthetic clips as sharp, contrasty, shiny and shadowed as the real ones?

Compares, on the training crop (clip box x1.5 -> 512 px, statistics inside the clip box), real MACS clip
boxes with one or more synthetic sets (NNNNN.png + NNNNN_mask.png). Two real sources (--real):
  crops  Inpaint_Realism/lora/clean_clips2/crops.jsonl, detector score >= 0.45, frames read from MACS_ROOT
         (laptop only: the DGX has no MACS frames)
  v2     real_eval_v2/frames + annotations/<annotator> (versioned, works on the DGX): box built from the
         annotated tip/back points (clip length L: +6 % along the axis, half-width 0.22 L across)
  auto   (default) crops when MACS_ROOT exists, else v2
Both keep boxes with max side <= --max-box (400 px, as in the 23 Oct tuning) so up/down-scaling to 512 is
comparable; a 600-px clip downscaled would look sharper than an 80-px one upscaled.
  lap      variance of the Laplacian after a 1-px denoise  (edge sharpness)
  grad90   90th percentile of the gradient magnitude        (edge strength)
  v_std    std of HSV value                                 (contrast)
  bright   % pixels with V > 225                            (specular highlights)
  dark     % pixels with V < 60                             (shadows, black metal, applier head)
  sat_med  median saturation
23 Oct 2026: anodized2 renders were flat (lap 0.4 vs 4.1 real, 0 % dark vs 19 %, bright 1.6 % vs 6.3 %), i.e.
real clips are SHARPER than the renders, not blurrier. The anodized_hc preset of restyle_appearance.py was
tuned with this script to lap 4.1 / grad 50 / v_std 50 / bright 3.5 % / dark 14 % (real 4.1 / 53 / 55 / 6.3 / 19).
Usage:  python measure_appearance_stats.py --dirs out_lclip out_lclip_hc [--n-real 250]
"""
import os
import json
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
MACS = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")
CROPS = os.path.join(REPO, "Inpaint_Realism", "lora", "clean_clips2", "crops.jsonl")
V2 = os.path.join(REPO, "real_eval_v2")


def crop512(img, x, y, w, h, margin=1.5):
    cx, cy, side = x + w / 2, y + h / 2, margin * max(w, h)
    s = 512 / side
    A = np.array([[s, 0, 256 - s * cx], [0, s, 256 - s * cy]], np.float64)
    return cv2.warpAffine(img, A, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), s


def stats(c, s, w, h):
    reg = np.zeros((512, 512), bool)
    reg[int(256 - s * h / 2):int(256 + s * h / 2), int(256 - s * w / 2):int(256 + s * w / 2)] = True
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gs = cv2.GaussianBlur(g, (0, 0), 1.0)                     # denoise first: sensor noise is not sharpness
    lap = cv2.Laplacian(gs, cv2.CV_32F)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2][reg], hsv[..., 1][reg]
    return dict(lap=float(lap[reg].var()), grad90=float(np.percentile(grad[reg], 90)), v_std=float(V.std()),
                bright=float((V > 225).mean() * 100), dark=float((V < 60).mean() * 100), sat_med=float(np.median(S)))


def real_stats(n, seed=0, max_box=400):
    """MACS detector boxes (crops.jsonl) read from the MACS root -- laptop only."""
    rng = np.random.default_rng(seed)
    rows = [json.loads(l) for l in open(CROPS)]
    rows = [r for r in rows if r["score"] >= 0.45 and max(r["box"][2], r["box"][3]) <= max_box]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        im = cv2.imread(os.path.join(MACS, r["src"].replace("\\", "/")))
        if im is None:
            continue
        x, y, w, h = r["box"]
        c, s = crop512(im, x, y, w, h)
        out.append(stats(c, s, w, h))
    return out


def box_from_points(pts, along=0.06, half_width=0.22):
    """Tight clip box from the annotated tip and back end: the clip axis plus a margin across it."""
    tip, back = np.array(pts["tip"], float), np.array(pts["back"], float)
    L = np.linalg.norm(back - tip)
    u = (back - tip) / max(L, 1e-6)
    v = np.array([-u[1], u[0]])
    corners = [tip - along * L * u + sgn * half_width * L * v for sgn in (-1, 1)] +               [back + along * L * u + sgn * half_width * L * v for sgn in (-1, 1)]
    corners = np.array(corners)
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def real_stats_v2(n, seed=0, max_box=400, annotator="ricardo"):
    """real_eval_v2 frames with the annotated clip axis (versioned with the repo -> runs on the DGX too)."""
    ann_dir = os.path.join(V2, "annotations", annotator)
    rows = []
    for f in sorted(os.listdir(ann_dir)):
        if not f.endswith(".json"):
            continue
        a = json.load(open(os.path.join(ann_dir, f)))
        pts = a.get("points") or {}
        if "tip" not in pts or "back" not in pts:
            continue
        x, y, w, h = box_from_points(pts)
        if max(w, h) <= max_box:
            rows.append((a["file"], (x, y, w, h)))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out = []
    for fname, (x, y, w, h) in rows[:n]:
        im = cv2.imread(os.path.join(V2, "frames", fname))
        if im is None:
            continue
        c, s = crop512(im, x, y, w, h)
        out.append(stats(c, s, w, h))
    return out


def synth_stats(d):
    dd = d if os.path.isabs(d) else os.path.join(HERE, d)
    out = []
    for f in sorted(os.listdir(dd)):
        if f.endswith(".json") and f[0].isdigit():
            st = f[:-5]
            img = cv2.imread(os.path.join(dd, st + ".png"))
            m = cv2.imread(os.path.join(dd, st + "_mask.png"), 0)
            if img is None or m is None or not (m > 127).any():
                continue
            ys, xs = np.nonzero(m > 127)
            x, y, w, h = xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1
            c, s = crop512(img, x, y, w, h)
            out.append(stats(c, s, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="synthetic sets to compare with the real clip boxes")
    ap.add_argument("--n-real", type=int, default=250)
    ap.add_argument("--max-synth", type=int, default=400, help="subsample large sets (first N stems)")
    ap.add_argument("--real", choices=("auto", "crops", "v2"), default="auto", help="real source (see module doc)")
    ap.add_argument("--max-box", type=int, default=400, help="keep real boxes with max side <= this (px)")
    ap.add_argument("--annotator", default="ricardo", help="--real v2: real_eval_v2/annotations/<annotator>")
    a = ap.parse_args()
    src = a.real
    if src == "auto":
        src = "crops" if os.path.isdir(MACS) else "v2"
    if src == "crops":
        real = real_stats(a.n_real, max_box=a.max_box)
    else:
        real = real_stats_v2(a.n_real, max_box=a.max_box, annotator=a.annotator)
    if not real:
        raise SystemExit(f"no real boxes from source '{src}' (MACS_ROOT={MACS}, max_box={a.max_box})")
    print(f"real source: {src} (max box side {a.max_box} px)")
    syn = {}
    for d in a.dirs:
        syn[d] = synth_stats(d)[:a.max_synth]

    def summ(rows, k):
        v = np.array([r[k] for r in rows])
        return f"{np.percentile(v, 25):6.1f} {np.median(v):6.1f} {np.percentile(v, 75):6.1f}"

    names = [os.path.basename(os.path.normpath(d)) for d in a.dirs]
    print(f"{'metric':8s} | {'REAL p25 med p75 (n=%d)' % len(real):24s} | " + " | ".join(f"{n[:22]:22s}" for n in names))
    for k in ("lap", "grad90", "v_std", "bright", "dark", "sat_med"):
        print(f"{k:8s} | {summ(real, k):24s} | " + " | ".join(f"{summ(syn[d], k):22s}" for d in a.dirs))
    print("lap/grad90 = sharpness after 1-px denoise; v_std = value std; bright = % V>225; dark = % V<60; sat_med = median saturation")


if __name__ == "__main__":
    main()
