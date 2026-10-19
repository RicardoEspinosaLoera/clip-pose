"""
sample_macs_frames.py — pick a diverse set of MACS frames to train a DOMAIN LoRA.

A domain LoRA trained on whole MACS frames teaches SDXL the real microsurgical
appearance (colour palette, microscope optics, tissue/blood texture, specular
lighting) so that `generate_scene.py` produces authentic scenes instead of the
CGI/abstract hallucinations generic SDXL falls into.

Unlike the clip-crop extractor, we do NOT localise anything — we just want varied,
representative full frames. Near-duplicate frames (consecutive video frames look
almost identical) are removed with a perceptual hash, and each kept frame is made
square (resize longest side + center crop) for training.

Usage:
  python sample_macs_frames.py --frames <dir_of_MACS_frames> --out macs_frames --size 768 --max 1500
"""

import os
import re
import glob
import argparse

import cv2
import numpy as np


def vid_key(path):
    """Top-level video id for a frame path (e.g. 'vid_07'), or '_' if none."""
    m = re.search(r"(vid_\d+)", path.replace("/", "\\"))
    return m.group(1) if m else "_"


def interleave_by_video(frames):
    """Round-robin the frames across their source videos so a max-cap stop still
    covers ALL videos instead of front-loading the alphabetically-first ones."""
    groups = {}
    for f in frames:
        groups.setdefault(vid_key(f), []).append(f)
    for k in groups:
        groups[k].sort()
    order = sorted(groups)
    out = []
    i = 0
    while True:
        added = False
        for k in order:
            g = groups[k]
            if i < len(g):
                out.append(g[i])
                added = True
        if not added:
            break
        i += 1
    return out, {k: len(v) for k, v in groups.items()}


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


def square(bgr, size):
    """Resize longest side to `size`, then center-crop to a square."""
    h, w = bgr.shape[:2]
    scale = size / min(h, w)
    r = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    H, W = r.shape[:2]
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return r[y0:y0 + size, x0:x0 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of MACS frames")
    ap.add_argument("--out", default="macs_frames")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--dedup-hamming", type=int, default=10,
                    help="reject a frame within this pHash distance of a kept one")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth file before dedup")
    ap.add_argument("--path-contains", default=None,
                    help=r"only scan frames whose path contains this (e.g. \1\ for label-1)")
    ap.add_argument("--balanced", action="store_true",
                    help="round-robin across vid_* folders so --max covers ALL videos")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        frames.extend(glob.glob(os.path.join(args.frames, "**", ext), recursive=True))
    frames = sorted(frames)
    if args.path_contains:
        frames = [f for f in frames if args.path_contains in f]
    frames = frames[::args.stride]
    if args.balanced:
        frames, counts = interleave_by_video(frames)
        print(f"Balanced round-robin across {len(counts)} videos: "
              + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"Scanning {len(frames)} frames -> diverse set in {out_dir} (dedup<={args.dedup_hamming})")

    kept_hashes = []
    kept_by_vid = {}
    n = skipped = 0
    for fp in frames:
        if n >= args.max:
            break
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        ph = phash(bgr)
        if any(hamming(ph, k) <= args.dedup_hamming for k in kept_hashes):
            skipped += 1
            continue
        kept_hashes.append(ph)
        cv2.imwrite(os.path.join(out_dir, f"macs_{n:05d}.png"), square(bgr, args.size))
        kept_by_vid[vid_key(fp)] = kept_by_vid.get(vid_key(fp), 0) + 1
        n += 1

    print(f"Wrote {n} diverse frames ({skipped} near-duplicates skipped).")
    print("Kept per video: " + ", ".join(f"{k}:{v}" for k, v in sorted(kept_by_vid.items())))
    print(f"Next: python train_lora_sdxl.py --crops {os.path.basename(out_dir)} "
          f"--out macs_lora --size {args.size} --steps 2000 "
          f"--instance-prompt \"MACS intraoperative microsurgical photograph\"")


if __name__ == "__main__":
    main()
