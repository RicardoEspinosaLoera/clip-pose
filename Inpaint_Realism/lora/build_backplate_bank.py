"""
build_backplate_bank.py — harvest a large, diverse bank of CLIP-FREE real MACS
tissue frames to use as backplates for GT_Generator/render_clip_poses_depth.py.

Today the renderer draws all 5000 samples from ~30 hand-picked images in
Data/noclips, so the pose net sees only 30 scenes. This builds hundreds–thousands
of real intraoperative backplates instead, grounding scene appearance in genuine
footage and multiplying diversity for free.

It reuses the existing tooling:
  * sample_macs_frames : balanced round-robin across videos + pHash dedup
  * extract_clip_crops : the validated GroundingDINO + titanium metal-gate clip
                         detector, used here INVERTED — to REJECT any frame that
                         already contains a clip (we want empty tissue to place
                         our own synthetic clip on; a leaked real clip would be an
                         un-annotated object that confuses training).

Two rejection detectors:
  --detector gdino     (default) semantic clip detector + metal gate. Accurate;
                       needs the ~700 MB GroundingDINO download and is slower.
                       Rejects frames with a clip-like titanium object.
  --detector heuristic colour-only, no download, fast. Rejects frames with a large
                       bright achromatic-metal blob (clip OR shiny instrument), so
                       it also drops some tool scenes — coarser but instant.
  --no-reject          keep every frame (fastest; only for a quick look).

Usage
-----
  # accurate clip-free bank of ~2000 frames, balanced across all videos
  python build_backplate_bank.py \
      --frames "C:/Users/ricar/OneDrive/Documentos/MAC_dataset/23533731" \
      --out ../../GT_Generator/Data/noclips_bank \
      --path-contains "\\1\\" --balanced --stride 5 --max 2000 \
      --detector gdino --reject-clip-score 0.35

  # fast dry run without the download
  python build_backplate_bank.py --frames <ds> --out <dir> --path-contains "\\1\\" \
      --stride 20 --max 300 --detector heuristic
"""

import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse the validated helpers
from sample_macs_frames import interleave_by_video, vid_key  # noqa: E402
from extract_clip_crops import (                              # noqa: E402
    phash, hamming, detect_gdino, load_gdino, metal_gate, candidate_boxes,
)


def resize_longest(bgr, size):
    """Downscale so the longest side == size (keep aspect); never upscale."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def quality_ok(bgr, min_bright, max_bright, min_sharp):
    """Reject too-dark / blown-out / blurry (out-of-focus transition) frames."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mv = float(g.mean())
    if mv < min_bright or mv > max_bright:
        return False, f"bright={mv:.0f}"
    sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
    if sharp < min_sharp:
        return False, f"sharp={sharp:.0f}"
    return True, ""


def has_clip(bgr, args):
    """True if the frame appears to already contain a clip (=> reject it)."""
    if args.detector == "none":
        return False
    if args.detector == "heuristic":
        # any large bright metallic blob scoring above threshold -> treat as clip/instrument
        cands = candidate_boxes(bgr, min_score=args.reject_clip_score, topk=1)
        return len(cands) > 0
    # gdino + metal gate: reject only genuine titanium-clip-like boxes
    cands = detect_gdino(bgr, args.text_prompt, args.box_threshold,
                         args.text_threshold, topk=4)
    for score, box in cands:
        if score < args.reject_clip_score:
            continue
        ok, _ = metal_gate(bgr, box, min_metal=args.metal_min)
        if ok:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="MACS dataset root (recursively scanned)")
    ap.add_argument("--out", required=True, help="output backplate-bank dir")
    ap.add_argument("--size", type=int, default=960, help="downscale longest side to this")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=5, help="take every Nth file before dedup")
    ap.add_argument("--path-contains", default=None,
                    help=r"only scan frames whose path contains this (e.g. \1\ for label-1)")
    ap.add_argument("--balanced", action="store_true",
                    help="round-robin across vid_* folders so --max covers ALL videos")
    ap.add_argument("--dedup-hamming", type=int, default=12,
                    help="reject a frame within this pHash distance of a kept one")
    # quality gates
    ap.add_argument("--min-bright", type=float, default=35.0)
    ap.add_argument("--max-bright", type=float, default=225.0)
    ap.add_argument("--min-sharp", type=float, default=40.0)
    # clip rejection
    ap.add_argument("--detector", choices=["gdino", "heuristic", "none"], default="gdino")
    ap.add_argument("--reject-clip-score", type=float, default=0.35,
                    help="reject frame if a clip is detected at >= this score")
    ap.add_argument("--text-prompt",
                    default="a metallic aneurysm clip. a titanium surgical clip closing a blood vessel.")
    ap.add_argument("--box-threshold", type=float, default=0.32)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--metal-min", type=float, default=0.10)
    ap.add_argument("--debug", action="store_true", help="save rejected-as-clip frames to out/_rejected")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    rej_dir = os.path.join(out_dir, "_rejected")
    if args.debug:
        os.makedirs(rej_dir, exist_ok=True)

    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        frames.extend(glob.glob(os.path.join(args.frames, "**", ext), recursive=True))
    frames = sorted(frames)
    if args.path_contains:
        frames = [f for f in frames if args.path_contains in f]
    frames = frames[::args.stride]
    if args.balanced:
        frames, counts = interleave_by_video(frames)
        print("Balanced across videos: " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"Scanning {len(frames)} frames -> bank in {out_dir} "
          f"(detector={args.detector}, dedup<={args.dedup_hamming})")
    if args.detector == "gdino":
        print(f"Loading GroundingDINO (first run downloads ~700 MB) | reject prompt: {args.text_prompt!r}")
        load_gdino()

    kept_hashes, kept_by_vid, meta = [], {}, []
    n = dup = badq = clip = 0
    for fp in frames:
        if n >= args.max:
            break
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        ok, why = quality_ok(bgr, args.min_bright, args.max_bright, args.min_sharp)
        if not ok:
            badq += 1
            continue
        ph = phash(bgr)
        if any(hamming(ph, k) <= args.dedup_hamming for k in kept_hashes):
            dup += 1
            continue
        if has_clip(bgr, args):
            clip += 1
            if args.debug:
                cv2.imwrite(os.path.join(rej_dir, os.path.basename(fp)), resize_longest(bgr, 512))
            continue
        kept_hashes.append(ph)
        out = resize_longest(bgr, args.size)
        name = f"bp_{n:05d}.png"
        cv2.imwrite(os.path.join(out_dir, name), out)
        kv = vid_key(fp)
        kept_by_vid[kv] = kept_by_vid.get(kv, 0) + 1
        meta.append({"file": name, "src": os.path.relpath(fp, args.frames), "vid": kv})
        n += 1
        if n % 100 == 0:
            print(f"  kept {n} (dup {dup}, badq {badq}, clip {clip})")

    with open(os.path.join(out_dir, "bank.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print(f"\nWrote {n} backplates "
          f"({dup} near-dups, {badq} low-quality, {clip} clip-containing rejected).")
    print("Kept per video: " + ", ".join(f"{k}:{v}" for k, v in sorted(kept_by_vid.items())))
    print(f"Next: point the renderer at it -> set BG_DIR / env BP_DIR = {out_dir}")


if __name__ == "__main__":
    main()
