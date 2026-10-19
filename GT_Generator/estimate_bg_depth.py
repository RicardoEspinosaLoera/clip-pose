"""
estimate_bg_depth.py
====================

Stage that complements the depth-aware generator (`render_clip_poses_depth.py`).

For each sample it:
  1. Estimates a monocular depth map of the tissue backplate `NNN_bg.png`
     using Depth Anything V2 (relative / affine-invariant depth).
  2. Loads the metric clip depth `NNN_depth_clip.npy` rendered by VTK.
  3. Fuses them into a single ControlNet-ready depth map `NNN_depth_fused.png`
     in which the clip is placed *at / just in front of* the local tissue
     surface, so the downstream diffusion inpainter renders a physically
     plausible clip-tissue contact instead of a floating clip.

Why two depth sources?
  * The clip depth from VTK is exact and metric (it is the render's ground
    truth), but the backplate has no geometry, so it carries no depth.
  * Monocular estimation gives the *tissue* depth, but only up to an unknown
    global scale + shift.  We therefore keep the clip metric, estimate the
    tissue relative depth, and align the clip into the tissue's value range
    around the contact ring so the two are mutually consistent.

Outputs per sample (written next to the inputs):
  NNN_depth_bg.npy      float32 relative inverse-depth of the backplate
  NNN_depth_bg.png      8-bit visualisation (closer = brighter)
  NNN_depth_fused.png   8-bit ControlNet-depth input (closer = brighter)
  NNN_contact_mask.png  8-bit ring around the clip = the free "contact" zone

Run:
  python estimate_bg_depth.py --dir out_rand_depth
Dependencies: torch, transformers, pillow, opencv-python, numpy
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np
from PIL import Image

# Depth Anything V2 via the transformers depth-estimation pipeline.
# Model choices (speed vs quality):
#   depth-anything/Depth-Anything-V2-Small-hf   (fast, default)
#   depth-anything/Depth-Anything-V2-Base-hf
#   depth-anything/Depth-Anything-V2-Large-hf   (best)
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def build_depth_pipeline(model_name=DEFAULT_MODEL):
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("depth-estimation", model=model_name, device=device)


def estimate_bg_depth(pipe, bg_path):
    """
    Return a float32 relative depth map where LARGER = CLOSER (disparity-like),
    normalised to [0, 1].  Depth Anything outputs disparity, so closer surfaces
    already get larger values; we just min-max normalise.
    """
    img = Image.open(bg_path).convert("RGB")
    out = pipe(img)
    pred = out.get("predicted_depth", None)
    if pred is not None and hasattr(pred, "cpu"):
        disp = pred.squeeze().cpu().numpy().astype(np.float32)   # raw disparity (H,W)
    else:
        disp = np.asarray(out["depth"], dtype=np.float32)         # normalised PIL fallback
    # ensure it matches the image size
    disp = cv2.resize(disp, (img.width, img.height), interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(disp.min()), float(disp.max())
    disp = (disp - dmin) / max(dmax - dmin, 1e-9)        # [0,1], closer brighter
    return disp


def metric_to_disparity(metric):
    """Convert metric clip distance (NaN outside) to a [0,1] disparity where
    closer = larger, normalised within the clip only."""
    valid = np.isfinite(metric)
    disp = np.zeros(metric.shape, dtype=np.float32)
    if valid.any():
        m = metric[valid]
        inv = 1.0 / np.clip(m, 1e-6, None)
        lo, hi = float(inv.min()), float(inv.max())
        disp[valid] = (inv - lo) / max(hi - lo, 1e-9)
    return disp, valid


def fuse_depths(clip_metric, bg_disp, contact_margin=0.06, ring_px=21):
    """
    Compose a single disparity map: tissue everywhere, clip pasted on top and
    shifted so its *contact* end matches the tissue level at the surrounding
    ring, plus a small margin so the clip reads as touching / pressing in.

    Returns (fused_disp [0,1], contact_mask uint8).
    """
    clip_disp, clip_valid = metric_to_disparity(clip_metric)

    # contact ring = dilated clip minus clip = the tissue immediately around it
    clip_u8 = (clip_valid * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    dilated = cv2.dilate(clip_u8, k)
    ring = cv2.subtract(dilated, clip_u8)
    ring_bool = ring > 0

    # tissue disparity level at the contact ring (median = robust surface level)
    if ring_bool.any():
        tissue_level = float(np.median(bg_disp[ring_bool]))
    else:
        tissue_level = float(np.median(bg_disp))

    # place the clip so its lowest-disparity (far) end sits at the tissue level,
    # and everything nearer is in front of the tissue (larger disparity).
    fused = bg_disp.copy()
    if clip_valid.any():
        placed = tissue_level + contact_margin + clip_disp * (1.0 - tissue_level - contact_margin)
        fused[clip_valid] = np.clip(placed[clip_valid], 0.0, 1.0)

    return fused.astype(np.float32), (dilated > 0).astype(np.uint8) * 255


def process_sample(pipe, json_path):
    d = os.path.dirname(json_path)
    with open(json_path) as f:
        meta = json.load(f)
    stem = os.path.splitext(os.path.basename(json_path))[0]

    bg_path   = os.path.join(d, meta["image"].get("file_bg", f"{stem}_bg.png"))
    clip_npy  = os.path.join(d, meta["image"].get("file_depth_clip_npy", f"{stem}_depth_clip.npy"))
    if not (os.path.exists(bg_path) and os.path.exists(clip_npy)):
        print(f"[SKIP] {stem}: missing bg or clip depth")
        return

    bg_disp = estimate_bg_depth(pipe, bg_path)
    clip_metric = np.load(clip_npy)
    if clip_metric.shape != bg_disp.shape:
        clip_metric = cv2.resize(clip_metric, (bg_disp.shape[1], bg_disp.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    fused, contact_mask = fuse_depths(clip_metric, bg_disp)

    np.save(os.path.join(d, f"{stem}_depth_bg.npy"), bg_disp)
    cv2.imwrite(os.path.join(d, f"{stem}_depth_bg.png"), (bg_disp * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(d, f"{stem}_depth_fused.png"), (fused * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(d, f"{stem}_contact_mask.png"), contact_mask)

    # record in JSON
    meta.setdefault("image", {})
    meta["image"]["file_depth_bg"] = f"{stem}_depth_bg.png"
    meta["image"]["file_depth_fused"] = f"{stem}_depth_fused.png"
    meta["image"]["file_contact_mask"] = f"{stem}_contact_mask.png"
    meta.setdefault("depth", {})["bg_source"] = DEFAULT_MODEL
    meta["depth"]["fused_convention"] = "disparity [0,1], closer=brighter, ControlNet-depth input"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] {stem}: bg/fused/contact depth written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_rand_depth", help="directory with NNN.json samples")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    work = args.dir if os.path.isabs(args.dir) else os.path.join(here, args.dir)
    jsons = sorted(glob.glob(os.path.join(work, "*.json")))
    if not jsons:
        print(f"No JSON samples found in {work}")
        return

    print(f"Loading depth model {args.model} ...")
    pipe = build_depth_pipeline(args.model)
    print(f"Processing {len(jsons)} samples in {work}")
    for jp in jsons:
        process_sample(pipe, jp)


if __name__ == "__main__":
    main()
