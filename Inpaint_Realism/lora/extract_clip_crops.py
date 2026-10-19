"""
extract_clip_crops.py — harvest real clip crops from MACS (or any real surgical
frames) to use as LoRA training data.

MACS has no clip annotations, so each candidate region is scored by the same kind
of complementary cues the paper's metallic-clip filter uses, combined into a
confidence in [0, 1]:

  * specular   — fraction of bright, achromatic (metallic) pixels
  * metallic   — fraction of low-saturation body pixels (the clip is grey, not red)
  * edges      — Canny edge density (clips have strong man-made edges)
  * texture    — Laplacian variance (sharp structure vs smooth tissue)
  * shape      — solidity + not-too-thin (rejects elongated vessel highlights)
  * sat-drop   — saturation is lower inside than around it (metal on red tissue)

Only regions whose score >= --min-score are kept (so frames without a clip are
skipped, not cropped), the top --topk regions per frame are taken, and near-
duplicate crops are removed with a perceptual hash (consecutive MACS frames look
almost identical). Detection is still imperfect — skim the output once — but it is
far more clip-specific and needs much less manual culling than a single bright blob.

Usage:
  python extract_clip_crops.py --frames <dir_of_real_frames> --out crops --size 768
  python extract_clip_crops.py --frames <dir> --min-score 0.5 --topk 2 --debug
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np


# ----------------------------- semantic detector (GroundingDINO) -----------------------------
# Zero-shot, open-vocabulary: localises the clip by MEANING, not colour, so it is
# not fooled by white gauze / specular tissue / instruments the way the colour
# heuristic is. Requires a ~700 MB model download on first use.
_GDINO = {"model": None, "processor": None}


def load_gdino(model_name="IDEA-Research/grounding-dino-tiny"):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if _GDINO["model"] is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _GDINO["processor"] = AutoProcessor.from_pretrained(model_name)
        _GDINO["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(dev)
        _GDINO["device"] = dev
    return _GDINO


def detect_gdino(bgr, text_prompt, box_threshold, text_threshold, topk,
                 min_area_frac=0.0008, max_area_frac=0.35):
    """Return [(score, (x,y,w,h)), ...] from GroundingDINO for `text_prompt`.

    Boxes larger than `max_area_frac` of the frame are dropped: a clip is a small
    object, never the whole field, so these are GroundingDINO's spurious global
    boxes (and would otherwise outrank the tight clip box under --topk)."""
    import torch
    from PIL import Image
    g = load_gdino()
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img_area = bgr.shape[0] * bgr.shape[1]
    inputs = g["processor"](images=pil, text=text_prompt, return_tensors="pt").to(g["device"])
    with torch.no_grad():
        outputs = g["model"](**inputs)
    post = g["processor"].post_process_grounded_object_detection
    common = dict(input_ids=inputs.input_ids, text_threshold=text_threshold,
                  target_sizes=[pil.size[::-1]])
    try:                                      # newer transformers: `threshold`
        res = post(outputs, threshold=box_threshold, **common)[0]
    except TypeError:                         # older transformers: `box_threshold`
        res = post(outputs, box_threshold=box_threshold, **common)[0]
    out = []
    for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
        x0, y0, x1, y1 = box
        w, h = int(x1 - x0), int(y1 - y0)
        area = w * h
        if area < min_area_frac * img_area or area > max_area_frac * img_area:
            continue
        out.append((float(score), (int(x0), int(y0), w, h)))
    out.sort(key=lambda t: t[0], reverse=True)
    return out[:topk]


# ----------------------------- titanium colour gate -----------------------------
def metal_gate(bgr, box, min_metal=0.10, max_yellow=0.22, max_blue=0.22,
               max_white=0.55, max_dark=0.70):
    """Reject a GroundingDINO box whose colours are NOT a titanium clip.

    GroundingDINO happily labels yellow forceps, blue vessel loops, white
    cottonoids and black sutures as "clip". A real aneurysm clip is achromatic
    blue-grey metal with specular glints. This colour gate keeps only boxes that
    look metallic and are NOT dominated by a saturated yellow / blue hue, a blown
    white patty, or a near-black instrument shaft.

    Returns (passes: bool, stats: dict) for logging.
    """
    x, y, w, h = box
    roi = bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return False, {}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f = roi.astype(np.int16)
    chroma = f.max(axis=2) - f.min(axis=2)

    n = roi.shape[0] * roi.shape[1]
    metal  = float(((chroma < 35) & (V > 55) & (V < 252)).mean())   # grey metal
    yellow = float(((H >= 18) & (H <= 45) & (S > 70) & (V > 70)).mean())
    blue   = float(((H >= 90) & (H <= 140) & (S > 70) & (V > 50)).mean())
    white  = float(((V > 205) & (S < 45)).mean())                   # blown cottonoid
    dark   = float((V < 45).mean())                                 # black shaft/suture

    passes = (metal >= min_metal and yellow <= max_yellow and
              blue <= max_blue and white <= max_white and dark <= max_dark)
    return passes, {"metal": round(metal, 3), "yellow": round(yellow, 3),
                    "blue": round(blue, 3), "white": round(white, 3),
                    "dark": round(dark, 3)}


# ----------------------------- metallic candidate mask -----------------------------
def achromatic_mask(bgr):
    """Grey/metallic pixels = low chroma (R≈G≈B), not pitch-black, not blown white.
    Chroma separates metal from red tissue at ANY brightness, unlike a brightness
    threshold which only catches glints."""
    f = bgr.astype(np.int16)
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    chroma = mx - mn
    return ((chroma < 35) & (mx > 55) & (mx < 252)).astype(np.uint8) * 255


def metallic_components(bgr):
    """Return (stats, specular_mask) for clip-like achromatic regions.

    A morphological OPENING removes the small bright tissue glints that plagued the
    brightness-based detector, leaving large contiguous metallic structures.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    specular = ((hsv[..., 2] > 210) & (hsv[..., 1] < 70)).astype(np.uint8)

    metal = achromatic_mask(bgr)
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    metal = cv2.morphologyEx(metal, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    _, _, stats, _ = cv2.connectedComponentsWithStats(metal, connectivity=8)
    return stats, specular


# ----------------------------- per-region scoring -----------------------------
def score_region(bgr, gray, hsv, specular, box):
    """Confidence in [0,1] that `box` (x,y,w,h) contains a metallic clip."""
    x, y, w, h = box
    H, W = gray.shape
    img_area = H * W
    area = w * h
    if area < 0.0008 * img_area or area > 0.30 * img_area:
        return 0.0
    if min(w, h) < 12:
        return 0.0

    roi = slice(y, y + h), slice(x, x + w)
    g = gray[roi]
    S = hsv[..., 1][roi]
    spec = specular[roi]
    roi_bgr = bgr[roi].astype(np.int16)
    chroma = roi_bgr.max(axis=2) - roi_bgr.min(axis=2)

    # 1) specular fraction
    f_spec = float(spec.mean())
    # 2) metallic fraction = achromatic (grey) pixels
    f_metal = float((chroma < 35).mean())
    # 3) edge density
    edges = cv2.Canny(g, 60, 160)
    f_edge = float((edges > 0).mean())
    # 4) texture (normalised Laplacian variance)
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())
    f_tex = float(np.clip(lap_var / 800.0, 0.0, 1.0))
    # 5) shape: solidity + not-too-thin
    aspect = max(w, h) / max(1.0, min(w, h))
    f_shape = float(np.clip(1.3 - aspect / 6.0, 0.0, 1.0))
    # 6) saturation drop vs a surrounding ring
    pad = int(0.4 * max(w, h))
    yy0, yy1 = max(0, y - pad), min(H, y + h + pad)
    xx0, xx1 = max(0, x - pad), min(W, x + w + pad)
    ring = hsv[..., 1][yy0:yy1, xx0:xx1].mean()
    f_sat = float(np.clip((ring - S.mean()) / 60.0, 0.0, 1.0))

    score = (0.26 * f_spec + 0.20 * f_metal + 0.18 * f_edge +
             0.16 * f_tex + 0.10 * f_shape + 0.10 * f_sat)
    # require at least one genuine glint — kills smooth pink/red tissue
    if f_spec < 0.01:
        score *= 0.3
    return float(np.clip(score, 0.0, 1.0))


def candidate_boxes(bgr, min_score, topk):
    """Return scored, sorted candidate boxes [(score, (x,y,w,h)), ...].

    Ranking is area-weighted so the large metallic clip body outranks any small
    leftover glint, while `min_score` still gates out non-metallic regions.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    stats, specular = metallic_components(bgr)
    img_area = bgr.shape[0] * bgr.shape[1]

    cands = []
    for i in range(1, stats.shape[0]):              # skip background label 0
        x, y, w, h, comp_area = stats[i]
        s = score_region(bgr, gray, hsv, specular, (x, y, w, h))
        if s >= min_score:
            rank = s * float(np.sqrt(comp_area / img_area))   # favour larger regions
            cands.append((rank, s, (int(x), int(y), int(w), int(h))))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [(s, box) for _, s, box in cands[:topk]]


# ----------------------------- cropping -----------------------------
def square_crop(bgr, box, size, pad=1.6):
    H, W = bgr.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(int(max(w, h) * pad), 64)
    x0 = int(np.clip(cx - side / 2, 0, W - 1))
    y0 = int(np.clip(cy - side / 2, 0, H - 1))
    x1 = int(np.clip(x0 + side, 0, W))
    y1 = int(np.clip(y0 + side, 0, H))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ----------------------------- perceptual-hash dedup -----------------------------
def phash(bgr, hash_size=8):
    """64-bit DCT perceptual hash for near-duplicate rejection."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)[:hash_size, :hash_size]
    med = np.median(dct[1:].flatten())              # ignore DC term for the median
    bits = (dct > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of real surgical frames")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--max", type=int, default=2000, help="max crops to write")
    ap.add_argument("--pad", type=float, default=1.6)
    ap.add_argument("--topk", type=int, default=1, help="max clip regions per frame")
    ap.add_argument("--min-score", type=float, default=0.45, help="confidence threshold")
    ap.add_argument("--dedup-hamming", type=int, default=8,
                    help="reject a crop within this pHash distance of a kept one")
    ap.add_argument("--debug", action="store_true", help="also save boxed frames to out/_debug")
    ap.add_argument("--detector", choices=["gdino", "heuristic"], default="gdino",
                    help="gdino = semantic open-vocab (robust); heuristic = colour-only, no download")
    ap.add_argument("--text-prompt", default="a surgical metal clip. an aneurysm clip.",
                    help="GroundingDINO query (lowercase, period-separated)")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1,
                    help="scan every Nth frame (subsample huge folders)")
    ap.add_argument("--path-contains", default=None,
                    help="only scan frames whose path contains this substring "
                         r"(e.g. \1\ to keep only the valid-content label folders)")
    ap.add_argument("--metal-gate", action="store_true",
                    help="reject gdino boxes that are not titanium-coloured "
                         "(kills yellow forceps / blue loops / white patties / black sutures)")
    ap.add_argument("--metal-min", type=float, default=0.10,
                    help="min achromatic-metal pixel fraction inside the box")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(out_dir, exist_ok=True)
    dbg_dir = os.path.join(out_dir, "_debug")
    if args.debug:
        os.makedirs(dbg_dir, exist_ok=True)

    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        frames.extend(glob.glob(os.path.join(args.frames, "**", ext), recursive=True))
    frames = sorted(frames)
    if args.path_contains:
        frames = [f for f in frames if args.path_contains in f]
    if args.stride > 1:
        frames = frames[::args.stride]
    print(f"Detector: {args.detector} | scanning {len(frames)} frames "
          f"(stride {args.stride}) -> crops in {out_dir} "
          f"(topk {args.topk}, dedup<={args.dedup_hamming}, "
          f"metal_gate={args.metal_gate})")
    if args.detector == "gdino":
        print(f"Loading GroundingDINO (first run downloads ~700 MB) | prompt: {args.text_prompt!r}")
        load_gdino()

    gate_stats = {"rejected": 0}

    def detect(bgr):
        if args.detector == "gdino":
            # over-fetch so the metal gate has candidates left after filtering
            k = args.topk * 4 if args.metal_gate else args.topk
            cands = detect_gdino(bgr, args.text_prompt, args.box_threshold,
                                 args.text_threshold, k)
            if args.metal_gate:
                kept = []
                for score, box in cands:
                    ok, _ = metal_gate(bgr, box, min_metal=args.metal_min)
                    if ok:
                        kept.append((score, box))
                    else:
                        gate_stats["rejected"] += 1
                cands = kept[:args.topk]
            return cands
        return candidate_boxes(bgr, args.min_score, args.topk)

    kept_hashes = []
    meta = []
    n = skipped_dup = 0
    for fp in frames:
        if n >= args.max:
            break
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        cands = detect(bgr)
        if not cands:
            continue

        dbg = bgr.copy() if args.debug else None
        for score, box in cands:
            if n >= args.max:
                break
            crop = square_crop(bgr, box, args.size, args.pad)
            if crop is None:
                continue
            ph = phash(crop)
            if any(hamming(ph, k) <= args.dedup_hamming for k in kept_hashes):
                skipped_dup += 1
                continue
            kept_hashes.append(ph)

            name = f"clip_{n:04d}_s{int(score*100):02d}.png"
            cv2.imwrite(os.path.join(out_dir, name), crop)
            meta.append({"file": name, "src": os.path.relpath(fp, args.frames),
                         "score": round(score, 3), "box": list(box)})
            if dbg is not None:
                x, y, w, h = box
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(dbg, f"{score:.2f}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            n += 1

        if dbg is not None and cands:
            cv2.imwrite(os.path.join(dbg_dir, os.path.basename(fp)), dbg)

    with open(os.path.join(out_dir, "crops.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print(f"Wrote {n} crops ({skipped_dup} near-duplicates skipped, "
          f"{gate_stats['rejected']} rejected by metal gate). "
          f"Filenames carry the confidence score (clip_NNNN_sSS.png).")
    print("Skim the lowest-score crops and delete any without a real clip before training.")


if __name__ == "__main__":
    main()
