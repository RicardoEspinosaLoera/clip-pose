"""
harvest_instruments.py — build a library of REAL instrument cutouts (RGBA) from MACS frames.

STATUS (2026-10): EXPERIMENTAL, low yield. GroundingDINO-tiny misses most dark shafts, a
colour mask merges them with the dark vignette, and SAM point/auto prompts around the clip
mostly return tissue regions. occlude.py therefore uses its procedural applier unless this
library exists and is manually vetted. A proper instrument detector/segmenter is needed.

Why: drawn occluders look fake. Real MACS clips are mostly occluded by instruments (the
applier holding the clip, suction, forceps): thick, dark, defocused shafts entering from the
frame border with glossy edges. Cutting those out of real frames and compositing them over
our renders gives occluders with real texture, blur and compression for free.

Per frame: GroundingDINO finds instrument boxes; a box is kept when it reaches a frame border
(the shaft enters from outside) and has a sane size; SAM segments it; the mask must be
achromatic-ish (metal/dark/white, not tissue), touch the border and be reasonably compact.
Output: Data/instruments/inst_NNNN.png (BGRA, bbox crop) + instruments.jsonl with the frame,
box, border side(s), mean colour, size in px and whether a clip box was adjacent (clip-holder).

Usage
  python harvest_instruments.py --max-frames 1106                  # frames that contain clips (crops.jsonl)
  python harvest_instruments.py --frames-dir <dir_of_frames> --max-frames 1500
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MACS_ROOT = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")
CROPS_JSONL = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2", "crops.jsonl"))
CROPS_DIR = os.path.dirname(CROPS_JSONL)
SAM_CKPT = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "models", "sam_vit_b_01ec64.pth"))
OUT_DIR = os.path.join(HERE, "Data", "instruments")
PROMPT = "clip applier. surgical forceps. bipolar forceps. suction tube. surgical instrument. metal instrument shaft."


def load_models(dev):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from segment_anything import sam_model_registry, SamPredictor
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(dev)
    return proc, gd, SamPredictor(sam)


def detect(proc, gd, im_bgr, dev, thr=0.28):
    H, W = im_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, text=PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = gd(**inp)
    post = proc.post_process_grounded_object_detection
    try:
        res = post(o, threshold=thr, input_ids=inp.input_ids, text_threshold=0.2, target_sizes=[(H, W)])[0]
    except TypeError:
        res = post(o, box_threshold=thr, input_ids=inp.input_ids, text_threshold=0.2, target_sizes=[(H, W)])[0]
    return [(float(s), np.array(b, float)) for s, b in zip(res["scores"].tolist(), res["boxes"].tolist())]


def color_candidates(im_bgr, min_area=0.008, max_area=0.40, min_elong=2.0):
    """Instrument candidates WITHOUT a detector: large, achromatic (dark shaft or bright
    suction/gauze), elongated, border-touching components. GroundingDINO-tiny misses most
    dark shafts, which are the typical occluder in MACS frames."""
    H, W = im_bgr.shape[:2]
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    f = im_bgr.astype(np.int16)
    chroma = f.max(2) - f.min(2)
    V, S = hsv[..., 2], hsv[..., 1]
    out = []
    for kind, m in (("dark", (V < 80) & (chroma < 45)), ("bright", (V > 195) & (chroma < 40) & (S < 60))):
        m8 = m.astype(np.uint8) * 255
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if not (min_area * H * W <= area <= max_area * H * W):
                continue
            if not border_sides(x, y, x + w, y + h, W, H):
                continue
            comp = lab == i
            ys, xs = np.nonzero(comp)
            pts = np.stack([xs, ys], 1).astype(np.float32)
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
            elong = max(rw, rh) / (min(rw, rh) + 1e-6)
            if elong < min_elong:
                continue
            out.append((0.5, np.array([x, y, x + w, y + h], float), comp, kind))
    return out


def border_sides(x0, y0, x1, y1, W, H, m=10):
    s = []
    if x0 < m: s.append("L")
    if y0 < m: s.append("T")
    if x1 > W - m: s.append("R")
    if y1 > H - m: s.append("B")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=None, help="dir of frames; default = source frames of clean_clips2")
    ap.add_argument("--max-frames", type=int, default=1106)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview-png", default=None, help="write a montage of the first 12 cutouts in context")
    a = ap.parse_args()
    a.preview = [] if a.preview_png else None
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc, gd, pred = load_models(dev)

    frames = []
    if a.frames_dir:
        frames = [(p, None) for p in sorted(glob.glob(os.path.join(a.frames_dir, "*.png")) + glob.glob(os.path.join(a.frames_dir, "*.jpg")))]
    else:
        rows = [json.loads(l) for l in open(CROPS_JSONL)]
        seen = set()
        for r in rows:
            if not os.path.exists(os.path.join(CROPS_DIR, r["file"])):
                continue
            p = os.path.join(MACS_ROOT, r["src"])
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            frames.append((p, r["box"]))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(frames)
    frames = frames[:a.max_frames]
    print(f"{len(frames)} frames | device {dev} | out {a.out}", flush=True)

    jl = open(os.path.join(a.out, "instruments.jsonl"), "a")
    n_saved = sum(1 for _ in glob.glob(os.path.join(a.out, "inst_*.png")))
    for fi, (fp, clipbox) in enumerate(frames):
        im = cv2.imread(fp)
        if im is None:
            continue
        H, W = im.shape[:2]
        cands = [(sc, b, None, "dino") for sc, b in sorted(detect(proc, gd, im, dev), key=lambda t: -t[0])[:4]]
        cands += color_candidates(im)
        kept = 0
        pred_set = False
        taken = []                                        # masks already saved from this frame (dedupe)
        for sc, b, comp, kind in cands:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            bw, bh = x1 - x0, y1 - y0
            area = bw * bh / (H * W)
            sides = border_sides(x0, y0, x1, y1, W, H)
            if not sides or area < 0.01 or area > 0.45:
                continue
            if not pred_set:
                pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                pred_set = True
            masks, scores, _ = pred.predict(box=np.array([x0, y0, x1, y1])[None, :], multimask_output=False)
            m = masks[0]
            if comp is not None:                          # SAM may leak into tissue: stay near the colour component
                m = m & (cv2.dilate(comp.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
            if clipbox is not None:                       # never carry the clip inside the cutout
                cx, cy, cw, ch = clipbox
                m[cy:cy + ch, cx:cx + cw] = False
            if any((m & t).sum() / max(1, (m | t).sum()) > 0.5 for t in taken):
                continue
            m8 = m.astype(np.uint8)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m8)
            if n < 2:
                continue
            big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
            m = lab == big
            if m.mean() < 0.004:
                continue
            ys, xs = np.nonzero(m)
            if not border_sides(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, W, H):
                continue
            # achromatic / dark / white gate: instruments are not red tissue
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            S, V = hsv[..., 1][m].astype(float), hsv[..., 2][m].astype(float)
            red_like = ((hsv[..., 0][m] < 12) | (hsv[..., 0][m] > 165)) & (S > 90)
            if red_like.mean() > 0.35:
                continue
            solidity = m.sum() / max(1, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            cut = np.dstack([im[by0:by1, bx0:bx1], (m[by0:by1, bx0:bx1] * 255).astype(np.uint8)])
            name = f"inst_{n_saved:04d}.png"
            cv2.imwrite(os.path.join(a.out, name), cut)
            adjacent = False
            if clipbox is not None:
                cx, cy, cw, ch = clipbox
                adjacent = bool(bx0 < cx + 1.5 * cw and bx1 > cx - 0.5 * cw and by0 < cy + 1.5 * ch and by1 > cy - 0.5 * ch)
            rec = dict(file=name, src=os.path.relpath(fp, MACS_ROOT) if fp.startswith(MACS_ROOT) else fp,
                       score=round(sc, 3), kind=kind, box=[int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)],
                       frame=[W, H], sides=sides, mean_bgr=[float(c) for c in im[m].mean(0)], mean_sat=float(S.mean()),
                       mean_val=float(V.mean()), solidity=round(float(solidity), 3),
                       clip_adjacent=adjacent, clip_box=clipbox)
            jl.write(json.dumps(rec) + "\n")
            taken.append(m)
            n_saved += 1
            kept += 1
            if a.preview is not None and len(a.preview) < 12:
                vis = im.copy()
                vis[m] = (0.45 * vis[m] + np.array([255, 0, 255]) * 0.55).astype(np.uint8)
                cv2.putText(vis, f"{kind} s{sc:.2f} {m.mean()*100:.1f}% {''.join(sides)}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
                a.preview.append(cv2.resize(vis, (480, 270)))
        if (fi + 1) % 50 == 0:
            print(f"[{fi+1}/{len(frames)}] cutouts so far: {n_saved}", flush=True)
    jl.close()
    print(f"done: {n_saved} cutouts in {a.out}")
    if a.preview:
        cells = a.preview + [np.zeros_like(a.preview[0])] * ((3 - len(a.preview) % 3) % 3)
        cv2.imwrite(a.preview_png, np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]))
        print("preview ->", a.preview_png)


if __name__ == "__main__":
    main()
