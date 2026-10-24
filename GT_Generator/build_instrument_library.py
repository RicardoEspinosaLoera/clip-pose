"""
build_instrument_library.py — REAL instrument cutouts (BGRA) for occlude.py, harvested from MACS.

Why: the procedural occluders (soft blob of tissue, shaded cylinder) look synthetic. Real MACS frames
are occluded by real instruments (applier, suction, bipolar) that enter from a frame border, carry the
frame's own lighting, blur and colour cast. occlude.py already prefers real cutouts from
Data/instruments/instruments.jsonl when they exist; this script builds that library.

For every backplate of the bank (Data/noclips_bank/bank.jsonl -> original 1920x1080 MACS frame):
  1. GroundingDINO (same model as the generator) with an instrument-only prompt;
  2. keep boxes that TOUCH a frame border (held from outside) and cover 1-45 % of the frame;
  3. GrabCut inside the box, seeded by a metal prior (low saturation / dark / specular), keep the
     component that reaches the border, drop thin or tiny results;
  4. feathered alpha -> Data/instruments/instr_NNNN.png (cropped BGRA) + a jsonl record
     {file, frame: [W, H], sides: [borders touched], bbox, src, score, area_frac}.
Usage:
  python build_instrument_library.py                      # all bank frames
  python build_instrument_library.py --limit 60 --montage # quick look: Data/instruments/_montage.jpg
Env: MACS_ROOT (default C:\\Users\\ricar\\OneDrive\\Documentos\\MAC_dataset\\23533731)
"""
import os
import json
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "Data", "noclips_bank")
OUT_DIR = os.path.join(HERE, "Data", "instruments")
MACS_ROOT = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")
DINO_MODEL = os.environ.get("DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
PROMPT = ("surgical forceps. bipolar forceps. clip applier. suction tube. suction cannula. "
          "surgical instrument. metal instrument. instrument shaft. scissors. dissector.")
BORDER = 12          # px: a box/mask within this distance of the frame edge "touches" it
_G = {}


def _import_transformers_dino():
    """`from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection`, tolerating a broken
    torchvision (on the laptop a Windows Application Control policy blocks torchvision/_C.pyd, WinError 4551):
    transformers only uses torchvision for optional fast image paths, so declare it absent and retry."""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except Exception:
        import transformers.utils as tu
        import transformers.utils.import_utils as iu
        iu.is_torchvision_available = lambda: False
        tu.is_torchvision_available = lambda: False
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    return AutoProcessor, AutoModelForZeroShotObjectDetection


def load_dino():
    import torch
    AutoProcessor, AutoModelForZeroShotObjectDetection = _import_transformers_dino()
    if not _G:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _G["proc"] = AutoProcessor.from_pretrained(DINO_MODEL)
        _G["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(dev).eval()
        _G["dev"] = dev
        print(f"[DINO] {DINO_MODEL} on {dev}", flush=True)
    return _G


def detect(bgr, thresh=0.25, topk=8):
    import torch
    from PIL import Image
    g = load_dino()
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inputs = g["proc"](images=pil, text=PROMPT, return_tensors="pt").to(g["dev"])
    with torch.no_grad():
        out = g["model"](**inputs)
    common = dict(input_ids=inputs.input_ids, text_threshold=0.2, target_sizes=[pil.size[::-1]])
    post = g["proc"].post_process_grounded_object_detection
    try:
        res = post(out, threshold=thresh, **common)[0]
    except TypeError:
        res = post(out, box_threshold=thresh, **common)[0]
    labels = res.get("text_labels") or res.get("labels") or [""] * len(res["scores"])
    boxes = sorted(zip(res["scores"].tolist(), res["boxes"].tolist(), [str(l) for l in labels]), key=lambda t: -t[0])[:topk]
    ok_words = ("forceps", "applier", "suction", "cannula", "scissors", "dissector", "shaft", "instrument")
    return [(s, [int(round(v)) for v in b], l) for s, b, l in boxes if any(w in l.lower() for w in ok_words)]


def sides_touched(x0, y0, x1, y1, W, H):
    s = []
    if x0 <= BORDER: s.append("L")
    if x1 >= W - 1 - BORDER: s.append("R")
    if y0 <= BORDER: s.append("T")
    if y1 >= H - 1 - BORDER: s.append("B")
    return s


def segment(bgr, box):
    """GrabCut inside `box` seeded by a metal prior. Returns a full-frame uint8 mask or None."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = box
    pad = 12
    rx0, ry0, rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    roi = bgr[ry0:ry1, rx0:rx1]
    if roi.size == 0 or min(roi.shape[:2]) < 24:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    Hh_ = hsv[..., 0].astype(np.int32)
    metal = ((S < 75) & (V < 165)) | (V < 70) | ((Hh_ >= 100) & (Hh_ <= 165) & (S > 60))
    gc = np.full(roi.shape[:2], cv2.GC_BGD, np.uint8)
    bx0, by0, bx1, by1 = x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0
    gc[by0:by1, bx0:bx1] = cv2.GC_PR_BGD
    inner = np.zeros_like(gc, bool)
    inner[by0:by1, bx0:bx1] = True
    gc[inner & metal] = cv2.GC_PR_FGD
    core = cv2.erode((inner & metal).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gc[core] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any():
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    # keep the components that reach the frame border (the instrument is held from outside)
    keep = np.zeros_like(fg, bool)
    for i in range(1, n):
        cx0, cy0, cw, chh, area = stats[i]
        if area < 1500:
            continue
        gx0, gy0, gx1, gy1 = rx0 + cx0, ry0 + cy0, rx0 + cx0 + cw - 1, ry0 + cy0 + chh - 1
        if not sides_touched(gx0, gy0, gx1, gy1, W, H):
            continue
        comp = lab == i
        hm, sm, vm = (float(np.median(hsv[..., c][comp])) for c in range(3))
        pts = np.column_stack(np.nonzero(comp))[:, ::-1].astype(np.float32)
        (rw, rh) = cv2.minAreaRect(pts)[1]
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        if elong < 1.8:                                # instruments are elongated shafts
            continue
        if vm > 165 and sm < 70 and elong < 3.5:       # white and blobby: cottonoid patty / gauze
            continue
        if (hm < 12 or hm > 168) and sm > 85:          # red / pink: tissue, blood
            continue
        keep |= comp
    if not keep.any():
        return None
    # GrabCut leaks a strip of adjacent tissue along the shaft: keep only metal-coloured pixels
    # (dark / grey / blue-purple, incl. specular highlights), then re-open and keep the border piece
    red_hue = (Hh_ < 16) | (Hh_ > 164)
    metal2 = (S < 35) | ((S < 95) & (V < 150)) | ((S < 95) & ~red_hue) | (V < 90) | ((Hh_ >= 95) & (Hh_ <= 170) & (S > 50))
    keep &= cv2.dilate(metal2.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    n2, lab3, st3, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
    if n2 <= 1:
        return None
    best = 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))
    keep = lab3 == best
    if keep.sum() < 1500:
        return None
    full = np.zeros((H, W), np.uint8)
    full[ry0:ry1, rx0:rx1] = keep.astype(np.uint8) * 255
    return full


def has_clip_blob(bgr, m):
    """True if the masked region holds a compact saturated-blue blob (an anodised clip); thin blue
    vessel loops (elongated) are allowed."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90) & (m > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 300:
            continue
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32)
        (rw, rh) = cv2.minAreaRect(pts)[1]
        if max(rw, rh) / max(min(rw, rh), 1.0) < 3.0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--bank", action="store_true", help="cut from the 800-px bank backplates even if the MACS originals exist")
    ap.add_argument("--extra-frames", type=int, default=0, help="also sample this many label-1 MACS frames (seeded, spread over the videos)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(BANK, "bank.jsonl"))]
    if a.limit:
        recs = recs[:a.limit]
    if a.extra_frames and os.path.isdir(MACS_ROOT):
        import glob
        rng = np.random.default_rng(0)
        vids = sorted(d for d in os.listdir(MACS_ROOT) if d.startswith("vid_") and os.path.isdir(os.path.join(MACS_ROOT, d)))
        per = max(1, a.extra_frames // max(1, len(vids)))
        for v in vids:
            fs = sorted(glob.glob(os.path.join(MACS_ROOT, v, v, "1", "*.jpg")))
            if not fs:
                continue
            for f in rng.choice(fs, size=min(per, len(fs)), replace=False):
                recs.append(dict(file=None, src=os.path.relpath(f, MACS_ROOT), vid=v))
        print(f"sources: {len(recs)} frames (bank + {a.extra_frames} extra)", flush=True)
    out_jl = open(os.path.join(OUT_DIR, "instruments.jsonl"), "w")
    n_cut, cells = 0, []
    per_vid, last_box = {}, {}
    MAX_PER_VID = int(os.environ.get("MAX_PER_VID", 16))
    for i, r in enumerate(recs):
        p = os.path.join(MACS_ROOT, r["src"].replace("\\", os.sep))
        bgr = cv2.imread(p) if (not a.bank and os.path.exists(p)) else None
        if bgr is None and r.get("file"):                 # no MACS originals here (DGX): use the 800-px bank backplate itself
            bgr = cv2.imread(os.path.join(BANK, r["file"]))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        for score, (x0, y0, x1, y1), label in detect(bgr, a.thresh):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
            frac = (x1 - x0) * (y1 - y0) / float(W * H)
            if not (0.01 <= frac <= 0.45) or not sides_touched(x0, y0, x1, y1, W, H):
                continue
            m = segment(bgr, (x0, y0, x1, y1))
            if m is None:
                continue
            area_frac = float((m > 0).sum()) / max(1, (x1 - x0) * (y1 - y0))
            if not (0.15 <= area_frac <= 0.95):
                continue
            ys, xs = np.nonzero(m)
            sides = sides_touched(xs.min(), ys.min(), xs.max(), ys.max(), W, H)
            if not sides:
                continue
            if has_clip_blob(bgr, m):                     # a real clip inside the cutout would corrupt the GT
                continue
            vid = r.get("vid", "?")
            if per_vid.get(vid, 0) >= MAX_PER_VID:
                continue
            bb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
            lb = last_box.get(vid)
            if lb is not None:
                ix0, iy0 = max(bb[0], lb[0]), max(bb[1], lb[1]); ix1, iy1 = min(bb[2], lb[2]), min(bb[3], lb[3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (lb[2] - lb[0]) * (lb[3] - lb[1]) - inter
                if union > 0 and inter / union > 0.6:      # same instrument, same place: a duplicate
                    continue
            per_vid[vid] = per_vid.get(vid, 0) + 1
            last_box[vid] = bb
            alpha = cv2.GaussianBlur(m, (0, 0), 1.5)
            cx0, cy0, cx1, cy1 = max(0, xs.min() - 4), max(0, ys.min() - 4), min(W, xs.max() + 5), min(H, ys.max() + 5)
            cut = np.dstack([bgr[cy0:cy1, cx0:cx1], alpha[cy0:cy1, cx0:cx1]])
            fn = f"instr_{n_cut:04d}.png"
            cv2.imwrite(os.path.join(OUT_DIR, fn), cut)
            out_jl.write(json.dumps(dict(file=fn, frame=[W, H], sides=sides, bbox=[int(cx0), int(cy0), int(cx1), int(cy1)],
                                         src=r["src"], bank=r.get("file"), vid=r.get("vid"), score=round(float(score), 3), label=label,
                                         area_frac=round(area_frac, 3))) + "\n")
            n_cut += 1
            if a.montage and len(cells) < 48:
                tile = np.full((220, 220, 3), 60, np.uint8)
                ch, cw = cut.shape[:2]
                s = 200.0 / max(ch, cw)
                small = cv2.resize(cut, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
                al = small[..., 3:4].astype(np.float32) / 255.0
                hh, ww = small.shape[:2]
                tile[10:10 + hh, 10:10 + ww] = (al * small[..., :3] + (1 - al) * tile[10:10 + hh, 10:10 + ww]).astype(np.uint8)
                cv2.putText(tile, f"{fn[6:10]} {''.join(sides)} {score:.2f}", (6, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cells.append(tile)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(recs)} frames, {n_cut} cutouts", flush=True)
    out_jl.close()
    print(f"{n_cut} cutouts -> {OUT_DIR}")
    if a.montage and cells:
        while len(cells) % 8:
            cells.append(np.full((220, 220, 3), 60, np.uint8))
        grid = np.vstack([np.hstack(cells[k:k + 8]) for k in range(0, len(cells), 8)])
        cv2.imwrite(os.path.join(OUT_DIR, "_montage.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("montage ->", os.path.join(OUT_DIR, "_montage.jpg"))


if __name__ == "__main__":
    main()
