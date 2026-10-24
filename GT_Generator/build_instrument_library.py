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


def load_dino():
    import torch
    import transformers
    if os.environ.get("NO_TORCHVISION", "1") == "1":
        # laptop env: torchvision is built for another torch ("operator torchvision::nms does not exist");
        # transformers only needs it for optional fast image paths, so declare it absent before the lazy import
        try:
            import transformers.utils.import_utils as _iu
            _iu._torchvision_available = False
        except Exception:
            pass
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
    boxes = sorted(zip(res["scores"].tolist(), res["boxes"].tolist()), key=lambda t: -t[0])[:topk]
    return [(s, [int(round(v)) for v in b]) for s, b in boxes]


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
    metal = (S < 75) | (V < 70) | ((V > 190) & (S < 90))
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
        if area < 2500:
            continue
        gx0, gy0, gx1, gy1 = rx0 + cx0, ry0 + cy0, rx0 + cx0 + cw - 1, ry0 + cy0 + chh - 1
        if sides_touched(gx0, gy0, gx1, gy1, W, H):
            keep |= lab == i
    if not keep.any():
        return None
    full = np.zeros((H, W), np.uint8)
    full[ry0:ry1, rx0:rx1] = keep.astype(np.uint8) * 255
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--bank", action="store_true", help="cut from the 800-px bank backplates even if the MACS originals exist")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(BANK, "bank.jsonl"))]
    if a.limit:
        recs = recs[:a.limit]
    out_jl = open(os.path.join(OUT_DIR, "instruments.jsonl"), "w")
    n_cut, cells = 0, []
    for i, r in enumerate(recs):
        p = os.path.join(MACS_ROOT, r["src"].replace("\\", os.sep))
        bgr = cv2.imread(p) if (not a.bank and os.path.exists(p)) else None
        if bgr is None:                                   # no MACS originals here (DGX): use the 800-px bank backplate itself
            bgr = cv2.imread(os.path.join(BANK, r["file"]))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        for score, (x0, y0, x1, y1) in detect(bgr, a.thresh):
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
            alpha = cv2.GaussianBlur(m, (0, 0), 1.5)
            cx0, cy0, cx1, cy1 = max(0, xs.min() - 4), max(0, ys.min() - 4), min(W, xs.max() + 5), min(H, ys.max() + 5)
            cut = np.dstack([bgr[cy0:cy1, cx0:cx1], alpha[cy0:cy1, cx0:cx1]])
            fn = f"instr_{n_cut:04d}.png"
            cv2.imwrite(os.path.join(OUT_DIR, fn), cut)
            out_jl.write(json.dumps(dict(file=fn, frame=[W, H], sides=sides, bbox=[int(cx0), int(cy0), int(cx1), int(cy1)],
                                         src=r["src"], bank=r["file"], score=round(float(score), 3),
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
