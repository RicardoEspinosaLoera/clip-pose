"""Segment the clip in each training crop (SAM box prompt from a colour-derived bbox, cleaned
by removing strongly-red tissue). Masks feed the MASKED diffusion loss so the LoRA learns the
clip metal, not the surrounding tissue (the 'reddish areas' contamination). Crops where no clip
is found are recorded (skipped from masked training)."""
import os, glob, cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
CROPS = os.path.join(HERE, "clean_clips2")
OUT = os.path.join(HERE, "clean_clips2_masks"); os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=os.path.join(HERE, "..", "models", "sam_vit_b_01ec64.pth")).to(dev)
pred = SamPredictor(sam)


def color_bbox(bgr):
    f = bgr.astype(np.int16); b, g, r = f[..., 0], f[..., 1], f[..., 2]; ch = f.max(2) - f.min(2); v = bgr.max(2)
    m = ((((b > r + 10) & (b >= g - 5)) | ((ch < 32) & (v > 125)) | ((ch < 20) & (v < 70))).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m); H, W = m.shape; best = -1; bi = 0
    for i in range(1, n):
        if st[i, 4] < 500:
            continue
        cx, cy = cen[i]; sc = st[i, 4] * (1 - (abs(cx - W / 2) + abs(cy - H / 2)) / W)
        if sc > best:
            best = sc; bi = i
    if not bi:
        return None
    x, y, w, h, _ = st[bi]; pad = 8
    return np.array([max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)])


def clip_mask(bgr):
    box = color_bbox(bgr)
    if box is None:
        return None
    pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = pred.predict(box=box[None, :], multimask_output=False)
    sam = masks[0].astype(bool)
    if sam.sum() < 400:
        return None
    f = bgr.astype(np.int16); b, g, r = f[..., 0], f[..., 1], f[..., 2]; ch = f.max(2) - f.min(2); v = bgr.max(2)
    # actual CLIP-coloured pixels: blue titanium OR achromatic metal (bright or dark) -> NOT tissue
    clipcol = (((b > r + 6) & (b >= g - 8)) | ((ch < 38) & (v > 110)) | ((ch < 22) & (v < 80)))
    frac = float((sam & clipcol).sum()) / float(sam.sum())
    if frac < 0.25:                                          # SAM grabbed mostly tissue -> drop
        return None
    m = (sam & clipcol).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # reconnect clip parts
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    bi = 1 + int(np.argmax(st[1:, 4]))
    out = (lab == bi).astype(np.uint8) * 255
    if out.sum() < 300 * 255:
        return None
    return out


def main():
    paths = sorted(glob.glob(os.path.join(CROPS, "*.png")))
    ok = skip = 0; qc = []
    for i, p in enumerate(paths):
        bgr = cv2.imread(p); m = clip_mask(bgr)
        stem = os.path.splitext(os.path.basename(p))[0]
        if m is None:
            skip += 1
            cv2.imwrite(os.path.join(OUT, stem + ".png"), np.zeros(bgr.shape[:2], np.uint8))  # empty = skip in training
        else:
            ok += 1
            cv2.imwrite(os.path.join(OUT, stem + ".png"), m)
        if i % 40 == 0 and len(qc) < 24:
            vis = bgr.copy()
            if m is not None:
                mm = m > 127; vis[mm] = (0.45 * vis[mm] + np.array([0, 255, 255]) * 0.55).astype(np.uint8)
            qc.append(cv2.resize(vis, (160, 160)))
        if i % 100 == 0:
            print(f"[{i}/{len(paths)}] ok={ok} skip={skip}", flush=True)
    rows = [np.hstack(qc[r * 6:r * 6 + 6]) for r in range(len(qc) // 6)]
    cv2.imwrite(os.path.join(HERE, "..", "..", "GT_Generator", "_clipmask_qc.png"), np.vstack(rows))
    print(f"DONE ok={ok} skip={skip} / {len(paths)}  masks -> {OUT} ; QC -> _clipmask_qc.png", flush=True)


if __name__ == "__main__":
    main()
