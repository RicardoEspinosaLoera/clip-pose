"""
metrics.py — clip-displacement check (the paper's edge phase-correlation measure).

Confirms the inpainting did not move the clip: compares Canny edges of the
original render and of the inpainted result *inside the clip mask* via phase
correlation, returning the estimated sub-pixel shift in pixels.
"""

import cv2
import numpy as np


def clip_edge_shift(original_bgr, result_bgr, clip_mask):
    """Return estimated (|shift| px) of the clip between original and result."""
    m = (clip_mask > 127).astype(np.uint8)
    if m.sum() < 50:
        return 0.0

    g0 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    e0 = cv2.Canny(g0, 50, 150).astype(np.float32)
    e1 = cv2.Canny(g1, 50, 150).astype(np.float32)

    # restrict to the clip region and window to suppress border effects
    e0 *= m
    e1 *= m
    if e0.sum() < 1 or e1.sum() < 1:
        return 0.0

    win = cv2.createHanningWindow((e0.shape[1], e0.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(e0 * win, e1 * win)
    return float(np.hypot(dx, dy))


def clip_shift_robust(original_bgr, result_bgr, clip_mask, search=28, pad=3):
    """Robust clip displacement (px) via masked template matching.

    phaseCorrelate on Canny edges is fooled when the surrounding tissue is heavily
    restyled (restyle-context): the new vascular edges dominate and it reports huge
    bogus shifts even when the clip never moved. This instead cuts the clip patch
    (masked to the silhouette) from the original and template-matches it within a
    +/-`search` px window of the result, so it measures where the CLIP actually
    went — robust to colour/texture restyle because it correlates the clip's own
    luminance structure, masked away from the tissue."""
    m = (clip_mask > 127).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) < 30:
        return 0.0
    H, W = m.shape
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    g0 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    templ = g0[y0:y1, x0:x1]
    tmask = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
    sx0, sy0 = max(0, x0 - search), max(0, y0 - search)
    sx1, sy1 = min(W, x1 + search), min(H, y1 + search)
    region = g1[sy0:sy1, sx0:sx1]
    if region.shape[0] < templ.shape[0] or region.shape[1] < templ.shape[1]:
        return 0.0
    try:
        resp = cv2.matchTemplate(region, templ, cv2.TM_CCORR_NORMED, mask=tmask)
    except cv2.error:
        resp = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
    resp = np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, _, _, maxloc = cv2.minMaxLoc(resp)
    dx = maxloc[0] - (x0 - sx0)
    dy = maxloc[1] - (y0 - sy0)
    return float(np.hypot(dx, dy))
