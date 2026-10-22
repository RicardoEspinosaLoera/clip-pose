"""
real_points.py — the three annotated points of a real MACS frame (annotation scheme 2, 2026-10-21):

  tip   end of the jaws (midpoint of the two jaw tips when open)          required
  head  centre of the clip head (blue anodised body, spring housing)     required
  back  back end of the clip, opposite the tip                           optional (None = hidden)

Clip axis = tip -> head (both nearly always visible); clip length L = |tip - head|.
Scheme 1 files (tip, spring = back end, coil = coil centre) are read as head = coil, back = spring;
their axis falls back to tip -> back when the coil was marked not visible.
Synthetic M1 keypoints map as tip = mean(tip_a, tip_b), head = coil_centre, back = spring.
"""

import numpy as np

KEYS = ("tip", "head", "back")
SYNTH_MAP = {"tip": ("tip_a", "tip_b"), "head": ("coil_centre",), "back": ("spring",)}


def norm_points(points):
    """Annotation 'points' dict (either scheme) -> {tip, head, back} with None for missing."""
    if "head" in points or "back" in points:
        return {k: points.get(k) for k in KEYS}
    return {"tip": points.get("tip"), "head": points.get("coil"), "back": points.get("spring")}


def axis_key(pts):
    """Second point of the clip axis: head when present, else back."""
    return "head" if pts.get("head") is not None else "back"


def clip_length(pts):
    return float(np.linalg.norm(np.subtract(pts["tip"], pts[axis_key(pts)])))


# Crop window for real frames. "box": the annotated box = whole clip incl. jaws, exactly what the
# synthetic training crop is built from (1.5 x mask bbox) -> default for scheme-2 annotations, whose box
# is always reviewed. "union": box U points, for scheme-1 annotations (their pre-loaded detector boxes
# were never corrected and often missed the clip). "points": square around the points (margin 1.75 =
# 1.5 x the synthetic mask-side / point-extent ratio 1.15); only valid if the points span the clip,
# which scheme-1 points often did not.
WINDOW_MARGIN = {"points": 1.75, "union": 1.5, "box": 1.5}


def auto_window(ann):
    return "box" if ann.get("scheme", 1) >= 2 else "union"


def crop_window(box, pts, mode="points"):
    """box (x, y, w, h), pts (N, 2) with NaN rows allowed -> (cx, cy, side) before the margin."""
    p = np.asarray(pts, float)
    p = p[~np.isnan(p[:, 0])]
    if mode == "box" or len(p) == 0:
        x, y, w, h = box
        return x + w / 2, y + h / 2, max(w, h)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    if mode == "union":
        x, y, w, h = box
        x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
    return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
