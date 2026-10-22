"""
real_points.py — the three annotated points of a real MACS frame. Canonical keys:

  tip   end of the jaws (midpoint of the two jaw tips when open)        click 1, required
  back  back end of the clip = end of the blue head                     click 2, required
  head  centre of the spring (coil) inside the head                     click 3, optional (None = not visible)

Clip axis and length L = tip -> back (spans the whole clip); fallback tip -> head.
Files on disk:
  scheme 3 (official, 2026-10-22): points {tip, back, coil}
  scheme 2 (2026-10-21, 'head' = 2nd click, 'back' = 3rd click): the reference annotator clicked
           back end 2nd and spring centre 3rd, so it is read as back = 'head', head = 'back'
  scheme 1 (first round): points {tip, spring = back end, coil = coil centre}
Synthetic M1 keypoints map as tip = mean(tip_a, tip_b), head = coil_centre, back = spring.
"""

import numpy as np

KEYS = ("tip", "head", "back")
SYNTH_MAP = {"tip": ("tip_a", "tip_b"), "head": ("coil_centre",), "back": ("spring",)}


def norm_points(points):
    """Annotation 'points' dict (any scheme) -> {tip, head, back} with None for missing."""
    if "spring" in points:                                        # scheme 1
        return {"tip": points.get("tip"), "head": points.get("coil"), "back": points.get("spring")}
    if "coil" in points:                                          # scheme 3
        return {"tip": points.get("tip"), "head": points.get("coil"), "back": points.get("back")}
    return {"tip": points.get("tip"), "head": points.get("back"), "back": points.get("head")}   # scheme 2 (clicks 3, 2)


def axis_key(pts):
    """Second point of the clip axis: the back end when present, else the spring centre."""
    return "back" if pts.get("back") is not None else "head"


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
    """Scheme 2: the box if it contains all points (it then frames the whole clip), else the points
    (the first scheme-2 round kept the pre-loaded detector box in every frame; 35/152 missed points,
    while the scheme-2 points are consistent: back end within 0.7 deg of the tip-head line).
    Scheme 1: box U points (its points often did not span the clip)."""
    if ann.get("scheme", 1) < 2:
        return "union"
    x, y, w, h = ann["box"]
    pts = [v for v in norm_points(ann["points"]).values() if v is not None]
    inside = all(x - 0.05 * w <= q[0] <= x + 1.05 * w and y - 0.05 * h <= q[1] <= y + 1.05 * h for q in pts)
    return "box" if inside else "points"


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
