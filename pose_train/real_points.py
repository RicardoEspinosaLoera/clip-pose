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
