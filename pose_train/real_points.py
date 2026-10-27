"""
real_points.py — the three annotated points of a real MACS frame. Canonical keys:

  tip   end of the jaws (midpoint of the two jaw tips when open)        click 1, required
  back  back end of the clip = end of the blue head                     click 2, required
  head  centre of the spring (coil) inside the head                     click 3, optional (None = not visible)

Clip axis and length L = tip -> back (spans the whole clip); fallback tip -> head.
Metrics and the main-set rule (L >= MIN_L_PX) live at the end of this file and are shared by eval_real.py
and finetune_real.py.
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


# ------------------------------------------------------------------ shared real-frame metrics (27 Oct 2026)
# eval_real.py (zero-shot) and finetune_real.py (cross-validation) used different clip axes (tip->back vs
# tip->head) and reported only the directed axis error. Both now call frame_metrics/summarize_real:
#   * axis and L = tip -> back end (fallback tip -> spring centre), as in the annotation guide
#   * axis_deg (directed, 0-180), axis_undir_deg = min(a, 180 - a) (line orientation only, chance median 45)
#     and flipped = axis_deg > 90 (tip/back swapped; the L-Clip head is nearly symmetric)
#   * MAIN SET = frames with L >= MIN_L_PX (60 px): below that the clip is seen end-on or is tiny, L is not a
#     usable scale and err/L explodes (2.6-3.7 L median on 20 of 152 v2 frames). Group "all" = main set;
#     "all_incl_small" and "small" are reported separately; attribute groups are within the main set.
MIN_L_PX = 60.0


def frame_metrics(pred, gt):
    """pred: {tip, head, back} -> xy; gt: same keys, missing ones absent or None. Returns the metric dict."""
    gt = {k: np.asarray(v, float) for k, v in gt.items() if v is not None}
    ak = axis_key(gt)
    L = float(np.linalg.norm(gt["tip"] - gt[ak])) + 1e-6
    row, fr = dict(L_px=L), []
    for k in KEYS:
        if k not in gt:
            continue
        e = float(np.linalg.norm(np.asarray(pred[k], float) - gt[k]))
        row[f"{k}_px"], row[f"{k}_frac"] = e, e / L
        fr.append(e / L)
    row["mean_frac"] = float(np.mean(fr))
    dp, dg = np.asarray(pred["tip"], float) - np.asarray(pred[ak], float), gt["tip"] - gt[ak]
    c = float(np.dot(dp, dg) / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-6))
    a = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    row.update(axis_deg=a, axis_undir_deg=min(a, 180.0 - a), flipped=int(a > 90.0), main=int(L >= MIN_L_PX))
    return row


def _summ(sel, name):
    v = np.array([r["mean_frac"] for r in sel])
    ax = np.array([r["axis_deg"] for r in sel])
    au = np.minimum(ax, 180.0 - ax)
    px = np.array([r["mean_frac"] * r["L_px"] for r in sel])
    return dict(name=name, n=len(sel), mean_frac_med=float(np.median(v)), mean_frac_p90=float(np.percentile(v, 90)),
                err_px_med=float(np.median(px)), pck_0_1=float(np.mean(v < 0.1)), pck_0_2=float(np.mean(v < 0.2)),
                tip_frac_med=float(np.median([r["tip_frac"] for r in sel])),
                head_frac_med=float(np.median([r["head_frac"] for r in sel if "head_frac" in r] or [np.nan])),
                axis_deg_med=float(np.median(ax)), axis_lt_10=float(np.mean(ax < 10)), axis_lt_20=float(np.mean(ax < 20)),
                axis_undir_med=float(np.median(au)), axis_undir_lt_10=float(np.mean(au < 10)),
                flip_rate=float(np.mean(ax > 90)))


def summarize_real(rows, attrs=("held", "open", "occluded", "blurry", "multi")):
    main = [r for r in rows if r["L_px"] >= MIN_L_PX]
    small = [r for r in rows if r["L_px"] < MIN_L_PX]
    groups = [(main, "all"), (rows, "all_incl_small"), (small, "small")]
    for k in attrs:
        groups += [([r for r in main if r.get(f"attr_{k}")], k), ([r for r in main if not r.get(f"attr_{k}")], f"not_{k}")]
    return [_summ(sel, name) for sel, name in groups if sel]


SUMMARY_HEADER = (f"{'group':14s} {'n':>4s} {'err/L':>6s} {'px':>5s} {'PCK.1':>6s} {'PCK.2':>6s} {'axis':>6s} {'<20':>5s} "
                  f"{'undir':>6s} {'flip':>5s}")


def summary_line(m):
    return (f"{m['name']:14s} {m['n']:4d} {m['mean_frac_med']:6.3f} {m['err_px_med']:5.0f} {m['pck_0_1']:6.2f} {m['pck_0_2']:6.2f} "
            f"{m['axis_deg_med']:6.1f} {m['axis_lt_20']:5.2f} {m['axis_undir_med']:6.1f} {m['flip_rate']:5.2f}")
