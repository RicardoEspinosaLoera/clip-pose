"""
occlude.py — put the clip INTO the scene: partial occlusion, clip applier, blood film.

Real MACS clips are rarely clean and fully visible: an instrument (most often the applier
still holding the clip) covers part of them, tissue folds over them, and the metal carries
blood. This post-process adds those effects to generated samples WITHOUT touching the 6-DoF
GT: pose, camera, depth and full mask stay as rendered; `NNN_mask_visible.png` (full mask
minus occluders) is written for visibility-aware training.

Design rules learned from a first, fake-looking version:
  * tissue occlusion is NOT a pasted patch (a sticker with a seam all around). The clip is
    rendered ON the real tissue, so "clip under tissue" = reveal the clip-free backplate again
    inside a soft region that enters from one side. Seamless by construction, plus a lip
    shadow on the clip where the tissue edge overhangs it, optionally translucent (membrane
    or blood pool).
  * the applier is a THICK (1.6-3x the clip width), strongly defocused cylinder (it sits
    closer to the camera than the clip), shaded like a cylinder, tinted by the surrounding
    tissue, with a specular edge, entering from the frame border roughly along the clip axis
    and gripping the SPRING end with two jaws. Real cutouts (Data/instruments) are used
    instead when a library exists.
  * blood is the frame's own dark-red colour, multiplied over part of the clip with a wet
    gloss streak, never a flat purple tint.
Modified samples get one more light JPEG generation so added pixels share the video footprint.

Usage
  python occlude.py --src out_pilot_v3 --out out_pilot_v3_occ          # copies then modifies
  python occlude.py --src out_v2 --in-place                            # modify a finished set
"""

import os
import sys
import json
import glob
import shutil
import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from articulate_clip import pca_frame

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.environ.get("CLIP_MESH", os.path.join(HERE, "Item.obj"))   # env-overridable (standalone runs only; the generator passes its own mesh)
INSTRUMENT_DIR = os.path.join(HERE, "Data", "instruments")

PROB = dict(tissue=0.50, applier=0.35, blood=0.25)      # applier = REAL cutout only (Data/instruments), never on tool_held samples
MIN_VISIBLE = 0.35                                      # an occluder is dropped if less of the clip would remain
JPEG_AFTER = (65, 90)


# ------------------------------------------------------------------ helpers
def _blob_mask(H, W, cx, cy, r, ax_ratio, angle, rng, wobble=0.25):
    """Organic blob: ellipse whose radius wobbles with a few low-frequency harmonics."""
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rad = 1.0 + wobble * (0.6 * np.sin(2 * th + rng.uniform(0, 6.28)) + 0.4 * np.sin(3 * th + rng.uniform(0, 6.28))
                          + 0.3 * np.sin(5 * th + rng.uniform(0, 6.28)))
    x = r * rad * np.cos(th)
    y = r * ax_ratio * rad * np.sin(th)
    ca, sa = np.cos(angle), np.sin(angle)
    pts = np.stack([cx + ca * x - sa * y, cy + sa * x + ca * y], 1)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


def _feather(m8, sigma):
    return np.clip(cv2.GaussianBlur(m8.astype(np.float32) / 255.0, (0, 0), max(sigma, 0.3)), 0, 1)


def _shadow_band(occ8, px, strength):
    """Darkening factor just outside an occluder (its shadow on what is below)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    soft = _feather(cv2.dilate(occ8, k), px * 0.6) * (1.0 - _feather(occ8, 0.8))
    return 1.0 - strength * soft


def _mask_geom(mask):
    m = mask > 127
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    cov = np.cov((pts - pts.mean(0)).T)
    ev = np.linalg.eigvalsh(cov)
    clip_w = 2.0 * np.sqrt(ev.min() + 1e-6)          # thickness across the minor axis
    clip_l = 2.0 * np.sqrt(ev.max() + 1e-6) * 1.7     # ~ extent along the major axis
    return m, xs, ys, clip_w, clip_l


def clip_axis_px(meta, V, blade_dir):
    """Image-space spring point and jaw tip from the (OpenCV) GT pose + camera."""
    q = meta["clip"]["pose_se3"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_se3"]["translation_m"], float)
    c = meta["camera"]
    P = (R @ V.T).T + t
    u = c["fx"] * P[:, 0] / P[:, 2] + c["cx"]
    v = c["fy"] * P[:, 1] / P[:, 2] + c["cy"]
    proj = (V - V.mean(0)) @ blade_dir
    tip = np.array([u[np.argmax(proj)], v[np.argmax(proj)]])
    spring = np.array([u[np.argmin(proj)], v[np.argmin(proj)]])
    return spring, tip


# ------------------------------------------------------------------ 1) clip under tissue (seamless reveal)
def _snap_to_edges(bg, blob, grid=14):
    """Re-shape a soft occluder blob so that its boundary follows real edges of the backplate:
    watershed over-segmentation of the backplate around the blob, keep the segments that lie
    mostly inside the blob. The reveal then stops at tissue folds / vessel borders instead of
    along a smooth ellipse (a smooth ellipse reads as a pasted sticker)."""
    ys, xs = np.nonzero(blob > 0)
    if len(xs) < 50:
        return blob
    H, W = blob.shape
    pad = 24
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    roi = cv2.GaussianBlur(bg[y0:y1, x0:x1], (0, 0), 1.0)
    grid = int(np.clip(0.22 * np.sqrt(len(xs)), 10, 26))     # segments scale with the occluder, not the image
    markers = np.zeros(roi.shape[:2], np.int32)
    k = 1
    for gy in range(grid // 2, roi.shape[0], grid):
        for gx in range(grid // 2, roi.shape[1], grid):
            markers[gy, gx] = k
            k += 1
    if k < 4:
        return blob
    cv2.watershed(roi, markers)
    lab = markers.copy()
    lab[lab < 1] = 0
    b = blob[y0:y1, x0:x1] > 0
    region = np.zeros_like(b)
    for l in np.unique(lab[lab > 0]):
        seg = lab == l
        if (seg & b).sum() >= 0.5 * seg.sum():
            region |= seg
    region = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8))               # fill the -1 ridges
    # in smooth tissue the watershed cells are grid squares: round the outline (keep the big
    # edge-following excursions, drop the stair steps) and keep one connected piece
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    region = cv2.morphologyEx(cv2.morphologyEx(region, cv2.MORPH_CLOSE, ker), cv2.MORPH_OPEN, ker)
    region = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.0) > 0.5
    region &= cv2.dilate((blob[y0:y1, x0:x1] > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    n, lab2, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    if n > 2:
        region = lab2 == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    out = np.zeros_like(blob)
    out[y0:y1, x0:x1] = region.astype(np.uint8) * 255
    return out if (out > 0).sum() > 0.4 * (blob > 0).sum() else blob


def tissue_cover(img, bg, mask, rng, meta=None, V=None, blade_dir=None, tries=12):
    """Reveal the clip-free backplate over part of the clip (tissue in front of the clip).
    70 %: anchored at the JAWS (the clip disappears into the aneurysm / tissue it is placed on,
    the common real case), else anywhere on the outline. The reveal boundary snaps to real
    backplate edges (watershed). Returns (image, cover_mask, coverage) or None."""
    H, W = mask.shape
    m, xs, ys, clip_w, clip_l = _mask_geom(mask)
    area = float(m.sum())
    edge = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    ey, ex = np.nonzero(edge)
    cxm, cym = xs.mean(), ys.mean()
    axis = None
    if meta is not None and V is not None and blade_dir is not None and rng.random() < 0.7:
        try:
            spring, tip = clip_axis_px(meta, V, blade_dir)
            axis = (np.asarray(spring, float), np.asarray(tip, float))
        except Exception:
            axis = None
    for _ in range(tries):
        if axis is not None:
            spring, tip = axis
            d = tip - spring
            d /= (np.linalg.norm(d) + 1e-6)
            r = float(np.sqrt(area) * rng.uniform(0.6, 1.3))
            centre = tip + d * r * rng.uniform(0.2, 0.7)              # tissue beyond / around the jaw tips
            blob = _blob_mask(H, W, centre[0], centre[1], r, rng.uniform(0.7, 1.0),
                              float(np.arctan2(d[1], d[0])) + rng.uniform(-0.4, 0.4), rng, wobble=0.35)
        else:
            i = rng.integers(len(ex))
            r = float(np.sqrt(area) * rng.uniform(0.5, 1.2))
            ov = np.array([ex[i] - cxm, ey[i] - cym], float)
            ov /= (np.linalg.norm(ov) + 1e-6)
            push = r * rng.uniform(0.35, 0.8)
            blob = _blob_mask(H, W, ex[i] + ov[0] * push, ey[i] + ov[1] * push, r,
                              rng.uniform(0.6, 1.0), rng.uniform(0, np.pi), rng, wobble=0.3)
        blob = _snap_to_edges(bg, blob)
        cov = float(((blob > 0) & m).sum() / max(area, 1))
        if not (0.12 <= cov <= 0.55):
            continue
        soft = rng.uniform(0.7, 1.6)                      # tissue edge softness (px): real edges are fairly crisp
        a_max = 1.0 if rng.random() < 0.6 else rng.uniform(0.6, 0.9)   # 40%: translucent film
        a = _feather(blob, soft) * a_max
        # lip shadow: the overhanging tissue darkens the CLIP just inside its edge. Restricted
        # to clip pixels: applied to the tissue too it drew a dark outline of the blob.
        lip = _feather(cv2.dilate(blob, np.ones((9, 9), np.uint8)), 2.0) * (1.0 - _feather(blob, 1.0))
        lip = lip * _feather(mask, 1.0)
        shaded = img.astype(np.float32) * (1.0 - 0.35 * lip)[..., None]
        out = a[..., None] * bg.astype(np.float32) + (1 - a[..., None]) * shaded
        cover = ((a > 0.5)).astype(np.uint8) * 255
        return np.clip(out, 0, 255).astype(np.uint8), cover, cov
    return None


# ------------------------------------------------------------------ 2) applier / instrument
_INSTR = {"loaded": False, "items": []}


def _load_instruments():
    if _INSTR["loaded"]:
        return _INSTR["items"]
    _INSTR["loaded"] = True
    jl = os.path.join(INSTRUMENT_DIR, "instruments.jsonl")
    if os.path.exists(jl):
        for line in open(jl):
            try:
                rec = json.loads(line)
                if os.path.exists(os.path.join(INSTRUMENT_DIR, rec["file"])):
                    _INSTR["items"].append(rec)
            except Exception:
                pass
    return _INSTR["items"]


def _cylinder_body(occ8, width, base_bgr, bg_blur, stripe_mask, rng):
    """Shade an occluder mask like a defocused dark metal cylinder tinted by its surroundings."""
    dist = cv2.distanceTransform((occ8 > 127).astype(np.uint8), cv2.DIST_L2, 3)
    r = max(width / 2.0, 1.0)
    prof = np.sqrt(np.clip(1.0 - ((r - np.minimum(dist, r)) / r) ** 2, 0, 1))      # 1 on the axis, 0 at the rim
    body = np.ones(bg_blur.shape, np.float32) * base_bgr
    body = body * 0.72 + bg_blur * 0.28                   # environment tint (red tissue bounce)
    body = body * (0.35 + 0.65 * prof)[..., None]
    sa = _feather(stripe_mask, width * 0.10) * rng.uniform(0.35, 0.9)
    body = body + sa[..., None] * (np.array([215, 205, 200], np.float32) - body)
    return body


def applier_procedural(img, mask, meta, V, blade_dir, rng):
    H, W = mask.shape
    m, xs, ys, clip_w, clip_l = _mask_geom(mask)
    spring, tip = clip_axis_px(meta, V, blade_dir)
    d = spring - tip
    L = float(np.linalg.norm(d)) + 1e-6
    d /= L
    ang = np.deg2rad(rng.uniform(-35, 35))                # appliers do not come exactly along the clip axis
    ca, sa = np.cos(ang), np.sin(ang)
    d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
    n = np.array([-d[1], d[0]])
    width = int(np.clip(clip_w * rng.uniform(1.6, 3.0), 10, 90))
    occ = np.zeros((H, W), np.uint8)
    # jaws: two prongs straddling the spring, tapering into the shaft
    jaw_len = rng.uniform(0.25, 0.45) * L
    for side in (-1.0, 1.0):
        lat = n * side * clip_w * rng.uniform(0.45, 0.7)
        p_tip = spring - d * 0.08 * L + lat
        p_back = spring + d * jaw_len + lat * 0.5
        cv2.line(occ, tuple(np.round(p_tip).astype(int)), tuple(np.round(p_back).astype(int)), 255,
                 max(3, int(width * 0.35)), cv2.LINE_AA)
    # shaft: widening from the jaws to full width, out past the frame border
    far = spring + d * (2.0 * max(H, W))
    for k, frac in enumerate(np.linspace(0.0, 1.0, 5)):
        p0 = spring + d * (jaw_len * (0.6 + 0.4 * frac))
        p1 = far if k == 4 else spring + d * (jaw_len * (0.6 + 0.4 * (frac + 0.25)) + 0.15 * L)
        cv2.line(occ, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 255,
                 max(3, int(width * (0.45 + 0.55 * frac))), cv2.LINE_AA)
    cov = float(((occ > 127) & m).sum() / max(m.sum(), 1))
    # appearance
    bg_blur = cv2.GaussianBlur(img, (0, 0), 12).astype(np.float32)
    base = np.array([38, 34, 36], np.float32) * rng.uniform(0.7, 1.7)
    stripe = np.zeros((H, W), np.uint8)
    off = n * width * rng.uniform(-0.3, 0.3)
    cv2.line(stripe, tuple(np.round(spring + d * jaw_len * 0.6 + off).astype(int)), tuple(np.round(far + off).astype(int)),
             255, max(1, int(width * 0.15)), cv2.LINE_AA)
    body = _cylinder_body(occ, width, base, bg_blur, stripe, rng)
    blur = rng.uniform(2.0, 5.0)                          # closer to the camera than the clip
    a = _feather(occ, blur)[..., None]
    body = cv2.GaussianBlur(body, (0, 0), blur)
    out = img.astype(np.float32) * _shadow_band(occ, int(max(4, width * 0.5)), 0.30)[..., None]
    out = a * body + (1 - a) * out
    return np.clip(out, 0, 255).astype(np.uint8), (a[..., 0] > 0.5).astype(np.uint8) * 255, cov


def applier_cutout(img, mask, meta, V, blade_dir, rng, items):
    """Paste a REAL instrument cutout (BGRA) so that its inner end sits on the clip's spring
    and its body runs out of the frame. Returns None when no cutout fits."""
    H, W = mask.shape
    m, xs, ys, clip_w, clip_l = _mask_geom(mask)
    spring, tip = clip_axis_px(meta, V, blade_dir)
    rec = items[int(rng.integers(len(items)))]
    cut = cv2.imread(os.path.join(INSTRUMENT_DIR, rec["file"]), cv2.IMREAD_UNCHANGED)
    if cut is None or cut.shape[2] != 4:
        return None
    # scale: cutout frames are 1920 px wide; our frames are 800. Keep the real relative scale
    # with jitter, then rotate so the cutout's border side points away from the clip.
    s = (W / float(rec["frame"][0])) * rng.uniform(0.8, 1.4)
    cut = cv2.resize(cut, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ch, cw = cut.shape[:2]
    a = cut[..., 3].astype(np.float32) / 255.0
    ys_c, xs_c = np.nonzero(a > 0.5)
    if len(xs_c) < 50:
        return None
    # inner end = the cutout point farthest from its border side(s); outer direction = toward them
    side = rec["sides"][0]
    if side == "L":
        inner = np.array([xs_c.max(), ys_c[np.argmax(xs_c)]], float); outward = np.array([-1.0, 0.0])
    elif side == "R":
        inner = np.array([xs_c.min(), ys_c[np.argmin(xs_c)]], float); outward = np.array([1.0, 0.0])
    elif side == "T":
        inner = np.array([xs_c[np.argmax(ys_c)], ys_c.max()], float); outward = np.array([0.0, -1.0])
    else:
        inner = np.array([xs_c[np.argmin(ys_c)], ys_c.min()], float); outward = np.array([0.0, 1.0])
    d = spring - tip
    d /= (np.linalg.norm(d) + 1e-6)
    ang = np.deg2rad(rng.uniform(-35, 35))
    ca, sa = np.cos(ang), np.sin(ang)
    d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])      # desired outward direction
    rot = np.degrees(np.arctan2(d[1], d[0]) - np.arctan2(outward[1], outward[0]))
    M = cv2.getRotationMatrix2D((float(inner[0]), float(inner[1])), -rot, 1.0)
    # place the inner end slightly before the spring (jaws over the spring)
    target = spring - d * clip_w * 0.6
    M[:, 2] += target - inner
    warped = cv2.warpAffine(cut, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    occ_a = warped[..., 3].astype(np.float32) / 255.0
    if occ_a.max() < 0.5:
        return None
    occ = (occ_a > 0.5).astype(np.uint8) * 255
    # must still reach a frame border (it is held from outside)
    oy, ox = np.nonzero(occ)
    if not (ox.min() < 3 or oy.min() < 3 or ox.max() > W - 4 or oy.max() > H - 4):
        return None
    cov = float(((occ > 127) & m).sum() / max(m.sum(), 1))
    blur = rng.uniform(1.0, 3.5)
    body = cv2.GaussianBlur(warped[..., :3].astype(np.float32), (0, 0), blur)
    a = cv2.GaussianBlur(occ_a, (0, 0), blur)[..., None]
    out = img.astype(np.float32) * _shadow_band(occ, 6, 0.25)[..., None]
    out = a * body + (1 - a) * out
    return np.clip(out, 0, 255).astype(np.uint8), occ, cov, rec["file"]


# ------------------------------------------------------------------ 3) blood film
def blood_film(img, bg, mask, rng):
    H, W = mask.shape
    m, xs, ys, clip_w, clip_l = _mask_geom(mask)
    i = rng.integers(len(xs))
    r = float(np.sqrt(m.sum()) * rng.uniform(0.5, 1.3))
    blob = _blob_mask(H, W, xs[i], ys[i], r, rng.uniform(0.4, 0.9), rng.uniform(0, np.pi), rng, wobble=0.35)
    region = (blob > 0) & m
    cov = float(region.sum() / max(m.sum(), 1))
    if not (0.1 <= cov <= 0.7):
        return None
    # blood colour = this frame's own dark saturated red
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    sel = ((hsv[..., 0] < 10) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 130) & (hsv[..., 2] > 30) & (hsv[..., 2] < 140)
    blood = bg[sel].mean(0).astype(np.float32) if sel.sum() > 200 else np.array([25, 12, 110], np.float32)
    # thin translucent film: moderate opacity and never near-black (a dark floor turned the
    # jaws into a black wedge that read as a transparent instrument, sample 032 of out_set50)
    a = _feather(region.astype(np.uint8) * 255, 1.2) * rng.uniform(0.25, 0.50)
    out = img.astype(np.float32)
    mult = np.clip(blood / 255.0 * 1.6, 0.40, 1.0)        # multiply toward the blood colour (floored)
    out = out * (1 - a[..., None]) + out * mult * a[..., None]
    gloss = np.zeros((H, W), np.uint8)
    ang = rng.uniform(0, np.pi)
    p0 = np.array([xs[i], ys[i]], float)
    dvec = np.array([np.cos(ang), np.sin(ang)]) * r * 0.8
    cv2.line(gloss, tuple(np.round(p0 - dvec).astype(int)), tuple(np.round(p0 + dvec).astype(int)), 255, 1, cv2.LINE_AA)
    g = _feather(gloss, 0.9) * region * rng.uniform(0.25, 0.6)
    out = out + g[..., None] * (np.array([235, 228, 250], np.float32) - out)
    return np.clip(out, 0, 255).astype(np.uint8), cov


# ------------------------------------------------------------------ driver
def process_sample(d, stem, V, blade_dir, rng, keep_clean=False):
    jp = os.path.join(d, f"{stem}.json")
    meta = json.load(open(jp))
    img = cv2.imread(os.path.join(d, meta["image"].get("file_rgb", f"{stem}.png")))
    bg = cv2.imread(os.path.join(d, meta["image"].get("file_bg", f"{stem}_bg.png")))
    mask = cv2.imread(os.path.join(d, meta["image"].get("file_mask", f"{stem}_mask.png")), 0)
    if img is None or bg is None or mask is None or not (mask > 127).any():
        return None
    m = mask > 127
    visible = m.astype(np.uint8) * 255
    prev = meta["image"].get("file_mask_visible")        # start from the renderer's visibility (3-D applier)
    if prev and os.path.exists(os.path.join(d, prev)):
        pv_ = cv2.imread(os.path.join(d, prev), 0)
        if pv_ is not None and pv_.shape == mask.shape:
            visible = ((pv_ > 127) & m).astype(np.uint8) * 255
    info = {"tissue": None, "applier": None, "blood": None}
    out = img

    def vis_frac(v):
        return float((v > 127).sum() / max(m.sum(), 1))

    if rng.random() < PROB["blood"]:
        r = blood_film(out, bg, mask, rng)
        if r is not None:
            out, cov = r
            info["blood"] = round(cov, 3)
    items = _load_instruments()
    if items and not meta["clip"].get("tool_held") and rng.random() < PROB["applier"]:
        r = applier_cutout(out, mask, meta, V, blade_dir, rng, items)
        if r is not None:
            out2, occ, cov, src = r
            vis2 = visible.copy()
            vis2[occ > 127] = 0
            if vis_frac(vis2) >= MIN_VISIBLE:
                out, visible, info["applier"], info["applier_src"] = out2, vis2, round(cov, 3), src
    if rng.random() < PROB["tissue"]:
        r = tissue_cover(out, bg, mask, rng, meta, V, blade_dir)
        if r is not None:
            out2, cover, cov = r
            vis2 = visible.copy()
            vis2[cover > 127] = 0
            if vis_frac(vis2) >= MIN_VISIBLE:
                out, visible, info["tissue"] = out2, vis2, round(cov, 3)
    modified = any(info.get(k) is not None for k in ("tissue", "applier", "blood"))
    if modified:
        q = int(rng.integers(JPEG_AFTER[0], JPEG_AFTER[1] + 1))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        info["jpeg_after"] = q
        if keep_clean:
            shutil.copy2(os.path.join(d, f"{stem}.png"), os.path.join(d, f"{stem}_clean.png"))
        cv2.imwrite(os.path.join(d, f"{stem}.png"), out)
    cv2.imwrite(os.path.join(d, f"{stem}_mask_visible.png"), visible)
    info["visible_frac"] = round(vis_frac(visible), 3)
    meta["image"]["file_mask_visible"] = f"{stem}_mask_visible.png"
    meta["occlusion"] = info
    with open(jp, "w") as f:
        json.dump(meta, f, indent=2)
    return out, mask, visible, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_pilot_v3")
    ap.add_argument("--out", default=None, help="copy src here first (default: --in-place required)")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-clean", action="store_true")
    ap.add_argument("--montage", default=None)
    ap.add_argument("--n-montage", type=int, default=24)
    ap.add_argument("--shard", default=None, help="K/N: process every N-th stem starting at K (run N processes in parallel, e.g. in-place on a copy)")
    ap.add_argument("--stem-seed", action="store_true",
                    help="seed per sample as seed*100003 + int(stem): reproduces the generator's in-line occlusion draws")
    a = ap.parse_args()
    src = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    if a.out:
        d = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
        if not os.path.exists(d):
            shutil.copytree(src, d)
    elif a.in_place:
        d = src
    else:
        print("give --out DIR (copy) or --in-place")
        return
    import pyvista as pv
    mesh = pv.read(MESH_PATH)
    V = np.asarray(mesh.points, np.float64)
    blade_dir = pca_frame(V)[1][:, 0]
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "[0-9]*.json")))
    if a.limit:
        stems = stems[:a.limit]
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        stems = stems[k::n]
        if not a.stem_seed:
            print("[shard] forcing --stem-seed so every process draws a different, reproducible occlusion per stem")
            a.stem_seed = True
    cells, counts, vis = [], {"tissue": 0, "applier": 0, "blood": 0, "clean": 0, "cutout": 0}, []
    print(f"instrument cutouts available: {len(_load_instruments())}")
    for i, stem in enumerate(stems):
        sid = int(stem) if a.stem_seed and stem.isdigit() else i
        r = process_sample(d, stem, V, blade_dir, np.random.default_rng(a.seed * 100003 + sid), a.keep_clean)
        if r is None:
            continue
        out, mask, visible, info = r
        vis.append(info["visible_frac"])
        mod = False
        for k in ("tissue", "applier", "blood"):
            if info.get(k) is not None:
                counts[k] += 1
                mod = True
        counts["cutout"] += "applier_src" in info
        counts["clean"] += not mod
        if a.montage and len(cells) < a.n_montage:
            ys, xs = np.nonzero(mask > 127)
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            s = int(max(64, 1.8 * max(xs.max() - xs.min(), ys.max() - ys.min())))
            H, W = mask.shape
            s = min(s, min(H, W))
            x0, y0 = int(np.clip(cx - s // 2, 0, W - s)), int(np.clip(cy - s // 2, 0, H - s))
            cell = cv2.resize(out[y0:y0 + s, x0:x0 + s], (224, 224), interpolation=cv2.INTER_AREA)
            lab = stem + " " + " ".join(k[0] + f"{info[k]:.2f}" for k in ("tissue", "applier", "blood") if info.get(k) is not None)
            if "applier_src" in info:
                lab += " R"
            cv2.rectangle(cell, (0, 0), (224, 16), (0, 0, 0), -1)
            cv2.putText(cell, lab if lab.strip() != stem else stem + " clean", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cells.append(cell)
    print(f"{os.path.basename(os.path.normpath(d))}: n={len(vis)} | tissue {counts['tissue']} applier {counts['applier']} "
          f"(real cutouts {counts['cutout']}) blood {counts['blood']} clean {counts['clean']} | visible fraction median "
          f"{np.median(vis):.2f} min {np.min(vis):.2f}")
    if a.montage and cells:
        while len(cells) % 6:
            cells.append(np.zeros_like(cells[0]))
        grid = np.vstack([np.hstack(cells[j:j + 6]) for j in range(0, len(cells), 6)])
        cv2.imwrite(a.montage, grid)
        print("montage ->", a.montage)


if __name__ == "__main__":
    main()
