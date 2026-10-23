"""
measure_pose_fidelity.py — how well does the FINAL image respect the pose GT in its JSON?

The saved mask is rendered from the same pose, so mask-vs-JSON agreement proves nothing about
the delivered PNG. Here the clip is found in the final image WITHOUT using the mask (change
against the clip-free backplate NNN_bg.png, applier region excluded) and compared with what
the GT pose + intrinsics predict:

  * silhouette : IoU / precision / recall of the detected clip vs the GT silhouette
                 (visible part, i.e. full mask minus the rendered tool)
  * 2-D pose   : centroid shift (px), principal-axis rotation (deg), scale change (%) between
                 the detected and the GT silhouette (scale <-> depth)
  * keypoints  : distance (px) from the projected 3-D jaw tip and spring end (GT pose) to the
                 nearest detected clip pixel
  * shift      : masked template correlation of a clean re-render of the GT pose against the
                 final image (the paper's clip-displacement measure, robust version)

Per-sample numbers go to <dir>/_pose_fidelity.csv, a summary to stdout and an overlay montage
(GT silhouette green, detected magenta, projected tip/spring as dots) to <dir>/_pose_fidelity.jpg.

Usage
  python measure_pose_fidelity.py --dir out_set50b [--n-montage 20]
"""

import os
import sys
import csv
import json
import glob
import argparse

import cv2
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as scipyR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism")))
from articulate_clip import articulate, pca_frame
from metrics import clip_shift_robust
import restyle_appearance as ra

MESH = pv.read(os.environ.get("CLIP_MESH", os.path.join(HERE, "Item.obj")))   # must match the mesh the set was rendered with
BLADE_DIR = pca_frame(np.asarray(MESH.points, np.float64))[1][:, 0]
_ART = {}


def verts(jaw):
    key = round(float(jaw) * 2) / 2.0
    if key not in _ART:
        m = articulate(MESH, key) if key > 0 else MESH
        _ART[key] = (m, np.asarray(m.points, np.float64))
    return _ART[key]


def project(meta, P):
    """3-D points in CLIP frame -> pixels via pose_se3 (OpenCV) + intrinsics."""
    q = meta["clip"]["pose_se3"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_se3"]["translation_m"], float)
    c = meta["camera"]
    X = (R @ P.T).T + t
    return np.stack([c["fx"] * X[:, 0] / X[:, 2] + c["cx"], c["fy"] * X[:, 1] / X[:, 2] + c["cy"]], 1)


def detect_clip(img, bg, applier_mask, gt_mask):
    """Clip pixels in the final image = change vs the clip-free backplate (Lab), minus the tool,
    minus JPEG noise, within a generous window (3x the GT bbox) so a real drift would be seen."""
    H, W = gt_mask.shape
    lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = np.abs(lab_i[..., 0] - lab_b[..., 0])
    dab = np.hypot(lab_i[..., 1] - lab_b[..., 1], lab_i[..., 2] - lab_b[..., 2])
    det = (dL > 30) | (dab > 20)
    det = cv2.morphologyEx(det.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    excl = np.zeros_like(det)
    if applier_mask is not None:                      # the defocused rod bleeds up to ~18 px past its mask
        excl = cv2.dilate(applier_mask.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        det &= ~excl
    ys, xs = np.nonzero(gt_mask)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    x0, x1 = max(0, xs.min() - bw), min(W, xs.max() + 1 + bw)
    y0, y1 = max(0, ys.min() - bh), min(H, ys.max() + 1 + bh)
    win = np.zeros_like(det)
    win[y0:y1, x0:x1] = True
    det &= win
    # keep components of a sensible size (drop speckle), no use of the GT mask position
    n, lab, st, _ = cv2.connectedComponentsWithStats(det.astype(np.uint8))
    keep = np.zeros_like(det)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 25:
            keep |= lab == i
    return keep, excl


def moments(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return None
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1])) % 180.0
    return cx, cy, ang, float(len(xs))


def clean_reference(meta, bg, jaw, cam, fovy):
    """Flat grey re-render of the GT pose over the backplate (template for the shift metric)."""
    m, _ = verts(jaw)
    q = meta["clip"]["pose_world"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_world"]["translation_m"], float)
    posed = ra.posed_mesh(MESH, R, t, jaw)
    H, W = bg.shape[:2]
    env = np.full((ra.ENV_H, ra.ENV_W, 3), 0.5, np.float32)
    lights = dict(warm=(1.0, 1.0, 1.0), head=0.5, fill=0.0, fill_pos=tuple(cam[0]), focal=tuple(cam[1]))
    rgba = ra.render_layer([(posed, dict(base=(0.5, 0.5, 0.55), rough=0.4, coat=0.0))], cam, fovy, env, lights, W, H)
    lay, a = ra._layer_linear(rgba, W, H)
    bg_lin = np.power(bg.astype(np.float32) / 255.0, 2.2)
    out = lay + (1 - a[..., None]) * bg_lin
    return (np.power(np.clip(out, 0, 1), 1 / 2.2) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_set50b")
    ap.add_argument("--n-montage", type=int, default=20)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(HERE, a.dir)
    rows, cells = [], []
    jsons = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        meta = json.load(open(jp))
        img = cv2.imread(os.path.join(d, f"{stem}.png"))
        bg = cv2.imread(os.path.join(d, meta["image"].get("file_bg", f"{stem}_bg.png")))
        gt = cv2.imread(os.path.join(d, meta["image"].get("file_mask", f"{stem}_mask.png")), 0) > 127
        vis_p = os.path.join(d, meta["image"].get("file_mask_visible", f"{stem}_mask_visible.png"))
        vis = (cv2.imread(vis_p, 0) > 127) if os.path.exists(vis_p) else gt
        app_p = os.path.join(d, f"{stem}_mask_applier.png")
        app = (cv2.imread(app_p, 0) > 127) if os.path.exists(app_p) else None
        if img is None or bg is None or not gt.any():
            continue
        jaw = float(meta["clip"].get("jaw_angle_deg", 0.0))
        det, excl = detect_clip(img, bg, app, gt)
        vis = vis & ~excl                               # compare only where the clip is not under the tool
        if vis.sum() < 20:
            continue
        inter = (det & vis).sum()
        iou = inter / max((det | vis).sum(), 1)
        prec = inter / max(det.sum(), 1)
        rec = inter / max(vis.sum(), 1)
        md, mg = moments(det), moments(vis)
        if md is None or mg is None:
            continue
        dxy = float(np.hypot(md[0] - mg[0], md[1] - mg[1]))
        dang = abs(md[2] - mg[2])
        dang = min(dang, 180.0 - dang)
        dscale = (np.sqrt(md[3] / mg[3]) - 1.0) * 100.0
        # projected 3-D keypoints of the GT pose -> distance to the nearest detected clip pixel
        _, V = verts(jaw)
        proj = (V - V.mean(0)) @ BLADE_DIR
        kp3 = np.stack([V[np.argmax(proj)], V[np.argmin(proj)]])
        kp = project(meta, kp3)
        dist_map = cv2.distanceTransform((~det).astype(np.uint8), cv2.DIST_L2, 3)
        H, W = gt.shape
        kerr = []
        for (u, v) in kp:
            ui, vi = int(round(u)), int(round(v))
            kerr.append(float(dist_map[vi, ui]) if 0 <= ui < W and 0 <= vi < H else np.nan)
        # tip may legitimately be hidden by the tool: report tip error only where the tip is visible
        def visible_at(uv):
            ui, vi = int(round(uv[0])), int(round(uv[1]))
            return bool(vis[vi, ui]) if 0 <= ui < W and 0 <= vi < H else False
        tip_visible, spring_visible = visible_at(kp[0]), visible_at(kp[1])
        # contour alignment: mean distance from the GT silhouette edge to the detected edge and back
        gt_edge = vis & ~(cv2.erode(vis.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        det_edge = det & ~(cv2.erode(det.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        d_to_det = cv2.distanceTransform((~det_edge).astype(np.uint8), cv2.DIST_L2, 3)
        d_to_gt = cv2.distanceTransform((~gt_edge).astype(np.uint8), cv2.DIST_L2, 3)
        shift = float(0.5 * (d_to_det[gt_edge].mean() + d_to_gt[det_edge].mean())) if det_edge.any() else np.nan
        rows.append(dict(stem=stem, tool=bool(meta["clip"].get("tool_held")), jaw=round(jaw, 1), iou=round(iou, 3),
                         precision=round(prec, 3), recall=round(rec, 3), centroid_px=round(dxy, 2), angle_deg=round(dang, 2),
                         scale_pct=round(dscale, 2), tip_px=(round(kerr[0], 2) if tip_visible else ""),
                         spring_px=(round(kerr[1], 2) if spring_visible else ""), contour_px=round(shift, 2)))
        if len(cells) < a.n_montage:
            vis_img = img.copy()
            cnt_g, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_d, _ = cv2.findContours(det.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, cnt_d, -1, (255, 0, 255), 1)
            cv2.drawContours(vis_img, cnt_g, -1, (0, 255, 0), 1)
            for (u, v), col in zip(kp, ((0, 255, 255), (255, 255, 0))):
                cv2.circle(vis_img, (int(round(u)), int(round(v))), 3, col, -1)
            lab = f"{stem} IoU{iou:.2f} c{dxy:.1f}px a{dang:.1f}deg s{dscale:+.0f}% edge{shift:.1f}px"
            cells.append(ra.zoom_cell(vis_img, gt.astype(np.uint8) * 255, 360, pad=3.0, label=lab))
        print(f"{stem} tool={int(bool(meta['clip'].get('tool_held')))} IoU {iou:.2f} P {prec:.2f} R {rec:.2f} | centroid {dxy:5.2f}px "
              f"angle {dang:5.2f}deg scale {dscale:+5.1f}% | tip {kerr[0]:.1f}px{'' if tip_visible else '(hidden)'} spring {kerr[1]:.1f}px{'' if spring_visible else '(hidden)'} | edge {shift:.2f}px", flush=True)

    if not rows:
        print("no samples")
        return
    with open(os.path.join(d, "_pose_fidelity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    arr = {k: np.array([r[k] for r in rows if r[k] != ""], float) for k in ("iou", "precision", "recall", "centroid_px", "angle_deg", "scale_pct", "tip_px", "spring_px", "contour_px")}
    print(f"\n{os.path.basename(os.path.normpath(d))}: n={len(rows)} (tool {sum(r['tool'] for r in rows)})")
    print(f"{'metric':12s} {'median':>8s} {'p95':>8s} {'max':>8s}")
    for k, v in arr.items():
        if k == "scale_pct":
            v = np.abs(v)
        print(f"{k:12s} {np.median(v):8.2f} {np.percentile(v, 95):8.2f} {v.max():8.2f}")
    while len(cells) % 4:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
    banner = np.zeros((56, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, "GT silhouette (green) vs clip detected in the final image (magenta); dots = projected jaw tip / spring", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    grid = np.vstack([banner, grid])
    small = cv2.resize(grid, (1200, int(grid.shape[0] * 1200 / grid.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(d, "_pose_fidelity.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("csv  ->", os.path.join(d, "_pose_fidelity.csv"))
    print("montage ->", os.path.join(d, "_pose_fidelity.jpg"))


if __name__ == "__main__":
    main()
