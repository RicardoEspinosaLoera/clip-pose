"""
define_keypoints.py — the 8 clip keypoints used by the pose estimator, as VERTEX INDICES of Item.obj
so that they follow the jaw articulation exactly (articulate() moves vertices, not a rig).

Keypoints (model frame, PCA axes e0 long / e1 wide / e2 thin):
  0 tip_a, 1 tip_b     : the two jaw tips (one per blade, found from what moves when the jaws open)
  2 hinge              : on the axis at the articulation pivot
  3 spring             : the spring/coil end (min e0)
  4 coil_centre        : vertex nearest the coil centroid
  5 body_a, 6 body_b   : on the two blades at 60 % of the length (shape of the opening)
  7 centroid           : vertex nearest the mesh centroid
Writes keypoints.json {names, indices, symmetric_pairs} and a check image where the points are
projected with the GT pose of a few samples (the same projection the training set will use).

Usage: python define_keypoints.py [--check-dir out_set50c]
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as scipyR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from articulate_clip import articulate, pca_frame

MESH = pv.read(os.path.join(HERE, "Item.obj"))
V0 = np.asarray(MESH.points, np.float64)
C, E = pca_frame(V0)
P = (V0 - C) @ E                                  # (n, 3): coordinates along e0, e1, e2


def nearest(target, cand=None):
    idx = np.arange(len(V0)) if cand is None else cand
    return int(idx[np.argmin(np.linalg.norm(V0[idx] - target, axis=1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-dir", default="out_set50c")
    ap.add_argument("--n-check", type=int, default=6)
    a = ap.parse_args()

    # which axis do the blades separate along when the jaws open? -> the one with the largest spread of displacement
    V20 = np.asarray(articulate(MESH, 20.0).points, np.float64)
    if len(V20) != len(V0):                         # hollow=True re-triangulates; use the non-hollow deformation
        V20 = np.asarray(articulate(MESH, 20.0, hollow=False).points, np.float64)
    D = (V20 - V0) @ E
    sep_axis = int(np.argmax(D.std(axis=0)[1:]) + 1)   # 1 = e1 or 2 = e2
    side = np.sign(D[:, sep_axis])
    moved = np.abs(D[:, sep_axis]) > 0.5
    print(f"blades separate along e{sep_axis}; moved vertices: {int(moved.sum())} of {len(V0)}")

    long_frac = (P[:, 0] - P[:, 0].min()) / np.ptp(P[:, 0])
    blade_a = moved & (side > 0)
    blade_b = moved & (side < 0)
    tip_a = int(np.flatnonzero(blade_a)[np.argmax(P[blade_a, 0])])
    tip_b = int(np.flatnonzero(blade_b)[np.argmax(P[blade_b, 0])])
    spring = int(np.argmin(P[:, 0]))
    hinge_t = P[:, 0].min() + 0.38 * np.ptp(P[:, 0])
    hinge = nearest(C + E[:, 0] * hinge_t)
    coil = P[:, 0] < P[:, 0].min() + 0.15 * np.ptp(P[:, 0])
    coil_centre = nearest(V0[coil].mean(0))
    body_t = P[:, 0].min() + 0.60 * np.ptp(P[:, 0])
    band = np.abs(P[:, 0] - body_t) < 0.03 * np.ptp(P[:, 0])
    body_a = nearest(V0[band & (side > 0)].mean(0), np.flatnonzero(band & (side > 0)))
    body_b = nearest(V0[band & (side < 0)].mean(0), np.flatnonzero(band & (side < 0)))
    centroid = nearest(V0.mean(0))
    names = ["tip_a", "tip_b", "hinge", "spring", "coil_centre", "body_a", "body_b", "centroid"]
    idx = [tip_a, tip_b, hinge, spring, coil_centre, body_a, body_b, centroid]
    out = dict(mesh="Item.obj", names=names, indices=idx, symmetric_pairs=[[0, 1], [5, 6]],
               separation_axis=f"e{sep_axis}", note="indices into Item.obj vertices; articulate() moves them with the jaws")
    json.dump(out, open(os.path.join(HERE, "keypoints.json"), "w"), indent=2)
    for n, i in zip(names, idx):
        print(f"  {n:12s} vertex {i:6d}  model {np.round(V0[i], 2)}  open20 {np.round(V20[i], 2)}")

    # check: project with the GT pose of a few samples and draw
    d = a.check_dir if os.path.isabs(a.check_dir) else os.path.join(HERE, a.check_dir)
    cells = []
    for jp in sorted(glob.glob(os.path.join(d, "[0-9]*.json")))[:a.n_check]:
        m = json.load(open(jp))
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = cv2.imread(os.path.join(d, f"{stem}.png"))
        jaw = float(m["clip"].get("jaw_angle_deg", 0.0))
        Vj = np.asarray((articulate(MESH, jaw, hollow=False) if jaw > 0 else MESH).points, np.float64)[idx]
        q = m["clip"]["pose_se3"]["quaternion_wxyz"]
        R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = np.asarray(m["clip"]["pose_se3"]["translation_m"], float)
        c = m["camera"]
        X = (R @ Vj.T).T + t
        u = c["fx"] * X[:, 0] / X[:, 2] + c["cx"]
        v = c["fy"] * X[:, 1] / X[:, 2] + c["cy"]
        mk = cv2.imread(os.path.join(d, f"{stem}_mask.png"), 0) > 127
        ys, xs = np.nonzero(mk)
        cx, cy = xs.mean(), ys.mean()
        s = int(max(120, 2.5 * max(np.ptp(xs), np.ptp(ys))))
        H, W = mk.shape
        x0, y0 = int(np.clip(cx - s // 2, 0, W - s)), int(np.clip(cy - s // 2, 0, H - s))
        vis = img.copy()
        cols = [(0, 255, 255), (0, 200, 255), (255, 255, 0), (255, 0, 255), (255, 128, 0), (0, 255, 0), (0, 160, 0), (255, 255, 255)]
        for k in range(8):
            cv2.circle(vis, (int(round(u[k])), int(round(v[k]))), 3, cols[k], -1)
        crop = cv2.resize(vis[y0:y0 + s, x0:x0 + s], (360, 360), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (360, 18), (0, 0, 0), -1)
        cv2.putText(crop, f"{stem} jaw {jaw:.0f} tool {int(bool(m['clip'].get('tool_held')))}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cells.append(crop)
    if cells:
        while len(cells) % 3:
            cells.append(np.zeros_like(cells[0]))
        grid = np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])
        cv2.imwrite(os.path.join(HERE, "_keypoints_check.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("check ->", os.path.join(HERE, "_keypoints_check.jpg"), "(tips yellow/orange, hinge cyan, spring magenta, coil orange, body greens, centroid white)")


if __name__ == "__main__":
    main()
