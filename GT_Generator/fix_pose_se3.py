"""
fix_pose_se3.py — repair the clip->camera pose (`clip.pose_se3`) in existing GT JSONs.

Why: until 2026-10 the generator built the camera frame as x right, y UP, z forward, which
is LEFT-handed (det -1). The exported pose_se3 "rotation" was therefore a reflection, the
quaternion written from it did not even represent that matrix, and the GT pose projected
with the stored intrinsics landed mirrored about cy (~80 px off the clip). `pose_world` and
the camera tuple were always correct, so the fix is a pure JSON rewrite -- no re-render.

What it does, per NNN.json:
  * recompute pose_se3 = (R_w2c @ R_world, R_w2c @ t_world + t_w2c) with the OpenCV frame
    (x right, y down, z forward), quaternion via scipy (proper rotation)
  * keep the old block as `pose_se3_legacy_lefthanded` (first run only)
  * rewrite the `conventions` block to describe the truth
  * VALIDATE: project the (jaw-articulated) clip mesh with the NEW pose + stored K and
    compare with NNN_mask.png -> centroid error (px) and fraction of projected vertices
    inside the (3 px dilated) mask. Samples above --max-centroid-px are reported.

Usage
  python fix_pose_se3.py --dirs out_pose100 out_pose100_appear        # repair + validate
  python fix_pose_se3.py --all                                        # every out_*/ with pose JSONs
  python fix_pose_se3.py --dirs out_rand_depth --dry-run              # validate only, write nothing
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from articulate_clip import articulate

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.environ.get("CLIP_MESH", os.path.join(HERE, "Item.obj"))   # must match the mesh the set was rendered with

CONVENTIONS = {
    "camera_frame": "OpenCV: x right, y down, z forward (right-handed); "
                    "pose_se3 maps clip model coords -> this frame",
    "projection": "u = fx*X/Z + cx, v = fy*Y/Z + cy; integer (u,v) = pixel centre, cx=(w-1)/2",
    "pixel_origin": "top_left",
    "handedness": "right", "cam_forward_axis": "+Z",
    "quaternion": "wxyz, unit, from a proper rotation (det +1)",
    "intrinsics": "square pixels, fx = fy (VTK vertical view angle); a legacy export had fx = fy*w/h",
    "repaired_by": "fix_pose_se3.py (2026-10): previous pose_se3 used a left-handed camera frame; "
                   "non-square fx (if any) reset to fy",
}


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def cam_w2c_opencv(pos, focal, up):
    pos, focal, up = (np.asarray(v, float) for v in (pos, focal, up))
    fwd = _norm(focal - pos)
    right = _norm(np.cross(fwd, up))
    down = -np.cross(right, fwd)
    R = np.stack([right, down, fwd], axis=0)
    return R, -R @ pos


def quat_wxyz(R):
    x, y, z, w = scipyR.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return (q if q[0] >= 0 else -q).tolist()


class MeshCache:
    def __init__(self):
        self.base = pv.read(MESH_PATH)
        self.cache = {}

    def verts(self, jaw):
        key = round(float(jaw) * 2) / 2.0            # 0.5 deg bins
        if key not in self.cache:
            m = articulate(self.base, key) if key > 0 else self.base
            self.cache[key] = np.asarray(m.points, np.float64)
        return self.cache[key]


def validate(V, R, t, K, mask):
    """Project vertices with (R,t,K) and compare with the GT mask.

    Returns (bbox_centre_err_px, inside_fraction, silhouette_iou). The vertex-density
    centroid is deliberately NOT used: vertices are much denser on the coil than on the
    blades, so it sits several px off the area centroid even for a perfect pose. The
    bbox centre and the splat IoU are density-free."""
    fx, fy, cx, cy = K
    P = (R @ V.T).T + t
    z = P[:, 2]
    ok = z > 1e-6
    H, W = mask.shape
    m = mask > 127
    if not ok.any() or not m.any():
        return np.inf, 0.0, 0.0
    u = fx * P[ok, 0] / z[ok] + cx
    v = fy * P[ok, 1] / z[ok] + cy
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not inb.any():
        return np.inf, 0.0, 0.0
    dil = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    inside = float(dil[vi[inb], ui[inb]].mean())
    splat = np.zeros((H, W), np.uint8)
    splat[vi[inb], ui[inb]] = 1
    splat = cv2.morphologyEx(cv2.dilate(splat, np.ones((3, 3), np.uint8)),
                             cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    iou = float((splat & m).sum() / max((splat | m).sum(), 1))

    def bbox_centre(b):
        ys, xs = np.nonzero(b)
        return np.array([(xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0])
    berr = float(np.linalg.norm(bbox_centre(splat) - bbox_centre(m)))
    return berr, inside, iou


def fix_dir(d, meshes, dry_run=False, max_centroid_px=3.0):
    jsons = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
    stats, bad, n_written, n_skipped, n_fx = [], [], 0, 0, 0
    for jp in jsons:
        with open(jp) as f:
            meta = json.load(f)
        if "clip" not in meta or "pose_world" not in meta["clip"] or "camera" not in meta:
            n_skipped += 1
            continue
        cam = meta["camera"]
        # Older exports (out_rand_depth*, Jul 2026) carry fx = fy*w/h, but VTK renders square
        # pixels (the aspect widens the horizontal FOV, it does not change the focal length):
        # the true fx equals fy. The wrong fx shifted projections horizontally by up to ~100 px.
        if abs(cam["fx"] / cam["fy"] - 1.0) > 1e-6:
            if "fx_legacy_nonsquare" not in cam:
                cam["fx_legacy_nonsquare"] = cam["fx"]
            cam["fx"] = float(cam["fy"])
            n_fx += 1
        R_w2c, t_w2c = cam_w2c_opencv(cam["position"], cam["focal_point"], cam["view_up"])
        qw = meta["clip"]["pose_world"]["quaternion_wxyz"]
        R_w = scipyR.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
        t_w = np.asarray(meta["clip"]["pose_world"]["translation_m"], float)
        R_co = R_w2c @ R_w
        t_co = R_w2c @ t_w + t_w2c
        new_se3 = {"quaternion_wxyz": quat_wxyz(R_co), "translation_m": [float(x) for x in t_co]}

        # validation against the saved mask
        stem = os.path.splitext(os.path.basename(jp))[0]
        mask_name = meta.get("image", {}).get("file_mask", f"{stem}_mask.png")
        mask = cv2.imread(os.path.join(d, mask_name), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            K = (cam["fx"], cam["fy"], cam["cx"], cam["cy"])
            Rq = scipyR.from_quat(np.roll(new_se3["quaternion_wxyz"], -1)).as_matrix()   # from the WRITTEN quaternion
            berr, inside, iou = validate(meshes.verts(meta["clip"].get("jaw_angle_deg", 0.0)),
                                         Rq, np.asarray(new_se3["translation_m"]), K, mask)
            stats.append((berr, inside, iou))
            if berr > max_centroid_px or inside < 0.9 or iou < 0.5:
                bad.append((stem, berr, inside, iou))

        if not dry_run:
            if "pose_se3_legacy_lefthanded" not in meta["clip"]:
                meta["clip"]["pose_se3_legacy_lefthanded"] = meta["clip"].get("pose_se3")
            meta["clip"]["pose_se3"] = new_se3
            meta["conventions"] = dict(CONVENTIONS)
            with open(jp, "w") as f:
                json.dump(meta, f, indent=2)
            n_written += 1

    arr = np.array(stats) if stats else np.zeros((0, 3))
    tag = "DRY-RUN " if dry_run else ""
    if len(arr):
        print(f"{tag}{os.path.basename(os.path.normpath(d)):24s} n={len(jsons):5d} written={n_written:5d} fx-fixed={n_fx:5d} "
              f"| bbox-centre err px: median {np.median(arr[:,0]):.2f} p95 {np.percentile(arr[:,0],95):.2f} "
              f"max {arr[:,0].max():.2f} | inside-mask min {arr[:,1].min():.3f} "
              f"| IoU median {np.median(arr[:,2]):.3f} min {arr[:,2].min():.3f} | flagged {len(bad)}", flush=True)
    else:
        print(f"{tag}{os.path.basename(os.path.normpath(d)):24s} n={len(jsons)} (no pose JSONs, skipped {n_skipped})")
    for stem, berr, inside, iou in bad[:10]:
        print(f"    flagged {stem}: bbox-centre {berr:.1f} px, inside {inside:.2f}, IoU {iou:.2f}")
    return arr, bad


def main():
    ap = argparse.ArgumentParser(description="Repair clip.pose_se3 in GT JSONs (left-handed camera frame bug)")
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every GT_Generator/out_*/ with NNN.json files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-centroid-px", type=float, default=3.0)
    a = ap.parse_args()

    if a.all:
        dirs = [d for d in sorted(glob.glob(os.path.join(HERE, "out_*")))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "[0-9]*.json"))]
    else:
        dirs = [d if os.path.isabs(d) else os.path.join(HERE, d) for d in (a.dirs or ["out_pose100"])]
    meshes = MeshCache()
    allstats, allbad = [], 0
    for d in dirs:
        arr, bad = fix_dir(d, meshes, a.dry_run, a.max_centroid_px)
        if len(arr):
            allstats.append(arr)
        allbad += len(bad)
    if allstats:
        arr = np.concatenate(allstats)
        print(f"\nTOTAL n={len(arr)}: bbox-centre err median {np.median(arr[:,0]):.2f} px, p95 "
              f"{np.percentile(arr[:,0],95):.2f}, max {arr[:,0].max():.2f} | inside-mask min {arr[:,1].min():.3f} "
              f"| IoU median {np.median(arr[:,2]):.3f} min {arr[:,2].min():.3f} | flagged {allbad}")


if __name__ == "__main__":
    main()
