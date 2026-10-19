"""
mv_fuse_check.py — multi-view depth-fusion sanity check.

Unprojects every view's depth (via the exported transforms.json c2w + intrinsics)
into a single WORLD point cloud, coloured by the RGB. If the multi-view set is
geometrically consistent, the K views fuse into ONE coherent surface + clip
(not K misaligned copies) -- the trivial-but-real form of multi-view 3-D
reconstruction, so it directly demonstrates the data supports MVS/NeRF/3DGS.

Outputs (in the scene dir): fused.ply, fused_views.png (the cloud rendered from a
few novel angles).

Run: python mv_fuse_check.py --scene out_mv/scene_0000
"""

import os
import json
import argparse

import cv2
import numpy as np
import pyvista as pv


def unproject_view(scene, fr, K):
    fx, fy, cx, cy = K
    d = np.load(os.path.join(scene, fr["depth_file"]))
    rgb = cv2.cvtColor(cv2.imread(os.path.join(scene, fr["file_path"])), cv2.COLOR_BGR2RGB)
    c2w = np.array(fr["transform_matrix"])
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    ok = np.isfinite(d)
    u = uu[ok]; v = vv[ok]; z = d[ok]
    # OpenGL camera: x right, y up, -z forward; saved depth = z-distance along forward
    xc = (u - cx) / fx * z
    yc = -(v - cy) / fy * z
    zc = -z
    pc = np.stack([xc, yc, zc, np.ones_like(z)], 0)
    pw = (c2w @ pc)[:3].T
    col = rgb[ok]
    return pw, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=3, help="subsample pixels for a lighter cloud")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    scene = args.scene if os.path.isabs(args.scene) else os.path.join(here, args.scene)

    T = json.load(open(os.path.join(scene, "transforms.json")))
    K = (T["fl_x"], T["fl_y"], T["cx"], T["cy"])

    pts, cols = [], []
    for fr in T["frames"]:
        pw, col = unproject_view(scene, fr, K)
        pw, col = pw[::args.stride], col[::args.stride]
        pts.append(pw); cols.append(col)
    P = np.concatenate(pts); C = np.concatenate(cols)
    print(f"fused {len(T['frames'])} views -> {len(P)} points")

    cloud = pv.PolyData(P)
    cloud["rgb"] = C.astype(np.uint8)
    cloud.save(os.path.join(scene, "fused.ply"))

    # render the fused cloud from a few angles to eyeball coherence
    shots = []
    for az in (0, 25, -25):
        p = pv.Plotter(off_screen=True, window_size=[420, 420])
        p.add_points(cloud, scalars="rgb", rgb=True, point_size=2.5)
        p.camera.azimuth = az
        p.camera.elevation = 12
        p.show(auto_close=False)
        shots.append(cv2.cvtColor(p.screenshot(return_img=True), cv2.COLOR_RGB2BGR))
        p.close()
    out = np.hstack(shots)
    cv2.putText(out, "fused point cloud (novel views) - coherent = consistent MV",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(scene, "fused_views.png"), out)
    print(f"wrote {scene}/fused.ply + fused_views.png")


if __name__ == "__main__":
    main()
