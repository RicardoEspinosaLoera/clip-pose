"""
render_scene3d_depth.py  (Phase E prototype)
============================================

Proof-of-concept for a depth-USEFUL synthetic dataset: instead of pasting the
metallic clip over a FLAT 2-D backplate (which can only give clip-only depth GT),
we build a real 3-D tissue SURFACE from the backplate and render the clip resting
on it. Because RGB and depth come from the SAME geometry, the depth GT is dense,
metric, and image-consistent for BOTH the clip and the tissue -- exactly what a
monocular depth predictor needs, while the clip keeps its valid 6-DoF pose GT.

Pipeline (one sample)
---------------------
  1. Depth Anything V2 on the backplate  -> relative height field (disparity).
  2. Back-project the image grid to camera space at a base depth Z0, displaced by
     the height field, and transform to world  -> a textured tissue mesh that
     exactly fills the view and is textured with the backplate.
  3. Place the clip mesh just in front of the surface (rests on tissue).
  4. Render RGB (textured surface + PBR clip).
  5. Render dense metric depth from the z-buffer (surface + clip, no NaN holes).
  6. Clip mask = clip-only depth pass (surface hidden).

Outputs (scratch/prototype): proto_rgb.png, proto_depth.png, proto_depth.npy,
proto_mask.png, proto_check.png (RGB | rendered-depth | DepthAnything).

Run:
  python render_scene3d_depth.py --bg Data/noclips/1.png --out _proto3d
"""

import os
import argparse

import cv2
import numpy as np
import pyvista as pv

from estimate_bg_depth import build_depth_pipeline, estimate_bg_depth

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- render / camera config (prototype: one fixed top-down camera) ----
W, H = 800, 544
ZNEAR, ZFAR = 0.01, 1000.0
VIEW_ANGLE = 30.0                      # vertical FOV (deg), matches VTK default

CAM_DIST = 70.0                        # camera distance from focal
FOCAL = np.array([0.0, 0.0, 0.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])   # camera looks down -Z toward focal

SURF_Z0 = 74.0                         # base camera-space depth of the tissue surface
SURF_AMP = 9.0                         # peak displacement of the height field (world units)
GRID_W, GRID_H = 300, 210              # tissue mesh resolution
OVERSCAN = 0.35                        # extend the surface beyond the frame so it always fills it

CLIP_FORWARD_OFFSET = 10.0             # place clip this far IN FRONT of the surface base
CLIP_MAT = dict(color=(0.62, 0.66, 0.72), metallic=0.7, roughness=0.35,
                specular=0.5, diffuse=0.8, ambient=0.25)   # brighter titanium than the 2-D path


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def intrinsics(w, h, view_angle=VIEW_ANGLE):
    fovy = np.deg2rad(view_angle)
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (w / h)
    return fx, fy, (w - 1) / 2.0, (h - 1) / 2.0


def camera_basis(pos, focal, up):
    """Return Rc2w (camera->world, axes as columns) for a (pos, focal, up) camera."""
    fwd = _norm(focal - pos)
    right = _norm(np.cross(fwd, up))
    up_cam = np.cross(right, fwd)
    return np.stack([right, up_cam, fwd], axis=1)   # columns: right, up, forward


def build_tissue_surface(bg_bgr, disp, cam_pos, Rc2w, fx, fy, cx, cy):
    """Back-project a grid of image pixels to a displaced surface in world space,
    textured with the backplate. Returns (pv.StructuredGrid, pv.Texture)."""
    # grid of pixel centres covering the whole image, plus overscan beyond the
    # frame edges so the surface always fills the view (no border holes).
    us = np.linspace(-OVERSCAN * W, (1 + OVERSCAN) * W, GRID_W)
    vs = np.linspace(-OVERSCAN * H, (1 + OVERSCAN) * H, GRID_H)
    uu, vv = np.meshgrid(us, vs)                      # (GH, GW)

    # height field sampled at the grid (disp in [0,1], closer=brighter)
    d_small = cv2.resize(disp, (GRID_W, GRID_H), interpolation=cv2.INTER_CUBIC)
    d_small = np.clip(d_small, 0, 1)
    # camera-space depth: nearer where disparity is high
    z = SURF_Z0 + SURF_AMP * (0.5 - d_small)          # (GH, GW)

    # back-project pixel (u,v) at depth z to camera space, then to world
    xc = (uu - cx) / fx * z
    yc = (vv - cy) / fy * z
    pts_cam = np.stack([xc, yc, z], axis=-1).reshape(-1, 3)   # (N,3), row-major (v outer, u inner)
    pts_world = (Rc2w @ pts_cam.T).T + cam_pos[None, :]

    grid = pv.StructuredGrid()
    grid.points = pts_world
    grid.dimensions = [GRID_W, GRID_H, 1]             # u fastest, matches reshape order

    # texture coords: u->s, v->t (flip t so the image is upright on the surface),
    # clamped to [0,1] so overscanned vertices sample the edge, not a tiled repeat.
    s = np.clip(uu / (W - 1), 0, 1).ravel()
    t = 1.0 - np.clip(vv / (H - 1), 0, 1).ravel()
    grid.active_texture_coordinates = np.column_stack([s, t]).astype(np.float32)

    rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    tex = pv.Texture(np.ascontiguousarray(rgb))
    tex.repeat = False                                # clamp-to-edge
    return grid, tex


def clip_depth_metric(plotter):
    plotter.render()
    z = plotter.get_image_depth(fill_value=np.nan)
    finite = np.isfinite(z)
    m = np.full(z.shape, np.nan, np.float32)
    m[finite] = -z[finite]                             # VTK reports negative in front
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="Data/noclips/1.png")
    ap.add_argument("--out", default="_proto3d")
    ap.add_argument("--mesh", default="Item.obj")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    bg_path = args.bg if os.path.isabs(args.bg) else os.path.join(HERE, args.bg)

    bg = cv2.resize(cv2.imread(bg_path, cv2.IMREAD_COLOR), (W, H))

    print("Estimating tissue height field (Depth Anything V2)...")
    pipe = build_depth_pipeline(args.model)
    tmp = os.path.join(out_dir, "_bg_tmp.png")
    cv2.imwrite(tmp, bg)
    disp = estimate_bg_depth(pipe, tmp)                # [0,1], closer brighter

    # ---- camera ----
    fwd = np.array([0.0, 0.0, -1.0])                   # look down -Z
    cam_pos = FOCAL - fwd * CAM_DIST                   # -> (0,0,70)
    view_up = np.array([0.0, 1.0, 0.0])
    Rc2w = camera_basis(cam_pos, FOCAL, view_up)
    fx, fy, cx, cy = intrinsics(W, H)

    # ---- scene ----
    plotter = pv.Plotter(off_screen=True, window_size=[W, H])
    plotter.camera.clipping_range = (ZNEAR, ZFAR)
    plotter.add_light(pv.Light(light_type="headlight", intensity=2.5, color="white"))
    plotter.add_light(pv.Light(position=(40, 30, 90), light_type="scene light", intensity=1.0))

    # Tissue surface is UNLIT: the backplate is already a real, lit photograph, so
    # displaying its texture flat reproduces the frame faithfully (no double-lighting
    # blow-out) while the geometry still yields correct depth/occlusion. Only the clip
    # is PBR-lit by the lights above.
    surf, tex = build_tissue_surface(bg, disp, cam_pos, Rc2w, fx, fy, cx, cy)
    surf_actor = plotter.add_mesh(surf, texture=tex, lighting=False)

    # ---- clip: place just in front of the surface, centered, mild tilt ----
    clip = pv.read(args.mesh if os.path.isabs(args.mesh) else os.path.join(HERE, args.mesh))
    clip = clip.rotate_z(25, inplace=False).rotate_x(12, inplace=False)
    clip_depth = SURF_Z0 - CLIP_FORWARD_OFFSET         # camera-space depth (in front of surface)
    clip_center_world = (Rc2w @ np.array([0.0, 0.0, clip_depth])) + cam_pos
    clip.translate(np.asarray(clip_center_world) - np.array(clip.center), inplace=True)
    clip_actor = plotter.add_mesh(clip, pbr=True, smooth_shading=True, **CLIP_MAT)

    plotter.camera_position = (tuple(cam_pos), tuple(FOCAL), tuple(view_up))
    plotter.camera.view_angle = VIEW_ANGLE
    plotter.show(auto_close=False)
    print(f"[cam] my view_angle={VIEW_ANGLE} vtk={plotter.camera.view_angle:.2f} "
          f"pos={np.round(plotter.camera.position,1)} up={np.round(plotter.camera.up,2)}")

    # ---- RGB ----
    rgb = plotter.screenshot(return_img=True)
    cv2.imwrite(os.path.join(out_dir, "proto_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # ---- dense metric depth (surface + clip) ----
    dense = clip_depth_metric(plotter)
    valid = np.isfinite(dense)
    np.save(os.path.join(out_dir, "proto_depth.npy"), dense)
    viz = np.zeros(dense.shape, np.uint8)
    if valid.any():
        dmin, dmax = np.nanmin(dense), np.nanmax(dense)
        viz[valid] = (255 * (1 - (dense[valid] - dmin) / max(dmax - dmin, 1e-9))).astype(np.uint8)
    viz = cv2.applyColorMap(viz, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, "proto_depth.png"), viz)

    # ---- clip mask: hide surface, depth pass = clip only ----
    surf_actor.SetVisibility(False)
    clip_only = clip_depth_metric(plotter)
    mask = (np.isfinite(clip_only).astype(np.uint8)) * 255
    cv2.imwrite(os.path.join(out_dir, "proto_mask.png"), mask)
    surf_actor.SetVisibility(True)

    # ---- 3-panel check ----
    da_viz = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    da_viz = cv2.resize(da_viz, (W, H))
    check = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), viz, da_viz])
    for x, lab in [(10, "RGB (clip on tissue)"), (W + 10, "rendered depth (dense)"),
                   (2 * W + 10, "DepthAnything input")]:
        cv2.putText(check, lab, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "proto_check.png"), check)

    frac = float(valid.mean())
    print(f"dense depth coverage: {frac*100:.1f}% of frame (want ~100%, i.e. no holes)")
    print(f"clip mask pixels: {int((mask>0).sum())}")
    print(f"wrote proto_rgb/depth/mask/check to {out_dir}")


if __name__ == "__main__":
    main()
