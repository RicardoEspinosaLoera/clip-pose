"""
render_clip_poses_depth.py
==========================

Recreated version of `render_clip_poses.py` for the new depth-aware approach.

In addition to the original artefacts per sample
    NNN.png        RGB composite (metallic clip over a tissue backplate)
    NNN_bg.png     the tissue backplate without the clip
    NNN_mask.png   binary clip mask (white = clip)
    NNN.json       6-DoF pose + camera (unchanged convention)

this version also emits a **clip depth pass**:
    NNN_depth_clip.npy   float32 metric depth (world units), NaN outside the clip
    NNN_depth_clip.png   16-bit normalised depth, for inspection
    NNN_normal.png       8-bit camera-frame normal map derived from the clip depth

The JSON is extended with a `depth` block recording units / convention so the
downstream stage can fuse the clip depth with a monocular estimate of the
background (tissue) depth -- see `estimate_bg_depth.py`.

Notes
-----
* The tissue frame is a 2-D background image (a backplate), NOT scene geometry,
  so the z-buffer contains only the clip.  `get_image_depth()` therefore returns
  finite values exactly on the clip and NaN everywhere else.
* Runs head-less (off_screen=True) so it can batch-render without a window.
* Paths default to this project's layout; override via the CONFIG block.
"""

import os
import sys
import json
import glob
import random
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyvista as pv
from PIL import Image
from scipy.spatial.transform import Rotation as scipyR

from articulate_clip import articulate, pca_frame   # jaw open/close deformation + PCA frame

np.set_printoptions(precision=2)

# Reproducible / sharded runs (DGX): SEED seeds numpy+random for the pose sampler, START_IDX
# offsets the stem counter so K shards can write disjoint stems into ONE output dir, and
# STEM_WIDTH widens the zero-padded stem (5 -> 00001..05000; 3 = legacy sets).
SEED       = os.environ.get("SEED")
if SEED is not None:
    np.random.seed(int(SEED))
    random.seed(int(SEED))
START_IDX  = int(os.environ.get("START_IDX", 0))
STEM_WIDTH = int(os.environ.get("STEM_WIDTH", 3))

# ------------------- CONFIG -------------------
HERE = os.path.dirname(os.path.abspath(__file__))

MESH_PATH    = os.environ.get("CLIP_MESH", os.path.join(HERE, "Item.obj"))   # clip mesh (env-overridable, e.g. clip_ti_std.obj from make_clip_mesh.py)
BG_DIR       = os.environ.get("BG_DIR", os.path.join(HERE, "Data", "noclips_bank"))   # 600 clip-free real backplates (env-overridable; Data/noclips = legacy 30)
OUT_DIR      = os.environ.get("OUT_DIR", os.path.join(HERE, "out_rand_depth"))  # output dir (env-overridable)
N            = int(os.environ.get("N_SAMPLES", 5000))   # number of samples (override via N_SAMPLES env)
OFF_SCREEN   = True                                    # head-less batch rendering

# Jaw articulation: each sample gets a random jaw-opening angle so the dataset
# spans clamping(0) .. open. A fraction are left fully closed (real clamping state).
JAW_MAX_DEG     = float(os.environ.get("JAW_MAX_DEG", 20.0))
JAW_CLOSED_PROB = float(os.environ.get("JAW_CLOSED_PROB", 0.35))
JAW_MIN_OPEN    = 5.0                                   # min angle when "open"
# Tool coupling (user 2026-10-17): the applier compresses the spring, so a HELD clip has its
# jaws open and a released clip is clamped shut. TOOL_PROB = fraction of samples held by the
# applier (rendered by restyle_appearance from `clip.tool_held`); held -> jaw in
# [JAW_MIN_OPEN_TOOL, JAW_MAX_DEG]; free -> closed when JAW_FREE_CLOSED=1 (else the legacy
# JAW_CLOSED_PROB mix).
TOOL_PROB         = float(os.environ.get("TOOL_PROB", 0.40))
JAW_MIN_OPEN_TOOL = float(os.environ.get("JAW_MIN_OPEN_TOOL", 8.0))
JAW_FREE_CLOSED   = int(os.environ.get("JAW_FREE_CLOSED", 1))

w, h         = 800, 544
znear, zfar  = 0.01, 1000.0

WORLD_UP     = np.array([0.0, 0.0, 1.0])   # "up" = +Z
MODEL_UP     = np.array([1.0, 0.0, 0.0])   # model's own up axis
# Phase B: env-overridable pose envelope. Defaults reproduce the original narrow
# near-top-down distribution; the 3-D generators widen these via env.
MIN_UP_DOT   = np.cos(np.deg2rad(float(os.environ.get("MIN_UP_DEG", 30))))  # reject if model-up tilts more than this from WORLD_UP
MAX_TILT_DEG = float(os.environ.get("MAX_TILT_DEG", 15.0))   # camera tilt away from straight-down
MAX_ROLL_DEG = float(os.environ.get("MAX_ROLL_DEG", 10.0))   # in-plane camera roll limit
# Extra random orientation perturbation applied to each sampled clip pose (deg).
# 0 = original behaviour. A POSE_STEEP_FRAC of samples get up to POSE_STEEP_MAX_DEG.
POSE_PERTURB_MAX_DEG = float(os.environ.get("POSE_PERTURB_MAX_DEG", 0.0))
POSE_STEEP_FRAC      = float(os.environ.get("POSE_STEEP_FRAC", 0.0))
POSE_STEEP_MAX_DEG   = float(os.environ.get("POSE_STEEP_MAX_DEG", 0.0))
# Cap the clip's MAXIMUM apparent size so it never appears too close/large (the
# sampler only guaranteed a MINIMUM size before). Defaults are effectively OFF
# (>1.0), so the flat path is unchanged; the 3-D paths set tighter values via env.
MAX_DIAG_FRAC = float(os.environ.get("MAX_DIAG_FRAC", 10.0))   # clip bbox diagonal / image diagonal
MAX_AREA_FRAC = float(os.environ.get("MAX_AREA_FRAC", 10.0))   # clip bbox area / image area
# Minimum apparent size (floor). Lowering these lets the clip sit FARTHER from the
# camera (smaller on screen). Defaults preserve the original flat-path behaviour.
MIN_DIAG_FRAC = float(os.environ.get("MIN_DIAG_FRAC", 0.10))
MIN_AREA_FRAC = float(os.environ.get("MIN_AREA_FRAC", 0.018))
# Clip SCREEN OCCUPANCY (fraction of the frame covered by the clip's mask). Measured on real
# MACS: median clip mask = 1.49% of the frame. Each sample draws a TARGET occupancy in
# [CLIP_FRAC_MIN, CLIP_FRAC_MAX] and on_save() dollies the camera along its view axis until
# the rendered mask matches it: OUT when too big, IN when too small (a dolly-in that pushes
# the clip against the frame border is undone). Only the camera moves, so the 6-DoF GT is
# simply re-measured at the final camera. (Until 2026-10 this was a fixed CAP of 0.3%, i.e.
# clips ~5x smaller than the real median -- too small to carry appearance or to learn from.)
CLIP_FRAC_MIN      = float(os.environ.get("CLIP_FRAC_MIN", 0.006))
CLIP_FRAC_MAX      = float(os.environ.get("CLIP_FRAC_MAX", 0.015))
# A thin clip can have a small mask yet SPAN a large bbox (arms reaching wide); limit the
# bbox area to BBOX_RATIO_MAX x the mask target so it still reads as a normal-size clip.
BBOX_RATIO_MAX     = float(os.environ.get("BBOX_RATIO_MAX", 5.0))
SIZE_CAP_TRIES     = int(os.environ.get("SIZE_CAP_TRIES", 6))
# Max fresh-pose resamples per saved sample: the size-cap dolly can nudge an off-centre
# clip onto a tool, so on_save re-checks tool overlap AFTER the dolly and rejects; the
# main loop then resamples. Guards against tool-crowded backplates never placing.
MAX_PLACE_ATTEMPTS = int(os.environ.get("MAX_PLACE_ATTEMPTS", 20))
# Focus gate: the clip must land on IN-FOCUS tissue (the surgical working plane), not on
# an out-of-focus region. Rejecting blurry placements fixes two things at once: (a) clips
# floating on defocused near-foreground/periphery (read as "too close to the camera"), and
# (b) clips landing on OUT-OF-FOCUS instruments that the colour/DINO tool detectors can't
# see (a blurred shaft has no sharp metal cues). Value = min sharpness PERCENTILE of the
# clip region within the frame (0.30 => clip must be sharper than the blurriest 30%).
FOCUS_MIN_PCT = float(os.environ.get("FOCUS_MIN_PCT", 0.30))
# Depth-PNG visualisation band (world units = distance from camera). FIXED/global (not
# per-sample) so the 16-bit depth PNG encodes ABSOLUTE distance -> its brightness now
# CHANGES with pose (a near clip is uniformly brighter than a far one; per-sample min-max
# normalisation erased that). Measured clip-depth span with the current placement is
# ~60..145; the band adds margin and values outside it clamp. The metric `.npy` is
# unaffected (always the true metric depth). Tune via env if the placement range changes.
DEPTH_VIZ_NEAR = float(os.environ.get("DEPTH_VIZ_NEAR", 45.0))
DEPTH_VIZ_FAR  = float(os.environ.get("DEPTH_VIZ_FAR", 175.0))
# Placement simulation: keep the clip's BLADES/JAWS pointing away from the camera, i.e.
# DOWN into the tissue (deeper), so the jaw end sits at greater depth than the spring/coil
# and the top-down microscope mostly sees the spring -- the physically correct way a clip
# is applied to a vessel. ON by default now (was off -> jaw direction was unconstrained,
# so the clip's depth orientation was arbitrary). BLADE_AWAY_MIN = min dot(blade_dir,
# camera-forward): 0 = just past perpendicular, 1 = straight down the view axis. 0.35
# gives a clear downward jaw tilt while keeping orientation variety.
BLADE_AWAY     = int(os.environ.get("BLADE_AWAY", 1))
BLADE_AWAY_MIN = float(os.environ.get("BLADE_AWAY_MIN", 0.35))
# MACS-like clip appearance (restyle_appearance.py) applied in-line after each save: blue
# anodized PBR + per-sample HDR environment + defocus/bloom/chroma matching. Same geometry
# and camera, so the GT written just before is untouched. APPEARANCE=0 keeps the flat look;
# KEEP_FLAT=1 also stores the flat composite as NNN_flat.png for comparison.
APPEARANCE        = int(os.environ.get("APPEARANCE", 1))
APPEARANCE_PRESET = os.environ.get("APPEARANCE_PRESET", "anodized2")   # A/B 2026-10: stats KID 2.62 -> 2.37
APPEARANCE_SEED   = int(os.environ.get("APPEARANCE_SEED", 0))
KEEP_FLAT         = int(os.environ.get("KEEP_FLAT", 0))
# Scene integration (occlude.py): partial tissue occlusion, clip applier gripping the spring,
# blood film. Pose/depth/full mask untouched; adds NNN_mask_visible.png + JSON `occlusion`.
OCCLUDE           = int(os.environ.get("OCCLUDE", 0))   # user (2026-10-17): no tissue cover / blood film
OCCLUDE_SEED      = int(os.environ.get("OCCLUDE_SEED", 0))

os.makedirs(OUT_DIR, exist_ok=True)
BACKGROUND_IMAGES = [f for f in os.listdir(BG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# ------------------- SCENE -------------------
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    try:                                              # head-less Linux (DGX): virtual X display
        pv.start_xvfb()
    except Exception as _e:
        print(f"[WARN] start_xvfb failed ({_e}); install xvfb or run under xvfb-run", flush=True)
plotter = pv.Plotter(off_screen=OFF_SCREEN, window_size=[w, h])
mesh = pv.read(MESH_PATH)

# Blade direction in the model frame = +e0 (long axis toward the HIGH-p0 end, which
# is the blades/jaws per the validated articulate convention; the low-p0 end is the
# spring/coil). Used to keep the blades pointing AWAY from the camera (into the
# tissue) so the render simulates the clip being placed onto a vessel.
_bc, _be = pca_frame(mesh.points.astype(np.float64))
BLADE_DIR_MODEL = _be[:, 0]                             # unit vector toward the jaws

plotter.camera.clipping_range = (znear, zfar)
plotter.add_light(pv.Light(light_type="headlight", intensity=40.0, color="white"))


def bbox_corners_world(mesh):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return np.array([
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax],
    ], dtype=float)


CORNERS_W = bbox_corners_world(mesh)

# ------------------- DATA (reference samples used to bound the sampling space) -------------------
SAMPLES = [
    dict(tvec=[-9.31,  2.95, -72.77],
         rotMat=[[-0.21, 0.95, 0.22], [-0.23, -0.27, 0.93], [0.95, 0.15, 0.28]]),
    dict(tvec=[3.64, -2.24, -44.53],
         rotMat=[[0.05, -0.99, -0.09], [-0.44, 0.06, -0.89], [0.89, 0.08, -0.44]]),
    dict(tvec=[-7.48,  3.80, -68.72],
         rotMat=[[-0.24, 0.93, 0.27], [-0.22, -0.32, 0.92], [0.95, 0.16, 0.28]]),
    dict(tvec=[-7.21,  3.31, -67.39],
         rotMat=[[0.23, 0.41, 0.88], [0.08, -0.91, 0.40], [0.97, -0.02, -0.25]]),
    dict(tvec=[2.78, -0.60, -34.20],
         rotMat=[[-0.62, -0.74, 0.28], [-0.13, -0.25, -0.96], [0.78, -0.63, 0.05]]),
    dict(tvec=[10.76, -22.03, -132.64],
         rotMat=[[-0.22, 0.80, 0.56], [0.19, -0.52, 0.83], [0.95, 0.29, -0.04]]),
    dict(tvec=[0.66, -1.16, -34.64],
         rotMat=[[-0.24, 0.15, 0.96], [-0.12, -0.99, 0.12], [0.96, -0.08, 0.25]]),
    dict(tvec=[-7.81,  7.71, -101.54],
         rotMat=[[-0.11, 0.27, -0.96], [-0.07, 0.96, 0.28], [0.99, 0.10, -0.08]]),
    dict(tvec=[-9.60, -4.50, -49.64],
         rotMat=[[0.66, 0.74, 0.14], [-0.21, 0.00, 0.98], [0.73, -0.67, 0.16]]),
    dict(tvec=[-0.33,  4.17, -39.69],
         rotMat=[[-0.38, -0.05, 0.92], [-0.21, -0.97, -0.14], [0.90, -0.25, 0.36]]),
    dict(tvec=[-1.60,  6.60, -48.27],
         rotMat=[[-0.27, 0.06, -0.96], [-0.30, 0.94, 0.14], [0.92, 0.32, -0.24]]),
    dict(tvec=[-13.01,  4.85, -83.36],
         rotMat=[[-0.12, 0.45, 0.88], [-0.13, -0.89, 0.44], [0.98, -0.06, 0.16]]),
    dict(tvec=[-4.02,  2.51, -79.57],
         rotMat=[[-0.11, -0.99, -0.03], [0.22, 0.01, -0.98], [0.97, -0.12, 0.21]]),
    dict(tvec=[-5.16,  1.46, -27.06],
         rotMat=[[-0.03, -1.00, -0.07], [-0.17, 0.07, -0.98], [0.99, -0.01, -0.17]]),
]

# ------------------- CAMERA / GEOMETRY HELPERS -------------------
def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def intrinsics_from_vtk(camera, w, h):
    """Pixel intrinsics (fx, fy, cx, cy) from a vtkCamera using the vertical FOV.

    VTK's ViewAngle is the VERTICAL FOV and it renders SQUARE pixels (the aspect
    ratio widens the horizontal FOV, it does NOT change the focal length), so
    fx == fy. (A previous version used fx = fy*(w/h), which disagreed with the
    actual rendered pixels by the aspect factor and distorted any reconstruction
    that consumed the exported intrinsics with the real images.)"""
    fovy = np.deg2rad(camera.GetViewAngle())
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return fx, fy, cx, cy


def cam_world_from_vtk(cam_tuple):
    """Return world->camera (R, t) for a (pos, focal, up) tuple in the OpenCV camera
    convention: x right, y DOWN, z forward (right-handed, det +1), so that
    u = fx*X/Z + cx, v = fy*Y/Z + cy lands exactly on the rendered pixel.

    History (2026-10 fix): this used to build x right, y UP, z forward -- a LEFT-handed
    frame (det -1). Its "rotation" was a reflection, so every exported pose_se3 quaternion
    was invalid and the GT pose, projected with the stored intrinsics, landed mirrored
    about cy (measured ~80 px off the clip). The symmetric in-frame checks hid it, while
    nudge_bbox_into_view pushed the clip the wrong way vertically and the 3-D path sampled
    the tissue disparity at the mirrored pixel. Existing JSONs are repaired from pose_world
    (always correct) by fix_pose_se3.py."""
    pos, focal, up = [np.asarray(v, float) for v in cam_tuple]
    fwd = _norm(focal - pos)
    right = _norm(np.cross(fwd, up))
    down = -np.cross(right, fwd)                      # image y grows downward
    R_w2c = np.stack([right, down, fwd], axis=0)      # rows = camera axes expressed in world
    t_w2c = -R_w2c @ pos
    return R_w2c, t_w2c


def cam_T_obj(R_obj_w, t_obj_w, cam_tuple):
    """Object->camera pose from the object->world pose and the camera tuple."""
    R_w2c, t_w2c = cam_world_from_vtk(cam_tuple)
    R_co = R_w2c @ R_obj_w
    t_co = R_w2c @ t_obj_w + t_w2c
    return R_co, t_co


def actor_mesh(mesh, R, t):
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    m.transform(T, inplace=True)
    return m


def camera_forward_from_tuple(cam):
    pos, foc, _up = [np.asarray(x, float) for x in cam]
    return _norm(foc - pos)


# ------------------- SAMPLING BOUNDS -------------------
tvecs = np.array([s["tvec"] for s in SAMPLES], dtype=float)
t_min, t_max = tvecs.min(axis=0), tvecs.max(axis=0)

rotvecs = np.array([scipyR.from_matrix(np.asarray(s["rotMat"])).as_rotvec() for s in SAMPLES], dtype=float)
r_min, r_max = rotvecs.min(axis=0), rotvecs.max(axis=0)

# focal points / distances are sampled around the data centroid
f_center = tvecs.mean(axis=0)
dists = np.array([np.linalg.norm(s["tvec"]) for s in SAMPLES], dtype=float)
dist_mean, dist_std = dists.mean(), dists.std()
d_low, d_high = max(dists.min() - 2.0, 1e-3), dists.max() + 2.0

PAD_T = np.array([0.5, 0.5, 1.0])
PAD_R = np.deg2rad([5, 5, 5])
PAD_F = np.array([5.0, 5.0, 5.0])

t_low, t_high = t_min - PAD_T, t_max + PAD_T
r_low, r_high = r_min - PAD_R, r_max + PAD_R
f_low, f_high = f_center - PAD_F, f_center + PAD_F


def sample_uniform(low, high):
    return low + np.random.rand(*low.shape) * (high - low)


# ------------------- TOP-VIEW CAMERA SAMPLER -------------------
def sample_camera_top(up_axis=WORLD_UP, max_tilt_deg=MAX_TILT_DEG, max_roll_deg=MAX_ROLL_DEG):
    """Return a (pos, focal, up) tuple looking almost straight down."""
    focal = sample_uniform(f_low, f_high)
    d = np.clip(np.random.normal(dist_mean, max(1e-3, dist_std)), d_low, d_high)

    up = _norm(up_axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _norm(np.cross(tmp, up))
    y = _norm(np.cross(up, x))

    az = np.random.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(np.random.uniform(0.0, max_tilt_deg))
    ground_dir = np.cos(az) * x + np.sin(az) * y
    forward = _norm(-up * np.cos(tilt) + ground_dir * np.sin(tilt))

    pos = focal - forward * d

    roll = np.deg2rad(np.random.uniform(-max_roll_deg, max_roll_deg))
    right = _norm(np.cross(forward, up))
    img_up = _norm(np.cross(right, forward))
    view_up = _norm(scipyR.from_rotvec(roll * forward).as_matrix() @ img_up)

    return (tuple(pos), tuple(focal), tuple(view_up))


# ------------------- VISIBILITY GUARANTEE -------------------
def _project_bbox_uv(R, t, cam, fx, fy, cx, cy):
    R_cw, t_cw = cam_world_from_vtk(cam)
    Cw = (R @ CORNERS_W.T).T + t[None, :]
    Cc = (R_cw @ Cw.T).T + t_cw[None, :]
    z = Cc[:, 2]
    u = fx * (Cc[:, 0] / z) + cx
    v = fy * (Cc[:, 1] / z) + cy
    return float(u.min()), float(u.max()), float(v.min()), float(v.max()), float(z.min()), float(np.median(z))


def all_corners_in_view(R, t, cam, fx, fy, cx, cy, w, h, znear=0.01, margin=4):
    R_cw, t_cw = cam_world_from_vtk(cam)
    Cw = (R @ CORNERS_W.T).T + t[None, :]
    Cc = (R_cw @ Cw.T).T + t_cw[None, :]
    z = Cc[:, 2]
    if np.any(z <= znear + 1e-9):
        return False
    u = fx * (Cc[:, 0] / z) + cx
    v = fy * (Cc[:, 1] / z) + cy
    return (np.all(u >= margin) & np.all(u < (w - margin)) &
            np.all(v >= margin) & np.all(v < (h - margin)))


def nudge_bbox_into_view(R, t, cam, fx, fy, cx, cy, w, h,
                         max_iters=6, step_px=80, z_pad=0.02, znear=0.01):
    R_cw, t_cw = cam_world_from_vtk(cam)
    R_wc = R_cw.T
    t_w = t.copy()
    for _ in range(max_iters):
        Cw = (R @ CORNERS_W.T).T + t_w[None, :]
        Cc = (R_cw @ Cw.T).T + t_cw[None, :]
        z = Cc[:, 2]
        if np.min(z) <= znear + 1e-6:
            t_w = t_w + R_wc @ np.array([0.0, 0.0, (znear + z_pad) - np.min(z)])
            continue
        u = fx * (Cc[:, 0] / z) + cx
        v = fy * (Cc[:, 1] / z) + cy
        leak_left   = max(0.0, np.max(0.0 - u))
        leak_right  = max(0.0, np.max(u - (w - 1)))
        leak_top    = max(0.0, np.max(0.0 - v))
        leak_bottom = max(0.0, np.max(v - (h - 1)))
        if leak_left == leak_right == leak_top == leak_bottom == 0.0:
            return t_w, True
        dx_px = leak_left - leak_right
        dy_px = leak_top - leak_bottom
        mag = (dx_px ** 2 + dy_px ** 2) ** 0.5
        if mag > step_px:
            s = step_px / (mag + 1e-9)
            dx_px *= s
            dy_px *= s
        z_med = float(np.median(z))
        delta_cam = np.array([(dx_px / fx) * z_med, (dy_px / fy) * z_med, 0.0])
        t_w = t_w + R_wc @ delta_cam
    return t_w, all_corners_in_view(R, t_w, cam, fx, fy, cx, cy, w, h, znear)


def sample_visible_pose_guaranteed(cam, fx, fy, cx, cy,
                                   max_pose_tries=200, nudge_step_px=90, z_pad=0.02,
                                   znear=0.01, margin=6, min_diag_frac=MIN_DIAG_FRAC,
                                   min_area_frac=MIN_AREA_FRAC, zoom_in_step=0.87,
                                   zoom_out_step=1.10, size_refine_iters=8,
                                   max_diag_frac=MAX_DIAG_FRAC, max_area_frac=MAX_AREA_FRAC):
    """Sample a pose guaranteed inside the frustum, not too small AND not too large."""
    img_diag = (w ** 2 + h ** 2) ** 0.5
    min_diag_px = min_diag_frac * img_diag
    min_area_px = min_area_frac * (w * h)
    max_diag_px = max_diag_frac * img_diag
    max_area_px = max_area_frac * (w * h)
    forward_world = camera_forward_from_tuple(cam)

    tries = 0
    R = t = None
    while tries < max_pose_tries:
        tries += 1
        t = sample_uniform(t_low, t_high)
        rv = sample_uniform(r_low, r_high)
        R = scipyR.from_rotvec(rv).as_matrix()

        # Phase B: perturb the (near-top-down) reference orientation by a random
        # rotation so the dataset spans steeper tilts / full roll while staying
        # anchored to a physically plausible clip-on-tissue pose.
        pmax = POSE_STEEP_MAX_DEG if (POSE_STEEP_FRAC > 0 and np.random.rand() < POSE_STEEP_FRAC) \
            else POSE_PERTURB_MAX_DEG
        if pmax > 0.0:
            ang = np.deg2rad(np.random.uniform(0.0, pmax))
            axis = np.random.randn(3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            R = scipyR.from_rotvec(ang * axis).as_matrix() @ R

        n_world = _norm(R @ MODEL_UP)
        if BLADE_AWAY:
            # simulate placement: blades must point away from the camera (into tissue)
            blade_w = _norm(R @ BLADE_DIR_MODEL)
            if float(np.dot(blade_w, forward_world)) < BLADE_AWAY_MIN:
                continue
        else:
            if float(np.dot(n_world, WORLD_UP)) < MIN_UP_DOT:
                continue
            if float(np.dot(n_world, -forward_world)) <= 0.0:
                continue

        R_cw, t_cw = cam_world_from_vtk(cam)
        R_wc = R_cw.T
        *_, zmin, _ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
        if zmin <= znear + z_pad:
            t = t + R_wc @ np.array([0.0, 0.0, (znear + z_pad) - zmin])

        if not all_corners_in_view(R, t, cam, fx, fy, cx, cy, w, h, znear, margin=margin):
            t_fix, ok = nudge_bbox_into_view(R, t, cam, fx, fy, cx, cy, w, h,
                                             max_iters=6, step_px=nudge_step_px, z_pad=z_pad, znear=znear)
            if ok:
                t = t_fix
            else:
                pos, focal, up = [np.asarray(v, float) for v in cam]
                dir_vec = focal - pos
                dist = np.linalg.norm(dir_vec) + 1e-12
                fwd_cam = dir_vec / dist
                cam_try = (tuple(focal - fwd_cam * (dist * zoom_out_step)), tuple(focal), tuple(up))
                if all_corners_in_view(R, t, cam_try, fx, fy, cx, cy, w, h, znear, margin=margin):
                    cam = cam_try
                else:
                    continue

        u0, u1, v0, v1, _, _ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
        diag = float(np.hypot(u1 - u0, v1 - v0))
        area = float(max(0.0, u1 - u0) * max(0.0, v1 - v0))

        if (diag < min_diag_px) or (area < min_area_px):
            for _ in range(size_refine_iters):
                pos, focal, up = [np.asarray(v, float) for v in cam]
                dir_vec = focal - pos
                dist = np.linalg.norm(dir_vec) + 1e-12
                fwd_cam = dir_vec / dist
                cam_try = (tuple(focal - fwd_cam * (dist * zoom_in_step)), tuple(focal), tuple(up))
                u0, u1, v0, v1, zmin, _ = _project_bbox_uv(R, t, cam_try, fx, fy, cx, cy)
                if zmin <= znear + z_pad:
                    break
                if not all_corners_in_view(R, t, cam_try, fx, fy, cx, cy, w, h, znear, margin=margin):
                    t_fix, ok = nudge_bbox_into_view(R, t, cam_try, fx, fy, cx, cy, w, h,
                                                     max_iters=6, step_px=nudge_step_px, z_pad=z_pad, znear=znear)
                    if ok:
                        t = t_fix
                        cam = cam_try
                    else:
                        break
                else:
                    cam = cam_try
                u0, u1, v0, v1, _, _ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
                diag = float(np.hypot(u1 - u0, v1 - v0))
                area = float(max(0.0, u1 - u0) * max(0.0, v1 - v0))
                if (diag >= min_diag_px) and (area >= min_area_px):
                    break
            if (diag < min_diag_px) or (area < min_area_px):
                continue

        # cap the MAXIMUM apparent size: zoom the camera OUT if the clip is too
        # close/large. Zooming out only shrinks the clip and adds frame margin, so
        # it never leaves the view -- no in-frame re-check needed.
        if (diag > max_diag_px) or (area > max_area_px):
            for _ in range(size_refine_iters):
                pos, focal, up = [np.asarray(v, float) for v in cam]
                dir_vec = focal - pos
                dist = np.linalg.norm(dir_vec) + 1e-12
                fwd_cam = dir_vec / dist
                cam = (tuple(focal - fwd_cam * (dist * zoom_out_step)), tuple(focal), tuple(up))
                u0, u1, v0, v1, _, _ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
                diag = float(np.hypot(u1 - u0, v1 - v0))
                area = float(max(0.0, u1 - u0) * max(0.0, v1 - v0))
                if (diag <= max_diag_px) and (area <= max_area_px):
                    break

        return R, t, cam

    return R, t, cam


# ------------------- STATE -------------------
@dataclass
class State:
    actor: Optional[object] = None
    idx: int = 0
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    cam: Optional[tuple] = None
    jaw: float = 0.0                  # jaw-opening angle (deg) of this sample
    tool: bool = False                # held by the applier (jaws open) or released (clamped)
    verts: Optional[np.ndarray] = None  # articulated clip vertices (model frame) of this sample
    backplate: str = ""               # source backplate file of this sample (for splits by scene)
    avoid: Optional[object] = None    # tool/instrument avoid mask for the current backplate


state = State()
state.idx = START_IDX


# ------------------- RELIT MATERIAL + SMART PLACEMENT -------------------
# Muted titanium material (was chrome) so the composite already looks like MACS;
# this bakes the old relight pass into the renderer -> no separate relight step.
CLIP_MAT = dict(color=(0.50, 0.48, 0.46), metallic=1.0, roughness=0.50,
                specular=0.30, diffuse=0.70, ambient=0.22)
LIGHT_INTENSITY = 8.0            # warm headlight (was 40 white)
SHADOW_STRENGTH = 0.45          # contact shadow on the tissue under the clip
SHADOW_SHIFT = (4, 7)
SHADOW_BLUR = 15
# Avoid placing the clip where the backplate already shows instruments/cautery/gauze.
# Now measured on the ACTUAL clip silhouette (projected vertices), not the coarse bbox.
AVOID_OVERLAP_MAX = float(os.environ.get("AVOID_OVERLAP_MAX", 0.0))   # user 2026-10-17: clip NEVER on another tool
AVOID_MARGIN_PX   = int(os.environ.get("AVOID_MARGIN_PX", 25))        # safety margin dilated around every detected tool
# Blade-direction probe (user 2026-10-17: the jaws must not POINT at another instrument): from the
# projected jaw tip, walk BLADE_PROBE_FRAC x the clip's on-screen length along the blade direction
# (a strip +/- BLADE_PROBE_HALFW px wide); any hit on the avoid mask rejects the pose.
BLADE_PROBE       = int(os.environ.get("BLADE_PROBE", 1))
BLADE_PROBE_FRAC  = float(os.environ.get("BLADE_PROBE_FRAC", 1.2))
BLADE_PROBE_HALFW = float(os.environ.get("BLADE_PROBE_HALFW", 6.0))
AVOID_TRIES = 45


def augment_backplate(bgr):
    """Randomly augment a tissue backplate so the same ~20 source images never
    repeat verbatim across 5000 samples. Geometric (hflip + rotate/zoom/shift,
    border-reflected so the frame stays filled) + photometric (hue/sat/value/
    contrast jitter). Only the background appearance changes; the clip geometry
    (pose/depth/mask) is unaffected, so the GT stays valid."""
    H, W = bgr.shape[:2]
    out = bgr
    if np.random.rand() < 0.5:
        out = out[:, ::-1].copy()                       # horizontal flip
    angle = np.random.uniform(-15.0, 15.0)
    scale = np.random.uniform(1.05, 1.30)               # >1 zooms in (no empty corners)
    cxr = W / 2.0 + np.random.uniform(-0.12, 0.12) * W
    cyr = H / 2.0 + np.random.uniform(-0.12, 0.12) * H
    M = cv2.getRotationMatrix2D((cxr, cyr), angle, scale)
    out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-8.0, 8.0)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.80, 1.20), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.80, 1.15), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    c = np.random.uniform(0.90, 1.15)                   # contrast
    out = np.clip((out.astype(np.float32) - 128.0) * c + 128.0, 0, 255).astype(np.uint8)
    return out


# ------------------- GroundingDINO tool avoidance -------------------
# Open-vocabulary detector to find instruments/forceps/etc the colour heuristic
# (instrument_avoid_mask) misses (yellow forceps, dark shafts, sutures, gauze). Their
# boxes are unioned into the avoid mask so the clip placement gate refuses to overlap
# any real tool. Runs once per sample on the ACTUAL augmented backplate (detecting on a
# warped source cache instead left tools unmasked after the border-reflect augmentation).
# Toggle with USE_DINO_AVOID=0.
USE_DINO_AVOID   = int(os.environ.get("USE_DINO_AVOID", 1))
DINO_MODEL       = os.environ.get("DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
DINO_TOOL_PROMPT = os.environ.get("DINO_TOOL_PROMPT",
    "surgical forceps. bipolar forceps. surgical instrument. instrument shaft. clip applier. suction tube. "
    "suction cannula. cautery. cottonoid patty. gauze. surgical needle. scissors. retractor. metal instrument. metal rod.")
DINO_BOX_THRESH  = float(os.environ.get("DINO_BOX_THRESH", 0.20))
DINO_TEXT_THRESH = float(os.environ.get("DINO_TEXT_THRESH", 0.20))
DINO_TOPK        = int(os.environ.get("DINO_TOPK", 10))
DINO_DILATE      = int(os.environ.get("DINO_DILATE", 25))    # margin (px) around detected tools
DINO_MAX_AREA    = float(os.environ.get("DINO_MAX_AREA", 0.35))  # drop over-inclusive boxes (real tools rarely >35% of frame)
_GDINO = {"model": None}


def _load_gdino():
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if _GDINO["model"] is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _GDINO["processor"] = AutoProcessor.from_pretrained(DINO_MODEL)
        _GDINO["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(dev)
        _GDINO["device"] = dev
        print(f"[DINO] loaded {DINO_MODEL} on {dev}", flush=True)
    return _GDINO


def gdino_tool_mask(bgr):
    """Filled+dilated binary mask (255) over instruments/forceps/gauze/etc detected by
    GroundingDINO. Empty mask if the model/deps are unavailable (fails safe)."""
    Hh, Ww = bgr.shape[:2]
    m = np.zeros((Hh, Ww), np.uint8)
    try:
        import torch
        from PIL import Image
        g = _load_gdino()
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = g["processor"](images=pil, text=DINO_TOOL_PROMPT, return_tensors="pt").to(g["device"])
        with torch.no_grad():
            outputs = g["model"](**inputs)
        common = dict(input_ids=inputs.input_ids, text_threshold=DINO_TEXT_THRESH,
                      target_sizes=[pil.size[::-1]])
        post = g["processor"].post_process_grounded_object_detection
        try:
            res = post(outputs, threshold=DINO_BOX_THRESH, **common)[0]
        except TypeError:
            res = post(outputs, box_threshold=DINO_BOX_THRESH, **common)[0]
        area = Hh * Ww
        boxes = sorted(zip(res["scores"].tolist(), res["boxes"].tolist()),
                       key=lambda t: t[0], reverse=True)[:DINO_TOPK]
        n = 0
        for score, (x0, y0, x1, y1) in boxes:
            if (x1 - x0) * (y1 - y0) > DINO_MAX_AREA * area:
                continue
            cv2.rectangle(m, (int(max(0, x0)), int(max(0, y0))),
                          (int(min(Ww, x1)), int(min(Hh, y1))), 255, -1)
            n += 1
    except Exception as e:
        if not _GDINO.get("warned"):
            print(f"[DINO] unavailable, tool-avoid disabled: {e}", flush=True)
            _GDINO["warned"] = True
        return m
    if DINO_DILATE > 0 and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DINO_DILATE, DINO_DILATE)))
    return m


def clip_silhouette_avoid_fraction(R, t, cam, fx, fy, cx, cy, avoid, verts):
    """Fraction of the clip's projected silhouette (its vertices) that lands on the
    instrument/cautery avoid mask. Far tighter than the bbox proxy, so the clip can
    be packed into clear tissue right next to a tool without overlapping it."""
    R_cw, t_cw = cam_world_from_vtk(cam)
    Pw = (R @ verts.T).T + t[None, :]
    Pc = (R_cw @ Pw.T).T + t_cw[None, :]
    z = Pc[:, 2]
    valid = z > 1e-6
    u = fx * (Pc[:, 0] / np.where(valid, z, 1.0)) + cx
    v = fy * (Pc[:, 1] / np.where(valid, z, 1.0)) + cy
    inb = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inb.any():
        return 1.0
    uu = u[inb].astype(np.int32)
    vv = v[inb].astype(np.int32)
    return float((avoid[vv, uu] > 127).mean())


def warm_light_color(bgr):
    """Light tint = mean colour of the brighter tissue pixels (warm)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sel = g >= np.percentile(g, 70)
    mean_bgr = bgr[sel].mean(axis=0) / 255.0 if sel.any() else np.array([0.7, 0.7, 0.7])
    rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
    return tuple(0.5 + 0.5 * c for c in rgb)


def instrument_avoid_mask(bgr):
    """Instrument regions to avoid: BRIGHT achromatic metal (forceps/suction/gauze),
    SPECULAR glints, DARK achromatic shafts/sutures (incl. out-of-focus ones the bright
    test and DINO both miss), and YELLOW forceps. Only LARGE or ELONGATED blobs are kept
    so ordinary dark TISSUE shadows/speckle are NOT masked (that would over-restrict)."""
    f = bgr.astype(np.int16)
    chroma = f.max(axis=2) - f.min(axis=2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright   = (chroma < 45) & (V > 115)                        # bright grey metal / gauze
    specular = (V > 220) & (S < 60)                             # blown glints
    dark     = (chroma < 28) & (V < 60)                         # dark shaft / suture (incl. blurry)
    # defocused shafts take a reddish tint from the tissue (chroma > 28) but stay far less
    # saturated than any tissue shadow: dark + low saturation = instrument
    dark_ts  = (V < 90) & (S < 110)
    greymid  = (chroma < 35) & (V >= 60) & (V <= 115)           # mid-grey shafts
    yellow   = (Hh >= 15) & (Hh <= 50) & (S > 90) & (V > 110)   # yellow forceps
    blue     = (Hh >= 95) & (Hh <= 130) & (S > 80) & (V > 60)   # blue instruments / drapes / silicone loops
    m = ((bright | specular | dark | dark_ts | greymid | yellow | blue).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); A = m.shape[0] * m.shape[1]
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        ew, eh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        elong = max(ew, eh) / (min(ew, eh) + 1e-6)
        if area > 0.004 * A or (area > 0.0015 * A and elong > 3.0):   # large OR long+thin = tool
            keep[lab == i] = 255
    k = 2 * AVOID_MARGIN_PX + 1
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def focus_percentile(bgr, region):
    """How in-focus `region` (bool mask) is, as its mean local-sharpness PERCENTILE within
    the whole frame (0..1). Sharpness = box-averaged |Laplacian|. 1 = sharpest area of the
    frame, low = out-of-focus foreground/periphery/defocused instrument."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), -1, (31, 31))
    if not region.any():
        return 1.0
    return float((F.ravel() < float(F[region].mean())).mean())


def bbox_avoid_fraction(R, t, cam, fx, fy, cx, cy, avoid):
    """Fraction of the clip's projected bbox that overlaps the avoid mask."""
    u0, u1, v0, v1, _, _ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
    x0, x1 = int(max(0, min(u0, u1))), int(min(w - 1, max(u0, u1)))
    y0, y1 = int(max(0, min(v0, v1))), int(min(h - 1, max(v0, v1)))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    region = avoid[y0:y1, x0:x1]
    return float((region > 127).mean()) if region.size else 1.0


def contact_shadow(bgr, clip_mask):
    """Darken the tissue just under/around the clip to ground it."""
    m = (clip_mask > 127).astype(np.float32)
    M = np.float32([[1, 0, SHADOW_SHIFT[0]], [0, 1, SHADOW_SHIFT[1]]])
    shifted = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
    k = SHADOW_BLUR * 2 + 1
    soft = np.clip(cv2.GaussianBlur(shifted, (k, k), 0) - m, 0, 1)
    factor = (1.0 - SHADOW_STRENGTH * soft)[..., None]
    return (bgr.astype(np.float32) * factor).astype(np.uint8)


def on_new_pose():
    cam = sample_camera_top()

    # --- augmented tissue backplate (so the ~20 sources never repeat verbatim) ---
    src = random.choice(BACKGROUND_IMAGES)
    state.backplate = src
    bg_src = cv2.imread(os.path.join(BG_DIR, src), cv2.IMREAD_COLOR)
    bg_aug = augment_backplate(cv2.resize(bg_src, (w, h)))
    tmp_bg = os.path.join(OUT_DIR, f"_bg_tmp_{os.getpid()}.png")   # per process: shards share OUT_DIR
    cv2.imwrite(tmp_bg, bg_aug)
    try:
        plotter.remove_background_image()
    except Exception:
        pass
    plotter.add_background_image(tmp_bg)

    # avoid mask = colour/specular heuristic UNION GroundingDINO-detected tools, so the
    # clip placement gate (clip_silhouette_avoid_fraction / AVOID_OVERLAP_MAX) refuses to
    # land the clip on any instrument, forceps, gauze, etc. DINO runs on THIS augmented
    # frame (not a warped source cache) so the avoid mask matches the tools actually
    # present -- incl. copies the border-reflect augmentation introduces at the edges.
    # cheap COLOUR pre-filter for the pose sampler (steers off bright tools). The
    # authoritative colour+DINO tool gate runs in on_save on the RENDER-space background
    # (pixel-aligned to the clip silhouette; the bg_aug array vs VTK's background render
    # can differ slightly, which previously let a few clips slip onto tools).
    avoid = instrument_avoid_mask(bg_aug)
    fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, w, h)

    # jaw articulation decided up-front so the avoid check uses the REAL silhouette.
    tool = bool(np.random.rand() < TOOL_PROB)
    if tool:                                                  # held by the applier -> jaws open
        theta = float(np.random.uniform(JAW_MIN_OPEN_TOOL, JAW_MAX_DEG))
    elif JAW_FREE_CLOSED:                                     # released -> clamped shut
        theta = 0.0
    else:                                                     # legacy mix
        theta = 0.0 if np.random.rand() < JAW_CLOSED_PROB \
            else float(np.random.uniform(JAW_MIN_OPEN, JAW_MAX_DEG))
    state.tool = tool
    clip_mesh = articulate(mesh, theta) if theta > 0.0 else mesh
    verts = clip_mesh.points.astype(np.float64)
    state.verts = verts

    # sample a pose whose clip silhouette does NOT overlap instruments/cautery/gauze
    R, t, cam = sample_visible_pose_guaranteed(cam, fx, fy, cx, cy)
    best = (R, t, cam, clip_silhouette_avoid_fraction(R, t, cam, fx, fy, cx, cy, avoid, verts))
    for _ in range(AVOID_TRIES):
        if best[3] <= AVOID_OVERLAP_MAX:
            break
        R2, t2, cam2 = sample_visible_pose_guaranteed(cam, fx, fy, cx, cy)
        frac = clip_silhouette_avoid_fraction(R2, t2, cam2, fx, fy, cx, cy, avoid, verts)
        if frac < best[3]:
            best = (R2, t2, cam2, frac)
    R, t, cam, _ = best

    if state.actor is not None:
        plotter.remove_actor(state.actor, render=False)

    # warm lighting sampled from this (augmented) backplate
    plotter.renderer.remove_all_lights()
    plotter.add_light(pv.Light(light_type="headlight",
                               intensity=LIGHT_INTENSITY, color=warm_light_color(bg_aug)))

    posed = actor_mesh(clip_mesh, R, t)
    state.actor = plotter.add_mesh(posed, pbr=True, smooth_shading=True,
                                   reset_camera=False, **CLIP_MAT)

    state.R, state.t, state.cam, state.jaw = R, t, cam, theta
    plotter.camera_position = cam
    plotter.render()


# ------------------- DEPTH PASS -------------------
def render_clip_depth():
    """
    Return per-pixel metric depth of the clip in world units (camera distance
    along the view axis), NaN outside the clip.

    The background frame is a 2-D backplate (no geometry), so the z-buffer only
    contains the clip; `get_image_depth` returns finite values there and NaN
    elsewhere.  PyVista reports depth as the camera-space z (negative in front of
    the camera), so we negate to obtain a positive distance.
    """
    plotter.render()
    zval = plotter.get_image_depth(fill_value=np.nan)   # negative in front, NaN bg
    finite = np.isfinite(zval)
    metric = np.full(zval.shape, np.nan, dtype=np.float32)
    metric[finite] = -zval[finite]
    return metric


def normals_from_depth(metric, fx, fy):
    """
    Cheap camera-frame normal map from the metric depth via finite differences.
    Encoded as 8-bit RGB with the usual (x,y,z)->(r,g,b) mapping in [0,1] and a
    neutral (0,0,1) background outside the clip.
    """
    z = np.nan_to_num(metric, nan=0.0).astype(np.float32)
    valid = np.isfinite(metric)
    # gradients of depth; scale by focal length to approximate metric slope
    dzdx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) * (fx / max(z[valid].mean(), 1e-6) if valid.any() else 1.0)
    dzdy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) * (fy / max(z[valid].mean(), 1e-6) if valid.any() else 1.0)
    nx, ny, nz = -dzdx, -dzdy, np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    rgb = np.stack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5], axis=-1)
    rgb[~valid] = np.array([0.5, 0.5, 1.0])             # flat background normal
    return (rgb * 255).astype(np.uint8)


def save_depth_artifacts(stem, metric, fx, fy):
    """Save metric depth (.npy), a 16-bit normalised viz, and a normal map."""
    npy_path    = os.path.join(OUT_DIR, f"{stem}_depth_clip.npy")
    png_path    = os.path.join(OUT_DIR, f"{stem}_depth_clip.png")
    normal_path = os.path.join(OUT_DIR, f"{stem}_normal.png")

    np.save(npy_path, metric)

    valid = np.isfinite(metric)
    viz = np.zeros(metric.shape, dtype=np.uint16)
    if valid.any():
        # FIXED/global normalisation over [DEPTH_VIZ_NEAR, DEPTH_VIZ_FAR] (NOT per-sample):
        # closer -> brighter, and crucially the SAME mapping for every sample, so absolute
        # distance (hence pose) shows in the brightness. Values outside the band clamp.
        rng = max(DEPTH_VIZ_FAR - DEPTH_VIZ_NEAR, 1e-9)
        norm = 1.0 - np.clip((metric - DEPTH_VIZ_NEAR) / rng, 0.0, 1.0)
        viz[valid] = (norm[valid] * 65535).astype(np.uint16)
    cv2.imwrite(png_path, viz)

    cv2.imwrite(normal_path, cv2.cvtColor(normals_from_depth(metric, fx, fy), cv2.COLOR_RGB2BGR))

    dmin = float(np.nanmin(metric)) if valid.any() else None
    dmax = float(np.nanmax(metric)) if valid.any() else None
    return {
        "depth_clip_npy": os.path.basename(npy_path),
        "depth_clip_png": os.path.basename(png_path),
        "normal_png": os.path.basename(normal_path),
        "depth_min": dmin,
        "depth_max": dmax,
    }


# ------------------- SAVE -------------------
def convert_to_binary_mask(image_path, output_path, threshold=10):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, mask)
    return mask


def _rotmat_to_quat_wxyz(R):
    """Trace-based matrix->quaternion (wxyz). Matches the original generator and
    tolerates the improper VTK camera frame that scipy's from_matrix rejects."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    return q.tolist()


def save_pose_and_camera(tvec_obj_w, R_obj_w, camera_position, plotter_camera,
                         out_path, w, h, znear, zfar, files, depth_info, jaw_angle_deg=0.0, tool_held=None,
                         backplate=None):
    R_co, t_co = cam_T_obj(R_obj_w, tvec_obj_w, camera_position)
    fx, fy, cx, cy = intrinsics_from_vtk(plotter_camera, w, h)

    data = {
        "image": {
            "file_rgb": files["rgb"], "file_bg": files["bg"], "file_mask": files["mask"],
            "file_depth_clip": depth_info["depth_clip_png"],
            "file_depth_clip_npy": depth_info["depth_clip_npy"],
            "file_normal": depth_info["normal_png"],
            "height": int(h), "width": int(w),
        },
        "camera": {
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "znear": float(znear), "zfar": float(zfar),
            "position": list(map(float, camera_position[0])),
            "focal_point": list(map(float, camera_position[1])),
            "view_up": list(map(float, camera_position[2])),
        },
        "clip": {
            "pose_se3": {
                "quaternion_wxyz": _rotmat_to_quat_wxyz(R_co),
                "translation_m": list(map(float, t_co)),
            },
            "pose_world": {
                "quaternion_wxyz": _rotmat_to_quat_wxyz(R_obj_w),
                "translation_m": list(map(float, tvec_obj_w)),
            },
            "jaw_angle_deg": float(jaw_angle_deg),   # 0 = clamping/closed, >0 = open
            "tool_held": tool_held,                  # True = applier on the spring (jaws open); None = legacy
            "mesh": os.path.basename(MESH_PATH),     # which clip mesh was rendered (Item.obj / clip_lazic_std.obj ...)
        },
        "depth": {
            "source": "vtk_zbuffer",
            "units": "world_units (same scale as translation_m)",
            "convention": "metric distance from camera along view axis; NaN outside clip",
            "viz_png": (f"16-bit, closer=brighter, background=0; FIXED global scale over "
                        f"[{DEPTH_VIZ_NEAR}, {DEPTH_VIZ_FAR}] world-units so brightness encodes "
                        f"ABSOLUTE distance (varies with pose); clamped. Use the .npy for exact metric."),
            "viz_near": DEPTH_VIZ_NEAR, "viz_far": DEPTH_VIZ_FAR,
            "clip_depth_min": depth_info["depth_min"],
            "clip_depth_max": depth_info["depth_max"],
        },
        "scene": {"backplate": backplate},
        "conventions": {
            "camera_frame": "OpenCV: x right, y down, z forward (right-handed); "
                            "pose_se3 maps clip model coords -> this frame",
            "projection": "u = fx*X/Z + cx, v = fy*Y/Z + cy; integer (u,v) = pixel centre, cx=(w-1)/2",
            "pixel_origin": "top_left",
            "handedness": "right", "cam_forward_axis": "+Z",
            "quaternion": "wxyz, unit, from a proper rotation (det +1)",
        },
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def on_save():
    """Render + save one sample. Returns True if it passed BOTH the size cap and the
    tool-avoidance gate (both evaluated on the FINAL, post-dolly clip silhouette). Returns
    False without saving if the clip lands on a tool after the size dolly, so the caller
    resamples a fresh pose."""
    fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, w, h)

    # --- depth pass (clip only, before we touch lights/background) ---
    metric = render_clip_depth()

    # --- SIZE CAP: clip must not occupy more of the screen than the MACS median.
    # metric is finite exactly on the clip, so its finite-fraction IS the clip mask
    # fraction. If it exceeds MAX_CLIP_FRAC, dolly the camera straight out (distance
    # x sqrt(frac/cap): projected area ~ 1/dist^2) and re-measure. Only the camera
    # moves -- the clip's world pose is untouched, so the saved 6-DoF GT recomputed
    # from the new camera stays exact; the backplate (a screen-fixed 2-D image) is
    # unchanged, so the clip simply shrinks on the same tissue. ---
    # Shrink until BOTH the clip mask AND its bbox extent are under their caps (a thin clip
    # can have a small mask but a wide bbox). Targets are 0.92x the caps because the SAVED
    # binary mask (separate antialiased pass) runs ~5% larger than this depth silhouette.
    def _clip_sizes(mt):
        sil = np.isfinite(mt); mfr = float(sil.mean())
        if sil.any():
            ys, xs = np.where(sil)
            bfr = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / sil.size
        else:
            bfr = 0.0
        return mfr, bfr
    def _touches_border(mt, margin=6):
        sil = np.isfinite(mt)
        if not sil.any():                                    # clip left the frame entirely: treat as cut
            return True
        return bool(sil[:margin].any() or sil[-margin:].any() or sil[:, :margin].any() or sil[:, -margin:].any())
    tgt = float(np.random.uniform(CLIP_FRAC_MIN, CLIP_FRAC_MAX)) * 0.92   # per-sample target occupancy
    bbox_tgt = tgt * BBOX_RATIO_MAX
    orig_mf, orig_bf = _clip_sizes(metric)
    mf, bf = orig_mf, orig_bf
    dollied = False
    for _ in range(SIZE_CAP_TRIES):
        over = max(mf / tgt, bf / bbox_tgt)                  # >1 => too big (both areas ~ 1/dist^2)
        if over > 1.0:
            f = min(1.8, over ** 0.5 * 1.03)                 # dolly OUT (+3% margin)
        elif mf / tgt < 0.80:
            f = max(0.60, (mf / tgt) ** 0.5)                 # dolly IN
        else:
            break
        prev_cam, prev_metric = state.cam, metric
        pos, focal, up = [np.asarray(v, float) for v in state.cam]
        pos = focal + (pos - focal) * f
        state.cam = (tuple(pos), tuple(focal), tuple(up))
        plotter.camera_position = state.cam
        metric = render_clip_depth()
        if (f < 1.0 and _touches_border(metric)) or not np.isfinite(metric).any():
            # dolly cut the clip at the border or lost it: undo the step and stop
            state.cam, metric = prev_cam, prev_metric
            plotter.camera_position = state.cam
            break
        dollied = True
        mf, bf = _clip_sizes(metric)
    fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, w, h)   # (unchanged by a dolly; refresh for safety)

    # --- render-space background (clip hidden), used BOTH for the tool gate below and
    # saved as {stem}_bg.png. Detecting tools on THIS (not the bg_aug array) keeps the
    # avoid mask pixel-aligned with the clip silhouette -- VTK's background render can
    # differ slightly from the raw array, which previously let a few clips onto tools. ---
    state.actor.SetVisibility(False)
    plotter.render()
    bg = plotter.screenshot(transparent_background=False, return_img=True)
    state.actor.SetVisibility(True)
    plotter.render()
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    # --- TOOL-AVOID GATE (authoritative, final camera): the clip's ACTUAL rendered
    # silhouette must not overlap the colour+DINO tool mask by more than AVOID_OVERLAP_MAX.
    # Checked AFTER the size dolly (which can shift an off-centre clip). Over -> reject
    # without saving; the main loop resamples a fresh pose. ---
    clip_sil = np.isfinite(metric)
    if clip_sil.sum() < 50:                                   # no usable clip in view: never save this
        print("[SIZE] clip not visible after placement; sample rejected", flush=True)
        return False
    avoid = instrument_avoid_mask(bg_bgr)
    if USE_DINO_AVOID:
        avoid = cv2.bitwise_or(avoid, gdino_tool_mask(bg_bgr))
    if clip_sil.any():
        if avoid.shape != clip_sil.shape:
            avoid = cv2.resize(avoid, (clip_sil.shape[1], clip_sil.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlap = float((clip_sil & (avoid > 127)).sum()) / float(clip_sil.sum())
        if overlap > AVOID_OVERLAP_MAX:
            return False
        # FOCUS gate: reject clips on out-of-focus tissue (defocused foreground/periphery,
        # or a blurred instrument the tool detectors can't see) -> resample onto the sharp
        # in-focus working plane, where a real clip would actually be.
        if focus_percentile(bg_bgr, clip_sil) < FOCUS_MIN_PCT:
            return False
        # BLADE-DIRECTION probe: the jaws must not point at an instrument
        if BLADE_PROBE and state.verts is not None:
            R_w2c, t_w2c = cam_world_from_vtk(state.cam)
            Pw = (state.R @ state.verts.T).T + state.t[None, :]
            Pc = (R_w2c @ Pw.T).T + t_w2c[None, :]
            ok3 = Pc[:, 2] > 1e-6
            if ok3.any():
                u = fx * Pc[ok3, 0] / Pc[ok3, 2] + cx
                v = fy * Pc[ok3, 1] / Pc[ok3, 2] + cy
                proj = (state.verts[ok3] - state.verts[ok3].mean(0)) @ BLADE_DIR_MODEL
                tip = np.array([u[np.argmax(proj)], v[np.argmax(proj)]])
                spr = np.array([u[np.argmin(proj)], v[np.argmin(proj)]])
                dvec = tip - spr
                L = float(np.linalg.norm(dvec))
                if L > 1e-3:
                    dvec /= L
                    nrm = np.array([-dvec[1], dvec[0]])
                    for sdist in np.linspace(4.0, BLADE_PROBE_FRAC * L, 24):
                        for lat in (-BLADE_PROBE_HALFW, 0.0, BLADE_PROBE_HALFW):
                            px = tip + dvec * sdist + nrm * lat
                            xi, yi = int(round(px[0])), int(round(px[1]))
                            if 0 <= xi < w and 0 <= yi < h and avoid[yi, xi] > 127:
                                return False                   # jaws aim at a tool: resample

    # --- passed both gates -> commit this sample ---
    state.idx += 1
    stem = f"{state.idx:0{STEM_WIDTH}d}"
    rgb_path  = os.path.join(OUT_DIR, f"{stem}.png")
    bg_path   = os.path.join(OUT_DIR, f"{stem}_bg.png")
    mask_path = os.path.join(OUT_DIR, f"{stem}_mask.png")
    json_path = os.path.join(OUT_DIR, f"{stem}.json")
    if dollied:
        print(f"[SIZE] {stem}: mask {orig_mf*100:.2f}->{mf*100:.2f}% (target {tgt/0.92*100:.2f}%)  "
              f"bbox {orig_bf*100:.2f}->{bf*100:.2f}%", flush=True)

    depth_info = save_depth_artifacts(stem, metric, fx, fy)
    Image.fromarray(bg).convert("RGB").save(bg_path)

    # --- RGB composite (muted titanium clip over backplate) + contact shadow ---
    rgb = plotter.screenshot(transparent_background=False, return_img=True)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    clip_mask = (np.isfinite(metric).astype(np.uint8)) * 255   # silhouette from depth
    rgb_bgr = contact_shadow(rgb_bgr, clip_mask)
    cv2.imwrite(rgb_path, rgb_bgr)

    # --- binary mask (black background, no lights) ---
    bg_backup = plotter.renderer.GetBackground()
    saved_lights = list(plotter.renderer.lights)
    plotter.set_background("black")
    plotter.renderer.remove_all_lights()
    plotter.remove_background_image()
    plotter.render()
    plotter.screenshot(mask_path, transparent_background=True)
    convert_to_binary_mask(mask_path, mask_path)
    plotter.set_background(bg_backup)
    for L in saved_lights:
        plotter.add_light(L)
    plotter.render()

    save_pose_and_camera(
        state.t, state.R, state.cam, plotter.camera, json_path,
        w=w, h=h, znear=znear, zfar=zfar,
        files={"rgb": os.path.basename(rgb_path),
               "bg": os.path.basename(bg_path),
               "mask": os.path.basename(mask_path)},
        depth_info=depth_info, jaw_angle_deg=state.jaw, tool_held=state.tool, backplate=state.backplate)

    # --- MACS-like appearance pass (reads back the sample just written; overwrites NNN.png,
    # adds an `appearance` block to the JSON; pose/camera/mask/depth stay as saved) ---
    look = "flat"
    if APPEARANCE:
        try:
            if KEEP_FLAT:
                shutil.copy2(rgb_path, os.path.join(OUT_DIR, f"{stem}_flat.png"))
            from restyle_appearance import restyle as _restyle
            _restyle(mesh, OUT_DIR, stem, APPEARANCE_PRESET, APPEARANCE_SEED + state.idx, out_dir=OUT_DIR)
            look = APPEARANCE_PRESET
        except Exception as e:                      # never lose a sample over the look
            print(f"[APPEARANCE] {stem}: failed ({e}); flat render kept", flush=True)
        if state.tool:
            # a held clip (jaws open) without its tool would contradict the GT -> drop & resample
            try:
                with open(json_path) as f:
                    _has_tool = bool(json.load(f).get("appearance", {}).get("applier"))
            except Exception:
                _has_tool = False
            if not _has_tool:
                for fp in glob.glob(os.path.join(OUT_DIR, f"{stem}_*")) + [rgb_path, json_path]:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                state.idx -= 1
                print(f"[TOOL] {stem}: no applier placement kept the clip visible; sample dropped", flush=True)
                return False
    if OCCLUDE:
        try:
            from occlude import process_sample as _occlude
            _occlude(OUT_DIR, stem, mesh.points.astype(np.float64), BLADE_DIR_MODEL,
                     np.random.default_rng(OCCLUDE_SEED * 100003 + state.idx))
            look += "+occ"
        except Exception as e:
            print(f"[OCCLUDE] {stem}: failed ({e})", flush=True)

    print(f"[INFO] saved {stem}: rgb({look})/bg/mask/depth/normal/json")
    return True


# ------------------- MAIN -------------------
def main():
    on_new_pose()
    # show() once initialises the render window (required by get_image_depth),
    # both on-screen and off-screen.
    plotter.show(auto_close=False, interactive_update=not OFF_SCREEN)
    guard = 0
    # keep sampling until N samples are SAVED; on_save returns False (no save, no idx
    # bump) when the clip lands on a tool after the size dolly, so we just resample.
    while state.idx < START_IDX + N and guard < N * MAX_PLACE_ATTEMPTS:
        guard += 1
        if not OFF_SCREEN:
            plotter.update()
        on_new_pose()
        on_save()
    if state.idx < START_IDX + N:
        print(f"[WARN] placed {state.idx - START_IDX}/{N} after {guard} tries -- tool-crowded "
              f"backplates; raise MAX_PLACE_ATTEMPTS or add backplates", flush=True)
    plotter.close()


if __name__ == "__main__":
    main()
