"""
restyle_appearance.py — re-render existing GT samples with a MACS-like clip appearance.

The 6-DoF pose, camera, jaw angle and backplate are read back from an existing sample
(NNN.json + NNN_bg.png), so the GT (pose JSON, mask, metric depth) stays valid by
construction: only the clip's LOOK changes. This is the "appearance in the renderer"
path: zero seconds of diffusion, exact geometry.

Per sample
  1. material  : blue anodized titanium PBR, jittered (base colour, roughness, coat)
  2. lighting  : image-based lighting from a synthetic HDR environment built from the
                 sample's own backplate -> tissue dome below (coloured reflections), dark
                 room above, bright coaxial microscope light toward the camera (HDR disc +
                 halo); plus a warm headlight and a weak off-axis fill
  3. render    : 2x supersampled RGBA clip layer (transparent background)
  4. applier   : (probabilistic) a 3-D clip applier gripping the SPRING end, rendered in
                 the SAME scene/lighting (real perspective, consistent speculars), defocused
                 because it sits closer to the camera; writes NNN_mask_visible.png
  5. optics    : local depth of field matched to the backplate at the placement (focus
                 percentile), lateral chromatic aberration, black level lifted to the
                 frame's own black, highlight bloom, 4:2:0 chroma, noise, JPEG generation,
                 occasional motion blur, contact shadow
Outputs: OUT/NNN.png + copies of json/mask/depth/normal/bg so downstream tools work
unchanged (JSON gains an `appearance` block). Montage OLD | NEW | REAL and an HSV
distance of clip pixels vs real clip pixels are produced for evaluation.

VTK facts measured for this build (VTK 9.6.2 / PyVista 0.48.4):
  * a float equirect env texture is linear radiance with 1.0 = white; values >1 give HDR
    light sources. (Pitfall: `color=(1,1,1)` with INTS is parsed by PyVista as 0..255,
    i.e. near-black F0 -- pass floats.)
  * the framebuffer output is gamma-encoded (~1/2.2); transparent screenshots are
    alpha-premultiplied in LINEAR light (edge pixel = enc(c_lin * a)).
  * equirect row 0 = EnvironmentUp (+Z here), column W/2 = EnvironmentRight (+X),
    azimuth runs clockwise seen from above (col 3W/4 = -Y).

Usage
  python restyle_appearance.py --src out_pose100 --out out_pose100_appear --limit 20
  python restyle_appearance.py --src out_pilot_v3 --out out_pilot_v3_r2 --reuse-seed
  python restyle_appearance.py --src out_pose100 --limit 8 --compare anodized2,navy
"""

import os
import sys
import json
import glob
import shutil
import argparse

import cv2
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as scipyR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from articulate_clip import articulate, pca_frame

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(HERE, "Item.obj")
REAL_CROPS = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2"))
REAL_MASKS = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2_masks"))

ENV_W, ENV_H = 256, 128
WHITE = 1.0              # linear radiance that a white metal reflects as white
SS = int(os.environ.get("RESTYLE_SS", 2))   # supersampling factor of the clip/applier layers (2 = paper quality; 1 = ~4x cheaper render, CPU-only hosts)

# ------------------------------------------------------------------ presets
# base = linear F0 reflectance of the metal (RGB); rough = (min,max) jitter range;
# coat = clear-coat strength (0 = off). Optional `P` = overrides of the P dict below.
PRESETS = {
    "anodized": dict(base=(0.25, 0.40, 0.80), rough=(0.22, 0.42), coat=0.0),
    # domain_gap.py breakdown on pilot v2: synthetic hue sat in the purple bins (157-180 in
    # OpenCV) while real anodized blue is cyan-blue (90-112), and the clip was too dark.
    "anodized2": dict(base=(0.18, 0.42, 0.82), rough=(0.22, 0.42), coat=0.0,
                      P=dict(tissue_gain=(0.5, 0.8), light_gain=(2.5, 5.0), halo_gain=(0.25, 0.45))),
    "anodized3": dict(base=(0.18, 0.42, 0.82), rough=(0.20, 0.40), coat=0.0,
                      P=dict(tissue_gain=(0.5, 0.8), light_gain=(3.5, 7.0), halo_gain=(0.35, 0.60),
                             shadow_strength=0.40, shadow_blur_frac=0.16, shadow_shift_frac=(0.04, 0.08))),
    # ablation ladder of the protocol (section 7, E2), all with --no-applier and --reuse-seed:
    #   A0 flat (generator's NNNNN_flat.png) -> A1 anodized + IBL -> A2 + optics (DoF, CA, black level)
    #   -> A3 + video footprint (bloom, chroma, noise, JPEG, motion blur) = anodized2 without applier
    #   -> A4 + applier = the full recipe (E0) -> A5 + SDXL repaint (inpaint_applier.py)
    "a1_ibl": dict(base=(0.18, 0.42, 0.82), rough=(0.22, 0.42), coat=0.0,
                   P=dict(tissue_gain=(0.5, 0.8), light_gain=(2.5, 5.0), halo_gain=(0.25, 0.45),
                          defocus_base=(0.3, 0.3), defocus_dof_gain=0.0, ca_strength=(0.0, 0.0), black_lift=0.0,
                          bloom_gain=0.0, chroma_sigma=0.3, noise_gain=0.0, jpeg_prob=0.0, motion_blur_prob=0.0)),
    "a2_optics": dict(base=(0.18, 0.42, 0.82), rough=(0.22, 0.42), coat=0.0,
                      P=dict(tissue_gain=(0.5, 0.8), light_gain=(2.5, 5.0), halo_gain=(0.25, 0.45),
                             bloom_gain=0.0, chroma_sigma=0.3, noise_gain=0.0, jpeg_prob=0.0, motion_blur_prob=0.0)),
    # ablation A0: the original flat look (grey PBR under a single headlight, no environment, no
    # optics / video footprint). Used with --no-applier.
    "flat": dict(base=(0.50, 0.48, 0.46), rough=(0.50, 0.50), coat=0.0,
                 P=dict(tissue_gain=(0.0, 0.0), light_gain=(0.0, 0.0), halo_gain=(0.0, 0.0), room_rgb=(0.04, 0.04, 0.04),
                        headlight=(6.0, 6.0), fill=(0.0, 0.0), defocus_base=(0.3, 0.3), defocus_dof_gain=0.0,
                        ca_strength=(0.0, 0.0), black_lift=0.0, bloom_gain=0.0, chroma_sigma=0.3, noise_gain=0.0,
                        jpeg_prob=0.0, motion_blur_prob=0.0)),
    "navy":  dict(base=(0.08, 0.15, 0.45), rough=(0.20, 0.35), coat=0.0),
    "blue":  dict(base=(0.14, 0.26, 0.60), rough=(0.20, 0.35), coat=0.0),
    "steel": dict(base=(0.26, 0.34, 0.56), rough=(0.30, 0.45), coat=0.0),
    "navy_coat": dict(base=(0.08, 0.15, 0.45), rough=(0.30, 0.45), coat=0.40),
}

# lighting / environment / post-processing parameters (jitter ranges where tuples)
P = dict(
    tissue_gain=(0.8, 1.2),          # radiance of the tissue dome relative to the backplate colour
    dome_scale=(0.8, 1.4),           # how far sideways directions reach on the backplate
    room_rgb=(0.010, 0.008, 0.008),  # dark OR above the field (linear, x WHITE)
    # The light is angularly LARGE on purpose: the mesh is round wire, so only a band of each
    # wire within +/- light_radius of the view axis lights up; a wide disc reproduces the
    # amount of bright pixels real (flat-faced) clips show.
    light_gain=(2.0, 4.0),
    light_radius_deg=(14.0, 24.0),
    halo_gain=(0.20, 0.40),
    halo_radius_deg=(30.0, 45.0),
    light_rgb=(1.00, 0.97, 0.92),    # slightly warm microscope light
    headlight=(0.30, 0.70),          # direct lights stay moderate (point light = blown stripe)
    fill=(0.10, 0.30),
    fill_angle_deg=(15.0, 35.0),
    # optics
    defocus_base=(0.45, 0.85),       # video softness even in focus (px)
    defocus_dof_gain=3.0,            # extra sigma = gain * (1 - focus_percentile)^2
    ca_strength=(0.0010, 0.0030),    # lateral chromatic aberration (radial scale split R/B)
    black_lift=1.0,                  # 1 = lift the clip's blacks fully to the frame's black level
    bloom_thresh=0.80, bloom_sigma=3.0, bloom_gain=0.35,   # threshold in display units
    chroma_sigma=1.6,
    noise_gain=1.0,
    shadow_strength=0.25, shadow_shift_frac=(0.03, 0.06), shadow_blur_frac=0.10,
    jpeg_prob=1.0, jpeg_quality=(50, 85),
    motion_blur_prob=0.35, motion_blur_len=(1.0, 3.0),
    # 3-D clip applier (gripping the spring; rendered in the same scene). Geometry follows a
    # Yasargil-type applier: two flat jaw blades pressing the WIDE sides of the spring arms,
    # a head wider than the shaft, a shaft that tapers toward the head and often has a
    # bayonet bend; it rises toward the camera and leaves the frame.
    applier_prob=0.40,
    # User choice: the plain v2 cylinder look (straight, constant radius, no bayonet, no bulky
    # head) with the v4 junction (prongs on the clip, collinear in-plane start, one gentle bend).
    # v7, designed from the user's real references (#2 #12 #13 #14): the instrument comes from
    # the surgeon's hand, i.e. it rises TOWARD the camera as it leaves the frame (huge and
    # very blurred at the border, thinner and sharper at the clip), the clip sits at its tip
    # with a dark head covering the coil, near-black metal with grey specular edges; either
    # one thick shaft or two diverging forceps arms.
    applier_radius=(2.1, 2.9),       # a bit thicker than V9 (clip half-width is 3.2); perspective does the rest
    applier_arm_radius=(1.7, 2.5),   # per-arm radius in the two-arm (forceps) variant
    applier_two_arms_prob=0.5,
    applier_jaw_len=(8.0, 10.0),     # head length over the clip (covers the coil / grip zone)
    applier_exit_jitter_deg=(-4.0, 4.0),     # user: the tool follows the clip line in the image, no angle
    applier_exit_depth=(0.40, 0.65), # camera-space depth of the exit point / depth of the clip (<1 = closer)
    applier_rough=(0.22, 0.35),      # shinier (user)
    applier_base=(0.33, 0.31, 0.31), # steel grey, a touch darker for solidity
    applier_metallic=(0.90, 1.00),
    applier_blur_near=(0.3, 0.8),    # at the tip the rod is at the CLIP depth: same focus as the clip ...
    applier_blur_far=(12.0, 18.0),   # ... and at the frame border (much closer to the camera)
    applier_min_visible=0.60,        # keep at least this fraction of the clip visible (user: little occlusion)
    applier_opacity_boost=1.7,       # re-solidify the blurred rod's core (edges stay soft)
    applier_bites=(0.35, 0.35, 0.2, 0.1, 0.0, -0.3, -0.6, -1.0),   # tip positions tried for a held clip
)


def U(rng, rngspec):
    return float(rng.uniform(*rngspec)) if isinstance(rngspec, (tuple, list)) else float(rngspec)


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def srgb_to_lin(c):
    return np.power(np.clip(c, 0.0, 1.0), 2.2)


def cam_frame(pos, focal, up):
    fwd = _norm(np.asarray(focal) - np.asarray(pos))
    right = _norm(np.cross(fwd, up))
    upc = np.cross(right, fwd)
    return right, upc, fwd


def equirect_dirs():
    th = np.pi * np.arange(ENV_H) / (ENV_H - 1)
    ph = -np.pi + 2.0 * np.pi * np.arange(ENV_W) / ENV_W
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    # measured VTK convention for EnvironmentUp=+Z, EnvironmentRight=+X
    return np.stack([np.sin(TH) * np.cos(PH), -np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)


DIRS = equirect_dirs()
_BASE_MESH = pv.read(MESH_PATH)
_CLIP_FRAME = pca_frame(np.asarray(_BASE_MESH.points, np.float64))[1]     # columns e0 (long), e1 (wide), e2 (thin)
BLADE_DIR_MODEL = _CLIP_FRAME[:, 0]


# ------------------------------------------------------------------ environment
def build_env(bg_bgr, cam, clip_uv, clip_world, fx, rng):
    """Synthetic HDR equirect environment seen from the clip (float32, RGB, linear)."""
    pos, focal, up = [np.asarray(v, float) for v in cam]
    right, upc, fwd = cam_frame(pos, focal, up)
    dx, dy, dz = DIRS @ right, DIRS @ upc, DIRS @ fwd          # dz>0: away from camera (toward tissue)
    H, W = bg_bgr.shape[:2]
    lin = srgb_to_lin(cv2.GaussianBlur(bg_bgr, (0, 0), 6).astype(np.float32) / 255.0)[..., ::-1]
    lin = np.ascontiguousarray(lin)
    denom = np.maximum(dz, 0.12)
    scale = U(rng, P["dome_scale"]) * fx * 0.25
    u = np.clip(clip_uv[0] + scale * dx / denom, 0, W - 1).astype(np.float32)
    v = np.clip(clip_uv[1] - scale * dy / denom, 0, H - 1).astype(np.float32)
    tissue = cv2.remap(lin, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    env = tissue * WHITE * U(rng, P["tissue_gain"])
    room = np.array(P["room_rgb"], np.float32) * WHITE
    wgt = np.clip((dz + 0.15) / 0.30, 0.0, 1.0)[..., None]
    env = wgt * env + (1.0 - wgt) * room
    to_cam = _norm(pos - clip_world)
    ang = np.arccos(np.clip(DIRS @ to_cam, -1.0, 1.0))
    disc = np.exp(-0.5 * (ang / np.deg2rad(U(rng, P["light_radius_deg"]))) ** 2)
    halo = np.exp(-0.5 * (ang / np.deg2rad(U(rng, P["halo_radius_deg"]))) ** 2)
    light = np.array(P["light_rgb"], np.float32) * WHITE
    env = env + (disc[..., None] * U(rng, P["light_gain"]) + halo[..., None] * U(rng, P["halo_gain"])) * light
    return env.astype(np.float32)


def warm_light_color(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sel = g >= np.percentile(g, 70)
    mean_bgr = bgr[sel].mean(axis=0) / 255.0 if sel.any() else np.array([0.7, 0.7, 0.7])
    rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
    return tuple(0.5 + 0.5 * c for c in rgb)


# ------------------------------------------------------------------ render
def make_lights(cam, clip_world, bg_bgr, rng):
    """Light set shared by every layer of one sample (so clip and applier match)."""
    pos = np.asarray(cam[0], float)
    to_cam = _norm(pos - clip_world)
    right, upc, _ = cam_frame(pos, cam[1], cam[2])
    az = rng.uniform(0, 2 * np.pi)
    tilt = np.deg2rad(U(rng, P["fill_angle_deg"]))
    fill_dir = _norm(np.cos(tilt) * to_cam + np.sin(tilt) * (np.cos(az) * right + np.sin(az) * upc))
    return dict(warm=warm_light_color(bg_bgr), head=U(rng, P["headlight"]), fill=U(rng, P["fill"]),
                fill_pos=tuple(clip_world + fill_dir * 100.0), focal=tuple(clip_world))


def render_layer(meshes, cam, fovy_deg, env, lights, w, h):
    """Render a list of (mesh, material) as one RGBA layer (SS x supersampled, transparent bg)."""
    pl = pv.Plotter(off_screen=True, window_size=[w * SS, h * SS])
    for mesh, mat in meshes:
        actor = pl.add_mesh(mesh, pbr=True, color=tuple(float(c) for c in mat["base"]),
                            metallic=float(mat.get("metallic", 1.0)),
                            roughness=float(mat["rough"]), smooth_shading=True, reset_camera=False)
        if mat.get("coat", 0.0) > 0:
            try:
                actor.prop.SetCoatStrength(float(mat["coat"]))
                actor.prop.SetCoatRoughness(0.12)
            except AttributeError:
                pass
    tex = pv.Texture(env)
    tex.mipmap = True
    tex.interpolate = True
    pl.renderer.SetEnvironmentUp(0.0, 0.0, 1.0)
    pl.renderer.SetEnvironmentRight(1.0, 0.0, 0.0)
    pl.set_environment_texture(tex, is_srgb=False, show_background=False)
    pl.remove_all_lights()
    pl.add_light(pv.Light(light_type="headlight", intensity=lights["head"], color=lights["warm"]))
    pl.add_light(pv.Light(position=lights["fill_pos"], focal_point=lights["focal"], intensity=lights["fill"],
                          color=lights["warm"]))
    pl.set_background("black")
    pl.camera_position = cam
    pl.camera.view_angle = fovy_deg
    rgba = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    return rgba


def _oriented_box(center, axes, size):
    """Box of `size` (along axes[0], axes[1], axes[2]) centred at `center`; axes = 3 unit vectors."""
    b = pv.Box(bounds=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2, -size[2] / 2, size[2] / 2))
    T = np.eye(4)
    T[:3, :3] = np.stack(axes, axis=1)
    T[:3, 3] = center
    return b.transform(T, inplace=False)


def _rot_about(v, axis, ang):
    axis = _norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1 - np.cos(ang))


def _cam_axes(meta):
    c = meta["camera"]
    pos, focal, up = (np.asarray(c[k], float) for k in ("position", "focal_point", "view_up"))
    right, upc, fwd = cam_frame(pos, focal, up)
    return pos, right, -upc, fwd                                # OpenCV axes: x right, y down, z forward


def cam_coords(meta, p_world):
    """World point -> (u, v, Z) with the sample's OpenCV camera."""
    c = meta["camera"]
    pos, xr, yd, fwd = _cam_axes(meta)
    X = np.asarray(p_world, float) - pos
    x, y, z = X @ xr, X @ yd, X @ fwd
    return c["fx"] * x / z + c["cx"], c["fy"] * y / z + c["cy"], z


def unproject(meta, u, v, Z):
    c = meta["camera"]
    pos, xr, yd, fwd = _cam_axes(meta)
    return pos + xr * ((u - c["cx"]) / c["fx"] * Z) + yd * ((v - c["cy"]) / c["fy"] * Z) + fwd * Z


def build_applier(pts, R, cam, meta, rng, bite=0.35, exit_depth=None):
    """v7 applier from the user's real references (#2 #12 #13 #14): the clip sits at the tip of
    a dark instrument that comes from the surgeon's hand, i.e. it leaves the frame while rising
    toward the camera (perspective makes it huge and defocused at the border). A dark head
    covers the coil; the shaft is one thick rod or two diverging forceps arms.
    Returns (mesh, material, info, spring_world)."""
    pos, focal, up = (np.asarray(v, float) for v in cam)
    fwd = _norm(focal - pos)
    e0, e1, e2 = (_norm(R @ _CLIP_FRAME[:, i]) for i in range(3))
    proj = (pts - pts.mean(0)) @ e0
    k = max(5, int(0.05 * len(pts)))
    order = np.argsort(proj)
    spring = pts[order[:k]].mean(0)
    tip = pts[order[-k:]].mean(0)
    back = _norm(spring - tip)                                  # along the clip, away from the jaws
    W, H = int(meta["image"]["width"]), int(meta["image"]["height"])

    # exit point: in the IMAGE, from the spring along the clip's back direction (+/- jitter)
    # until well outside the frame; in DEPTH, closer to the camera than the clip.
    su, sv, sz = cam_coords(meta, spring)
    bu, bv, _ = cam_coords(meta, spring + back * 5.0)
    dir2 = np.array([bu - su, bv - sv])
    dir2 = dir2 / (np.linalg.norm(dir2) + 1e-6)
    a = np.deg2rad(U(rng, P["applier_exit_jitter_deg"]))
    dir2 = np.array([np.cos(a) * dir2[0] - np.sin(a) * dir2[1], np.sin(a) * dir2[0] + np.cos(a) * dir2[1]])
    L = 1.6 * np.hypot(W, H)
    zf = U(rng, P["applier_exit_depth"]) if exit_depth is None else float(exit_depth)
    exit_w = unproject(meta, su + dir2[0] * L, sv + dir2[1] * L, sz * zf)
    d = _norm(exit_w - spring)                                  # instrument direction, head -> hand
    n_lat = _norm(np.cross(d, fwd))                             # sideways in the image

    # V8 (user: "hazla mas simple"): ONE rod with a rounded tip. The tip sits on the clip's
    # coil (it holds it there), the rod runs toward the surgeon's hand rising to the camera;
    # perspective alone makes it wide and soft near the border. No head block, no neck, no arms.
    jaw_len = U(rng, P["applier_jaw_len"])
    r_full = U(rng, P["applier_radius"])
    two_arms = False
    tip_c = spring - back * (r_full * bite)                     # bite>0: tip over the coil end (grip); <0: behind it
    shaft_len = float(np.linalg.norm(exit_w - tip_c)) + 120.0
    parts = [pv.Sphere(radius=r_full, center=tip_c, theta_resolution=32, phi_resolution=32),
             pv.Cylinder(center=tip_c + d * shaft_len / 2, direction=d, radius=r_full, height=shaft_len, resolution=48)]
    mesh = pv.merge(parts)
    mat = dict(base=tuple(np.clip(np.array(P["applier_base"]) * rng.uniform(0.85, 1.25), 0, 1)),
               rough=U(rng, P["applier_rough"]), metallic=U(rng, P.get("applier_metallic", (1.0, 1.0))), coat=0.0)
    info = dict(radius=round(r_full, 2), jaw_len=round(jaw_len, 2), two_arms=bool(two_arms), bite=round(float(bite), 2),
                exit_jitter_deg=round(float(np.degrees(a)), 1), roughness=round(mat["rough"], 3))
    return mesh, mat, info, spring


# ------------------------------------------------------------------ post-processing
def sharpness_map(gray):
    return cv2.boxFilter(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)), -1, (31, 31))


def focus_percentile(bgr, region):
    """How in-focus `region` is, as its sharpness percentile within the whole frame (0..1)."""
    F = sharpness_map(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if not region.any():
        return 1.0
    return float((F.ravel() < float(F[region].mean())).mean())


def noise_sigma(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = gm < np.percentile(gm, 40)
    return float(1.4826 * np.median(np.abs(hp[flat] - np.median(hp[flat]))))


def contact_shadow(bgf, mask, strength=None, blur_frac=None, shift_frac=None):
    """Soft contact shadow under an object, sized RELATIVE to its bbox diagonal."""
    strength = P["shadow_strength"] if strength is None else strength
    blur_frac = P["shadow_blur_frac"] if blur_frac is None else blur_frac
    shift_frac = P["shadow_shift_frac"] if shift_frac is None else shift_frac
    m = (mask > 127).astype(np.float32)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return bgf
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    dx = int(np.clip(round(shift_frac[0] * diag), 1, 6))
    dy = int(np.clip(round(shift_frac[1] * diag), 1, 10))
    blur = float(np.clip(blur_frac * diag, 1.5, 12.0))
    shifted = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (m.shape[1], m.shape[0]))
    soft = np.clip(cv2.GaussianBlur(shifted, (0, 0), blur) - m, 0, 1)
    return bgf * (1.0 - strength * soft)[..., None]


def motion_kernel(length_px, angle_deg, size=9):
    k = np.zeros((size, size), np.float32)
    c = (size - 1) / 2.0
    dx, dy = np.cos(np.deg2rad(angle_deg)) * length_px / 2.0, np.sin(np.deg2rad(angle_deg)) * length_px / 2.0
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
    k[int(c), int(c)] = max(k[int(c), int(c)], 1.0)
    return k / k.sum()


def chroma_aberration(lay, k, cx, cy):
    """Lateral CA: blue scaled outward, red inward about the image centre (premultiplied layer)."""
    H, W = lay.shape[:2]
    out = lay.copy()
    for ch, s in ((0, 1.0 + k), (2, 1.0 - k)):
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        out[..., ch] = cv2.warpAffine(lay[..., ch], M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def _layer_linear(rgba_ss, W, H):
    """Supersampled transparent screenshot -> (premultiplied linear BGR, alpha) at frame size."""
    rgba = rgba_ss.astype(np.float32) / 255.0
    lay = cv2.resize(np.ascontiguousarray(np.power(rgba[..., :3][..., ::-1], 2.2)), (W, H), interpolation=cv2.INTER_AREA)
    a = cv2.resize(np.ascontiguousarray(rgba[..., 3]), (W, H), interpolation=cv2.INTER_AREA)
    return lay, a


def project_point(meta, p_world):
    """World point -> pixel (u, v) with the sample's OpenCV camera."""
    c = meta["camera"]
    pos, focal, up = (np.asarray(c[k], float) for k in ("position", "focal_point", "view_up"))
    right, upc, fwd = cam_frame(pos, focal, up)
    Rw2c = np.stack([right, -upc, fwd], axis=0)
    X = Rw2c @ (np.asarray(p_world, float) - pos)
    return np.array([c["fx"] * X[0] / X[2] + c["cx"], c["fy"] * X[1] / X[2] + c["cy"]])


def composite(bg_bgr, rgba_clip, mask, rng, rgba_app=None, app_anchor=None):
    """Compose clip (+ applier) layers onto the backplate with matched optics.
    `app_anchor` = pixel of the applier jaws (defocus grows with the distance from it).
    Returns (image, info, visible_mask)."""
    H, W = bg_bgr.shape[:2]
    m = mask > 127
    m8 = m.astype(np.uint8)
    lay, a = _layer_linear(rgba_clip, W, H)
    bg_lin = np.power(bg_bgr.astype(np.float32) / 255.0, 2.2)

    # black level: the frame's own darkest 1% (per channel, linear) -- a CG object darker
    # than anything in the video reads as pasted
    lum = bg_lin.mean(-1)
    dark = lum <= np.percentile(lum, 1.0)
    c_lo = bg_lin[dark].mean(0) * P["black_lift"] if dark.any() else np.zeros(3, np.float32)

    def finish_layer(lay_, a_, extra_blur=0.0, ca_k=None):
        lay_ = lay_ * (1.0 - c_lo) + a_[..., None] * c_lo
        if ca_k:
            lay_ = chroma_aberration(lay_, ca_k, W / 2.0, H / 2.0)
        return lay_, a_

    # local depth of field: how in-focus is the tissue right around the clip?
    ring = (cv2.dilate(m8, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(m8, np.ones((7, 7), np.uint8)) > 0)
    pct = focus_percentile(bg_bgr, ring)
    s = float(np.clip(U(rng, P["defocus_base"]) + P["defocus_dof_gain"] * (1.0 - pct) ** 2, 0.4, 3.5))
    ca_k = U(rng, P["ca_strength"])

    mb_len = 0.0
    if rng.random() < P["motion_blur_prob"]:
        mb_len = round(U(rng, P["motion_blur_len"]), 2)
        kern = motion_kernel(mb_len, float(rng.uniform(0.0, 180.0)))
        lay, a = cv2.filter2D(lay, -1, kern), cv2.filter2D(a, -1, kern)

    lay, a = finish_layer(lay, a, ca_k=ca_k)
    layb, ab = cv2.GaussianBlur(lay, (0, 0), s), cv2.GaussianBlur(a, (0, 0), s)
    out_lin = layb + (1.0 - ab[..., None]) * contact_shadow(bg_lin, mask)
    bloom_src = layb
    visible = m8 * 255

    app_info = None
    if rgba_app is not None:
        lay2, a2 = _layer_linear(rgba_app, W, H)
        occ8 = (a2 > 0.5).astype(np.uint8) * 255
        visible = visible.copy()
        visible[occ8 > 127] = 0
        lay2, a2 = finish_layer(lay2, a2, ca_k=ca_k * 1.5)
        # the frame's own light falloff (vignette / cavity) relative to the clip location
        lum_s = cv2.GaussianBlur(lum, (0, 0), 60.0)
        anchor = app_anchor if app_anchor is not None else np.array([W / 2.0, H / 2.0])
        ax, ay = int(np.clip(anchor[0], 0, W - 1)), int(np.clip(anchor[1], 0, H - 1))
        vig = np.clip(lum_s / max(lum_s[ay, ax], 1e-4), 0.25, 1.3)
        lay2 = lay2 * vig[..., None]
        # defocus grows with the distance from the jaws (the shaft rises toward the camera)
        s_near = s + U(rng, P["applier_blur_near"])
        s_far = s + U(rng, P["applier_blur_far"])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        wfar = (np.clip(np.hypot(xx - float(anchor[0]), yy - float(anchor[1])) / (0.55 * np.hypot(W, H)), 0, 1) ** 1.0).astype(np.float32)
        ln, an = cv2.GaussianBlur(lay2, (0, 0), s_near), cv2.GaussianBlur(a2, (0, 0), s_near)
        lf, af = cv2.GaussianBlur(lay2, (0, 0), s_far), cv2.GaussianBlur(a2, (0, 0), s_far)
        lay2b = ln * (1 - wfar)[..., None] + lf * wfar[..., None]
        a2b = an * (1 - wfar) + af * wfar
        # solidity (user): a heavy blur thins the alpha of a slim rod and it reads translucent.
        # Keep the colour blur but re-solidify the core: boost alpha, keep edges soft.
        a_solid = np.clip(a2b * P["applier_opacity_boost"], 0.0, 1.0)
        col = lay2b / np.maximum(a2b, 1e-4)[..., None]
        lay2b = col * a_solid[..., None]
        a2b = a_solid
        # its shadow on tissue and clip, then the defocused applier on top
        out_lin = contact_shadow(out_lin, occ8, strength=0.30, blur_frac=0.05, shift_frac=(0.015, 0.03))
        out_lin = lay2b + (1.0 - a2b[..., None]) * out_lin
        bloom_src = bloom_src + lay2b
        app_info = dict(blur_near=round(s_near, 2), blur_far=round(s_far, 2),
                        covers=round(float(((occ8 > 127) & m).sum() / max(m.sum(), 1)), 3))

    thr = P["bloom_thresh"] ** 2.2
    lumb = bloom_src @ np.array([0.114, 0.587, 0.299], np.float32)
    hl = np.clip((lumb - thr) / (1.0 - thr), 0, 1)[..., None] * bloom_src
    out_lin = out_lin + cv2.GaussianBlur(hl, (0, 0), P["bloom_sigma"]) * P["bloom_gain"]

    out = np.power(np.clip(out_lin, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)
    f_clip = float(sharpness_map((out.mean(-1) * 255).astype(np.uint8))[m].mean())

    reg = cv2.GaussianBlur(cv2.dilate(m8 * 255, np.ones((9, 9), np.uint8)).astype(np.float32) / 255.0, (0, 0), 2.0)
    if rgba_app is not None:
        reg = np.maximum(reg, cv2.GaussianBlur(cv2.dilate(occ8, np.ones((9, 9), np.uint8)).astype(np.float32) / 255.0, (0, 0), 2.0))
    ycc = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_BGR2YCrCb)
    cb = cv2.GaussianBlur(ycc[..., 1:], (0, 0), P["chroma_sigma"])
    ycc[..., 1:] = cb * reg[..., None] + ycc[..., 1:] * (1.0 - reg[..., None])
    out = cv2.cvtColor(np.clip(ycc, 0.0, 1.0), cv2.COLOR_YCrCb2BGR) * 255.0
    sig = noise_sigma(bg_bgr) * P["noise_gain"]
    if sig > 0:
        out = out + rng.normal(0.0, sig, out.shape).astype(np.float32) * reg[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    q = None
    if rng.random() < P["jpeg_prob"]:
        q = int(round(U(rng, P["jpeg_quality"])))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    info = dict(defocus_sigma=round(s, 2), focus_percentile=round(pct, 3), ca=round(ca_k, 4),
                motion_blur_px=mb_len, jpeg_quality=q, F_clip=round(f_clip, 2), applier=app_info)
    info["_applier_mask"] = occ8 if rgba_app is not None else None
    return out, info, visible


# ------------------------------------------------------------------ sample I/O
def load_sample(src, stem):
    with open(os.path.join(src, f"{stem}.json")) as f:
        meta = json.load(f)
    img = meta["image"]
    bg = cv2.imread(os.path.join(src, img.get("file_bg", f"{stem}_bg.png")), cv2.IMREAD_COLOR)
    old = cv2.imread(os.path.join(src, img.get("file_rgb", f"{stem}.png")), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(src, img.get("file_mask", f"{stem}_mask.png")), cv2.IMREAD_GRAYSCALE)
    if bg is None or mask is None or old is None:
        raise FileNotFoundError(stem)
    cam = (tuple(meta["camera"]["position"]), tuple(meta["camera"]["focal_point"]), tuple(meta["camera"]["view_up"]))
    q = meta["clip"]["pose_world"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_world"]["translation_m"], float)
    jaw = float(meta["clip"].get("jaw_angle_deg", 0.0))
    fy = float(meta["camera"]["fy"])
    h = int(img["height"])
    fovy = float(np.degrees(2.0 * np.arctan((h / 2.0) / fy)))
    return meta, bg, old, mask, cam, R, t, jaw, fovy


def posed_mesh(mesh, R, t, jaw):
    m = articulate(mesh, jaw) if jaw > 0 else mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return m.transform(T, inplace=False)


def jitter_material(preset, rng):
    base = np.array(PRESETS[preset]["base"], float) * (1.0 + rng.uniform(-0.12, 0.12, 3))
    return dict(base=tuple(np.clip(base, 0.0, 1.0)), rough=U(rng, PRESETS[preset]["rough"]),
                coat=PRESETS[preset]["coat"])


def restyle(mesh, src, stem, preset, seed, out_dir=None):
    over = PRESETS[preset].get("P")
    if over:                                           # preset-specific lighting/post overrides
        saved = dict(P)
        P.update(over)
        try:
            return _restyle(mesh, src, stem, preset, seed, out_dir)
        finally:
            P.clear()
            P.update(saved)
    return _restyle(mesh, src, stem, preset, seed, out_dir)


def _restyle(mesh, src, stem, preset, seed, out_dir=None):
    meta, bg, old, mask, cam, R, t, jaw, fovy = load_sample(src, stem)
    H, W = bg.shape[:2]
    if seed is None:                                   # re-style in place: reuse the sample's own seed
        seed = int(meta.get("appearance", {}).get("seed", 0))
    rng = np.random.default_rng(seed)
    posed = posed_mesh(mesh, R, t, jaw)
    pts = np.asarray(posed.points, float)
    clip_world = pts.mean(axis=0)
    ys, xs = np.where(mask > 127)
    clip_uv = (float(xs.mean()), float(ys.mean()))
    env = build_env(bg, cam, clip_uv, clip_world, float(meta["camera"]["fx"]), rng)
    mat = jitter_material(preset, rng)
    lights = make_lights(cam, clip_world, bg, rng)
    rgba = render_layer([(posed, mat)], cam, fovy, env, lights, W, H)

    rgba_app, app_geom, app_anchor = None, None, None
    held = meta.get("clip", {}).get("tool_held")               # decided by the generator (jaws coupled)
    if held is None or P.get("ignore_tool_held"):
        want_tool = rng.random() < P["applier_prob"]
    else:
        want_tool = bool(held)
    if want_tool:
        m = mask > 127
        # a held clip MUST get its tool: retreat the tip step by step until enough clip stays visible
        for bite in P["applier_bites"]:
            amesh, amat, ainfo, spring_w = build_applier(pts, R, cam, meta, rng, bite=bite)
            cand = render_layer([(amesh, amat)], cam, fovy, env, lights, W, H)
            a1 = cv2.resize(cand[..., 3].astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA) > 0.5
            vis_frac = float((m & ~a1).sum() / max(m.sum(), 1))
            if vis_frac >= P["applier_min_visible"]:
                rgba_app, app_geom, app_anchor = cand, ainfo, project_point(meta, spring_w)
                break
        if rgba_app is None:
            # fallback for clips seen along their axis: rod almost at the clip depth (thin, no
            # perspective widening) so it cannot blanket the clip
            for bite in (0.1, 0.0, -0.5):
                amesh, amat, ainfo, spring_w = build_applier(pts, R, cam, meta, rng, bite=bite, exit_depth=0.9)
                cand = render_layer([(amesh, amat)], cam, fovy, env, lights, W, H)
                a1 = cv2.resize(cand[..., 3].astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA) > 0.5
                if float((m & ~a1).sum() / max(m.sum(), 1)) >= P["applier_min_visible"]:
                    ainfo["shallow"] = True
                    rgba_app, app_geom, app_anchor = cand, ainfo, project_point(meta, spring_w)
                    break

    new, info, visible = composite(bg, rgba, mask, rng, rgba_app, app_anchor)
    if out_dir is None:
        info.pop("_applier_mask", None)
    if app_geom is not None:
        info["applier"].update(app_geom)
    info.update(preset=preset, base=[round(float(c), 3) for c in mat["base"]], roughness=round(mat["rough"], 3),
                coat=mat["coat"], seed=seed)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), new)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_mask_visible.png"), visible)
        amask = info.pop("_applier_mask", None)
        if amask is not None:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_mask_applier.png"), amask)
            meta["image"]["file_mask_applier"] = f"{stem}_mask_applier.png"
        else:
            meta["image"].pop("file_mask_applier", None)
        for key in ("file_bg", "file_mask", "file_depth_clip", "file_depth_clip_npy", "file_normal"):
            fn = meta["image"].get(key)
            if fn and os.path.exists(os.path.join(src, fn)) and not os.path.exists(os.path.join(out_dir, fn)):
                shutil.copy2(os.path.join(src, fn), os.path.join(out_dir, fn))
        meta2 = dict(meta)
        meta2["image"]["file_mask_visible"] = f"{stem}_mask_visible.png"
        meta2["appearance"] = info
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(meta2, f, indent=2)
    return old, new, mask, info


# ------------------------------------------------------------------ evaluation
def zoom_cell(img, mask, size=256, pad=2.5, label=None):
    ys, xs = np.where(mask > 127)
    H, W = img.shape[:2]
    if len(xs) == 0:
        cell = cv2.resize(img, (size, size))
    else:
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        side = int(max(96, pad * max(xs.max() - xs.min(), ys.max() - ys.min())))
        side = min(side, min(W, H))
        x0 = int(np.clip(cx - side / 2, 0, W - side))
        y0 = int(np.clip(cy - side / 2, 0, H - side))
        cell = cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_CUBIC)
    if label:
        cv2.rectangle(cell, (0, 0), (size, 18), (0, 0, 0), -1)
        cv2.putText(cell, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return cell


def real_crop_cell(rng, size=256):
    paths = sorted(glob.glob(os.path.join(REAL_CROPS, "*.png")))
    p = paths[int(rng.integers(len(paths)))]
    cell = cv2.resize(cv2.imread(p), (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (size, 18), (0, 0, 0), -1)
    cv2.putText(cell, "REAL " + os.path.basename(p)[:13], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return cell


def clip_pixels(img, mask, erode=1):
    m = mask > 127
    if erode > 0:
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    return img[m]


def real_clip_pixels(n_crops=300, scale=240, seed=0):
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(REAL_CROPS, "*.png")))
    pick = rng.choice(len(paths), size=min(n_crops, len(paths)), replace=False)
    pix, fr = [], []
    for i in pick:
        im = cv2.imread(paths[i])
        mk = cv2.imread(os.path.join(REAL_MASKS, os.path.basename(paths[i])), 0)
        if im is None or mk is None:
            continue
        im = cv2.resize(im, (scale, scale), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, (scale, scale), interpolation=cv2.INTER_NEAREST)
        px = clip_pixels(im, mk, erode=2)
        if len(px) < 50:
            continue
        pix.append(px[rng.choice(len(px), size=min(len(px), 1500), replace=False)])
        F = sharpness_map(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        m = mk > 127
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~m
        fr.append(F[m].mean() / max(F[ring].mean(), 1e-3))
    return np.concatenate(pix), float(np.median(fr))


def hsv_w1(pix_a, pix_b):
    """Per-channel Wasserstein-1 between two pixel sets (OpenCV HSV scale)."""
    ha = cv2.cvtColor(pix_a.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hb = cv2.cvtColor(pix_b.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    qs = np.linspace(0, 100, 201)
    return [float(np.mean(np.abs(np.percentile(ha[:, c], qs) - np.percentile(hb[:, c], qs)))) for c in range(3)]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Re-render GT samples with a MACS-like clip appearance")
    ap.add_argument("--src", default="out_pose100")
    ap.add_argument("--out", default="out_pose100_appear")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="anodized2", choices=sorted(PRESETS))
    ap.add_argument("--compare", default=None, help="comma list of presets -> montage only, no outputs")
    ap.add_argument("--montage", default=None, help="montage path (default: <out>/_montage.png)")
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--reuse-seed", action="store_true",
                    help="take each sample's seed from its JSON `appearance` block (re-style in place)")
    ap.add_argument("--no-applier", action="store_true")
    ap.add_argument("--applier-prob", type=float, default=None, help="override P['applier_prob'] (1.0 = always)")
    a = ap.parse_args()
    if a.no_applier:
        P["applier_prob"] = 0.0
        P["ignore_tool_held"] = True                       # also for samples whose JSON says tool_held
    if a.applier_prob is not None:
        P["applier_prob"] = a.applier_prob
        P["ignore_tool_held"] = True                       # explicit CLI override wins over the JSON flag

    src = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    out_dir = None if a.compare else (a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(src, "[0-9]*.json")))]
    stems = stems[a.start:a.start + a.limit]
    if not stems:
        print(f"no samples in {src}")
        return
    presets = a.compare.split(",") if a.compare else [a.preset]
    default_name = "_montage.png" if not a.compare else f"_compare_{'_'.join(presets)}.png"
    montage_path = a.montage or os.path.join(out_dir or HERE, default_name)

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception as _e:
            print(f"[WARN] start_xvfb failed ({_e}); install xvfb or run under xvfb-run", flush=True)
    mesh = pv.read(MESH_PATH)
    rows, old_pix, new_pix, infos = [], [], [], []
    rng_real = np.random.default_rng(a.seed + 999)
    print(f"restyle | {len(stems)} samples | presets {presets} | out {out_dir or '(compare only)'}", flush=True)
    for i, stem in enumerate(stems):
        cells = []
        for k, preset in enumerate(presets):
            old, new, mask, info = restyle(mesh, src, stem, preset, None if a.reuse_seed else a.seed + i,
                                           out_dir if k == 0 else None)
            if k == 0:
                cells.append(zoom_cell(old, mask, a.cell, label=f"{stem} OLD"))
                old_pix.append(clip_pixels(old, mask))
                new_pix.append(clip_pixels(new, mask))
                infos.append(info)
            lab = f"{preset} s{info['defocus_sigma']} r{info['roughness']:.2f}" + (" +app" if info.get("applier") else "")
            cells.append(zoom_cell(new, mask, a.cell, label=lab))
        cells.append(real_crop_cell(rng_real, a.cell))
        rows.append(np.hstack(cells))
        print(f"[{i+1}/{len(stems)}] {stem}: sigma {info['defocus_sigma']} focus-pct {info['focus_percentile']} "
              f"applier {'yes' if info.get('applier') else 'no'}", flush=True)

    two_up = rows[0].shape[1] <= 4 * a.cell and len(rows) > 1
    if two_up:
        if len(rows) % 2:
            rows.append(np.zeros_like(rows[0]))
        grid = np.vstack([np.hstack([rows[j], rows[j + 1]]) for j in range(0, len(rows), 2)])
    else:
        grid = np.vstack(rows)
    os.makedirs(os.path.dirname(montage_path), exist_ok=True)
    cv2.imwrite(montage_path, grid)

    real_pix, real_ratio = real_clip_pixels(seed=a.seed)
    op, npx = np.concatenate(old_pix), np.concatenate(new_pix)
    w_old, w_new = hsv_w1(op, real_pix), hsv_w1(npx, real_pix)

    def vstd(px):
        return float(cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[..., 2].std())
    print("\nclip-pixel HSV W1 distance to REAL clip pixels (lower = closer):")
    print(f"  OLD render : H {w_old[0]:6.1f}  S {w_old[1]:6.1f}  V {w_old[2]:6.1f}   mean BGR {op.mean(0).round(0)}  V-std {vstd(op):5.1f}")
    print(f"  NEW render : H {w_new[0]:6.1f}  S {w_new[1]:6.1f}  V {w_new[2]:6.1f}   mean BGR {npx.mean(0).round(0)}  V-std {vstd(npx):5.1f}")
    print(f"  REAL       : mean BGR {real_pix.mean(0).round(0)}  V-std {vstd(real_pix):5.1f}")
    print(f"applier in {sum(1 for i in infos if i.get('applier'))}/{len(infos)} samples | "
          f"defocus sigma median {np.median([i['defocus_sigma'] for i in infos]):.2f}")
    print(f"montage -> {montage_path}")


if __name__ == "__main__":
    main()
