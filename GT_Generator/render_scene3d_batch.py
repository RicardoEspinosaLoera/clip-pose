"""
render_scene3d_batch.py  (Phase E, production batch)
====================================================

Batch generator for the depth-USEFUL synthetic dataset. Same job as
`render_clip_poses_depth.py`, but the clip rests on a real 3-D tissue SURFACE
(built from the backplate via Depth Anything V2) instead of a flat 2-D backplate,
so every sample also carries a DENSE, metric, image-consistent scene depth map --
usable to train a monocular depth predictor -- while keeping the valid 6-DoF clip
pose GT.

It REUSES the pose sampler, camera math, and GT writers from
render_clip_poses_depth.py (so the pose distribution and JSON conventions are
identical) and adds the 3-D surface + dense depth from the validated prototype
render_scene3d_depth.py.

Per sample (backward-compatible superset of the old format):
    NNN.png              RGB (metallic clip on the textured 3-D tissue surface)
    NNN_bg.png           the tissue surface WITHOUT the clip
    NNN_mask.png         binary clip mask
    NNN_depth_clip.npy   metric clip depth (NaN off clip)          [as before]
    NNN_depth_clip.png   16-bit clip-depth viz                      [as before]
    NNN_normal.png       clip normal map                            [as before]
    NNN_depth_scene.npy  NEW dense metric depth (clip + tissue), no holes
    NNN_depth_scene.png  NEW 8-bit turbo viz of the dense depth
    NNN.json             6-DoF pose + camera + depth block (extended)

Run:
    OUT_DIR=out_scene3d N_SAMPLES=200 BP_DIR=Data/noclips \
        python render_scene3d_batch.py
"""

import os
import json
import random

# Phase B: the 3-D paths widen the clip-pose envelope by default (the flat path
# stays narrow). setdefault so explicit env still wins; MUST precede the import of
# render_clip_poses_depth, which reads these at import time.
os.environ.setdefault("POSE_PERTURB_MAX_DEG", "35")   # most samples: up to 35deg extra tilt/roll
os.environ.setdefault("POSE_STEEP_FRAC", "0.25")      # a quarter get steeper
os.environ.setdefault("POSE_STEEP_MAX_DEG", "65")
os.environ.setdefault("MIN_UP_DEG", "80")             # allow the perturbed poses through the up-axis gate
os.environ.setdefault("MAX_TILT_DEG", "22")           # a bit more camera tilt too
# keep the clip FARTHER from the camera (smaller on screen): lower both the min
# floor and the max cap on apparent size.
os.environ.setdefault("MIN_AREA_FRAC", "0.004")       # clip bbox >= 0.4% of frame area
os.environ.setdefault("MIN_DIAG_FRAC", "0.05")        # clip bbox diagonal >= 5% of image diagonal
os.environ.setdefault("MAX_AREA_FRAC", "0.035")       # clip bbox <= 3.5% of frame area (was 10%)
os.environ.setdefault("MAX_DIAG_FRAC", "0.24")        # clip bbox diagonal <= 24% of image diagonal (was 42%)
# placement simulation: clip blades point INTO the tissue (brain wall), not shallow.
# Higher BLADE_AWAY_MIN = blades aim more directly down the view axis into the surface.
os.environ.setdefault("BLADE_AWAY", "1")
os.environ.setdefault("BLADE_AWAY_MIN", "0.55")

import cv2
import numpy as np
import pyvista as pv
from PIL import Image
from vtk import vtkTexture

# --- reuse the validated machinery from the flat-backplate generator ---
import render_clip_poses_depth as R
from render_clip_poses_depth import (
    sample_camera_top, sample_visible_pose_guaranteed, cam_world_from_vtk,
    intrinsics_from_vtk, actor_mesh, cam_T_obj, warm_light_color,
    normals_from_depth, save_depth_artifacts, save_pose_and_camera,
    convert_to_binary_mask, _project_bbox_uv,
)
from articulate_clip import articulate
from estimate_bg_depth import build_depth_pipeline, estimate_bg_depth

# the imported module built its own (flat-path) plotter at import time; close it
# so we drive a single plotter of our own.
try:
    R.plotter.close()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = R.w, R.h                                  # 800, 544 (same as the flat path)
ZNEAR, ZFAR = R.znear, R.zfar

BP_DIR  = os.environ.get("BP_DIR", os.path.join(HERE, "Data", "noclips"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "out_scene3d"))
N       = int(os.environ.get("N_SAMPLES", 200))
DA_MODEL = os.environ.get("DA_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")

# jaw articulation (same policy as the flat path)
JAW_MAX_DEG     = float(os.environ.get("JAW_MAX_DEG", 20.0))
JAW_CLOSED_PROB = float(os.environ.get("JAW_CLOSED_PROB", 0.35))
JAW_MIN_OPEN    = 5.0

# tissue-surface shape
GRID_W, GRID_H = 300, 210
OVERSCAN   = 0.35
SURF_REL   = 0.09        # displacement amplitude as a fraction of the surface depth
CONTACT_CLEARANCE = 0.4  # tiny gap left after dropping the clip onto the tissue (world units)
# Depth-consistent co-location: %% of the clip pressed BELOW the predicted tissue
# surface. 0 = clip fully on top (visible, but pops toward the camera in depth);
# higher = more embedded (depth co-located with tissue, but that part is occluded).
# EMBED_PCT may be a fixed number OR a "lo-hi" range sampled uniformly PER SCENE
# (default "0-40" -> a mix of fully-visible and co-located clips across the dataset).
def _parse_embed(s):
    s = str(s).strip()
    if "-" in s[1:]:                      # "lo-hi" range (dash not a leading minus)
        lo, hi = s.split("-", 1)
        return float(lo), float(hi)
    v = float(s)
    return v, v


EMBED_PCT_LO, EMBED_PCT_HI = _parse_embed(os.environ.get("EMBED_PCT", "0-20"))


def sample_embed_pct():
    """A per-scene embed %% (random in the range, or the fixed value)."""
    if EMBED_PCT_HI <= EMBED_PCT_LO:
        return EMBED_PCT_LO
    return float(np.random.uniform(EMBED_PCT_LO, EMBED_PCT_HI))
# Less mirror-metallic + more diffuse than a pure metal so the clip reads BRIGHT in every
# orientation (pure metallic reflects the dark background -> near-black clips); keeps a
# metallic sheen via the specular term. Proper env-map is a Phase C follow-up.
CLIP_MAT = dict(color=(0.66, 0.70, 0.76), metallic=0.45, roughness=0.42,
                specular=0.6, diffuse=0.9, ambient=0.35)

# PBR_CLIP=1: realistic blue-anodized titanium (full metal) reflecting a surgical
# environment cube-map -> photoreal clip, consistent across all views (env is world-
# fixed). Overrides CLIP_MAT; the env is attached to the plotter via setup_pbr_env().
PBR_CLIP = int(os.environ.get("PBR_CLIP", "0"))
if PBR_CLIP:
    from pbr_clip import PBR_CLIP_MAT
    CLIP_MAT = PBR_CLIP_MAT


def setup_pbr_env(plotter):
    """Attach the surgical environment map for PBR clip reflections (if PBR_CLIP)."""
    if PBR_CLIP:
        from pbr_clip import set_pbr_environment
        set_pbr_environment(plotter)

os.makedirs(OUT_DIR, exist_ok=True)
BACKPLATES = [f for f in os.listdir(BP_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
assert BACKPLATES, f"no backplates in {BP_DIR}"

_DISP_CACHE = {}                                  # backplate file -> smoothed disparity


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


# --- Meaning A: reconstruct the image to FIT the predicted depth -----------------
# Instead of pasting the flat photo (whose baked shading encodes the REAL surface
# normals) onto the estimated surface, de-shade the photo to an ALBEDO and let the
# tissue be LIT by the height-field geometry, so the shading is derived from the
# depth -> (image, depth) pair is self-consistent.
RESHADE         = int(os.environ.get("RESHADE", "0"))
RESHADE_AMBIENT = float(os.environ.get("RESHADE_AMBIENT", 0.50))
RESHADE_DIFFUSE = float(os.environ.get("RESHADE_DIFFUSE", 0.28))
RESHADE_ALBEDO_MEAN = float(os.environ.get("RESHADE_ALBEDO_MEAN", 105.0))


def albedo_from_bgr(bgr, sigma_frac=0.06):
    """Retinex-style intrinsic albedo: divide out the large-scale (low-frequency)
    luminance = the coarse baked shading, keeping chromaticity + fine texture
    (vessels). The removed coarse shading is what we re-derive from the geometry.
    Its mean is normalised so re-lighting has headroom and doesn't blow out."""
    f = bgr.astype(np.float32) + 1.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    sigma = sigma_frac * max(bgr.shape[:2])
    shading = cv2.GaussianBlur(gray, (0, 0), sigma)
    shading = np.clip(shading / (shading.mean() + 1e-6), 0.55, 1.8)   # normalised ~1
    albedo = f / shading[..., None]
    m = float(cv2.cvtColor(albedo.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()) + 1e-6
    albedo = np.clip(albedo * (RESHADE_ALBEDO_MEAN / m), 0, 255).astype(np.uint8)
    return albedo


def surface_add_kwargs():
    """add_mesh kwargs for the tissue surface: unlit flat photo (default) or, when
    RESHADE, lit so the shading comes from the height-field normals."""
    if RESHADE:
        return dict(lighting=True, smooth_shading=True, ambient=RESHADE_AMBIENT,
                    diffuse=RESHADE_DIFFUSE, specular=0.12, specular_power=8)
    return dict(lighting=False)


def trim_dark_border(bgr, thresh=18, frac=0.80, max_trim=0.22):
    """Crop near-black vignette/letterbox edges off a backplate. Those dark edges,
    duplicated by the mirror-repeat texture wrap at the frame boundary, appear as a
    hard black LINE across orbited views; trimming them makes the mirror fold land
    on tissue instead. Only trims edge rows/cols that are >`frac` near-black."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    Hh, Ww = g.shape
    dark = g < thresh
    top, bot, left, right = 0, Hh, 0, Ww
    while top < int(Hh * max_trim) and dark[top].mean() > frac:
        top += 1
    while bot > int(Hh * (1 - max_trim)) and dark[bot - 1].mean() > frac:
        bot -= 1
    while left < int(Ww * max_trim) and dark[:, left].mean() > frac:
        left += 1
    while right > int(Ww * (1 - max_trim)) and dark[:, right - 1].mean() > frac:
        right -= 1
    crop = bgr[top:bot, left:right]
    if crop.size and crop.shape[0] > 0.55 * Hh and crop.shape[1] > 0.55 * Ww:
        return crop
    return bgr


def camera_basis(pos, focal, up):
    fwd = _norm(focal - pos)
    right = _norm(np.cross(fwd, up))
    up_cam = np.cross(right, fwd)
    return np.stack([right, up_cam, fwd], axis=1)   # cam->world (columns: right, up, fwd)


def cam_space_depth(pt_world, cam_tuple):
    R_wc, t_wc = cam_world_from_vtk(cam_tuple)
    return float((R_wc @ np.asarray(pt_world, float) + t_wc)[2])


# disparity backend: "depthanything" (default) or "hadepth" (endoscopy-specialised)
DEPTH_BACKEND = os.environ.get("DEPTH_BACKEND", "depthanything").lower()


def get_disparity(pipe, bp_name, bgr):
    """Relative disparity [0,1] (higher=closer) for a backplate, bilateral-smoothed,
    cached per source file. Backend = Depth Anything (default) or HADepth."""
    if bp_name in _DISP_CACHE:
        return _DISP_CACHE[bp_name]
    if DEPTH_BACKEND == "hadepth":
        from hadepth_infer import hadepth_disparity
        disp = hadepth_disparity(bgr)                        # [0,1], closer brighter
    else:
        tmp = os.path.join(OUT_DIR, "_da_tmp.png")
        cv2.imwrite(tmp, bgr)
        disp = estimate_bg_depth(pipe, tmp)                  # [0,1], closer brighter
    disp = cv2.bilateralFilter(disp.astype(np.float32), 9, 0.08, 7)
    _DISP_CACHE[bp_name] = disp
    _DISP_CACHE.clear() if len(_DISP_CACHE) > 64 else None    # bound the cache
    _DISP_CACHE[bp_name] = disp
    return disp


def _edge_fade_1d(n, edge_lo, edge_hi, full_lo, full_hi):
    """1.0 in [full_lo, full_hi], smoothstep down to 0 AT the frame edges
    (edge_lo, edge_hi), and 0 outside them (the padding)."""
    x = np.arange(n, dtype=np.float32)
    r_lo = np.clip((x - edge_lo) / max(full_lo - edge_lo, 1e-6), 0, 1)
    r_hi = np.clip((edge_hi - x) / max(edge_hi - full_hi, 1e-6), 0, 1)
    f = np.clip(np.minimum(r_lo, r_hi), 0, 1)
    return f * f * (3.0 - 2.0 * f)


def build_surface(bgr, disp, cam_tuple, fx, fy, cx, cy, Z0, amp):
    """Textured tissue mesh: back-project an over-scanned image grid to a displaced
    surface at base camera-depth Z0, transform to world, texture with the frame.

    The off-frame over-scan (revealed at the edges when the camera orbits) is filled
    with REFLECT-padded content that then FADES to a soft dark vignette starting just
    inside the frame edge. This hides the reflection's symmetry "join" inside the
    darkening so it reads as a natural microscope FOV edge rather than a mirror seam,
    while the frame interior is untouched. Both texture and height field are padded
    consistently. (fx == fy since the VTK intrinsics fix.)"""
    pos, focal, up = [np.asarray(v, float) for v in cam_tuple]
    Rc2w = camera_basis(pos, focal, up)

    pad_w = min(int(round(OVERSCAN * W)), W - 2)
    pad_h = min(int(round(OVERSCAN * H)), H - 2)
    base = albedo_from_bgr(bgr) if RESHADE else bgr           # de-shaded albedo when re-shading
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)
    disp_ext = cv2.copyMakeBorder(disp, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    rgb_ext = cv2.copyMakeBorder(rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    He, We = disp_ext.shape[:2]

    # vignette: full brightness only in the inner frame (minus an `inset` border),
    # smoothstep to black across that border + the whole padding -> the mirror join
    # at the frame boundary sits in the fade and disappears.
    inset_w, inset_h = int(0.12 * W), int(0.12 * H)
    Vx = _edge_fade_1d(We, pad_w, pad_w + (W - 1), pad_w + inset_w, pad_w + (W - 1) - inset_w)
    Vy = _edge_fade_1d(He, pad_h, pad_h + (H - 1), pad_h + inset_h, pad_h + (H - 1) - inset_h)
    V = np.outer(Vy, Vx)[..., None]
    rgb_ext = np.ascontiguousarray((rgb_ext * V).astype(np.uint8))

    us = np.linspace(-pad_w, (W - 1) + pad_w, GRID_W)
    vs = np.linspace(-pad_h, (H - 1) + pad_h, GRID_H)
    uu, vv = np.meshgrid(us, vs)

    d_small = np.clip(cv2.resize(disp_ext, (GRID_W, GRID_H), interpolation=cv2.INTER_CUBIC), 0, 1)
    z = Z0 + amp * (0.5 - d_small)                            # nearer where disparity high

    xc = (uu - cx) / fx * z          # fx == fy (VTK square pixels), correct since the intrinsics fix
    yc = (vv - cy) / fy * z
    pts_cam = np.stack([xc, yc, z], axis=-1).reshape(-1, 3)
    pts_world = (Rc2w @ pts_cam.T).T + pos[None, :]

    grid = pv.StructuredGrid()
    grid.points = pts_world
    grid.dimensions = [GRID_W, GRID_H, 1]
    # texcoords map each vertex into the (padded) texture; base frame -> original region
    s = np.clip((uu + pad_w) / (We - 1), 0, 1).ravel()
    t = 1.0 - np.clip((vv + pad_h) / (He - 1), 0, 1).ravel()
    grid.active_texture_coordinates = np.column_stack([s, t]).astype(np.float32)

    tex = pv.Texture(rgb_ext)
    tex.repeat = False
    return grid, tex


def drop_clip_onto_surface(Rm, t, clip_mesh, cam, fx, fy, cx, cy, disp, Z0, amp,
                           clearance=CONTACT_CLEARANCE, iters=2, embed_pct=0.0):
    """Shift the clip along the view axis until its tightest vertex just rests on the
    tissue surface (no float, no burial). Returns the corrected world translation t."""
    R_wc, t_wc = cam_world_from_vtk(cam)
    pos, focal, _ = [np.asarray(v, float) for v in cam]
    fwd = _norm(focal - pos)                              # camera -> scene (toward tissue)
    P = clip_mesh.points.astype(np.float64)
    for _ in range(iters):
        vw = (Rm @ P.T).T + t
        vc = (R_wc @ vw.T).T + t_wc
        vz = vc[:, 2]
        ok = vz > 1e-6
        u = np.clip(fx * vc[:, 0] / np.where(ok, vz, 1) + cx, 0, W - 1).astype(int)
        v = np.clip(fy * vc[:, 1] / np.where(ok, vz, 1) + cy, 0, H - 1).astype(int)
        surf_z = Z0 + amp * (0.5 - disp[v, u])            # surface depth under each vertex
        gap = (surf_z - vz)[ok]                           # >0 = vertex floats above tissue
        if gap.size == 0:
            break
        # EMBED_PCT=0 -> min gap (tightest vertex just contacts, clip fully in front);
        # higher -> deeper reference vertex so that %% of the clip embeds below the
        # local tissue depth -> the clip's depth is co-located with the tissue.
        ref = float(np.min(gap)) if embed_pct <= 0 else float(np.percentile(gap, embed_pct))
        shift = ref - clearance
        t = t + fwd * shift
    return t


def metric_depth(plotter):
    plotter.render()
    z = plotter.get_image_depth(fill_value=np.nan)
    finite = np.isfinite(z)
    m = np.full(z.shape, np.nan, np.float32)
    m[finite] = -z[finite]
    return m


def save_scene_depth(stem, dense):
    npy = os.path.join(OUT_DIR, f"{stem}_depth_scene.npy")
    png = os.path.join(OUT_DIR, f"{stem}_depth_scene.png")
    np.save(npy, dense)
    valid = np.isfinite(dense)
    viz = np.zeros(dense.shape, np.uint8)
    if valid.any():
        dmin, dmax = float(np.nanmin(dense)), float(np.nanmax(dense))
        viz[valid] = (255 * (1 - (dense[valid] - dmin) / max(dmax - dmin, 1e-9))).astype(np.uint8)
    cv2.imwrite(png, cv2.applyColorMap(viz, cv2.COLORMAP_TURBO))
    return os.path.basename(npy), os.path.basename(png), (float(valid.mean()))


def main():
    pipe = build_depth_pipeline(DA_MODEL)
    print(f"[3d-batch] N={N} BP_DIR={BP_DIR} OUT_DIR={OUT_DIR}")

    plotter = pv.Plotter(off_screen=True, window_size=[W, H])
    plotter.camera.clipping_range = (ZNEAR, ZFAR)
    plotter.show(auto_close=False)                            # init render window
    setup_pbr_env(plotter)

    mesh = R.mesh                                             # Item.obj (loaded by the import)

    idx = 0
    tries = 0
    while idx < N and tries < N * 20:
        tries += 1
        # --- backplate + its height field ---
        bp = random.choice(BACKPLATES)
        bgr = cv2.resize(trim_dark_border(cv2.imread(os.path.join(BP_DIR, bp), cv2.IMREAD_COLOR)), (W, H))
        disp = get_disparity(pipe, bp, bgr)

        # --- camera + clip pose (reuse the flat path's samplers / guarantees) ---
        cam = sample_camera_top()
        plotter.camera_position = cam
        fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, W, H)

        theta = 0.0 if np.random.rand() < JAW_CLOSED_PROB \
            else float(np.random.uniform(JAW_MIN_OPEN, JAW_MAX_DEG))
        clip_mesh = articulate(mesh, theta) if theta > 0.0 else mesh

        Rm, t, cam = sample_visible_pose_guaranteed(cam, fx, fy, cx, cy)
        if Rm is None:
            continue
        plotter.camera_position = cam
        fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, W, H)

        # surface base at the clip centroid depth; then DROP the clip so it rests on the
        # tissue (tightest vertex contacts the local surface) instead of floating.
        centroid_w = Rm @ np.asarray(clip_mesh.center, float) + t
        z_clip = cam_space_depth(centroid_w, cam)
        if z_clip <= 1.0:
            continue
        amp = SURF_REL * z_clip
        Z0 = z_clip
        t = drop_clip_onto_surface(Rm, t, clip_mesh, cam, fx, fy, cx, cy, disp, Z0, amp,
                                   embed_pct=sample_embed_pct())

        # --- build actors ---
        for a in list(plotter.renderer.actors.values()):
            plotter.remove_actor(a, render=False)
        plotter.renderer.remove_all_lights()
        plotter.add_light(pv.Light(light_type="headlight", intensity=2.4,
                                   color=warm_light_color(bgr)))
        # two off-axis fills so a metallic sheen catches in more orientations
        for off in ((45, 30, 20), (-40, -25, 15)):
            plotter.add_light(pv.Light(position=tuple(np.asarray(cam[0]) + np.array(off)),
                                       light_type="scene light", intensity=0.7))

        surf, tex = build_surface(bgr, disp, cam, fx, fy, cx, cy, Z0, amp)
        surf_actor = plotter.add_mesh(surf, texture=tex, **surface_add_kwargs())

        posed = actor_mesh(clip_mesh, Rm, t)
        clip_actor = plotter.add_mesh(posed, pbr=True, smooth_shading=True,
                                      reset_camera=False, **CLIP_MAT)
        plotter.camera_position = cam
        plotter.render()

        # --- renders ---
        idx += 1
        stem = f"{idx:03d}"

        rgb = plotter.screenshot(return_img=True)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        dense = metric_depth(plotter)
        scene_npy, scene_png, cover = save_scene_depth(stem, dense)

        # clip-only depth + mask (hide surface)
        surf_actor.SetVisibility(False)
        clip_metric = metric_depth(plotter)
        depth_info = save_depth_artifacts(stem, clip_metric, fx, fy)
        mask = (np.isfinite(clip_metric).astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_mask.png"), mask)

        # RGB with a soft contact shadow to ground the clip on the tissue
        rgb_bgr = R.contact_shadow(rgb_bgr, mask)
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}.png"), rgb_bgr)

        # tissue-only RGB (hide clip, show surface) -> bg
        surf_actor.SetVisibility(True)
        clip_actor.SetVisibility(False)
        plotter.render()
        bgimg = plotter.screenshot(return_img=True)
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_bg.png"), cv2.cvtColor(bgimg, cv2.COLOR_RGB2BGR))
        clip_actor.SetVisibility(True)
        plotter.render()

        # --- GT json (same writer) + dense-depth block ---
        json_path = os.path.join(OUT_DIR, f"{stem}.json")
        save_pose_and_camera(
            t, Rm, cam, plotter.camera, json_path,
            w=W, h=H, znear=ZNEAR, zfar=ZFAR,
            files={"rgb": f"{stem}.png", "bg": f"{stem}_bg.png", "mask": f"{stem}_mask.png"},
            depth_info=depth_info, jaw_angle_deg=theta)
        with open(json_path) as f:
            meta = json.load(f)
        meta["image"]["file_depth_scene"] = scene_npy
        meta["image"]["file_depth_scene_png"] = scene_png
        meta["image"]["backplate_src"] = bp                # true scene-source id for eval
        meta["depth"]["scene_source"] = f"3d_surface(DepthAnything={DA_MODEL})"
        meta["depth"]["scene_convention"] = "dense metric camera-space distance (world units), clip+tissue"
        meta["depth"]["scene_coverage"] = round(cover, 4)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        if idx % 25 == 0 or idx == 1:
            print(f"  [{idx}/{N}] {bp} jaw={theta:.0f} z_clip={z_clip:.1f} cover={cover*100:.1f}%")

    plotter.close()
    print(f"[3d-batch] done: {idx} samples -> {OUT_DIR}")


if __name__ == "__main__":
    main()
