"""
render_scene3d_multiview.py  (Phase E, multi-view)
==================================================

Multi-view variant of the 3-D scene generator, for training 3-D reconstruction
networks (MVS / NeRF / 3D Gaussian Splatting / SfM). For each scene it fixes the
tissue surface + clip in world space and renders K posed views by orbiting the
camera over a SMALL baseline (surgical-microscope-like). Because every view is a
render of the SAME known meshes, the multi-view geometry GT is EXACT and
cross-view consistent (the tissue depth per view IS the height-field mesh, not an
estimate); the tissue SHAPE is still the Depth-Anything surface (2.5-D pseudo-
anatomy) -- a realism caveat, not a GT-correctness one.

Per scene -> a NeRF/instant-ngp-style folder:
    scene_XXXX/
      transforms.json     intrinsics + per-view camera-to-world (OpenGL) + clip pose GT
      images/vNN.png       RGB (clip on tissue), world-fixed lights so speculars move
      depth/vNN.npy/.png   dense metric depth (exact w.r.t. the rendered meshes)
      masks/vNN.png        clip mask

Lights are FIXED in world (not a headlight) so the clip's specular highlights are
genuinely view-dependent across the K views -- the real signal a reconstruction
net should exploit.

Run:
    BP_DIR=Data/noclips_bank OUT_DIR=out_mv N_SCENES=8 N_VIEWS=16 \
        python render_scene3d_multiview.py
"""

import os
import json
import random

import cv2
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as Rot

# reuse the validated single-view machinery
import render_scene3d_batch as B
from render_scene3d_batch import (
    build_surface, drop_clip_onto_surface, cam_space_depth, metric_depth,
    get_disparity, camera_basis, trim_dark_border, surface_add_kwargs, sample_embed_pct,
    setup_pbr_env, CLIP_MAT, W, H, ZNEAR, ZFAR, SURF_REL,
)
from render_clip_poses_depth import (
    sample_camera_top, sample_visible_pose_guaranteed, intrinsics_from_vtk,
    actor_mesh, _rotmat_to_quat_wxyz, warm_light_color, LIGHT_INTENSITY,
    CLIP_MAT as POSE_CLIP_MAT, _project_bbox_uv,
    instrument_avoid_mask, clip_silhouette_avoid_fraction, AVOID_OVERLAP_MAX,
    BLADE_DIR_MODEL, cam_world_from_vtk,
)
from articulate_clip import articulate
from estimate_bg_depth import build_depth_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))

BP_DIR   = os.environ.get("BP_DIR", os.path.join(HERE, "Data", "noclips_bank"))
OUT_DIR  = os.environ.get("OUT_DIR", os.path.join(HERE, "out_mv"))
N_SCENES = int(os.environ.get("N_SCENES", 8))
N_VIEWS  = int(os.environ.get("N_VIEWS", 16))
DA_MODEL = os.environ.get("DA_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")

# Orbit baseline (deg): sampled PER SCENE from [MIN,MAX] so the dataset spans tight
# microscope-like baselines (realistic, small disocclusion) AND wider baselines
# (better triangulation for reconstruction). Wide orbits expose the 2.5-D flatness,
# so the max stays moderate.
# Small, microscope-REALISTIC baseline. A surgical scope barely orbits; large angles
# relative to the ~30deg FOV also reveal too much off-frame border (the single
# backplate has no content there), which reads as a mirror/vignette join. Kept small
# so borders stay clean AND the parallax is physically plausible.
AZ_MIN = float(os.environ.get("AZ_MIN", 4.0))
AZ_MAX = float(os.environ.get("AZ_MAX", 9.0))
EL_MIN = float(os.environ.get("EL_MIN", 3.0))
EL_MAX = float(os.environ.get("EL_MAX", 6.0))
D_JIT  = float(os.environ.get("D_JIT", 0.08))
# Depth-aware placement: sample this many candidate poses per scene and drop the clip onto
# the FARTHEST (lowest-disparity) tissue, so it avoids near instruments / raised structures
# in the backplate. 1 = old behaviour (first valid pose). Higher = stronger far-bias.
POSE_DEPTH_TRIES = int(os.environ.get("POSE_DEPTH_TRIES", 24))
# Reject a scene unless at least this fraction of the clip's silhouette is actually VISIBLE
# (in front of the tissue) at the base view -> guarantees the whole clip shows, no hidden parts.
VIS_MIN = float(os.environ.get("VIS_MIN", 0.98))
# Tool avoidance: a real instrument / gauze pad is a LARGE bright achromatic blob; scattered
# specular sparkles on wet tissue are tiny. Keep only large blobs so the clip is rejected for
# landing on a TOOL, not for touching a highlight. Reject if >TOOL_OVERLAP_MAX of the clip is on one.
MIN_TOOL_AREA_FRAC = float(os.environ.get("MIN_TOOL_AREA_FRAC", 0.004))
TOOL_OVERLAP_MAX = float(os.environ.get("TOOL_OVERLAP_MAX", 0.03))   # ~can't touch a tool
# Blade-direction gate: the clip's jaws must point at tissue/vessel, not at a tool. Probe
# forward from the jaw tips and reject if that region is a bright tool, or sits markedly in
# FRONT of the clip's contact depth (a raised instrument -> catches dark instruments too).
BLADE_PROBE_LEN = int(os.environ.get("BLADE_PROBE_LEN", 110))    # px probed ahead of the tips
BLADE_CHROMA_MAX = int(os.environ.get("BLADE_CHROMA_MAX", 28))   # chroma below this = achromatic (tool)
BLADE_DARK_V = int(os.environ.get("BLADE_DARK_V", 45))           # brightness below this = dark instrument
BLADE_TOOL_RUN = int(os.environ.get("BLADE_TOOL_RUN", 8))        # reject if a blade ray crosses this many
                                                                 # consecutive tool px (a thin thread)


def large_tool_mask(bgr):
    """Surgical INSTRUMENT / gauze mask: BRIGHT achromatic (metal, gauze, cottonoid) OR DARK
    achromatic (black sutures / forceps in shadow). Tissue & vessels are chromatic (red/blue)
    -> excluded. Kept as LARGE blobs, and thin DARK structures only if elongated (a real
    instrument, not a shadow speck). Catches the dark instruments the old bright-only mask
    missed, so the clip can't cross them. (Colour heuristic -> best-effort, not perfect.)"""
    f = bgr.astype(np.int16); chroma = f.max(2) - f.min(2); v = bgr.max(2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); hue = hsv[..., 0]; sat = hsv[..., 1]
    bright = (chroma < 45) & (v > 115)                       # metal / gauze / cottonoid
    dark = (chroma < 28) & (v < 60)                          # dark instrument / suture
    yellow = (hue >= 15) & (hue <= 50) & (sat > 90) & (v > 110)   # yellow/gold forceps, retractors
    m = ((bright | dark | yellow).astype(np.uint8)) * 255    # (red tissue hue~0-10, blue vessels ~110 excluded)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros(m.shape, np.uint8); big = MIN_TOOL_AREA_FRAC * m.size
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        elong = max(w, h) / max(1, min(w, h))
        if a >= big and (elong > 2.0 or a >= 6 * big):       # elongated instrument OR big blob
            out[lab == i] = 255
    return cv2.dilate(out, np.ones((7, 7), np.uint8))

JAW_MAX_DEG, JAW_CLOSED_PROB, JAW_MIN_OPEN = 20.0, 0.35, 5.0

# the surface is built for the base view; widen its overscan so orbited views
# still find geometry out to their frame corners (no edge holes).
B.OVERSCAN = 0.65

if not os.path.isabs(BP_DIR):
    BP_DIR = os.path.join(HERE, BP_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
BACKPLATES = [f for f in os.listdir(BP_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
assert BACKPLATES, f"no backplates in {BP_DIR}"


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def orbit_cameras(base_cam, k, az_max, el_max, d_jit):
    """K cameras orbiting the focal point over a small baseline. View 0 = base."""
    pos, focal, up = [np.asarray(v, float) for v in base_cam]
    rel = pos - focal
    d0 = np.linalg.norm(rel)
    dir0 = rel / (d0 + 1e-12)
    up0 = _norm(up)
    right0 = _norm(np.cross(dir0, up0))
    cams = []
    for i in range(k):
        if i == 0:
            az = el = 0.0; dj = 0.0
        else:
            az = np.random.uniform(-az_max, az_max)
            el = np.random.uniform(-el_max, el_max)
            dj = np.random.uniform(-d_jit, d_jit)
        R_az = Rot.from_rotvec(np.deg2rad(az) * up0)
        R_el = Rot.from_rotvec(np.deg2rad(el) * right0)
        dirk = _norm(R_el.apply(R_az.apply(dir0)))
        posk = focal + dirk * d0 * (1.0 + dj)
        cams.append((tuple(posk), tuple(focal), tuple(up0)))
    return cams


def c2w_opengl(cam):
    """Camera-to-world 4x4 in OpenGL/NeRF convention (x right, y up, z back)."""
    pos, focal, up = [np.asarray(v, float) for v in cam]
    fwd = _norm(focal - pos)
    right = _norm(np.cross(fwd, up))
    up_cam = np.cross(right, fwd)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = up_cam
    M[:3, 2] = -fwd                     # OpenGL camera looks down -Z
    M[:3, 3] = pos
    return M


def save_depth(path_npy, path_png, dense):
    np.save(path_npy, dense)
    valid = np.isfinite(dense)
    viz = np.zeros(dense.shape, np.uint8)
    if valid.any():
        dmin, dmax = float(np.nanmin(dense)), float(np.nanmax(dense))
        viz[valid] = (255 * (1 - (dense[valid] - dmin) / max(dmax - dmin, 1e-9))).astype(np.uint8)
    cv2.imwrite(path_png, cv2.applyColorMap(viz, cv2.COLORMAP_TURBO))


def main():
    pipe = build_depth_pipeline(DA_MODEL)
    print(f"[mv] N_SCENES={N_SCENES} N_VIEWS={N_VIEWS} baseline az +-[{AZ_MIN},{AZ_MAX}] "
          f"el +-[{EL_MIN},{EL_MAX}] deg BP_DIR={BP_DIR}")

    plotter = pv.Plotter(off_screen=True, window_size=[W, H])
    plotter.camera.clipping_range = (ZNEAR, ZFAR)
    plotter.show(auto_close=False)
    setup_pbr_env(plotter)
    mesh = B.R.mesh

    seed = os.environ.get("SEED")
    if seed is not None:                          # reproducible scene/pose selection
        np.random.seed(int(seed)); random.seed(int(seed))

    scene = 0
    tries = 0
    while scene < N_SCENES and tries < N_SCENES * 20:
        tries += 1
        bp = random.choice(BACKPLATES)
        bgr = cv2.resize(trim_dark_border(cv2.imread(os.path.join(BP_DIR, bp), cv2.IMREAD_COLOR)), (W, H))
        disp = get_disparity(pipe, bp, bgr)

        # base camera + clip pose (reuse guarantees), then drop clip onto tissue
        base_cam = sample_camera_top()
        plotter.camera_position = base_cam
        fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, W, H)
        theta = 0.0 if np.random.rand() < JAW_CLOSED_PROB \
            else float(np.random.uniform(JAW_MIN_OPEN, JAW_MAX_DEG))
        clip_mesh = articulate(mesh, theta) if theta > 0.0 else mesh
        clip_verts = np.asarray(clip_mesh.points, float)
        avoid = large_tool_mask(bgr)                         # LARGE tool/gauze blobs (no sparkles)
        # Placement + ORIENTATION: sample several valid poses; keep those whose clip silhouette
        # is off a tool AND whose JAWS point toward the DEEPEST tissue (lowest disparity ahead)
        # -> down into the surgical corridor, spring up toward the scope, away from near tools.
        # Also pre-reject on the flat backplate if the blade ray crosses a tool. Then pick the
        # deepest-pointing candidate (with a little variety). disp: high=near, low=far/deep.
        cands = []
        for _ in range(POSE_DEPTH_TRIES):
            Rm_c, t_c, cam_c = sample_visible_pose_guaranteed(base_cam, fx, fy, cx, cy)
            if Rm_c is None:
                continue
            if clip_silhouette_avoid_fraction(Rm_c, t_c, cam_c, fx, fy, cx, cy,
                                              avoid, clip_verts) > TOOL_OVERLAP_MAX:
                continue                                     # clip would sit on a tool -> skip
            # blade direction in image (OpenGL c2w, matches saved views)
            w2c_c = np.linalg.inv(c2w_opengl(cam_c))
            Wv = (Rm_c @ clip_verts.T).T + t_c; Cw = Wv.mean(0)
            bw = Rm_c @ BLADE_DIR_MODEL; bw = bw / (np.linalg.norm(bw) + 1e-9)
            tw = Cw + bw * float(((Wv - Cw) @ bw).max())

            def _pj(P):
                Pc = w2c_c @ np.append(P, 1.0); Z = Pc[2]
                return np.array([fx * (Pc[0] / (-Z)) + cx, cy - fy * (Pc[1] / (-Z))])
            tv = _pj(tw); dv = tv - _pj(Cw); dnn = float(np.linalg.norm(dv))
            if dnn < 1e-3:
                continue
            di = dv / dnn
            # walk ahead of the jaws: gather tissue disparity (deep=low) + flat-backplate tool run
            da = []; run = 0; toolrun = 0
            for sd in range(8, 95, 3):
                p = tv + di * sd; x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < W and 0 <= y < H:
                    da.append(float(disp[y, x]))
                    px = bgr[y, x].astype(int); ch = int(px.max() - px.min()); vv = int(px.max())
                    if ch < BLADE_CHROMA_MAX or vv < BLADE_DARK_V or vv > 228:
                        run += 1; toolrun = max(toolrun, run)
                    else:
                        run = 0
            if toolrun >= BLADE_TOOL_RUN or not da:
                continue                                     # jaws point at a tool (flat pre-filter)
            cands.append((float(np.median(da)), Rm_c, t_c, cam_c))   # disp ahead of jaws (low=deep)
        if not cands:
            if os.environ.get("DEBUG_PLACE"): print(f"  [rej] no deep/tool-free candidate pose", flush=True)
            continue                                         # none point at deep tissue -> new backplate
        cands.sort(key=lambda c: c[0])                       # jaws point DEEPEST (lowest disp ahead) first
        _, Rm, t, base_cam = random.choice(cands[:max(1, len(cands) // 4)])
        plotter.camera_position = base_cam
        fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, W, H)
        centroid_w = Rm @ np.asarray(clip_mesh.center, float) + t
        z_clip = cam_space_depth(centroid_w, base_cam)
        if z_clip <= 1.0:
            continue
        amp = SURF_REL * z_clip
        Z0 = z_clip
        # embed_pct=0 -> the clip rests ON the tissue (lowest point tangent), nothing pressed
        # below the surface, so the WHOLE clip stays visible (no buried/hidden parts).
        t = drop_clip_onto_surface(Rm, t, clip_mesh, base_cam, fx, fy, cx, cy, disp, Z0, amp,
                                   embed_pct=0.0)

        # build the FIXED world scene: tissue surface (for the base view) + clip
        for a in list(plotter.renderer.actors.values()):
            plotter.remove_actor(a, render=False)
        plotter.renderer.remove_all_lights()
        # Reproduce render_clip_poses_depth EXACTLY (out_rand_depth / out_zoom_artic look):
        # a single warm headlight so the clip renders the SAME neutral-grey full-metal colour
        # that the SDXL restyle was tuned on. Speculars become view-locked, but every non-v0
        # clip is replaced by the geometric warp of v0's restyle, so only v0's clip colour
        # matters -- and it must match the pose renderer for the LoRA/inpaint to reproduce.
        plotter.add_light(pv.Light(light_type="headlight", intensity=LIGHT_INTENSITY,
                                   color=warm_light_color(bgr)))
        surf, tex = build_surface(bgr, disp, base_cam, fx, fy, cx, cy, Z0, amp)
        surf_actor = plotter.add_mesh(surf, texture=tex, **surface_add_kwargs())
        posed = actor_mesh(clip_mesh, Rm, t)
        clip_actor = plotter.add_mesh(posed, pbr=True, smooth_shading=True,
                                      reset_camera=False, **POSE_CLIP_MAT)   # pose-renderer material

        # VISIBILITY GUARANTEE: at the base view, the clip's full silhouette (clip alone) must
        # be almost entirely IN FRONT of the tissue (not occluded). Otherwise the tissue hides
        # part of the clip in the RGB -> reject and resample the scene.
        plotter.camera_position = base_cam
        plotter.camera.view_angle = 30.0
        plotter.render()
        full0 = metric_depth(plotter)                        # frontmost (clip + surface)
        surf_actor.SetVisibility(False)
        clip0 = metric_depth(plotter)                        # clip alone = full silhouette
        surf_actor.SetVisibility(True)
        full_sil = np.isfinite(clip0)
        vis_sil = full_sil & np.isfinite(full0) & (clip0 <= full0 + 1e-3)
        vis_frac = float(vis_sil.sum()) / float(max(1, full_sil.sum()))
        if vis_frac < VIS_MIN:
            if os.environ.get("DEBUG_PLACE"): print(f"  [rej] visibility {vis_frac:.2f} < {VIS_MIN}", flush=True)
            continue                                         # part of clip hidden by tissue

        # TOOL-OVERLAP gate in RENDERED space: the flat-backplate avoid mask does NOT align
        # with the warped/over-scanned tissue, so check on the actual rendered tissue. Render
        # tissue-only at the base view, detect instruments/gauze there, and reject if the
        # VISIBLE clip lands on one (> AVOID_OVERLAP_MAX).
        clip_actor.SetVisibility(False)
        plotter.render()                                     # refresh framebuffer BEFORE screenshot
        tissue_rgb = plotter.screenshot(return_img=True)     # (else it returns a stale clip-only frame)
        surf0 = metric_depth(plotter)                        # surface-only depth (clip hidden)
        clip_actor.SetVisibility(True)
        # Only judge tool overlap where the clip actually sits OVER tissue. The clip stands up
        # (embed=0), so much of its silhouette is against the empty background above the surface
        # -> those pixels are not on any tool; restricting to isfinite(surf0) excludes them.
        region = vis_sil & np.isfinite(surf0)
        avoid_r = large_tool_mask(cv2.cvtColor(tissue_rgb, cv2.COLOR_RGB2BGR)) > 127
        if os.environ.get("DEBUG_PLACE"):
            cv2.imwrite("_dbg_tissue.png", cv2.cvtColor(tissue_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite("_dbg_avoid.png", avoid_r.astype(np.uint8) * 255)
        tool_ov = float((avoid_r & region).sum()) / float(max(1, region.sum()))
        if tool_ov > TOOL_OVERLAP_MAX:
            if os.environ.get("DEBUG_PLACE"):
                print(f"  [rej] tool {tool_ov:.2f} region={int(region.sum())} "
                      f"surfcov={np.isfinite(surf0).mean():.2f} avoidcov={avoid_r.mean():.2f}", flush=True)
            continue                                         # clip sits on a tool/gauze

        # BLADE-DIRECTION gate: the jaws must point at tissue/vessel, not at a tool. Probe
        # ahead of the jaw tips (in the projected blade direction) and reject if that region is
        # a bright tool OR sits well in FRONT of the clip's contact depth (a raised instrument;
        # this depth test also catches DARK instruments that colour-based masks miss). Vessels
        # and tissue walls lie at surface depth -> not "near" -> allowed.
        # Project with the SAME OpenGL c2w that is written to transforms.json (matches the
        # saved v00 image exactly); the VTK cam_world_from_vtk convention gave a slightly
        # different image direction, making the probe miss tools.
        w2c_gl = np.linalg.inv(c2w_opengl(base_cam))
        Wv = (Rm @ clip_verts.T).T + t
        C = Wv.mean(0); bw = Rm @ BLADE_DIR_MODEL; bw = bw / (np.linalg.norm(bw) + 1e-9)
        tipw = C + bw * float(((Wv - C) @ bw).max())

        def _proj(P):
            Pc = w2c_gl @ np.append(P, 1.0); Z = Pc[2]
            return np.array([fx * (Pc[0] / (-Z)) + cx, cy - fy * (Pc[1] / (-Z))])
        ci, ti = _proj(C), _proj(tipw)
        dvec = ti - ci; dn = float(np.linalg.norm(dvec))
        if dn > 1e-3:
            di = dvec / dn; perp = np.array([-di[1], di[0]])
            tb = cv2.cvtColor(tissue_rgb, cv2.COLOR_RGB2BGR).astype(np.int16)   # tissue behind clip
            # Walk several parallel rays forward across the jaw width; a tool pixel is achromatic
            # (metal/gauze), dark (thin instrument/suture) or blown-out. The longest RUN of
            # consecutive tool pixels along any ray = the clip's jaws crossing/aimed at a tool
            # (even a THIN thread). Vessels/tissue-walls are chromatic mid-tones -> no run.
            maxrun = 0
            for lat in (-16, -8, 0, 8, 16):
                run = 0
                for sdist in range(4, BLADE_PROBE_LEN):
                    p = ti + di * sdist + perp * lat
                    x, y = int(round(p[0])), int(round(p[1]))
                    if 0 <= x < W and 0 <= y < H:
                        px = tb[y, x]; ch = int(px.max() - px.min()); v = int(px.max())
                        if ch < BLADE_CHROMA_MAX or v < BLADE_DARK_V or v > 228:
                            run += 1; maxrun = max(maxrun, run)
                        else:
                            run = 0
                    else:
                        run = 0
            if os.environ.get("DEBUG_PLACE"):
                dbg = cv2.cvtColor(tissue_rgb, cv2.COLOR_RGB2BGR).copy()
                for lat in (-16, -8, 0, 8, 16):
                    for sdist in range(4, BLADE_PROBE_LEN):
                        p = ti + di * sdist + perp * lat; x, y = int(round(p[0])), int(round(p[1]))
                        if 0 <= x < W and 0 <= y < H:
                            px = tb[y, x]; ch = int(px.max() - px.min()); v = int(px.max())
                            dbg[y, x] = (0, 0, 255) if (ch < BLADE_CHROMA_MAX or v < BLADE_DARK_V or v > 228) else (0, 255, 0)
                cv2.circle(dbg, (int(ti[0]), int(ti[1])), 6, (255, 0, 0), -1)
                cv2.imwrite("_gate_probe.png", dbg)
            if maxrun >= BLADE_TOOL_RUN:
                if os.environ.get("DEBUG_PLACE"):
                    print(f"  [rej] blades point at tool (maxrun={maxrun}px)", flush=True)
                continue

        if os.environ.get("DEBUG_PLACE"): print(f"  [ok] vis {vis_frac:.2f} tool {tool_ov:.2f}", flush=True)

        # --- render K orbit views ---
        sdir = os.path.join(OUT_DIR, f"scene_{scene:04d}")
        for sub in ("images", "depth", "masks"):
            os.makedirs(os.path.join(sdir, sub), exist_ok=True)
        az_max = np.random.uniform(AZ_MIN, AZ_MAX)
        el_max = np.random.uniform(EL_MIN, EL_MAX)
        # adaptive over-scan: extend the surface just enough that orbited views still
        # find geometry at their corners (a single backplate has no content beyond
        # its frame, so wide orbits otherwise open edge holes). Sized from the
        # baseline vs the horizontal half-FOV, with a 3 deg margin.
        fov_half = np.arctan(W / (2.0 * fx))
        need = 1.25 * np.tan(np.deg2rad(max(az_max, el_max) + 3.0)) / np.tan(fov_half)
        B.OVERSCAN = float(np.clip(need, 0.6, 0.95))   # <1 so REFLECT_101 padding is valid
        cams = orbit_cameras(base_cam, N_VIEWS, az_max, el_max, D_JIT)

        frames = []
        cover_min = 1.0
        for vi, cam in enumerate(cams):
            plotter.camera_position = cam
            plotter.camera.view_angle = 30.0
            plotter.render()
            rgb = plotter.screenshot(return_img=True)
            cv2.imwrite(os.path.join(sdir, "images", f"v{vi:02d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            dense = metric_depth(plotter)
            cover_min = min(cover_min, float(np.isfinite(dense).mean()))
            save_depth(os.path.join(sdir, "depth", f"v{vi:02d}.npy"),
                       os.path.join(sdir, "depth", f"v{vi:02d}.png"), dense)

            surf_actor.SetVisibility(False)
            clip_metric = metric_depth(plotter)
            surf_actor.SetVisibility(True)
            # VISIBLE clip silhouette only: clip pixels that are in front of the tissue (dense
            # = frontmost of clip+surface). Excludes any part occluded by tissue -> GT mask
            # matches what the RGB actually shows (no hidden parts claimed).
            visible = np.isfinite(clip_metric) & np.isfinite(dense) & (clip_metric <= dense + 1e-3)
            cv2.imwrite(os.path.join(sdir, "masks", f"v{vi:02d}.png"),
                        (visible.astype(np.uint8)) * 255)

            frames.append({
                "file_path": f"images/v{vi:02d}.png",
                "depth_file": f"depth/v{vi:02d}.npy",
                "mask_file": f"masks/v{vi:02d}.png",
                "transform_matrix": c2w_opengl(cam).tolist(),
            })

        transforms = {
            "w": W, "h": H, "fl_x": float(fx), "fl_y": float(fy),
            "cx": float(cx), "cy": float(cy),
            "camera_angle_x": float(2 * np.arctan(W / (2 * fx))),
            "convention": "OpenGL camera-to-world (x right, y up, z back); depth = metric camera-space distance, world units",
            "depth_note": "EXACT w.r.t. rendered meshes; tissue SHAPE from DepthAnything (2.5-D pseudo-anatomy)",
            "backplate_src": bp,
            "clip": {
                "pose_world": {
                    "quaternion_wxyz": _rotmat_to_quat_wxyz(Rm),
                    "translation_m": list(map(float, t)),
                },
                "jaw_angle_deg": float(theta),
            },
            "frames": frames,
        }
        with open(os.path.join(sdir, "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=2)

        scene += 1
        print(f"  scene {scene}/{N_SCENES}: {bp} views={N_VIEWS} jaw={theta:.0f} "
              f"az={az_max:.0f} el={el_max:.0f} min_depth_cover={cover_min*100:.1f}%")

    plotter.close()
    print(f"[mv] done: {scene} scenes x {N_VIEWS} views -> {OUT_DIR}")


if __name__ == "__main__":
    main()
