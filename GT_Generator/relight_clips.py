"""
relight_clips.py — re-render the clip layer so it matches the MACS look.

Two problems with the original composites are fixed here, both at render time
(where they actually originate, not in the diffusion stage):

  1. The clip ignores the scene light / casts no shadow -> we light it with a warm
     colour sampled from the tissue backplate, add the backplate as an environment
     texture so the metal REFLECTS the surgical field, and paint a contact shadow.
  2. The clip is too bright / chrome-like vs. real MACS clips -> we use a muted
     titanium material (rougher, less specular, darker base, lower light).

It reuses the existing 6-DoF poses from each NNN.json, so nothing is re-sampled and
all ground truth stays valid. Output: NNN_relit.png (clip composited over its
backplate) and NNN_relit_sbs.png (bg | relit) for comparison.

Run:
  python relight_clips.py --dir out_rand_depth --limit 5
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np
import pyvista as pv
try:                                   # laptop: Windows App Control blocks vtkInteractionWidgets.pyd (see pv_compat.py)
    import pv_compat; pv_compat.install()
except Exception:
    pass
from scipy.spatial.transform import Rotation as scipyR

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.environ.get("CLIP_MESH", os.path.join(HERE, "Item.obj"))   # env-overridable (must match the mesh the set was rendered with)

# ---- titanium-MACS material (tune these to match your real frames) ----
# Real MACS clips are mostly a MIRROR of the warm surgical field (reddish/golden
# reflections, dark bands), darker than chrome. So the look must come from the
# environment reflection (IBL), not from a strong white direct light.
MAT = dict(
    color=(0.50, 0.48, 0.46),   # muted warm-grey titanium (between chrome and black)
    metallic=1.0,
    roughness=0.50,
    specular=0.30,
    diffuse=0.70,
    ambient=0.22,
)
LIGHT_INTENSITY = 8.0           # moderate; IBL adds the warm tint on top
SHADOW_STRENGTH = 0.45          # 0..1 darkening under the clip
SHADOW_SHIFT = (4, 7)          # px (x,y) offset of the contact shadow
SHADOW_BLUR = 15


def quat_wxyz_to_R(q):
    return scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def actor_mesh(mesh, R, t):
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    m.transform(T, inplace=True)
    return m


def warm_light_color(bgr):
    """Mean colour of the brighter tissue pixels -> the scene's light tint (RGB)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(g, 70)
    sel = g >= thr
    mean_bgr = bgr[sel].mean(axis=0) / 255.0
    rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
    # keep it warm but not fully saturated
    return tuple(0.5 + 0.5 * c for c in rgb)


def make_env_texture(bg_rgb):
    """Approximate environment from the backplate (6 identical cubemap faces)."""
    try:
        return pv.Texture([bg_rgb] * 6)
    except Exception:
        return None


def contact_shadow(bg_bgr, clip_mask):
    """Darken the tissue just under/around the clip to ground it."""
    m = (clip_mask > 127).astype(np.float32)
    M = np.float32([[1, 0, SHADOW_SHIFT[0]], [0, 1, SHADOW_SHIFT[1]]])
    shifted = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
    k = SHADOW_BLUR * 2 + 1
    soft = cv2.GaussianBlur(shifted, (k, k), 0)
    soft = np.clip(soft - m, 0, 1)                 # shadow only outside the clip
    factor = (1.0 - SHADOW_STRENGTH * soft)[..., None]
    return (bg_bgr.astype(np.float32) * factor).astype(np.uint8)


def relight_one(plotter, mesh, meta, d, stem):
    cam = meta["camera"]
    pw = meta["clip"]["pose_world"]
    R = quat_wxyz_to_R(pw["quaternion_wxyz"])
    t = np.array(pw["translation_m"], dtype=float)

    bg_path = os.path.join(d, meta["image"]["file_bg"])
    mask_path = os.path.join(d, meta["image"]["file_mask"])
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    clip_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if bg is None or clip_mask is None:
        return None
    h, w = bg.shape[:2]
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    plotter.clear()
    plotter.set_background("black")
    plotter.camera.clipping_range = (cam["znear"], cam["zfar"])

    env = make_env_texture(bg_rgb)
    env_ok = False
    if env is not None:
        try:
            plotter.set_environment_texture(env)
            try:
                plotter.renderer.UseImageBasedLightingOn()
                plotter.renderer.SetEnvironmentTexture(env)
            except Exception:
                pass
            env_ok = True
        except Exception as e:
            print(f"   [env texture failed: {e}]")
    relight_one.env_ok = env_ok
    plotter.add_light(pv.Light(light_type="headlight",
                               intensity=LIGHT_INTENSITY, color=warm_light_color(bg)))

    posed = actor_mesh(mesh, R, t)
    plotter.add_mesh(posed, pbr=True, smooth_shading=True, reset_camera=False, **MAT)
    plotter.camera_position = (tuple(cam["position"]), tuple(cam["focal_point"]),
                               tuple(cam["view_up"]))
    plotter.render()
    clip_rgb = plotter.screenshot(transparent_background=False, return_img=True)
    clip_bgr = cv2.cvtColor(clip_rgb, cv2.COLOR_RGB2BGR)

    # composite: contact shadow on tissue, then clip pixels via the silhouette
    bg_shadowed = contact_shadow(bg, clip_mask)
    alpha = cv2.GaussianBlur((clip_mask > 127).astype(np.float32), (3, 3), 0)[..., None]
    out = (alpha * clip_bgr.astype(np.float32) +
           (1 - alpha) * bg_shadowed.astype(np.float32)).astype(np.uint8)

    out_path = os.path.join(d, f"{stem}_relit.png")
    cv2.imwrite(out_path, out)
    cv2.imwrite(os.path.join(d, f"{stem}_relit_sbs.png"), np.concatenate([bg, out], axis=1))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_rand_depth")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    work = args.dir if os.path.isabs(args.dir) else os.path.join(HERE, args.dir)
    jsons = sorted(glob.glob(os.path.join(work, "[0-9]*.json")))[args.start_index:]
    if args.limit is not None:
        jsons = jsons[:args.limit]
    if not jsons:
        print(f"No samples in {work}")
        return

    mesh = pv.read(MESH_PATH)
    plotter = pv.Plotter(off_screen=True, window_size=[800, 544])
    # one show() call to initialise the off-screen render window
    plotter.show(auto_close=False)

    n = 0
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp) as f:
            meta = json.load(f)
        if relight_one(plotter, mesh, meta, work, stem):
            n += 1
            print(f"[OK] {stem}_relit.png")
    plotter.close()
    print(f"Done. {n} relit composites written to {work}")


if __name__ == "__main__":
    main()
