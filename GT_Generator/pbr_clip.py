"""
pbr_clip.py — realistic PBR titanium material + surgical environment map for the clip.

A metallic surface in a PBR renderer reflects its ENVIRONMENT; with no environment
map it just goes dark (why the old clip looked grey/flat). This builds a small
procedural cube-map that mimics a surgical microscope scene — a strong warm coaxial
light overhead for sharp speculars, reddish tissue tones around, dark below — so the
clip renders as real blue-anodized titanium, consistently across all views (the env
is fixed in world space, so speculars are view-dependent but coherent).

Use:
    from pbr_clip import set_pbr_environment, PBR_CLIP_MAT
    set_pbr_environment(plotter)
    plotter.add_mesh(clip, pbr=True, **PBR_CLIP_MAT)
"""

import os
import cv2
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))

# blue-anodized titanium: bluish albedo, full metal, low-ish roughness for sharp glints
PBR_CLIP_MAT = dict(color=(0.28, 0.42, 0.62), metallic=1.0, roughness=0.30)


def _face(sz, base, top_boost=0.0, hotspot=None):
    """One cube face: warm base color, brighter toward the top, optional bright hotspot."""
    img = np.zeros((sz, sz, 3), np.float32)
    grad = np.linspace(1.0 + top_boost, 0.35, sz)[:, None]        # bright top -> dark bottom
    img[:] = (np.array(base, np.float32)[None, None, :] * grad[..., None])
    if hotspot is not None:
        cy, cx, r, inten = hotspot
        yy, xx = np.mgrid[0:sz, 0:sz]
        d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / float(r * r)
        img += inten * np.exp(-d2)[..., None]                     # soft bright light blob
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def make_env_cubemap(sz=256):
    """6-face surgical-microscope cube-map, written to _envmap/ and loaded as a
    cubemap texture (order posx,negx,posy,negy,posz,negz)."""
    warm = (0.55, 0.32, 0.30)        # reddish tissue reflection (RGB)
    top = _face(sz, (0.95, 0.85, 0.78), top_boost=0.4,
                hotspot=(sz * 0.42, sz * 0.5, sz * 0.16, 1.6))   # coaxial light + hotspot
    bottom = _face(sz, (0.12, 0.08, 0.08))                        # dark below
    side = lambda: _face(sz, warm, top_boost=0.25)
    faces = dict(posx=side(), negx=side(), posy=top, negy=bottom, posz=side(), negz=side())
    d = os.path.join(HERE, "_envmap")
    os.makedirs(d, exist_ok=True)
    paths = []
    for name in ("posx", "negx", "posy", "negy", "posz", "negz"):
        p = os.path.join(d, f"{name}.png")
        cv2.imwrite(p, cv2.cvtColor(faces[name], cv2.COLOR_RGB2BGR))
        paths.append(p)
    return pv.cubemap_from_filenames(paths)


def set_pbr_environment(plotter, sz=256):
    """Attach the surgical env cube-map so PBR metals reflect it."""
    try:
        plotter.set_environment_texture(make_env_cubemap(sz))
        return True
    except Exception as e:
        print(f"[pbr_clip] env texture failed ({e}); metal will be dimmer")
        return False


if __name__ == "__main__":
    import cv2, os
    here = os.path.dirname(os.path.abspath(__file__))
    clip = pv.read(os.path.join(here, "Item.obj"))
    p = pv.Plotter(off_screen=True, window_size=[600, 400])
    import glob
    bg = cv2.imread(sorted(glob.glob(os.path.join(here, "Data", "noclips_bank", "bp_*.png")))[0])
    if bg is not None:
        cv2.imwrite(os.path.join(here, "_bgtmp.png"), cv2.resize(bg, (600, 400)))
        p.add_background_image(os.path.join(here, "_bgtmp.png"))
    set_pbr_environment(p)
    p.add_light(pv.Light(light_type="scene light", position=(40, 60, 40), intensity=0.6))
    p.add_mesh(clip, pbr=True, smooth_shading=True, **PBR_CLIP_MAT)
    p.camera_position = "xy"
    p.show(auto_close=False)
    img = p.screenshot(return_img=True)
    cv2.imwrite(os.path.join(here, "_pbr_test.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote _pbr_test.png")
