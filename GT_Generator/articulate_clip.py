"""
articulate_clip.py — open/close the jaws of the fused (closed) aneurysm-clip mesh.

The clip is a single rigid mesh, so there is no hinge to rotate. We synthesize the
opening as a length-weighted fan deformation in the PCA frame:

  * e0 = long axis (blade length), e1 = width, e2 = thinnest (jaws stack along it)
  * each vertex is assigned to a blade by sign(p2) (which side of the mid-plane)
  * it is rotated about the e1 hinge axis in the e0-e2 plane by  s * (theta/2) * w,
    where w grows 0 -> 1 from the hinge end to the tip.

Because w -> 0 at the hinge, the spring/hinge region barely moves (no tearing) while
the blade tips fan apart by `theta` degrees — a clamping(0) .. open(theta) clip.

API:  articulate(polydata, theta_deg, hinge='low') -> new pyvista.PolyData
CLI:  python articulate_clip.py   # renders previews at 0/15/30 deg to _articulate_preview
"""

import os
import json
import numpy as np
import pyvista as pv

MESH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Item.obj")


# ---- rigid-hinge mode (true pivot clips, e.g. the Lazic L-Clip from make_lclip_mesh.py) ----------
# A sidecar <mesh stem>.parts.json (pivot, axis, per-part vertex indices + rotation sign) switches
# articulate() from the fan deformation to rigid rotations of each clip part about the pin axis.
# It is found through CLIP_PARTS, or next to the mesh named by CLIP_MESH. Item.obj has none.
_PARTS_CACHE = {}


def _rigid_parts(n_points):
    path = os.environ.get("CLIP_PARTS")
    if not path and os.environ.get("CLIP_MESH"):
        path = os.path.splitext(os.environ["CLIP_MESH"])[0] + ".parts.json"
    if not path or not os.path.exists(path):
        return None
    if path not in _PARTS_CACHE:
        with open(path) as fh:
            _PARTS_CACHE[path] = json.load(fh)
    parts = _PARTS_CACHE[path]
    return parts if parts.get("n_points") == n_points else None


def _articulate_rigid(mesh, theta_deg, parts):
    out = mesh.copy()
    V = out.points.astype(np.float64)
    pivot = np.asarray(parts["pivot"], float)
    k = np.asarray(parts["axis"], float)
    k /= np.linalg.norm(k)
    for part in parts["parts"]:
        idx = np.asarray(part["indices"], int)
        ang = np.deg2rad(theta_deg) * 0.5 * float(part["sign"])
        p = V[idx] - pivot
        V[idx] = pivot + p * np.cos(ang) + np.cross(k, p) * np.sin(ang) + np.outer(p @ k, k) * (1.0 - np.cos(ang))
    out.points = V.astype(np.float32)
    return out


def pca_frame(V):
    c = V.mean(0)
    Vc = V - c
    cov = np.cov(Vc.T)
    w, vec = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    e = vec[:, order]                       # columns: e0 long, e1 mid, e2 thin
    return c, e


def articulate(mesh, theta_deg, hinge="low", pivot_frac=0.38, blend=0.12, hollow=True):
    """Return a copy of `mesh` with its jaws opened by theta_deg (0 = closed/clamping).

    The two blades open like rigid tweezers: every blade vertex rotates by the SAME
    angle (+/- theta/2 by blade) about the hinge, so the blades stay straight and
    part with a clean V-gap (not a curved trumpet). A smoothstep over `blend` near
    the hinge keeps the spring rigid so the coil never tears.

      * blades = the high-p0 end (hinge='low'); spring = the low-p0 end
      * pivot_frac = where along the long axis the hinge sits (0=spring tip .. 1=jaw tip)
      * hollow: the closed mesh fuses the two blades at the mid-plane; rotating them
        apart would stretch those bridging faces into a solid web across the gap.
        With hollow=True those bridging faces (mixed-blade, in the opened region) are
        deleted so the opening is empty, not filled with clip material.
    """
    parts = _rigid_parts(mesh.n_points)
    if parts is not None:                             # true-hinge clip: rigid rotation of each part
        return _articulate_rigid(mesh, theta_deg, parts)
    out = mesh.copy()
    if hollow and theta_deg > 0:
        out = out.triangulate()                       # uniform triangles for face culling
    V = out.points.astype(np.float64)
    c, e = pca_frame(V)
    Vc = V - c
    p0 = Vc @ e[:, 0]
    p1 = Vc @ e[:, 1]
    p2 = Vc @ e[:, 2]

    span = p0.max() - p0.min()
    p0n = (p0 - p0.min()) / (span + 1e-12)            # 0 at low end .. 1 at high end
    frac = p0n if hinge == "low" else (1.0 - p0n)     # 0 at the spring side
    pivot_p0 = (p0.min() + pivot_frac * span) if hinge == "low" \
        else (p0.max() - pivot_frac * span)

    # rigid-blade weight: 0 over the spring, smoothstep up to 1 over the blades
    w = np.clip((frac - pivot_frac) / blend, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)

    # The clip is flat (thin along e2 = the plane normal); its jaws open IN-PLANE,
    # separating along e1 and rotating about the e2 normal — i.e. in the spring's
    # plane (horizontal), not out of plane (vertical).
    s = np.sign(p1 - np.median(p1))                   # which blade (in-plane side)
    s[s == 0] = 1.0

    ang = np.deg2rad(theta_deg) * 0.5 * s * w         # rotation about the e2 (normal) axis
    a = p0 - pivot_p0                                  # along e0 from the hinge
    b = p1                                             # along e1 (in-plane opening direction)
    ca, sa = np.cos(ang), np.sin(ang)
    p0n2 = pivot_p0 + (a * ca - b * sa)
    p1n2 = a * sa + b * ca

    Vn = c + np.outer(p0n2, e[:, 0]) + np.outer(p1n2, e[:, 1]) + np.outer(p2, e[:, 2])
    out.points = Vn.astype(np.float32)

    if hollow and theta_deg > 0:
        # drop faces that bridge the two blades within the opened region (mixed
        # blade sign AND past the hinge) -> the V opening becomes empty
        tri = out.faces.reshape(-1, 4)[:, 1:4]
        sv = s[tri]
        wv = w[tri]
        bridges = (sv.min(axis=1) < 0) & (sv.max(axis=1) > 0) & (wv.min(axis=1) > 0.5)
        if bridges.any():
            kept = out.faces.reshape(-1, 4)[~bridges]
            out.cell_data.clear()        # per-cell arrays (e.g. the OBJ reader's GroupIds) no longer match
            out.faces = kept.ravel()
    return out


def _preview():
    out_dir = os.path.join(os.path.dirname(MESH_PATH), "_articulate_preview")
    os.makedirs(out_dir, exist_ok=True)
    base = pv.read(MESH_PATH)
    print("loaded", base.n_points, "points")
    # face-on to the opening plane = look along e1 (the width axis)
    _, e = pca_frame(base.points.astype(np.float64))
    view_e1 = tuple(e[:, 1])
    for hinge in ("low", "high"):
        for theta in (0, 25):
            m = articulate(base, theta, hinge=hinge)
            pl = pv.Plotter(off_screen=True, window_size=(600, 360))
            pl.add_mesh(m, color=(0.55, 0.55, 0.6), specular=0.6, smooth_shading=True)
            pl.view_vector(view_e1)                  # face-on to the jaw-opening plane
            pl.set_background("black")
            png = os.path.join(out_dir, f"face_{hinge}_{theta:02d}.png")
            pl.screenshot(png)
            pl.close()
            print("wrote", png)


if __name__ == "__main__":
    _preview()
