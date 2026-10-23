"""
make_clip_mesh.py — parametric flat-ribbon titanium aneurysm clip (Aesculap YASARGIL type),
drawn from the manufacturer's spec sheet.

Why: the clips in the MACS videos are permanent YASARGIL titanium clips (IFU: "can be identified
by their colour: blue"). Item.obj is a round-wire clip with a big open coil and sugita_clip.obj
is also a round-wire model, so neither matches the real silhouette. No downloadable CAD of these
clips exists, hence this generator.

Dimensions come from the Aesculap brochure "YASARGIL Aneurysm Clip System":
  * p.6 table + 1:1 vector side-view drawings (checked: the length differences between FT740T,
    FT750T, FT760T, FT780D, FT790D equal exactly 2.0/2.0/4.0/5.0 mm at 72/25.4 pt per mm):
      total length = L + 10.83 mm, box lock = L+1.4 .. L+3.4, blade stock ends at L+3.6,
      spring stock (blue) from L+4.56, blade height 1.35 mm, spring stock height 0.94 mm,
      the two coil ends stacked at the far end.
  * p.5 plan-view diagram (scaled by its own L bracket; its total/L ratio = FT750T):
      blade thickness 0.5 mm at the tip -> 1.0 mm at L (per blade), legs cross at ~L+2.9 inside
      the box lock, legs run at |y| ~ 2.0 and join the coil tangentially, coil centre at L+8.57,
      coil OD 4.38 mm (outer r 2.19, inner r 1.28 -> ribbon ~0.9 radial).
  Default L = 9.0 mm (FT750T, max opening 7.0 mm). --blade 7 / 11 / 15 give FT740T / FT760T / FT780D.

Model frame (PCA-compatible with articulate_clip.py / define_keypoints.py):
  x = long axis (jaw tips at +x, coil at -x), y = in-plane width (the blades part along y),
  z = thin normal (coil axis). Units: mm (same convention as Item.obj).
The closed blades sit at y = +-blade_gap/2 (contact faces); articulate() opens them in-plane.
NOTE: the finished mesh may be rotated 180 deg about z (or y) so that pca_frame() puts the jaws
at the high-p0 end, as the renderer/trainer assume (see orient_for_pipeline).

Usage:
  python make_clip_mesh.py                       -> clip_ti_std.obj (+ _clip_mesh_preview.png)
  python make_clip_mesh.py --blade 7 --out clip_ti_740.obj
  python make_clip_mesh.py --no-preview
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def catmull_rom(points, n_per_seg=24):
    """Uniform Catmull-Rom spline through `points`, with clamped end tangents."""
    P = np.asarray(points, float)
    P2 = np.vstack([P[0] - (P[1] - P[0]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    t = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)[:, None]
    for i in range(1, len(P2) - 2):
        p0, p1, p2, p3 = P2[i - 1], P2[i], P2[i + 1], P2[i + 2]
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        out.append(0.5 * (a + b * t + c * t ** 2 + d * t ** 3))
    out.append(P[-1][None])
    return np.vstack(out)


def resample(xy, step):
    """Resample a 2-D polyline at ~constant arc length `step`."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = max(2, int(np.ceil(d[-1] / step)) + 1)
    u = np.linspace(0.0, d[-1], n)
    return np.c_[np.interp(u, d, xy[:, 0]), np.interp(u, d, xy[:, 1])]


# ------------------------------------------------------------------ geometry
def default_params(blade_len=9.0):
    return dict(
        blade_len=blade_len,           # L in the Aesculap table
        blade_t_tip=0.50,              # in-plane (y) thickness of each blade at the tip   (plan view)
        blade_t_base=1.00,             # ... and at x = L                                   (plan view)
        blade_h=1.35,                  # blade height (z)                                   (1:1 side view)
        blade_gap=0.15,                # y gap between the closed blades (~1 px at MACS scale; keeps
                                       # articulate()'s blade assignment unambiguous)
        tip_round=0.675,               # semicircular tip in the side view
        box_start=1.4, box_len=2.0,    # box lock at L+1.4 .. L+3.4                         (1:1 side view)
        box_plate=0.3,                 # box lock plate thickness (brochure p.5)
        spring_start=4.56,             # spring stock (0.94 high) from L+4.56               (1:1 side view)
        leg_t=1.05, leg_w=0.94,        # spring stock: in-plane thickness (plan view 1.0-1.1) / height (side view)
        leg_y=2.10,                    # |y| of the parallel leg section (edges 1.6..2.6)   (plan view)
        cross_x=2.4,                   # the legs cross at L+2.4, inside the box lock       (plan view)
        flare_end=4.2,                 # legs reach leg_y by L+4.2                          (plan view)
        coil_dx=8.57,                  # coil centre at L+8.57                              (plan view)
        coil_r=1.75,                   # coil centreline radius (OD 4.38 -> far end at L+10.8)
        coil_turns=1.5,                # "1.5 spring coil" (brochure p.3)
        coil_pitch=0.63,               # z advance per turn: the side view shows the two coil ends stacked
                                       # 1.88 mm high in total, i.e. 1.5 turns climb only ~0.94 mm; the
                                       # turns therefore interpenetrate slightly (0.3 mm) - harmless for
                                       # rendering / masks / PCA, which is all this mesh is used for
        attach_ramp=4.0,               # mm before the coil over which a leg climbs to its helix end height
                                       # (side view: the legs start separating in z right after L+4.56)
        step=0.07,                     # sweep sampling (mm)
        chamfer=0.10,                  # edge chamfer of the ribbon cross-section
    )


def centerline(P):
    """Single continuous ribbon path in the DRAWING frame (tip at x = 0, coil toward +x):
    blade A tip -> box -> leg A -> coil -> leg B -> box -> blade B tip.
    Returns xy (N,2), z (N,), w (N,), t (N,) and a segment id per sample."""
    L, g = P["blade_len"], P["blade_gap"] / 2
    xc, R = L + P["coil_dx"], P["coil_r"]
    tt, tb = P["blade_t_tip"], P["blade_t_base"]
    yA_tip, yA_base = g + tt / 2, g + tb / 2            # blade A centreline (y > 0)
    zspan = P["coil_pitch"] * P["coil_turns"]
    zA, zB = -zspan / 2, +zspan / 2                      # helix end heights (leg A end, leg B start)

    legA_pts = [(L, yA_base), (L + 0.7, 0.8 * yA_base), (L + P["cross_x"], 0.0), (L + 3.4, -1.1),
                (L + P["flare_end"], -P["leg_y"] + 0.1), (L + 5.2, -P["leg_y"]), (xc - 1.9, -P["leg_y"]),
                (xc - 0.7, -R - 0.08), (xc, -R)]
    legB_pts = [(x, -y) for (x, y) in legA_pts][::-1]   # mirror in y, coil -> blade

    segs = []
    nb = max(2, int(L / P["step"]))
    xb = np.linspace(0.0, L, nb)
    segs.append(("bladeA", np.c_[xb, yA_tip + (yA_base - yA_tip) * xb / L]))
    segs.append(("legA", catmull_rom(legA_pts)))
    u = np.linspace(0.0, 1.0, 256)                       # CCW from the bottom (-90 deg): entered travelling +x
    a = np.deg2rad(-90.0) + u * 2 * np.pi * P["coil_turns"]
    segs.append(("coil", np.c_[xc + R * np.cos(a), R * np.sin(a)]))
    segs.append(("legB", catmull_rom(legB_pts)))
    xb2 = xb[::-1]
    segs.append(("bladeB", np.c_[xb2, -(yA_tip + (yA_base - yA_tip) * xb2 / L)]))

    H, hw = P["blade_h"], P["blade_h"] / 2 - 0.03        # full / half (stacked at the crossing) height
    x_box0, x_box1 = L + P["box_start"], L + P["box_start"] + P["box_len"]
    xy, sid, zz, ww, tt_ = [], [], [], [], []
    for name, pts in segs:
        pts = resample(pts, P["step"])
        n = len(pts)
        x = pts[:, 0]
        if name in ("bladeA", "bladeB"):
            t = tt + (tb - tt) * np.clip(x / L, 0, 1)
            r = np.clip((P["tip_round"] - x) / P["tip_round"], 0.0, 1.0)
            w = H * np.sqrt(np.clip(1.0 - r ** 2, 0.03, 1.0))
            z = np.zeros(n)
        elif name in ("legA", "legB"):
            sgn = 1.0 if name == "legA" else -1.0
            # height: blade stock -> half height through the box lock (the two arms are stacked) -> spring stock
            s_in = smoothstep(L + 0.8, x_box0, x)
            s_out = smoothstep(x_box1, L + P["spring_start"], x)
            w = np.where(x < L + P["cross_x"], H + (hw - H) * s_in, hw + (P["leg_w"] - hw) * s_out)
            t = tb + (P["leg_t"] - tb) * smoothstep(L, x_box0, x)
            # z: stacked arms inside the box lock, back to the mid-plane, then up/down to the helix end
            z_box = sgn * (hw / 2 + 0.02) * s_in * (1.0 - smoothstep(x_box1, L + P["flare_end"], x))
            z_end = (zA if name == "legA" else zB) * smoothstep(xc - P["attach_ramp"], xc, x)
            z = z_box + z_end
        else:  # coil
            w = np.full(n, P["leg_w"])
            t = np.full(n, P["leg_t"])
            z = np.linspace(zA, zB, n)
        xy.append(pts)
        zz.append(z)
        ww.append(w)
        tt_.append(t)
        sid += [name] * n
    xy = np.vstack(xy)
    z = np.concatenate(zz)
    w = np.concatenate(ww)
    t = np.concatenate(tt_)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]   # drop duplicated joints
    return xy[keep], z[keep], w[keep], t[keep], np.asarray(sid)[keep]


def sweep(xy, z, w, t, chamfer):
    """Sweep a chamfered rectangle (thickness t along the in-plane normal, height w along z)."""
    n = len(xy)
    d = np.gradient(xy, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[-d[:, 1], d[:, 0]]                    # in-plane normal
    C = np.c_[xy, z]
    N3 = np.c_[nrm, np.zeros(n)]
    Z3 = np.tile([0.0, 0.0, 1.0], (n, 1))
    c = np.minimum(chamfer, 0.3 * np.minimum(w, t))[:, None]
    ht, hw = (t / 2)[:, None], (w / 2)[:, None]
    # 8-gon cross-section (counter-clockwise), corners cut by c: (sign_n, sign_z, dn, dz)
    ring = [(+1, +1, -1, 0), (+1, +1, 0, -1), (+1, -1, 0, +1), (+1, -1, -1, 0),
            (-1, -1, +1, 0), (-1, -1, 0, +1), (-1, +1, 0, -1), (-1, +1, +1, 0)]
    V = []
    for sn, sz, dn, dz in ring:
        V.append(C + N3 * (sn * ht + dn * c) + Z3 * (sz * hw + dz * c))
    V = np.stack(V, axis=1)                           # (n, 8, 3)
    verts = V.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for j in range(8):
            a, b = i * 8 + j, i * 8 + (j + 1) % 8
            c2, d2 = (i + 1) * 8 + j, (i + 1) * 8 + (j + 1) % 8
            F.append([a, b, d2])
            F.append([a, d2, c2])
    for i, flip in ((0, False), (n - 1, True)):      # caps (fan) at both ends
        base = i * 8
        for j in range(1, 7):
            tri = [base, base + j, base + j + 1]
            F.append(tri[::-1] if flip else tri)
    return verts, np.asarray(F, int)


def box(xmin, xmax, ymin, ymax, zmin, zmax):
    v = np.array([[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], int)
    return v, f


def drop_degenerate(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return faces[(area2 > 1e-9) & distinct]


def build(P, return_ids=False):
    xy, z, w, t, sid = centerline(P)
    v, f = sweep(xy, z, w, t, P["chamfer"])
    L = P["blade_len"]
    # box lock: a thin sleeve around the crossing arms, right behind the jaws
    pl, H = P["box_plate"], P["blade_h"]
    x0, x1 = L + P["box_start"], L + P["box_start"] + P["box_len"]
    inside = (xy[:, 0] > x0) & (xy[:, 0] < x1) & np.isin(sid, ["legA", "legB"])
    yhalf = (np.abs(xy[inside, 1]) + t[inside] / 2).max() if inside.any() else 1.0
    vb, fb = box(x0, x1, -(yhalf + pl), yhalf + pl, -(H / 2 + pl), H / 2 + pl)
    verts = np.vstack([v, vb])
    faces = np.vstack([f, fb + len(v)])
    faces = drop_degenerate(verts, faces)
    ids = np.r_[np.repeat(sid, 8), np.full(len(vb), "box")]
    jaw = int(np.flatnonzero(ids == "bladeA")[np.argmin(verts[ids == "bladeA", 0])])   # the tip is at x = 0
    verts = orient_for_pipeline(verts, jaw)
    return (verts, faces, ids) if return_ids else (verts, faces)


def orient_for_pipeline(verts, jaw_idx):
    """The generator/trainer take the jaws to be at the HIGH end of PCA axis e0 (articulate
    hinge='low'). numpy's eigh fixes the eigenvector sign arbitrarily, so try 180-deg rotations
    (first about z, which also puts the jaws at +x) until the jaw tip lands at p0 > 0. Pure
    rotations: the geometry is unchanged, only the model-frame axes flip sign."""
    sys.path.insert(0, HERE)
    from articulate_clip import pca_frame
    for R in (np.diag([-1.0, -1.0, 1.0]), np.eye(3), np.diag([-1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0])):
        W = verts @ R.T
        c, e = pca_frame(W)
        if ((W[jaw_idx] - c) @ e[:, 0]) > 0:
            return W
    return verts


def write_obj(path, verts, faces, header=""):
    with open(path, "w") as fh:
        fh.write("# " + header.replace("\n", "\n# ") + "\n")
        for p in verts:
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for tri in faces + 1:
            fh.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blade", type=float, default=9.0, help="blade length L in mm (Aesculap table)")
    ap.add_argument("--out", default="clip_ti_std.obj")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    P = default_params(a.blade)
    verts, faces, ids = build(P, return_ids=True)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    write_obj(out, verts, faces,
              f"YASARGIL-type flat-ribbon titanium aneurysm clip, blade L = {P['blade_len']} mm, units mm\n"
              f"generated by make_clip_mesh.py from the Aesculap spec sheet (total length L+10.83 mm);\n"
              f"x = long axis (+x jaw tips), y = in-plane width, z = coil axis")
    ext = verts.max(0) - verts.min(0)
    print(f"wrote {out}: {len(verts)} verts, {len(faces)} tris; extents x/y/z = {np.round(ext, 2)} mm "
          f"(spec: total {P['blade_len'] + 10.83:.2f}, coil OD {2 * (P['coil_r'] + P['leg_t'] / 2):.2f})")

    # PCA + articulation sanity check (same code the generator and the trainer use)
    import pyvista as pv
    sys.path.insert(0, HERE)
    from articulate_clip import articulate, pca_frame
    m = pv.read(out)
    if m.n_cells != len(faces):
        print(f"WARNING: pyvista read {m.n_cells} cells but {len(faces)} faces were written")
    V = np.asarray(m.points, float)
    c, e = pca_frame(V)
    Pp = (V - c) @ e
    pe = Pp.max(0) - Pp.min(0)
    blade = np.flatnonzero(np.char.startswith(ids.astype(str), "blade"))
    jaw_idx = int(blade[np.argmax(np.abs(V[blade, 0]))])
    tips_hi = Pp[jaw_idx, 0] > 0
    print(f"PCA extents e0/e1/e2 = {np.round(pe, 2)}  |e0.x|={abs(e[0, 0]):.3f} |e2.z|={abs(e[2, 2]):.3f}  "
          f"jaws at high-p0 end: {tips_hi} (articulate/renderer convention requires True)")
    m20 = articulate(m, 20.0, hollow=False)
    D = (np.asarray(m20.points) - V) @ e
    sep = int(np.argmax(D.std(axis=0)[1:]) + 1)
    tips = Pp[:, 0] > Pp[:, 0].max() - 0.5
    gap = np.ptp(D[tips, sep])
    print(f"articulate(20 deg): blades separate along e{sep}; tip separation {gap:.2f} mm")
    s_side = np.sign(Pp[:, 1] - np.median(Pp[:, 1]))
    a_side, b_side = s_side[ids == "bladeA"], s_side[ids == "bladeB"]
    bad = min((a_side < 0).sum() + (b_side > 0).sum(), (a_side > 0).sum() + (b_side < 0).sum())
    tilt = np.degrees(np.arccos(min(1.0, abs(e[1, 1]))))
    print(f"PCA e1 vs model y tilt = {tilt:.1f} deg; blade vertices mis-assigned by articulate(): {int(bad)} (must be 0)")

    if a.no_preview:
        return
    from PIL import Image, ImageDraw
    pv.OFF_SCREEN = True
    T = 520
    tiles = []
    for label, mesh in (("closed", m), ("open 20 deg", m20)):
        mp = pv.PolyData((np.asarray(mesh.points) - c) @ e, mesh.faces).compute_normals(auto_orient_normals=True)
        for vname, pos, up in (("top", (0, 0, 1), (0, 1, 0)), ("side", (0, 1, 0), (0, 0, 1)), ("3/4", (0.5, 0.9, 1.0), (0, 0, 1))):
            pl = pv.Plotter(off_screen=True, window_size=(T, T))
            pl.set_background("white")
            pl.add_mesh(mp, color=(0.30, 0.42, 0.80), smooth_shading=False, specular=0.8, specular_power=40)
            pl.camera_position = [np.array(pos, float) * 100.0, (0, 0, 0), up]
            pl.camera.parallel_projection = True
            pl.reset_camera()
            pl.camera.zoom(0.95)
            img = Image.fromarray(pl.screenshot(return_img=True))
            pl.close()
            ImageDraw.Draw(img).text((8, 8), f"{os.path.basename(out)}  {label}  {vname}", fill=(0, 0, 0))
            tiles.append(img)
    sheet = Image.new("RGB", (3 * T, 2 * T), "white")
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % 3) * T, (i // 3) * T))
    prev = os.path.join(HERE, "_clip_mesh_preview.png")
    sheet.save(prev)
    print("preview ->", prev)


if __name__ == "__main__":
    main()
