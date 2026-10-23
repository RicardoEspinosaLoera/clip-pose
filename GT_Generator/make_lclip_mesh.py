"""
make_lclip_mesh.py — parametric Peter Lazic L-Aneurysm-Clip (the clip seen in the MACS videos).

Identification (2026-10-22): MACS clips are blue, with a box-shaped head carrying a visible coil
spring and two lateral tabs the applier grips -> Peter Lazic L-Aneurysm-Clip, permanent standard
series (blue; 45.7xx). Yasargil (folded ribbon, alpha coil, plain legs) and Sugita (green/grey, bridge
wire) do not match. Design per US 9,386,987 B2 (Peter Lazic GmbH): two identical flat clip parts,
each = clamping arm (blade) + annular section with a circular opening + short operating arm on the
OPPOSITE side of the axis; a bearing sleeve through both openings forms the pivot; a leg (torsion)
spring whose winding sits in the sleeve bore pretensions the parts closed; the applier opens the
clip by spreading the two operating arms (the "tabs").

Dimensions: Lazic poster "L-Aneurysm-Clip" (dbsurgical.com/wp-content/uploads/2015/10/Lazic.pdf),
whose icons are vector drawings at 1:1 (the "9.0" blade dimension of 45.750 measures 26 pt =
9.17 mm). For 45.750 (blade 9.0 mm, max opening 8.9 mm, 1.77 N):
  side view (looking along the pin): closed jaws 1.7 mm wide at the tip -> 2.1 mm at 5.8 mm from
  the tip, shoulders flaring to 4.1 mm at 10.4 mm, round head OD 5.0 mm centred 11.6 mm from the
  tip, pin bore 1.3 mm, spring winding OD 3.3 mm, two operating arms ending at 15.46 mm with a V
  notch between them (2.3 mm wide, apex at 14.2 mm);
  front view (looking along the opening direction): blade face 1.3 mm wide, head/arms 2.74 mm wide
  (= two 1.37 mm clip parts side by side).
Other sizes keep the head and change the blade: --blade 7 (45.740), 12 (45.760), 15 (45.780), 18 (45.790).

Model frame: x = long axis (tip at x = 0, head toward -x), y = opening direction (blade A at +y, its
operating arm at -y), z = pin axis (part A at z > 0, part B at z < 0). Units mm. The final mesh may be
rotated 180 deg so that pca_frame() puts the jaws at high p0 (pipeline convention), like
make_clip_mesh.py does.

Articulation: besides the OBJ, a sidecar <stem>.parts.json is written (pivot, axis, per-part vertex
indices + rotation sign). articulate_clip.articulate() uses it (when CLIP_MESH points at this mesh, or
CLIP_PARTS at the JSON) to open the clip as a TRUE HINGE: part A rotates +theta/2, part B -theta/2
about the pin axis, sleeve + spring stay. Max opening in the spec (8.9 mm at the tips) ~ 2 x 27 deg.

Usage:
  python make_lclip_mesh.py                         -> clip_lazic_std.obj + clip_lazic_std.parts.json
  python make_lclip_mesh.py --blade 7 --out clip_lazic_740.obj
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class MeshBuilder:
    def __init__(self):
        self.V, self.F, self.groups = [], [], {}
        self._n = 0

    def add(self, verts, faces, group):
        verts = np.asarray(verts, float)
        faces = np.asarray(faces, int) + self._n
        self.V.append(verts)
        self.F.append(faces)
        self.groups.setdefault(group, []).append(np.arange(self._n, self._n + len(verts)))
        self._n += len(verts)

    def finish(self):
        V = np.vstack(self.V)
        F = np.vstack(self.F)
        G = {k: np.concatenate(v) for k, v in self.groups.items()}
        return V, F, G


def octagon(y_lo, y_hi, z_lo, z_hi, c):
    """8 corners (counter-clockwise in the y-z plane) of a chamfered rectangle."""
    c = min(c, 0.3 * (y_hi - y_lo), 0.3 * (z_hi - z_lo))
    return [(y_hi - c, z_lo), (y_hi, z_lo + c), (y_hi, z_hi - c), (y_hi - c, z_hi),
            (y_lo + c, z_hi), (y_lo, z_hi - c), (y_lo, z_lo + c), (y_lo + c, z_lo)]


def loft(stations, chamfer=0.15):
    """Loft chamfered rectangles along x. stations: list of (x, y_lo, y_hi, z_lo, z_hi)."""
    verts, faces = [], []
    for (x, y0, y1, z0, z1) in stations:
        verts += [(x, y, z) for (y, z) in octagon(y0, y1, z0, z1, chamfer)]
    n = len(stations)
    for i in range(n - 1):
        for j in range(8):
            a, b = i * 8 + j, i * 8 + (j + 1) % 8
            c2, d2 = (i + 1) * 8 + j, (i + 1) * 8 + (j + 1) % 8
            faces += [[a, b, d2], [a, d2, c2]]
    for i, flip in ((0, False), (n - 1, True)):
        base = i * 8
        for j in range(1, 7):
            tri = [base, base + j, base + j + 1]
            faces.append(tri[::-1] if flip else tri)
    return np.array(verts), np.array(faces)


def annulus(cx, cy, R, r, z0, z1, n=48):
    """Ring (tube) around the z axis through (cx, cy)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    co, si = np.cos(a), np.sin(a)
    verts = []
    for (rad, z) in ((R, z0), (R, z1), (r, z1), (r, z0)):      # 4 loops
        verts += [(cx + rad * c, cy + rad * s, z) for c, s in zip(co, si)]
    faces = []
    for k in range(4):                                          # outer wall, top, inner wall, bottom
        l0, l1 = k * n, ((k + 1) % 4) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[l0 + i, l0 + j, l1 + j], [l0 + i, l1 + j, l1 + i]]
    return np.array(verts), np.array(faces)


def wire(points, radius, n=8):
    """Sweep a circle along a 3-D polyline (frames by parallel transport)."""
    P = np.asarray(points, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    N = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], [0, 1.0, 0])
    N /= np.linalg.norm(N)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i in range(len(P)):
        if i > 0:
            N = N - T[i] * (N @ T[i])
            N /= np.linalg.norm(N) + 1e-12
        B = np.cross(T[i], N)
        verts += [P[i] + radius * (np.cos(a) * N + np.sin(a) * B) for a in ang]
    for i in range(len(P) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c2, d2 = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [[a, b, d2], [a, d2, c2]]
    for i, flip in ((0, False), (len(P) - 1, True)):
        base = i * n
        for j in range(1, n - 1):
            tri = [base, base + j, base + j + 1]
            faces.append(tri[::-1] if flip else tri)
    return np.array(verts), np.array(faces)


# ------------------------------------------------------------------ geometry
def default_params(blade_len=9.0):
    return dict(
        blade_len=blade_len,      # L (poster dimension), tip -> start of the head shoulders
        blade_gap=0.10,           # y gap between the closed blade faces
        t_tip=0.85, t_mid=1.05,   # blade thickness (y) at the tip / at 5.8 mm            (side view)
        shoulder_hw=2.07,         # half width of the head where the shoulders end        (side view)
        face_w=1.30,              # blade face width (z)                                   (front view)
        part_th=1.37,             # thickness of one clip part along the pin (head = 2.74) (front view)
        pin_x=11.62,              # pin axis distance from the tip                        (side view)
        ring_R=2.50, ring_r=1.65, # annular section outer / bore radius (OD 5.0, bore 3.3) (side view)
        bore_r=0.65,              # sleeve bore (visible hole, dia 1.3)                    (side view)
        sleeve_r=0.95,            # sleeve outer radius
        coil_R=1.25, coil_wire=0.20, coil_turns=3.0, coil_z=0.90,   # torsion spring in the ring bore
        arm_end=15.46,            # end of the operating arms (total length)              (side view)
        notch_apex=14.18, notch_hw=1.14,   # V notch between the arms: apex x, half width at 14.81
        arm_out0=2.20, arm_out1=1.93,      # outer edge of an arm at the head / at the end
        tip_round=0.75,
        chamfer=0.15,
    )


def build(P):
    mb = MeshBuilder()
    L, g = P["blade_len"], P["blade_gap"] / 2
    th, fw = P["part_th"], P["face_w"]
    xp = -P["pin_x"] - (L - 9.0)          # the whole head shifts with the blade length
    x_sh0, x_sh1 = -(L - 3.2), -(L + 1.4)   # shoulders: 5.8 -> 10.4 for L = 9
    x_fl0, x_fl1 = -(L - 2.0), -(L + 1.2)   # z flare of the parts: 7.0 -> 10.2 for L = 9

    def blade(sign):
        """Clamping arm + shoulder of one clip part (sign=+1: blade at +y, part at z>0)."""
        xs = np.linspace(0.0, x_sh1, 90)
        st = []
        for x in xs:
            d = -x                                                     # distance from the tip
            t = P["t_tip"] + (P["t_mid"] - P["t_tip"]) * np.clip(d / (L - 3.2), 0, 1)
            t = t + (P["shoulder_hw"] - g - P["t_mid"]) * smoothstep(-x_sh0, -x_sh1, d) ** 1.3
            r = np.clip((P["tip_round"] - d) / P["tip_round"], 0.0, 1.0)
            k = np.sqrt(max(1.0 - r ** 2, 0.03))
            s = smoothstep(-x_fl0, -x_fl1, d)
            z_lo, z_hi = -fw / 2 * (1 - s) + 0.0 * s, fw / 2 * (1 - s) + th * s
            y_lo, y_hi = g, g + t * k
            zc, zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2 * k
            st.append((x, sign * y_hi if sign > 0 else -y_hi, sign * y_lo if sign > 0 else -y_lo, sign * (zc - zh), sign * (zc + zh)))
        st = [(x, min(a, b), max(a, b), min(c, d), max(c, d)) for (x, a, b, c, d) in st]
        return loft(st, P["chamfer"])

    def arm(sign):
        """Operating arm of one clip part, on the side opposite to its blade, full head thickness."""
        xs = np.linspace(xp - 0.4, -P["arm_end"], 40)
        st = []
        x_notch0, x_notch1 = -P["notch_apex"], -(P["notch_apex"] + 0.63)
        for x in xs:
            u = (xp - 0.4 - x) / (P["arm_end"] + xp - 0.4)
            y_out = P["arm_out0"] + (P["arm_out1"] - P["arm_out0"]) * u
            if x > x_notch0:
                y_in = 0.05
            elif x > x_notch1:
                y_in = 0.05 + (P["notch_hw"] - 0.05) * (x_notch0 - x) / (x_notch0 - x_notch1)
            else:
                y_in = P["notch_hw"] + 0.45 * (x_notch1 - x) / max(1e-6, (x_notch1 + P["arm_end"]))
            y_lo, y_hi = (-y_out, -y_in) if sign > 0 else (y_in, y_out)
            st.append((x, y_lo, y_hi, -th, th))
        return loft(st, P["chamfer"])

    for sign, name in ((+1, "partA"), (-1, "partB")):
        v, f = blade(sign)
        mb.add(v, f, name)
        z0, z1 = (0.0, th) if sign > 0 else (-th, 0.0)
        v, f = annulus(xp, 0.0, P["ring_R"], P["ring_r"], z0, z1)
        mb.add(v, f, name)
        v, f = arm(sign)
        mb.add(v, f, name)
    # bearing sleeve (visible bore) and the torsion spring inside the ring bore
    v, f = annulus(xp, 0.0, P["sleeve_r"], P["bore_r"], -th, th, n=32)
    mb.add(v, f, "static")
    u = np.linspace(0, 1, int(60 * P["coil_turns"]))
    a = 2 * np.pi * P["coil_turns"] * u
    helix = np.c_[xp + P["coil_R"] * np.cos(a), P["coil_R"] * np.sin(a), -P["coil_z"] + 2 * P["coil_z"] * u]
    v, f = wire(helix, P["coil_wire"])
    mb.add(v, f, "static")
    # one spring leg: leaves the top of the winding toward the blades and hooks over the head
    leg = np.array([[xp + P["coil_R"], 0.0, P["coil_z"]], [xp + 1.9, 0.0, 1.15], [xp + 2.6, 0.0, 1.55],
                    [xp + 3.1, 0.0, 1.45], [xp + 3.2, 0.0, 1.0]])
    v, f = wire(leg, P["coil_wire"])
    mb.add(v, f, "static")
    V, F, G = mb.finish()
    return V, F, G, np.array([xp, 0.0, 0.0])


def orient_for_pipeline(V, pivot, jaw_idx):
    sys.path.insert(0, HERE)
    from articulate_clip import pca_frame
    for R in (np.diag([-1.0, -1.0, 1.0]), np.eye(3), np.diag([-1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0])):
        W = V @ R.T
        c, e = pca_frame(W)
        if ((W[jaw_idx] - c) @ e[:, 0]) > 0:
            return W, pivot @ R.T, R
    return V, pivot, np.eye(3)


def write_obj(path, verts, faces, header=""):
    with open(path, "w") as fh:
        fh.write("# " + header.replace("\n", "\n# ") + "\n")
        for p in verts:
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for tri in faces + 1:
            fh.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blade", type=float, default=9.0, help="blade length (mm): 7 / 9 / 12 / 15 / 18 = 45.740/750/760/780/790")
    ap.add_argument("--out", default="clip_lazic_std.obj")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    P = default_params(a.blade)
    V, F, G, pivot = build(P)
    jaw = int(G["partA"][np.argmax(V[G["partA"], 0])])          # tip of blade A (x = 0 before orientation)
    V, pivot, R = orient_for_pipeline(V, pivot, jaw)
    axis = np.array([0.0, 0.0, 1.0]) @ R.T
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    stem = os.path.splitext(out)[0]
    write_obj(out, V, F, f"Peter Lazic L-Aneurysm-Clip (permanent standard), blade {P['blade_len']} mm, units mm\n"
                         f"generated by make_lclip_mesh.py from the 1:1 Lazic poster icons; hinge data in {os.path.basename(stem)}.parts.json")
    # sign convention: rotating part A by +s*theta/2 about `axis` must move its blade tip AWAY from part B
    tipA = V[jaw]
    rel = tipA - pivot
    moved = rel + 1e-3 * np.cross(axis, rel)                      # tiny +rotation
    sA = 1.0 if abs((pivot + moved)[1] - 0) > abs(tipA[1]) or np.sign((pivot + moved)[1]) == np.sign(tipA[1]) and abs((pivot + moved)[1]) > abs(tipA[1]) else -1.0
    # simpler and robust: evaluate the blade-A y displacement under a small positive rotation
    def rot(points, ang):
        k = axis / np.linalg.norm(axis)
        p = points - pivot
        return pivot + p * np.cos(ang) + np.cross(k, p) * np.sin(ang) + np.outer(p @ k, k) * (1 - np.cos(ang))
    dA = rot(V[[jaw]], np.deg2rad(5.0))[0] - tipA
    e1_dir = np.sign(tipA[1] - V[G["partB"]][:, 1].mean())        # which side of part B blade A sits on (model y)
    sA = 1.0 if np.sign(dA[1]) == e1_dir else -1.0
    parts = dict(mesh=os.path.basename(out), n_points=int(len(V)), pivot=[float(v) for v in pivot], axis=[float(v) for v in axis],
                 parts=[dict(name="partA", sign=sA, indices=G["partA"].tolist()),
                        dict(name="partB", sign=-sA, indices=G["partB"].tolist())],
                 static=G["static"].tolist(),
                 note="true hinge: each part rotates sign*theta/2 about axis through pivot; articulate_clip uses this when CLIP_MESH/CLIP_PARTS point here")
    json.dump(parts, open(stem + ".parts.json", "w"))
    ext = V.max(0) - V.min(0)
    print(f"wrote {out}: {len(V)} verts, {len(F)} tris; extents {np.round(ext, 2)} mm (spec 45.750: 15.46 x 5.0 x 2.74); parts -> {stem}.parts.json")

    # checks with the pipeline code
    import pyvista as pv
    os.environ["CLIP_MESH"] = out
    sys.path.insert(0, HERE)
    import articulate_clip
    articulate_clip._PARTS_CACHE.clear()
    from articulate_clip import articulate, pca_frame
    m = pv.read(out)
    if m.n_cells != len(F):
        print(f"WARNING: pyvista read {m.n_cells} cells but {len(F)} faces were written")
    Vm = np.asarray(m.points, float)
    c, e = pca_frame(Vm)
    Pp = (Vm - c) @ e
    pe = Pp.max(0) - Pp.min(0)
    print(f"PCA extents e0/e1/e2 = {np.round(pe, 2)}  |e1.y|={abs(e[1, 1]):.3f} |e2.z|={abs(e[2, 2]):.3f}  jaws at high-p0: {Pp[jaw, 0] > 0}")
    m20 = articulate(m, 20.0)
    D = (np.asarray(m20.points) - Vm) @ e
    tips = Pp[:, 0] > Pp[:, 0].max() - 0.5
    print(f"articulate(20 deg) rigid hinge: tip separation along e1 = {np.ptp(D[tips, 1]):.2f} mm, along e2 = {np.ptp(D[tips, 2]):.2f} mm; static verts moved: {np.abs(D[G['static']]).max():.4f}")

    if a.no_preview:
        return
    from PIL import Image, ImageDraw
    pv.OFF_SCREEN = True
    T = 520
    tiles = []
    views = (("plan (along pin)", (0, 0, 1), (0, 1, 0)), ("side (along opening)", (0, 1, 0), (0, 0, 1)),
             ("3/4", (0.5, 0.9, 1.0), (0, 0, 1)), ("from the arms (applier view)", (-1.0, 0.25, 0.6), (0, 0, 1)))
    for label, mesh in (("closed", m), ("open 20 deg", m20)):
        mp = pv.PolyData((np.asarray(mesh.points) - c) @ e, mesh.faces).compute_normals(auto_orient_normals=True)
        for vname, pos, up in views:
            pl = pv.Plotter(off_screen=True, window_size=(T, T))
            pl.set_background("white")
            pl.add_mesh(mp, color=(0.30, 0.42, 0.80), smooth_shading=True, specular=0.8, specular_power=40)
            pl.camera_position = [np.array(pos, float) * 100.0, (0, 0, 0), up]
            pl.camera.parallel_projection = True
            pl.reset_camera()
            pl.camera.zoom(1.0)
            img = Image.fromarray(pl.screenshot(return_img=True))
            pl.close()
            ImageDraw.Draw(img).text((8, 8), f"{os.path.basename(out)}  {label}  {vname}", fill=(0, 0, 0))
            tiles.append(img)
    sheet = Image.new("RGB", (4 * T, 2 * T), "white")
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % 4) * T, (i // 4) * T))
    prev = os.path.join(HERE, "_lclip_mesh_preview.png")
    sheet.save(prev)
    print("preview ->", prev)


if __name__ == "__main__":
    main()
