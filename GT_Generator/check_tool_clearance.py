"""
check_tool_clearance.py — verify on a GENERATED set that the clip neither lies on nor points at
another instrument.

Independent re-check (not the generator's own gate): detects instruments on the clip-free
backplate NNN_bg.png with the colour heuristic (grey / black / white / yellow / BLUE) AND
GroundingDINO, then measures per sample
  * on-tool %   : fraction of the clip silhouette on an instrument
  * clearance   : distance (px) from the clip to the nearest instrument pixel
  * pointing    : walking from the projected jaw tip along the blade direction (GT pose), the
                  distance (px) to the first instrument pixel within 1.2 x the clip length
Draws instrument contours (cyan), the clip (green) and the probe line (magenta) on a montage.

Usage
  python check_tool_clearance.py --dir out_set50c [--no-dino]
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as scipyR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restyle_appearance import zoom_cell
from articulate_clip import articulate, pca_frame

# reuse the generator's colour heuristic without importing its module (it opens a render window)
_src = open(os.path.join(HERE, "render_clip_poses_depth.py"), encoding="utf-8").read()
_fn = _src[_src.index("def instrument_avoid_mask"):_src.index("def focus_percentile")]
_g = {"cv2": cv2, "np": np, "AVOID_MARGIN_PX": 0}
exec(_fn, _g)
instrument_avoid_mask = _g["instrument_avoid_mask"]      # margin 0 here: we measure clearance ourselves

MESH = pv.read(os.path.join(HERE, "Item.obj"))
BLADE_DIR = pca_frame(np.asarray(MESH.points, np.float64))[1][:, 0]
PROMPT = ("surgical forceps. bipolar forceps. surgical instrument. instrument shaft. clip applier. suction tube. "
          "suction cannula. cautery. cottonoid patty. gauze. surgical needle. scissors. retractor. metal instrument. metal rod.")
PROBE_FRAC = 1.2
_ART = {}


def verts(jaw):
    key = round(float(jaw) * 2) / 2.0
    if key not in _ART:
        m = articulate(MESH, key) if key > 0 else MESH
        _ART[key] = np.asarray(m.points, np.float64)
    return _ART[key]


def tip_and_dir(meta, jaw):
    V = verts(jaw)
    q = meta["clip"]["pose_se3"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_se3"]["translation_m"], float)
    c = meta["camera"]
    X = (R @ V.T).T + t
    u = c["fx"] * X[:, 0] / X[:, 2] + c["cx"]
    v = c["fy"] * X[:, 1] / X[:, 2] + c["cy"]
    proj = (V - V.mean(0)) @ BLADE_DIR
    tip = np.array([u[np.argmax(proj)], v[np.argmax(proj)]])
    spr = np.array([u[np.argmin(proj)], v[np.argmin(proj)]])
    d = tip - spr
    L = float(np.linalg.norm(d))
    return tip, (d / L if L > 1e-6 else d), L


def dino_mask(proc, mdl, bgr, dev, thr=0.20, max_area=0.35):
    import torch
    from PIL import Image
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, text=PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = mdl(**inp)
    post = proc.post_process_grounded_object_detection
    try:
        r = post(o, threshold=thr, input_ids=inp.input_ids, text_threshold=0.2, target_sizes=[(H, W)])[0]
    except TypeError:
        r = post(o, box_threshold=thr, input_ids=inp.input_ids, text_threshold=0.2, target_sizes=[(H, W)])[0]
    m = np.zeros((H, W), np.uint8)
    for sc, b in sorted(zip(r["scores"].tolist(), r["boxes"].tolist()), key=lambda t: -t[0])[:10]:
        x0, y0, x1, y1 = [int(v) for v in b]
        if (x1 - x0) * (y1 - y0) > max_area * H * W:
            continue
        cv2.rectangle(m, (max(0, x0), max(0, y0)), (min(W, x1), min(H, y1)), 255, -1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out_set50c")
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--n-montage", type=int, default=50)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(HERE, a.dir)
    proc = mdl = None
    dev = "cpu"
    if not a.no_dino:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        mdl = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
    rows, cells = [], []
    for jp in sorted(glob.glob(os.path.join(d, "[0-9]*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        meta = json.load(open(jp))
        img = cv2.imread(os.path.join(d, f"{stem}.png"))
        bg = cv2.imread(os.path.join(d, meta["image"].get("file_bg", f"{stem}_bg.png")))
        clip = cv2.imread(os.path.join(d, meta["image"].get("file_mask", f"{stem}_mask.png")), 0) > 127
        if img is None or bg is None or not clip.any():
            continue
        H, W = clip.shape
        tools = instrument_avoid_mask(bg) > 127
        if mdl is not None:
            tools |= dino_mask(proc, mdl, bg, dev) > 127
        overlap = float((clip & tools).sum() / clip.sum() * 100)
        if tools.any():
            dist = cv2.distanceTransform((~tools).astype(np.uint8), cv2.DIST_L2, 3)
            clearance = float(dist[clip].min())
        else:
            clearance = float("inf")
        # pointing probe along the blade direction from the projected jaw tip
        tip, dvec, L = tip_and_dir(meta, float(meta["clip"].get("jaw_angle_deg", 0.0)))
        point_hit = float("inf")
        if L > 1e-3 and tools.any():
            nrm = np.array([-dvec[1], dvec[0]])
            for sdist in np.linspace(2.0, PROBE_FRAC * L, 40):
                hit = False
                for lat in (-6.0, 0.0, 6.0):
                    px = tip + dvec * sdist + nrm * lat
                    xi, yi = int(round(px[0])), int(round(px[1]))
                    if 0 <= xi < W and 0 <= yi < H and tools[yi, xi]:
                        hit = True
                        break
                if hit:
                    point_hit = float(sdist)
                    break
        rows.append((stem, overlap, clearance, point_hit, bool(meta["clip"].get("tool_held"))))
        if len(cells) < a.n_montage:
            ov = img.copy()
            c_t, _ = cv2.findContours(tools.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_c, _ = cv2.findContours(clip.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, c_t, -1, (255, 255, 0), 2)
            cv2.drawContours(ov, c_c, -1, (0, 255, 0), 1)
            if L > 1e-3:
                p1 = tip + dvec * PROBE_FRAC * L
                cv2.line(ov, (int(tip[0]), int(tip[1])), (int(p1[0]), int(p1[1])), (255, 0, 255), 1)
            lab = f"{stem} on-tool {overlap:.0f}% clear {clearance:.0f}px" if np.isfinite(clearance) else f"{stem} no instruments"
            lab += f" AIM {point_hit:.0f}px" if np.isfinite(point_hit) else ""
            cells.append(zoom_cell(ov, clip.astype(np.uint8) * 255, 300, pad=6.0, label=lab))
        print(f"{stem} tool_held={int(rows[-1][4])} on-instrument {overlap:5.1f}% clearance {clearance:6.1f}px "
              f"pointing-at {('%.0f px' % point_hit) if np.isfinite(point_hit) else 'none'}", flush=True)
    ov = np.array([r[1] for r in rows])
    cl = np.array([r[2] for r in rows if np.isfinite(r[2])])
    aim = [r[0] for r in rows if np.isfinite(r[3])]
    print(f"\n{os.path.basename(os.path.normpath(d))}: n={len(rows)} | clip on instrument: {int((ov > 0).sum())} samples (max {ov.max():.1f}%) "
          f"| clearance median {np.median(cl):.0f} px, min {cl.min():.0f} px | jaws pointing at an instrument (within {PROBE_FRAC}x clip length): "
          f"{len(aim)} {aim if aim else ''}")
    while len(cells) % 5:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)])
    banner = np.zeros((56, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, "cyan = instruments on the backplate (colour incl. blue + GroundingDINO), green = clip, magenta = blade probe; AIM = probe hits a tool",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    grid = np.vstack([banner, grid])
    small = cv2.resize(grid, (1200, int(grid.shape[0] * 1200 / grid.shape[1])), interpolation=cv2.INTER_AREA)
    out = os.path.join(d, "_tool_clearance.jpg")
    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("montage ->", out)


if __name__ == "__main__":
    main()
