"""
triage_tool.py — fast pre-selection of real MACS frames for the matched evaluation set: one key per frame
(clip shape + occlusion), before the slow point annotation.

Shape classes (compared with the synthetic mesh, a straight clip):
  1 straight      jaws in line with the clip body (like the mesh)
  2 curved        jaws curved sideways
  3 bent          angled / bayonet / fenestrated, or clearly deformed
  0 discard       no clip, clip not recognisable, or a duplicate of a frame just seen
Toggle before choosing the shape:  o = heavily occluded (tissue, blood or instruments hide a large part)

Order: the 200 frames of the first evaluation set, then the rest of the detector pool
(Inpaint_Realism/lora/clean_clips2/crops.jsonl) interleaved across videos, so any prefix stays balanced.
Results: real_eval/triage.json (resumable). Then: python select_matched_frames.py

Keys: 1 / 2 / 3 / 0 = classify and next   o = occluded on/off   z = zoom / full frame
      BACKSPACE = previous   q = quit
Usage: python triage_tool.py [--limit 600]
"""

import os
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RE = os.path.join(ROOT, "real_eval")
POOL = os.path.join(ROOT, "Inpaint_Realism", "lora", "clean_clips2", "crops.jsonl")
MACS = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")
SHAPES = {ord("1"): "straight", ord("2"): "curved", ord("3"): "bent", ord("0"): "discard"}
VIEW = 900


def candidates(seed=0):
    v1 = json.load(open(os.path.join(RE, "frames.json")))
    out, seen = [], set()
    for f in v1:
        out.append(dict(src=f["src"], video=f["video"], box=f["hint_box"], v1_file=f["file"]))
        seen.add(f["src"])
    by_vid = defaultdict(list)
    for line in open(POOL):
        r = json.loads(line)
        if r["src"] in seen:
            continue
        seen.add(r["src"])
        by_vid[r["src"].replace("/", "\\").split("\\")[0]].append(dict(src=r["src"], video=r["src"].replace("/", "\\").split("\\")[0], box=r["box"]))
    rng = np.random.default_rng(seed)
    vids = sorted(by_vid)
    for v in vids:
        rng.shuffle(by_vid[v])
    while any(by_vid.values()):
        for v in vids:
            if by_vid[v]:
                out.append(by_vid[v].pop())
    return out


def load(c):
    p = os.path.join(RE, "frames", c["v1_file"]) if c.get("v1_file") else os.path.join(MACS, c["src"])
    return cv2.imread(p)


def render(img, c, rec, zoom, i, n, counts):
    x, y, w, h = c["box"]
    H, W = img.shape[:2]
    if zoom:
        side = max(320.0, 3.0 * max(w, h))
        cx, cy = x + w / 2, y + h / 2
        s = VIEW / side
        A = np.array([[s, 0, VIEW / 2 - s * cx], [0, s, VIEW / 2 - s * cy]], np.float64)
        vis = cv2.warpAffine(img, A, (VIEW, VIEW), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        p0 = (int(A[0, 0] * x + A[0, 2]), int(A[1, 1] * y + A[1, 2]))
        p1 = (int(A[0, 0] * (x + w) + A[0, 2]), int(A[1, 1] * (y + h) + A[1, 2]))
        cv2.rectangle(vis, p0, p1, (0, 255, 255), 1)
        th = cv2.resize(img, (300, int(300 * H / W)))
        cv2.rectangle(th, (int(x * 300 / W), int(y * 300 / W)), (int((x + w) * 300 / W), int((y + h) * 300 / W)), (0, 255, 255), 2)
        vis[VIEW - th.shape[0] - 8:VIEW - 8, VIEW - 308:VIEW - 8] = th
    else:
        s = VIEW / W
        vis = cv2.resize(img, (VIEW, int(H * s)))
        cv2.rectangle(vis, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), (0, 255, 255), 2)
    bar = np.zeros((70, vis.shape[1], 3), np.uint8)
    prev = f"  saved: {rec['shape']}{' + occluded' if rec.get('occluded') else ''}" if rec.get("shape") else ""
    cv2.putText(bar, f"[{i + 1}/{n}] {c['video']}  occluded(o)={int(rec.get('occ_toggle', False))}{prev}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(bar, f"1 straight  2 curved  3 bent  0 discard  | z zoom  BKSP back  q quit   "
                     f"| so far: {counts['straight']} straight, {counts['straight_clean']} straight+clear", (8, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, vis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after this many candidates")
    ap.add_argument("--start", type=int, default=None, help="open candidate N (1-based)")
    a = ap.parse_args()
    cands = candidates()
    if a.limit:
        cands = cands[:a.limit]
    path = os.path.join(RE, "triage.json")
    tri = json.load(open(path)) if os.path.exists(path) else {}

    def counts():
        st = [r for r in tri.values() if r.get("shape") == "straight"]
        return dict(straight=len(st), straight_clean=sum(not r.get("occluded") for r in st))
    i = (a.start - 1) if a.start else next((k for k, c in enumerate(cands) if c["src"] not in tri), len(cands))
    print(f"{len(tri)} frames already triaged; {len(cands)} candidates; starting at {i + 1}")
    cv2.namedWindow("triage", cv2.WINDOW_AUTOSIZE)
    zoom = True
    while 0 <= i < len(cands):
        c = cands[i]
        img = load(c)
        if img is None:
            i += 1
            continue
        rec = dict(tri.get(c["src"], {}))
        rec["occ_toggle"] = rec.get("occluded", False)
        while True:
            cv2.imshow("triage", render(img, c, rec, zoom, i, len(cands), counts()))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("o"):
                rec["occ_toggle"] = not rec["occ_toggle"]
            elif k == ord("z"):
                zoom = not zoom
            elif k in SHAPES:
                tri[c["src"]] = dict(shape=SHAPES[k], occluded=bool(rec["occ_toggle"]), video=c["video"], box=c["box"],
                                     v1_file=c.get("v1_file"))
                json.dump(tri, open(path, "w"), indent=0)
                i += 1
                break
            elif k == 8:
                i = max(0, i - 1)
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                print(f"saved {len(tri)} -> {path} | {counts()}")
                return
    cv2.destroyAllWindows()
    print(f"saved {len(tri)} -> {path} | {counts()}")


if __name__ == "__main__":
    main()
