"""
select_matched_frames.py — builds the matched real evaluation set from triage_tool.py decisions:
straight clips (shape like the mesh) without heavy occlusion, balanced across videos, near-duplicates removed.

Output: real_eval_v2/frames/*.jpg + real_eval_v2/frames.json (same format as real_eval/frames.json),
plus real_eval_v2/selection.json with the criteria and the triage counts (for the paper).
Annotate it with: python annotate_tool.py --root ../real_eval_v2 --annotator <name>

Usage: python select_matched_frames.py [--n 200] [--per-video 20] [--allow-occluded]
"""

import os
import json
import shutil
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RE = os.path.join(ROOT, "real_eval")
MACS = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")


def phash(bgr, size=8):
    g = cv2.cvtColor(cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.dct(g)[:size, :size]
    return (d > np.median(d[1:, 1:])).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-video", type=int, default=20)
    ap.add_argument("--hamming", type=int, default=12)
    ap.add_argument("--allow-occluded", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "real_eval_v2"))
    a = ap.parse_args()
    tri = json.load(open(os.path.join(RE, "triage.json")))
    keep = {s: r for s, r in tri.items() if r["shape"] == "straight" and (a.allow_occluded or not r["occluded"])}
    by_vid = defaultdict(list)
    for s, r in sorted(keep.items()):
        by_vid[r["video"]].append((s, r))
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    chosen, hashes, per = [], [], Counter()
    vids = sorted(by_vid)
    k = 0
    while len(chosen) < a.n and any(by_vid[v] for v in vids) and k < 100000:
        v = vids[k % len(vids)]
        k += 1
        while by_vid[v] and per[v] < a.per_video:
            s, r = by_vid[v].pop(0)
            p = os.path.join(RE, "frames", r["v1_file"]) if r.get("v1_file") else os.path.join(MACS, s)
            im = cv2.imread(p)
            if im is None:
                continue
            h = phash(im)
            if any((h != hh).sum() < a.hamming for hh in hashes):
                continue
            name = s.replace("\\", "_").replace("/", "_").rsplit(".", 1)[0] + ".jpg"
            if r.get("v1_file"):
                shutil.copy2(p, os.path.join(a.out, "frames", name))
            else:
                cv2.imwrite(os.path.join(a.out, "frames", name), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
            chosen.append(dict(file=name, src=s, video=v, hint_box=r["box"], det_score=None, in_first_set=bool(r.get("v1_file"))))
            hashes.append(h)
            per[v] += 1
            break
    json.dump(chosen, open(os.path.join(a.out, "frames.json"), "w"), indent=1)
    shape_counts = Counter(r["shape"] for r in tri.values())
    sel = dict(criteria=dict(shape="straight", occluded=a.allow_occluded, per_video=a.per_video, hamming=a.hamming),
               triaged=len(tri), shapes=dict(shape_counts),
               straight_occluded=sum(r["shape"] == "straight" and r["occluded"] for r in tri.values()),
               selected=len(chosen), per_video=dict(sorted(per.items())),
               from_first_set=sum(c["in_first_set"] for c in chosen))
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    print(json.dumps(sel, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
