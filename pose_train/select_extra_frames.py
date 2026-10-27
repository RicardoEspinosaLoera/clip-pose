"""
select_extra_frames.py -- new real frames for more annotators (27 Oct 2026), from the same triage as real_eval_v2.

Candidates: real_eval/triage.json frames with shape 'straight' that are NOT in real_eval_v2 (487 on 27 Oct).
Near-duplicates are removed against real_eval_v2 and among themselves: same video segment (file name
VVSSFFFFFF = video, segment, frame) closer than --min-gap frames, or a nearly identical image (pHash Hamming
< --hamming, default 4; passes with --gaps 15 8 5 relax the spacing only if needed; whole-frame pHash at 12 rejected 90 % of the candidates because frames of one surgery
look alike globally). Frames are
taken round-robin across videos (so small videos are used fully and vid_19, 150 candidates, does not dominate),
then split into --kits packages that alternate frames within each video (same video mix in every package).
Output per package: <out_prefix><letter>/frames/*.jpg + frames.json (same format as real_eval_v2, so
annotate_tool.py / make_annotation_kit.py --set work unchanged) + selection.json.

Usage: python select_extra_frames.py --per-kit 200 --kits 2            -> real_eval_v3a, real_eval_v3b
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
V2 = os.path.join(ROOT, "real_eval_v2")
MACS = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")


def phash(bgr, size=8):
    g = cv2.cvtColor(cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.dct(g)[:size, :size]
    return (d > np.median(d[1:, 1:])).flatten()


def seg_frame(src):
    """'vid_00\vid_00\1\0011001680.jpg' -> ('vid_00', '11', 1680)"""
    parts = src.replace("\\", "/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    return parts[0], stem[2:4], int(stem[4:])


def too_close(key, taken, gap):
    v, sg, f = key
    return any(f2 is not None and abs(f - f2) < gap for f2 in taken.get((v, sg), []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kit", type=int, default=200)
    ap.add_argument("--kits", type=int, default=2)
    ap.add_argument("--hamming", type=int, default=4)
    ap.add_argument("--gaps", type=int, nargs="+", default=[15, 8, 5],
                    help="min frame-number gap inside one video segment, one pass per value (strict first)")
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "real_eval_v3"))
    a = ap.parse_args()
    need = a.per_kit * a.kits
    tri = json.load(open(os.path.join(RE, "triage.json")))
    v2 = json.load(open(os.path.join(V2, "frames.json")))
    used = {d["src"] for d in v2}
    hashes, taken = [], defaultdict(list)
    for d in v2:                                           # do not re-ship near-copies of annotated frames
        v, sg, f = seg_frame(d["src"])
        taken[(v, sg)].append(f)
        im = cv2.imread(os.path.join(V2, "frames", d["file"]))
        if im is not None:
            hashes.append(phash(im))
    by_vid = defaultdict(list)
    for s, r in sorted(tri.items()):
        if r["shape"] == "straight" and s not in used:
            by_vid[r["video"]].append((s, r))
    n_cand = sum(len(v) for v in by_vid.values())
    print(f"{n_cand} straight candidates not in real_eval_v2, {len(by_vid)} videos; need {need}")

    chosen, per, dup, gap_of = defaultdict(list), Counter(), 0, {}
    vids = sorted(by_vid)
    imcache = {}
    for gap in a.gaps:                                     # strict temporal spacing first, then relaxed passes
        left = {v: list(by_vid[v]) for v in vids}
        while sum(per.values()) < need and any(left[v] for v in vids):
            for v in vids:
                if sum(per.values()) >= need:
                    break
                while left[v]:
                    s, r = left[v].pop(0)
                    key = seg_frame(s)
                    if too_close(key, taken, gap):
                        continue                           # may still qualify in a relaxed pass
                    by_vid[v].remove((s, r))
                    p = os.path.join(RE, "frames", r["v1_file"]) if r.get("v1_file") else os.path.join(MACS, s)
                    im = imcache.get(p)
                    if im is None:
                        im = cv2.imread(p)
                    if im is None:
                        continue
                    h = phash(im)
                    if any((h != hh).sum() < a.hamming for hh in hashes):
                        dup += 1
                        continue
                    hashes.append(h)
                    taken[key[:2]].append(key[2])
                    chosen[v].append((s, r, p, im))
                    gap_of[s] = gap
                    per[v] += 1
                    break
        print(f"  after pass min-gap {gap:3d}: {sum(per.values())} frames")
    total = sum(per.values())
    print(f"selected {total} (near-duplicates removed: {dup}); per video {dict(sorted(per.items()))}")
    if total < need:
        print(f"WARNING: only {total} unique frames available for {need} requested; packages will be smaller")

    kits = [[] for _ in range(a.kits)]
    j = 0
    for v in vids:                                         # alternate within each video -> same video mix per kit
        for item in chosen[v]:
            kits[j % a.kits].append((v, item))
            j += 1
    for k, items in enumerate(kits):
        out = a.out_prefix + "abcdefgh"[k]
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(os.path.join(out, "frames"))
        frames = []
        for v, (s, r, p, im) in items:
            name = s.replace("\\", "_").replace("/", "_").rsplit(".", 1)[0] + ".jpg"
            if r.get("v1_file"):
                shutil.copy2(p, os.path.join(out, "frames", name))
            else:
                cv2.imwrite(os.path.join(out, "frames", name), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
            frames.append(dict(file=name, src=s, video=v, hint_box=r["box"], det_score=None,
                               in_first_set=bool(r.get("v1_file")), min_gap=gap_of[s]))
        json.dump(frames, open(os.path.join(out, "frames.json"), "w"), indent=1)
        sel = dict(criteria=dict(shape="straight", not_in="real_eval_v2", hamming=a.hamming, gaps=a.gaps, round_robin_by_video=True),
                   candidates=n_cand, near_duplicates_removed=dup, selected=len(frames),
                   by_gap_pass=dict(sorted(Counter(f["min_gap"] for f in frames).items())),
                   per_video=dict(sorted(Counter(f["video"] for f in frames).items())))
        json.dump(sel, open(os.path.join(out, "selection.json"), "w"), indent=1)
        print(f"-> {out}: {len(frames)} frames, videos {sel['per_video']}")


if __name__ == "__main__":
    main()
