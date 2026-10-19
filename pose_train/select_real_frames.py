"""
select_real_frames.py — pick the 200 real MACS frames of the evaluation set (protocol section 3).

Source: the validated clip crops in Inpaint_Realism/lora/clean_clips2 (crops.jsonl -> native frame).
Rules: balanced by video (max per video), pHash near-duplicate removal, two strata by the
detector score as a proxy for 'held by applier' is NOT reliable, so stratification is done later
by the annotator's attribute; here we only guarantee video balance and diversity.
Output: real_eval/frames/<video>_<frame>.jpg (native 1920x1080) + real_eval/frames.json
(list with source path and the detector box as a starting hint for the annotator).

Usage: python select_real_frames.py --n 200 --per-video 15 --seed 0
"""

import os
import json
import argparse
import shutil
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MACS_ROOT", r"C:\Users\ricar\OneDrive\Documentos\MAC_dataset\23533731")   # native MACS videos/frames
CROPS = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "clean_clips2"))
OUT = os.path.join(HERE, "..", "real_eval")


def phash(bgr, size=8):
    g = cv2.cvtColor(cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.dct(g)[:size, :size]
    return (d > np.median(d[1:, 1:])).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-video", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hamming", type=int, default=12)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(CROPS, "crops.jsonl"))]
    rows = [r for r in rows if os.path.exists(os.path.join(CROPS, r["file"])) and os.path.exists(os.path.join(ROOT, r["src"]))]
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["src"].split("\\")[0]].append(r)
    rng = np.random.default_rng(a.seed)
    for v in by_vid:
        rng.shuffle(by_vid[v])
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    chosen, hashes, counts = [], [], defaultdict(int)
    vids = sorted(by_vid)
    k, cap, stall = 0, a.per_video, 0
    while len(chosen) < a.n and any(by_vid[v] for v in vids):
        if k and k % len(vids) == 0:                 # one full round without progress -> relax the per-video cap
            if len(chosen) == stall:
                cap += 1
                print(f"  cap -> {cap} per video ({len(chosen)} chosen)", flush=True)
            stall = len(chosen)
        v = vids[k % len(vids)]
        k += 1
        while by_vid[v] and counts[v] < cap:
            r = by_vid[v].pop()
            im = cv2.imread(os.path.join(ROOT, r["src"]))
            if im is None:
                continue
            h = phash(im)
            if any((h != hh).sum() < a.hamming for hh in hashes):
                continue
            name = r["src"].replace("\\", "_").replace("/", "_").rsplit(".", 1)[0] + ".jpg"
            cv2.imwrite(os.path.join(OUT, "frames", name), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
            chosen.append(dict(file=name, src=r["src"], video=v, hint_box=r["box"], det_score=r["score"]))
            hashes.append(h)
            counts[v] += 1
            break
    json.dump(chosen, open(os.path.join(OUT, "frames.json"), "w"), indent=1)
    print(f"selected {len(chosen)} frames from {len([v for v in counts if counts[v]])} videos -> {os.path.normpath(OUT)}")
    print("per video:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
