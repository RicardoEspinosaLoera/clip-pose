"""
annotation_agreement.py — inter-annotator agreement, consensus and arbitration list for the real
evaluation frames (protocol section 3).

For every frame of real_eval/frames.json and the given annotators:
  * exclusion: majority vote; a split vote is flagged.
  * box IoU (pairwise, min), point distances (pairwise, max) for tip / back end / coil, in px and
    as a fraction of the clip length L (median |tip - back end|), coil visibility agreement.
  * attributes: per-attribute Fleiss kappa over the frames everybody annotated.
Consensus (written as an annotator folder, so annotate_tool.py can edit it and eval_real.py can read it):
  median point / box per coordinate, majority attributes, coil = median of the annotators who saw it
  (None if the majority marked it not visible). A frame is FLAGGED for arbitration when any pair
  exceeds the thresholds (points max(4 px, 5% of clip length), box IoU 0.8) or exclusion / coil visibility split.

Usage:
  python annotation_agreement.py --annotators ricardo ana luis
  -> real_eval/annotations/consensus/*.json, real_eval/agreement/{summary.json, frames.csv, arbitration.txt, _flagged_montage.jpg}
  Arbitration: python annotate_tool.py --annotator consensus --start N   (N from arbitration.txt)
"""

import os
import csv
import json
import argparse
from itertools import combinations

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.normpath(os.path.join(HERE, "..", "real_eval"))
ATTRS = ("held", "open", "occluded", "blurry", "multi")
PTS = ("tip", "spring", "coil")
COLS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (255, 0, 255), (0, 220, 220)]


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-6)


def fleiss_binary(votes):
    """votes: (n_items, n_raters) 0/1 -> Fleiss kappa (nan if undefined)."""
    v = np.asarray(votes, float)
    if v.size == 0 or v.shape[1] < 2:
        return float("nan")
    n = v.shape[1]
    ones = v.sum(1)
    P_i = (ones * (ones - 1) + (n - ones) * (n - ones - 1)) / (n * (n - 1))
    p1 = v.mean()
    Pe = p1 ** 2 + (1 - p1) ** 2
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def main():
    global RE
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--root", default=None, help="real_eval folder (frames.json, frames/, annotations/)")
    ap.add_argument("--px", type=float, default=4.0, help="absolute floor of the point threshold (px)")
    ap.add_argument("--frac", type=float, default=0.05, help="flag if two annotators' points differ more than max(px, frac * clip length)")
    ap.add_argument("--iou", type=float, default=0.8, help="flag if two boxes have IoU below this")
    ap.add_argument("--consensus-name", default="consensus")
    ap.add_argument("--n-montage", type=int, default=40)
    a = ap.parse_args()
    RE = a.root or RE
    frames = json.load(open(os.path.join(RE, "frames.json")))
    ann_root = os.path.join(RE, "annotations")
    out_cons = os.path.join(ann_root, a.consensus_name)
    out_dir = os.path.join(RE, "agreement")
    os.makedirs(out_cons, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rows, cells, attr_votes = [], [], {k: [] for k in ATTRS}
    pair_d = {k: [] for k in PTS}
    pair_dL = {k: [] for k in PTS}
    pair_iou = []
    for i, fr in enumerate(frames):
        anns = {}
        for who in a.annotators:
            p = os.path.join(ann_root, who, fr["file"] + ".json")
            if os.path.exists(p):
                anns[who] = json.load(open(p))
        row = dict(n=i + 1, file=fr["file"], n_raters=len(anns), missing=" ".join(w for w in a.annotators if w not in anns))
        if not anns:
            row["status"] = "not annotated"
            rows.append(row)
            continue
        excl = [w for w, d in anns.items() if d.get("excluded")]
        keep = {w: d for w, d in anns.items() if not d.get("excluded")}
        flags = []
        if excl and keep:
            flags.append(f"exclusion split ({len(excl)} excl / {len(keep)} annotated)")
        if len(excl) > len(keep):
            json.dump(dict(file=fr["file"], src=fr["src"], video=fr["video"], excluded=True, reason="majority excluded",
                           annotator=a.consensus_name, n_raters=len(anns), flagged=bool(flags)),
                      open(os.path.join(out_cons, fr["file"] + ".json"), "w"), indent=1)
            row.update(status="excluded", flags="; ".join(flags))
            rows.append(row)
            continue
        L = float(np.median([np.linalg.norm(np.subtract(d["points"]["tip"], d["points"]["spring"])) for d in keep.values()]))
        row["L_px"] = round(L, 1)
        # pairwise
        ious, worst = [], {k: 0.0 for k in PTS}
        for (w1, d1), (w2, d2) in combinations(keep.items(), 2):
            j = iou(d1["box"], d2["box"])
            ious.append(j)
            pair_iou.append(j)
            for k in PTS:
                p1, p2 = d1["points"].get(k), d2["points"].get(k)
                if p1 is None or p2 is None:
                    continue
                dd = float(np.linalg.norm(np.subtract(p1, p2)))
                worst[k] = max(worst[k], dd)
                pair_d[k].append(dd)
                pair_dL[k].append(dd / max(L, 1e-6))
        if ious:
            row["iou_min"] = round(min(ious), 3)
            if min(ious) < a.iou:
                flags.append(f"box IoU {min(ious):.2f}")
        thr = max(a.px, a.frac * L)
        for k in PTS:
            row[f"{k}_maxdiff_px"] = round(worst[k], 1)
            if worst[k] > thr:
                flags.append(f"{k} {worst[k]:.1f}px")
        coil_vis = [d["points"].get("coil") is not None for d in keep.values()]
        if 0 < sum(coil_vis) < len(coil_vis):
            flags.append("coil visibility split")
        # consensus
        box = np.median(np.array([d["box"] for d in keep.values()], float), 0)
        pts = {k: np.median(np.array([d["points"][k] for d in keep.values()], float), 0).tolist() for k in ("tip", "spring")}
        coils = [d["points"]["coil"] for d in keep.values() if d["points"].get("coil") is not None]
        pts["coil"] = np.median(np.array(coils, float), 0).tolist() if len(coils) * 2 > len(keep) else None
        attrs = {k: bool(sum(bool(d["attrs"].get(k, False)) for d in keep.values()) * 2 > len(keep)) for k in ATTRS}
        for k in ATTRS:
            split = 0 < sum(bool(d["attrs"].get(k, False)) for d in keep.values()) < len(keep)
            if split and k in ("held", "open"):
                flags.append(f"{k} split")
        if len(keep) == len(a.annotators):
            for k in ATTRS:
                attr_votes[k].append([int(bool(d["attrs"].get(k, False))) for d in keep.values()])
        any_d = next(iter(keep.values()))
        json.dump(dict(file=fr["file"], src=fr["src"], video=fr["video"], width=any_d.get("width"), height=any_d.get("height"),
                       box=box.tolist(), points=pts, attrs=attrs, annotator=a.consensus_name, n_raters=len(keep),
                       flagged=bool(flags), flags=flags),
                  open(os.path.join(out_cons, fr["file"] + ".json"), "w"), indent=1)
        row.update(status="flagged" if flags else "agreed", flags="; ".join(flags))
        rows.append(row)
        if flags and len(cells) < a.n_montage:
            img = cv2.imread(os.path.join(RE, "frames", fr["file"]))
            x, y, w, h = box
            side = max(w, h) * 1.6
            cx, cy = x + w / 2, y + h / 2
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            vis = img.copy()
            for c, (who, d) in enumerate(keep.items()):
                for k in PTS:
                    p = d["points"].get(k)
                    if p is not None:
                        cv2.circle(vis, (int(p[0]), int(p[1])), {"tip": 5, "spring": 4, "coil": 3}[k], COLS[c % len(COLS)], -1)
            crop = cv2.resize(vis[y0:int(y0 + side), x0:int(x0 + side)], (300, 300))
            cv2.rectangle(crop, (0, 0), (300, 34), (0, 0, 0), -1)
            cv2.putText(crop, f"#{i + 1} {'; '.join(flags)[:44]}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cv2.putText(crop, "  ".join(f"{w}" for w in keep), (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cells.append(crop)

    keys = ["n", "file", "status", "flags", "n_raters", "missing", "L_px", "iou_min"] + [f"{k}_maxdiff_px" for k in PTS]
    with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r.get("status") == "flagged" or (r.get("status") == "excluded" and r.get("flags"))]
    with open(os.path.join(out_dir, "arbitration.txt"), "w", encoding="utf-8") as f:
        f.write(f"Frames to arbitrate: python annotate_tool.py --annotator {a.consensus_name} --start N\n")
        for r in flagged:
            f.write(f"{r['n']:4d}  {r['file']}  {r.get('flags', '')}\n")

    def stat(v):
        v = np.asarray(v, float)
        return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                    p90=float(np.percentile(v, 90)) if len(v) else None)
    summary = dict(annotators=a.annotators, thresholds=dict(px=a.px, frac_L=a.frac, iou=a.iou),
                   frames=dict(total=len(frames), agreed=sum(r.get("status") == "agreed" for r in rows),
                               flagged=len(flagged), excluded=sum(r.get("status") == "excluded" for r in rows),
                               not_annotated=sum(r.get("status") == "not annotated" for r in rows)),
                   box_iou_pairwise=stat(pair_iou),
                   point_diff_px_pairwise={k: stat(pair_d[k]) for k in PTS},
                   point_diff_fracL_pairwise={k: stat(pair_dL[k]) for k in PTS},
                   attribute_fleiss_kappa={k: fleiss_binary(attr_votes[k]) for k in ATTRS})
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    s = summary
    print(f"annotators {' '.join(a.annotators)} | frames: {s['frames']}")
    print(f"box IoU pairwise median {s['box_iou_pairwise']['median']}")
    for k in PTS:
        d, dl = s["point_diff_px_pairwise"][k], s["point_diff_fracL_pairwise"][k]
        if d["n"]:
            print(f"  {k:7s} pairwise diff: median {d['median']:.1f} px ({100 * dl['median']:.1f}% L), p90 {d['p90']:.1f} px")
    print("attribute Fleiss kappa:", {k: (round(v, 2) if v == v else None) for k, v in s["attribute_fleiss_kappa"].items()})
    for thr in (4, 6, 10):
        n = sum(1 for r in rows if r.get("status") in ("agreed", "flagged") and max(r.get("tip_maxdiff_px", 0), r.get("spring_maxdiff_px", 0)) > thr)
        print(f"  frames with tip/back-end disagreement > {thr} px: {n}")
    if cells:
        while len(cells) % 5:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(os.path.join(out_dir, "_flagged_montage.jpg"),
                    np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"consensus -> {out_cons}\nreport    -> {out_dir} (arbitration.txt lists {len(flagged)} frames)")


if __name__ == "__main__":
    main()
