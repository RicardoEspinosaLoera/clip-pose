"""
eval_synth.py — evaluate a trained model on a synthetic split with exact 6-DoF GT (protocol section 6).

Per image: keypoint error (original px), PnP success, rotation error (deg, symmetric), translation
and depth error (mesh units), ADD-S (fraction of clip length), 2D reprojection of the 8 keypoints
with the predicted pose (original px); subgroups: tool / no tool, jaws open / closed, size terciles.
M1 is evaluated with the jaw searched (inference condition) and, for reference, with the GT jaw.

Usage: python eval_synth.py --data ../GT_Generator/out_v3 --split ../GT_Generator/out_v3/splits/test.txt \
           --ckpt runs/e0_m1_s0/best.pt --model m1 --out runs/e0_m1_s0/test
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ClipModel, ClipPoseDataset, read_split, decode_heatmaps, pose_pnp, pose_pnp_jawsearch, project
from model import KeypointNet, DirectPoseNet, geodesic_sym

CLIP_LEN = 24.5


def rot_err(Rp, Rg, R_sym):
    return float(geodesic_sym(torch.from_numpy(Rp[None]).float(), torch.from_numpy(Rg[None]).float(), torch.from_numpy(R_sym[None]).float())[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["m1", "m2"], default="m1")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jaw-step", type=float, default=6.0, help="jaw-angle grid of the PnP search, 0..20 deg (6 = original 0/6/12/18)")
    ap.add_argument("--image-suffix", default="", help='"_flat" evaluates on the flat renders (ablation A0)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = ClipModel()
    R_sym = cm.symmetry_rotation()
    Vsub = cm.V0[::10]
    ds = ClipPoseDataset(a.data, read_split(a.split), cm, train=False, image_suffix=a.image_suffix)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    model = (KeypointNet(len(cm.idx), a.arch, pretrained=False) if a.model == "m1" else DirectPoseNet(a.arch, pretrained=False)).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device)["model"])
    model.eval()
    rows = []
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(device)
            B = x.shape[0]
            if a.model == "m1":
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    hm = model(x).float().cpu().numpy()
            else:
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    Rp_b, tp_b = model(x, b["K_crop"].to(device))
                Rp_b, tp_b = Rp_b.float().cpu().numpy(), tp_b.float().cpu().numpy()
            for i in range(B):
                Rg, tg = b["R"][i].numpy().astype(np.float64), b["t"][i].numpy().astype(np.float64)
                K, Kc, s = b["K"][i].numpy().astype(np.float64), b["K_crop"][i].numpy().astype(np.float64), float(b["scale"][i])
                jaw, tool = float(b["jaw"][i]), bool(b["tool"][i] > 0.5)
                row = dict(stem=b["stem"][i], tool=int(tool), jaw_open=int(jaw > 0), bbox_diag=float(b["bbox_diag"][i]))
                if a.model == "m1":
                    pts, conf = decode_heatmaps(hm[i])
                    pc = pts * (512 / 128)
                    vis = b["vis"][i].numpy() > 0
                    row["kp_px"] = float(np.linalg.norm(pc[vis] - b["kp"][i].numpy()[vis], axis=1).mean() / s) if vis.any() else np.nan
                    res = pose_pnp_jawsearch(pc, conf, cm, Kc, jaws=tuple(np.arange(0.0, 20.0 + 1e-6, a.jaw_step)))
                    res_gt = pose_pnp(pc, conf, cm.keypoints3d(jaw), Kc)
                    if res_gt is not None:
                        row["rot_deg_gtjaw"] = rot_err(res_gt[0], Rg, R_sym)
                    if res is None:
                        row["pnp_ok"] = 0
                        rows.append(row)
                        continue
                    Rp, tp, _, jaw_hat = res
                    row["pnp_ok"], row["jaw_hat"] = 1, float(jaw_hat)
                else:
                    Rp, tp = Rp_b[i].astype(np.float64), tp_b[i].astype(np.float64)
                    row["pnp_ok"] = 1
                row["rot_deg"] = rot_err(Rp, Rg, R_sym)
                row["trans"] = float(np.linalg.norm(tp - tg))
                row["depth"] = float(abs(tp[2] - tg[2]))
                add = min(np.linalg.norm((Vsub @ Rp.T + tp) - (Vsub @ Rg.T + tg), axis=1).mean(),
                          np.linalg.norm((Vsub @ Rp.T + tp) - (Vsub @ (Rg @ R_sym).T + tg), axis=1).mean())
                row["add_s"] = float(add)
                row["add_s_frac"] = float(add / CLIP_LEN)
                P3 = cm.keypoints3d(jaw)
                uv_p, _ = project(P3, Rp, tp, K)
                uv_g, _ = project(P3, Rg, tg, K)
                uv_ps, _ = project(P3, Rp, tp, K)
                uv_gs, _ = project(P3, Rg @ R_sym, tg, K)
                row["reproj_px"] = float(min(np.linalg.norm(uv_p - uv_g, axis=1).mean(), np.linalg.norm(uv_ps - uv_gs, axis=1).mean()))
                rows.append(row)
    if not rows:
        print("no samples in", a.split)
        return
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def summ(sel, name):
        ok = [r for r in sel if r.get("pnp_ok")]
        def med(k):
            v = np.array([r[k] for r in ok if k in r and not np.isnan(r[k])], float)
            return (float(np.median(v)), float(np.percentile(v, 90))) if len(v) else (np.nan, np.nan)
        out = dict(name=name, n=len(sel), pnp_ok=len(ok) / max(len(sel), 1))
        for k in ("kp_px", "rot_deg", "rot_deg_gtjaw", "trans", "depth", "add_s_frac", "reproj_px"):
            out[k + "_med"], out[k + "_p90"] = med(k)
        out["add_s_lt_0.1"] = float(np.mean([r["add_s_frac"] < 0.1 for r in ok])) if ok else np.nan
        out["rot_lt_5"] = float(np.mean([r["rot_deg"] < 5 for r in ok])) if ok else np.nan
        return out
    diag = np.array([r["bbox_diag"] for r in rows])
    t1, t2 = np.percentile(diag, [33, 66])
    groups = [(rows, "all"), ([r for r in rows if r["tool"]], "tool"), ([r for r in rows if not r["tool"]], "no_tool"),
              ([r for r in rows if r["jaw_open"]], "jaw_open"), ([r for r in rows if not r["jaw_open"]], "jaw_closed"),
              ([r for r in rows if r["bbox_diag"] < t1], "small"), ([r for r in rows if t1 <= r["bbox_diag"] < t2], "medium"),
              ([r for r in rows if r["bbox_diag"] >= t2], "large")]
    summary = [summ(sel, name) for sel, name in groups if sel]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{'group':10s} {'n':>5s} {'pnp':>5s} {'kp px':>7s} {'rot':>6s} {'rotGT':>6s} {'trans':>6s} {'depth':>6s} {'ADD-S':>6s} {'<0.1':>5s} {'reproj':>7s}")
    for m in summary:
        print(f"{m['name']:10s} {m['n']:5d} {m['pnp_ok']:5.2f} {m['kp_px_med']:7.2f} {m['rot_deg_med']:6.2f} {m['rot_deg_gtjaw_med']:6.2f} {m['trans_med']:6.2f} "
              f"{m['depth_med']:6.2f} {m['add_s_frac_med']:6.3f} {m['add_s_lt_0.1']:5.2f} {m['reproj_px_med']:7.2f}")
    print("->", a.out)


if __name__ == "__main__":
    main()
