"""
train.py — train M1 (keypoint heatmaps + PnP) or M2 (direct regression) on a synthetic set.

  python train.py --data ../GT_Generator/out_v3 --split-dir ../GT_Generator/out_v3/splits \
                  --model m1 --iters 30000 --batch 16 --out runs/e0_m1_s0 --seed 0
  python train.py ... --train-list subsets/train_2500.txt      # learning-curve subset

Validation every --val-every iterations on the val split: mean keypoint error (px, crop space,
visible points), PnP success rate, rotation error (deg, symmetric), translation error, ADD-S.
Checkpoints: last.pt every validation, best.pt on lowest val keypoint error (M1) or ADD-S (M2).
Everything is logged to <out>/log.csv and <out>/config.json (data hash, seed, git hash).
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ClipModel, ClipPoseDataset, read_split, decode_heatmaps, pose_pnp, pose_pnp_jawsearch
from model import KeypointNet, DirectPoseNet, heatmap_loss, geodesic_sym


def add_s(R_p, t_p, R_g, t_g, V, R_sym):
    a = np.linalg.norm((V @ R_p.T + t_p) - (V @ R_g.T + t_g), axis=1).mean()
    b = np.linalg.norm((V @ R_p.T + t_p) - (V @ (R_g @ R_sym).T + t_g), axis=1).mean()
    return float(min(a, b))


@torch.no_grad()
def evaluate(model, loader, cm, args, device, max_batches=None):
    model.eval()
    V = cm.V0[::25]                                     # subsample for ADD
    R_sym = cm.symmetry_rotation()
    R_sym_t = torch.from_numpy(R_sym.astype(np.float32)).to(device)
    kp_err, rot_err, tr_err, adds, ok_n, n = [], [], [], [], 0, 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = b["image"].to(device)
        B = x.shape[0]
        if args.model == "m1":
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                hm = model(x).float()
            hm_np = hm.cpu().numpy()
            for i in range(B):
                pts, conf = decode_heatmaps(hm_np[i])
                pts_crop = pts * (args.crop / args.heat)
                vis = b["vis"][i].numpy() > 0
                if vis.any():
                    kp_err.append(float(np.linalg.norm(pts_crop[vis] - b["kp"][i].numpy()[vis], axis=1).mean()))
                res = pose_pnp_jawsearch(pts_crop, conf, cm, b["K_crop"][i].numpy())   # jaw unknown at inference
                n += 1
                if res is None:
                    continue
                Rp, tp, _, _ = res
                Rg, tg = b["R"][i].numpy().astype(np.float64), b["t"][i].numpy().astype(np.float64)
                ok_n += 1
                rot_err.append(float(geodesic_sym(torch.from_numpy(Rp[None]).float(), torch.from_numpy(Rg[None]).float(), torch.from_numpy(R_sym[None]).float())[0]))
                tr_err.append(float(np.linalg.norm(tp - tg)))
                adds.append(add_s(Rp, tp, Rg, tg, V, R_sym))
        else:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                Rp, tp = model(x, b["K_crop"].to(device))
            Rp, tp = Rp.float(), tp.float()
            Rg, tg = b["R"].to(device), b["t"].to(device)
            rot_err += geodesic_sym(Rp, Rg, R_sym_t.expand(B, 3, 3)).cpu().tolist()
            tr_err += (tp - tg).norm(dim=1).cpu().tolist()
            for i in range(B):
                adds.append(add_s(Rp[i].cpu().numpy().astype(np.float64), tp[i].cpu().numpy().astype(np.float64),
                                  Rg[i].cpu().numpy().astype(np.float64), tg[i].cpu().numpy().astype(np.float64), V, R_sym))
            n += B
            ok_n += B
    model.train()
    med = lambda v: float(np.median(v)) if v else float("nan")
    return dict(kp_px=med(kp_err), pnp_ok=ok_n / max(n, 1), rot_deg=med(rot_err), trans=med(tr_err), add_s=med(adds), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--train-list", default=None, help="override train stems (learning-curve subsets)")
    ap.add_argument("--model", choices=["m1", "m2"], default="m1")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--heat", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amodal", action="store_true", help="supervise keypoints hidden by the applier too")
    ap.add_argument("--image-suffix", default="", help='"_flat" trains on the flat renders (ablation A0)')
    ap.add_argument("--aug", choices=["normal", "strong"], default="normal", help="strong = wide photometric + geometric jitter for real transfer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    split_dir = a.split_dir or os.path.join(a.data, "splits")
    train_stems = read_split(a.train_list or os.path.join(split_dir, "train.txt"))
    val_stems = read_split(os.path.join(split_dir, "val.txt"))
    cm = ClipModel()
    ds_tr = ClipPoseDataset(a.data, train_stems, cm, crop=a.crop, heat=a.heat, train=True, image_suffix=a.image_suffix, amodal=a.amodal, aug=a.aug)
    ds_va = ClipPoseDataset(a.data, val_stems, cm, crop=a.crop, heat=a.heat, train=False, image_suffix=a.image_suffix, amodal=a.amodal)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=max(1, a.workers // 2))

    model = (KeypointNet(len(cm.idx), a.arch) if a.model == "m1" else DirectPoseNet(a.arch)).to(device)
    bb = [p for n_, p in model.named_parameters() if n_.startswith("backbone")]
    hd = [p for n_, p in model.named_parameters() if not n_.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": hd, "lr": a.lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    R_sym = torch.from_numpy(cm.symmetry_rotation().astype(np.float32)).to(device)

    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "n/a"
    cfg = dict(vars(a), n_train=len(train_stems), n_val=len(val_stems), git=git, device=str(device),
               train_sha1=hashlib.sha1("\n".join(train_stems).encode()).hexdigest()[:12])
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    log = open(os.path.join(a.out, "log.csv"), "a")
    log.write("iter,loss,lr,val_kp_px,val_pnp_ok,val_rot_deg,val_trans,val_add_s,sec\n")
    print(f"{a.model} {a.arch} | train {len(train_stems)} val {len(val_stems)} | {a.iters} iters, batch {a.batch} | {device}", flush=True)

    it, best, t0, run_loss = 0, float("inf"), time.time(), 0.0
    model.train()
    while it < a.iters:
        for b in dl_tr:
            x = b["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                if a.model == "m1":
                    loss = heatmap_loss(model(x).float(), b["heatmaps"].to(device), b["vis"].to(device)) * 1000.0
                else:
                    Rp, tp = model(x, b["K_crop"].to(device))
                    Rg, tg = b["R"].to(device), b["t"].to(device)
                    rot = geodesic_sym(Rp.float(), Rg, R_sym.expand(x.shape[0], 3, 3)).mean() / 45.0
                    loss = rot + (tp.float() - tg).abs().mean() / 10.0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            run_loss = 0.98 * run_loss + 0.02 * float(loss) if it > 1 else float(loss)
            if it % 100 == 0:
                print(f"it {it:6d} loss {run_loss:.4f} lr {opt.param_groups[1]['lr']:.2e} {(time.time() - t0) / it:.2f} s/it", flush=True)
            if it % a.val_every == 0 or it == a.iters:
                m = evaluate(model, dl_va, cm, a, device, max_batches=a.val_batches)
                key = m["kp_px"] if a.model == "m1" else m["add_s"]
                log.write(f"{it},{run_loss:.5f},{opt.param_groups[1]['lr']:.3e},{m['kp_px']:.3f},{m['pnp_ok']:.3f},{m['rot_deg']:.3f},{m['trans']:.3f},{m['add_s']:.3f},{time.time() - t0:.0f}\n")
                log.flush()
                torch.save({"model": model.state_dict(), "iter": it, "cfg": cfg}, os.path.join(a.out, "last.pt"))
                if key < best:
                    best = key
                    torch.save({"model": model.state_dict(), "iter": it, "cfg": cfg, "val": m}, os.path.join(a.out, "best.pt"))
                print(f"[val {it}] kp {m['kp_px']:.2f} px | pnp ok {m['pnp_ok']:.2f} | rot {m['rot_deg']:.2f} deg | trans {m['trans']:.2f} | ADD-S {m['add_s']:.3f} | best {best:.3f}", flush=True)
            if it >= a.iters:
                break
    log.close()
    print("done", a.out)


if __name__ == "__main__":
    main()
