#!/usr/bin/env python3
import os, json, argparse, math
import numpy as np
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your codebase imports ---
from src.datasets import TripletDataset            # <-- uses your dataset exactly as given
from src.models import Regressor
from src.models_dinov3 import DinoV3Regressor, DinoV3RegressorLoRA
from src.camera import sixd_to_rotmat
from src.losses import sample_mesh_points

import trimesh


# ------------------------------
# Utils
# ------------------------------

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_mesh(path: str):
    """Load mesh without altering scale/center."""
    m = trimesh.load(path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    V = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).contiguous()
    F = torch.from_numpy(np.asarray(m.faces, dtype=np.int64)).contiguous()
    return V, F


def geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """B×3×3 geodesic rotation error in degrees."""
    R_delta = torch.transpose(R_gt, -1, -2) @ R_pred
    tr = torch.diagonal(R_delta, dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1 + eps, 1 - eps)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return ang


def transform_points(P: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """P: (N,3), R: (3,3), t: (3,) -> (N,3)"""
    return (P @ R.T) + t


def project_points(P_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """P_cam: (N,3), K: (3,3) -> (N,2) pixels"""
    Z = P_cam[:, 2:3].clamp(min=1e-6)
    xy = P_cam[:, :2] / Z
    u = xy[:, 0] * K[0, 0] + K[0, 2]
    v = xy[:, 1] * K[1, 1] + K[1, 2]
    return torch.stack([u, v], dim=-1)


def add_metric(P_obj: torch.Tensor,
               R_pred: torch.Tensor, t_pred: torch.Tensor,
               R_gt: torch.Tensor,   t_gt: torch.Tensor) -> float:
    """
    P_obj: (M,3) points in object frame (torch, on device ok)
    R_*, t_*: (3,3), (3,)
    Returns mean L2 distance in *object units*.
    """
    Pp = transform_points(P_obj, R_pred, t_pred)
    Pg = transform_points(P_obj, R_gt, t_gt)
    return float(torch.linalg.norm(Pp - Pg, dim=1).mean().item())


def reproj_rms(P_obj: torch.Tensor,
               R_pred: torch.Tensor, t_pred: torch.Tensor, K: torch.Tensor,
               R_gt: torch.Tensor,   t_gt: torch.Tensor) -> float:
    """2D RMS reprojection error over sampled object points (pixels)."""
    Pp_cam = transform_points(P_obj, R_pred, t_pred)
    Pg_cam = transform_points(P_obj, R_gt, t_gt)
    up = project_points(Pp_cam, K)
    ug = project_points(Pg_cam, K)
    return float(torch.sqrt(((up - ug) ** 2).sum(dim=1).mean()).item())


def load_model(arch: str, ckpt: str, device: str):
    if arch == "regressor":
        model = Regressor().to(device)
    elif arch == "dinov3":
        model = DinoV3Regressor(unfreeze_last_blocks=1, freeze_backbone=False).to(device)
    elif arch == "dinov3_lora":
        model = DinoV3RegressorLoRA(unfreeze_last_blocks=1, freeze_backbone=False).to(device)
    else:
        raise ValueError("--arch must be one of ['regressor','dinov3','dinov3_lora']")

    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    sd = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with your TripletDataset JSONs")
    ap.add_argument("--mesh", required=True, help="Path to mesh (OBJ/PLY)")
    ap.add_argument("--ckpt", required=True, help="Path to model weights (.pth)")
    ap.add_argument("--arch", default="regressor", choices=["regressor", "dinov3", "dinov3_lora"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_points", type=int, default=1500, help="Sampled mesh points for ADD/reproj")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--compute_reproj", action="store_true", help="Also compute 2D reprojection RMS (uses K from sample)")
    ap.add_argument("--denorm_t", action="store_true",
                    help="Multiply predicted t by D_obj (use if your training predicted t normalized by D_obj).")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & loader (uses your TripletDataset exactly)
    ds = TripletDataset(args.data_root, train=False, transform=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device == "cuda"),
                    collate_fn=lambda x: x)  # keep list of dicts

    # Mesh & points
    verts, faces = load_mesh(args.mesh)
    verts = verts.to(device)
    faces = faces.to(device)

    with torch.no_grad():
        # Points for metrics (object space)
        P_obj = sample_mesh_points(verts, faces, n=args.num_points).to(device)  # (M,3)
        # Diameter for normalization
        samp = sample_mesh_points(verts, faces, n=2048).to(device)
        D_obj = torch.cdist(samp[None], samp[None]).amax().item()
        print(f"Mesh diameter (D_obj): {D_obj:.6f}")

    # Model
    model = load_model(args.arch, args.ckpt, device)

    # Aggregates
    n_samples = 0
    sum_add = 0.0
    sum_addn = 0.0
    sum_rdeg = 0.0
    sum_tn   = 0.0
    sum_reproj = 0.0

    # Per-sample logs
    per_sample = []

    with torch.no_grad():
        for batch in dl:
            for s in batch:
                # Pull metadata
                stem = s.get("stem", "sample")
                # K, R, t are tensors from your dataset
                K_t  = s["K"]           # [3,3] (float, CPU)
                R_gt = s["R_co"].squeeze(0)  # [3,3]
                t_gt = s["t_co"].squeeze(0)  # [3]

                # Image
                I_t = s["image"]        # CHW in [0,1]
                if I_t is None:
                    # Your dataset always has images, but guard anyway
                    print(f"[WARN] No image for {stem}; skipping.")
                    continue
                I = I_t.unsqueeze(0).to(device)  # (1,3,H,W)

                # Forward
                D_tensor = torch.as_tensor([D_obj], device=device, dtype=I.dtype)
                out = model(I, D_obj=D_tensor) if "dinov3" in args.arch or "regressor" in args.arch else model(I)
                if isinstance(out, tuple) and len(out) == 2:
                    r6, t_pred = out
                else:
                    # fallback if your model returns dict or different layout
                    r6, t_pred = out["r6"], out["t"]

                R_pred = sixd_to_rotmat(r6)[0]   # (3,3)
                t_pred = t_pred[0]               # (3,)

                # Optional de-normalization of translation
                if args.denorm_t:
                    t_pred = t_pred * D_obj

                # Metrics
                rdeg = float(geodesic_deg(R_pred.unsqueeze(0), R_gt.unsqueeze(0))[0].item())
                tn   = float(torch.linalg.norm(t_pred - t_gt).item()) / (D_obj + 1e-8)
                add  = add_metric(P_obj, R_pred, t_pred, R_gt, t_gt)
                addn = add / (D_obj + 1e-8)

                reproj = None
                if args.compute_reproj:
                    K = K_t.to(device) if isinstance(K_t, torch.Tensor) else torch.from_numpy(K_t).to(device)
                    reproj = reproj_rms(P_obj, R_pred, t_pred, K, R_gt, t_gt)

                # Accumulate
                n_samples += 1
                sum_add += add
                sum_addn += addn
                sum_rdeg += rdeg
                sum_tn   += tn
                if reproj is not None:
                    sum_reproj += reproj

                per_sample.append({
                    "sample": stem,
                    "R_deg": rdeg,
                    "Tn": tn,
                    "ADD": add,
                    "ADDn": addn,
                    **({"Reproj_RMS_px": reproj} if reproj is not None else {})
                })

    if n_samples == 0:
        print("No evaluable samples.")
        return

    # Averages
    avg_add   = sum_add / n_samples
    avg_addn  = sum_addn / n_samples
    avg_rdeg  = sum_rdeg / n_samples
    avg_tn    = sum_tn   / n_samples
    avg_reproj = (sum_reproj / n_samples) if args.compute_reproj else None

    print("\n===== EVAL RESULTS =====")
    print(f"Samples: {n_samples}")
    print(f"R_deg (↓):   {avg_rdeg:.3f}")
    print(f"Tn (↓):      {avg_tn:.4f}")
    print(f"ADD (↓):     {avg_add:.6f}")
    print(f"ADDn (↓):    {avg_addn:.6f}")
    if avg_reproj is not None:
        print(f"Reproj RMS px (↓): {avg_reproj:.3f}")

    # Write per-sample CSV next to ckpt
    out_csv = os.path.join(os.path.dirname(args.ckpt), "eval_results_triplet.csv")
    try:
        import csv
        with open(out_csv, "w", newline="") as f:
            fieldnames = list(per_sample[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in per_sample:
                w.writerow(row)
        print(f"Per-sample results written to: {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not write CSV: {e}")


if __name__ == "__main__":
    main()
