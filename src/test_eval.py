#!/usr/bin/env python3
import os, json, glob, argparse, math
import numpy as np
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models import Regressor
from src.models_dinov3 import DinoV3RegressorLoRA
from src.camera import sixd_to_rotmat
from src.losses import sample_mesh_points

import trimesh


# ------------------------------
# Utilities
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


def quat_wxyz_to_R(q: List[float]) -> np.ndarray:
    """Quaternion [w,x,y,z] → 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)


def parse_intrinsics(d: Dict[str, Any]) -> np.ndarray:
    """
    Accepts:
      - 'K': 3x3
      - or {fx, fy, cx, cy}
    Returns K (3x3).
    """
    if "K" in d and d["K"] is not None:
        K = np.array(d["K"], dtype=np.float32).reshape(3, 3)
    else:
        fx = float(d.get("fx"))
        fy = float(d.get("fy", fx))
        cx = float(d.get("cx"))
        cy = float(d.get("cy"))
        K = np.array([[fx, 0, cx],
                      [0,  fy, cy],
                      [0,  0,  1]], dtype=np.float32)
    return K


def parse_pose(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts:
      - 'R_co', 't_co'
      - or {'quaternion_wxyz', 'translation_m'}
    Returns R (3x3), t (3,)
    """
    if "R_co" in d and "t_co" in d:
        R = np.array(d["R_co"], dtype=np.float32).reshape(3, 3)
        t = np.array(d["t_co"], dtype=np.float32).reshape(3)
        return R, t
    # fallback quaternion format
    if "quaternion_wxyz" in d and "translation_m" in d:
        R = quat_wxyz_to_R(d["quaternion_wxyz"])
        t = np.array(d["translation_m"], dtype=np.float32).reshape(3)
        return R, t
    raise ValueError("Pose not found: need (R_co,t_co) or (quaternion_wxyz,translation_m).")


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
    """
    P_cam: (N,3) in camera space, K: (3,3) -> (N,2) pixels
    """
    Z = P_cam[:, 2:3].clamp(min=1e-6)
    xy = P_cam[:, :2] / Z
    u = xy[:, 0] * K[0, 0] + K[0, 2]
    v = xy[:, 1] * K[1, 1] + K[1, 2]
    return torch.stack([u, v], dim=-1)


def add_metric(P_obj: torch.Tensor,
               R_pred: torch.Tensor, t_pred: torch.Tensor,
               R_gt: torch.Tensor,   t_gt: torch.Tensor) -> float:
    """
    P_obj: (M,3) points in object frame (torch, cpu or gpu ok)
    R_*, t_*: (3,3), (3,)
    Returns mean L2 distance in *object units*.
    """
    Pp = transform_points(P_obj, R_pred, t_pred)
    Pg = transform_points(P_obj, R_gt, t_gt)
    return float(torch.linalg.norm(Pp - Pg, dim=1).mean().item())


def reproj_rms(P_obj: torch.Tensor,
               R_pred: torch.Tensor, t_pred: torch.Tensor, K: torch.Tensor,
               R_gt: torch.Tensor,   t_gt: torch.Tensor) -> float:
    """
    2D RMS reprojection error over sampled object points.
    Returns pixels.
    """
    Pp_cam = transform_points(P_obj, R_pred, t_pred)
    Pg_cam = transform_points(P_obj, R_gt, t_gt)
    up = project_points(Pp_cam, K)
    ug = project_points(Pg_cam, K)
    return float(torch.sqrt(((up - ug) ** 2).sum(dim=1).mean()).item())


# ------------------------------
# Dataset that reads JSON GT files
# ------------------------------

class JsonGTDataset(Dataset):
    """
    Expects a folder with one or more *.json files.
    Each JSON should contain either:
      - { "image": <path>, "K": 3x3 or {fx,fy,cx,cy}, "R_co":3x3, "t_co":3 }
        (optional "mask","bg" allowed but not required for metrics)
    or
      - { "image": <path>, "fx","fy","cx","cy", "quaternion_wxyz":[w,x,y,z], "translation_m":[x,y,z] }
    """
    def __init__(self, root: str, image_size: Tuple[int,int]=(544,800)):
        self.jsons = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
        if not self.jsons:
            raise FileNotFoundError(f"No JSON files found under: {root}")
        self.image_size = image_size

    def __len__(self): return len(self.jsons)

    def __getitem__(self, idx):
        path = self.jsons[idx]
        with open(path, "r") as f:
            d = json.load(f)

        K = parse_intrinsics(d)
        R, t = parse_pose(d)

        # Optional image load (not needed for metrics; only for running the model)
        # We'll load the image if 'image' is present; otherwise assume the test set
        # is features/embeddings or skip visual input.
        I = None
        if "image" in d and d["image"]:
            import PIL.Image as Image
            from torchvision import transforms as T
            img = Image.open(d["image"]).convert("RGB")
            H, W = self.image_size
            tf = T.Compose([
                T.Resize((H, W)),
                T.ToTensor()
            ])
            I = tf(img)  # (3,H,W)

        sample = {
            "json_path": path,
            "image": I,                 # may be None
            "K": K.astype(np.float32),  # (3,3)
            "R_gt": R.astype(np.float32),
            "t_gt": t.astype(np.float32),
        }
        return sample


# ------------------------------
# Main evaluation
# ------------------------------

def load_model(arch: str, ckpt: str, device: str):
    if arch == "regressor":
        model = Regressor().to(device)
    elif arch == "dinov3":
        # default: unfreeze last block; adjust if needed
        model = DinoV3Regressor(unfreeze_last_blocks=1, freeze_backbone=False).to(device)
    else:
        raise ValueError("--arch must be one of ['regressor','dinov3']")

    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    # Support both full checkpoint and raw state_dict
    if "model_state" in state:
        sd = state["model_state"]
    else:
        sd = state
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder containing *.json GT files")
    ap.add_argument("--mesh", required=True, help="Path to mesh (OBJ/PLY)", default="../meshes/Item.obj")
    ap.add_argument("--ckpt", required=True, help="Path to model weights (.pth)")
    ap.add_argument("--arch", default="regressor", choices=["regressor","dinov3"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_points", type=int, default=1500, help="Sampled mesh points for ADD/reproj")
    ap.add_argument("--image_h", type=int, default=544)
    ap.add_argument("--image_w", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compute_reproj", action="store_true", help="Also compute 2D reprojection RMS")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    ds = JsonGTDataset(args.data_root, image_size=(args.image_h, args.image_w))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda x: x)

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

    # Per-sample logs (optional)
    per_sample = []

    with torch.no_grad():
        for batch in dl:
            # Collated as a list; process sequentially for simplicity (since images may be None)
            for s in batch:
                json_path = s["json_path"]
                K_np = s["K"]; R_gt_np = s["R_gt"]; t_gt_np = s["t_gt"]

                K = torch.from_numpy(K_np).to(device)
                R_gt = torch.from_numpy(R_gt_np).to(device)
                t_gt = torch.from_numpy(t_gt_np).to(device)

                # Forward pass only if we have an image
                if s["image"] is not None:
                    I = s["image"].unsqueeze(0).to(device)  # (1,3,H,W)
                    # For models expecting D_obj in forward:
                    D_tensor = torch.as_tensor([D_obj], device=device, dtype=I.dtype)
                    r6, t_pred = model(I, D_obj=D_tensor)
                    R_pred = sixd_to_rotmat(r6)
                    R_pred = R_pred[0]    # (3,3)
                    t_pred = t_pred[0]    # (3,)
                else:
                    # If no image provided, we cannot predict. Skip.
                    print(f"[WARN] No image in {json_path}; skipping prediction.")
                    continue

                # Metrics
                # Rotation error (deg)
                rdeg = float(geodesic_deg(R_pred.unsqueeze(0), R_gt.unsqueeze(0))[0].item())
                # Translation normalized by D_obj
                tn = float(torch.linalg.norm(t_pred - t_gt).item()) / (D_obj + 1e-8)
                # ADD and normalized ADD
                add = add_metric(P_obj, R_pred, t_pred, R_gt, t_gt)
                addn = add / (D_obj + 1e-8)

                # Optional 2D reprojection RMS
                reproj = None
                if args.compute_reproj:
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
                    "json": os.path.relpath(json_path, args.data_root),
                    "R_deg": rdeg,
                    "Tn": tn,
                    "ADD": add,
                    "ADDn": addn,
                    **({"Reproj_RMS_px": reproj} if reproj is not None else {})
                })

    if n_samples == 0:
        print("No evaluable samples (no images found in JSONs).")
        return

    # Averages
    avg_add = sum_add / n_samples
    avg_addn = sum_addn / n_samples
    avg_rdeg = sum_rdeg / n_samples
    avg_tn   = sum_tn   / n_samples
    avg_reproj = (sum_reproj / n_samples) if args.compute_reproj else None

    print("\n===== EVAL RESULTS =====")
    print(f"Samples: {n_samples}")
    print(f"R_deg (↓):   {avg_rdeg:.3f}")
    print(f"Tn (↓):      {avg_tn:.4f}")
    print(f"ADD (↓):     {avg_add:.6f}")
    print(f"ADDn (↓):    {avg_addn:.6f}")
    if avg_reproj is not None:
        print(f"Reproj RMS px (↓): {avg_reproj:.3f}")

    # Optional: write per-sample CSV next to ckpt
    out_csv = os.path.join(os.path.dirname(args.ckpt), "eval_results.csv")
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
