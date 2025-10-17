# src/camera.py
import math
import numpy as np
import torch

def world_to_camera_from_vtk(C_w, F_w, u_w):
    C, F, U = map(lambda x: np.asarray(x, dtype=np.float64), (C_w, F_w, u_w))

    z = F - C; z /= (np.linalg.norm(z)+1e-12)         # +Z forward
    x = np.cross(z, U); x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(x, z)

    R_wc = np.stack([x, -y, z], 0).astype(np.float32)  # world→cam
    t_wc = (-R_wc @ C).astype(np.float32)
    return R_wc, t_wc


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + z*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def kaolin_cam_to_K(cam):
    """Reads Kaolin-style intrinsics."""
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    K = np.array([[fx, 0.,  cx],
                  [0.,  fy, cy],
                  [0.,  0.,  1.]], dtype=np.float32)
    return K
	
def sixd_to_rotmat(a):
    a1, a2 = a[..., :3], a[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)