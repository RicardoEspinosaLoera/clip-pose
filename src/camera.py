# src/camera.py
import math
import numpy as np
import torch

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + z*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

# -------------------------------------------------------------------
# ✅ New: Kaolin-friendly camera/object composition
# -------------------------------------------------------------------
def _kaolin_cam_to_K(cam):
    """Reads Kaolin-style intrinsics."""
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    K = np.array([[fx, 0.,  cx],
                  [0.,  fy, cy],
                  [0.,  0.,  1.]], dtype=np.float32)
    return K

def _world_obj_to_obj2cam(clip):
    """
    Reads object pose in camera coordinates directly from Kaolin JSON.
    JSON structure:
      clip["pose_se3"]["rotation"]: 3x3
      clip["pose_se3"]["translation_m"]: [3]
    """
    pose = clip.get("pose_se3", {})
    if "rotation" in pose:
        R_co = np.array(pose["rotation"], dtype=np.float32)
        t_co = np.array(pose["translation_m"], dtype=np.float32)
        return R_co, t_co
    elif "quaternion_wxyz" in pose:
        q = np.array(pose["quaternion_wxyz"], dtype=np.float32)
        R_co = _quat_wxyz_to_R(q)
        t_co = np.array(pose["translation_m"], dtype=np.float32)
        return R_co, t_co
    elif "R_co" in clip and "t_co" in clip:
        return np.array(clip["R_co"], dtype=np.float32), np.array(clip["t_co"], dtype=np.float32)
    else:
        raise KeyError("clip must contain rotation/translation in pose_se3.")

def compose_camera_object(cam, clip, H, W, strict=True):
    """
    Returns: K (3x3), R_co (3x3), t_co (3,)
    - Compatible with Kaolin ground truth generator.
    - K is directly read from JSON (no PyVista conversion).
    """
    K = _kaolin_cam_to_K(cam)
    R_co, t_co = _world_obj_to_obj2cam(clip)

    if strict and not (t_co[2] > 0.0):
        raise ValueError(f"compose_camera_object: tz<=0 (tz={t_co[2]:.6f}). Object behind camera?")

    return K.astype(np.float32), R_co.astype(np.float32), t_co.astype(np.float32)

# -------------------------------------------------------------------
# Rotation 6D → Matrix (unchanged)
# -------------------------------------------------------------------
def sixd_to_rotmat(a):
    a1, a2 = a[..., :3], a[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)
