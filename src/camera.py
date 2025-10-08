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

def _rescale_K(K, Hsrc, Wsrc, Hdst, Wdst):
    if (Hsrc, Wsrc) == (Hdst, Wdst):
        return K.astype(np.float32)
    sx = float(Wdst) / float(Wsrc)
    sy = float(Hdst) / float(Hsrc)
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2

def _kaolin_cam_to_K(cam):
    """Reads Kaolin-style intrinsics."""
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    K = np.array([[fx, 0.,  cx],
                  [0.,  fy, cy],
                  [0.,  0.,  1.]], dtype=np.float32)
    return K

def camera_extrinsics_from_pyvista(cam):
    """
    Converts a PyVista-style camera definition to Kaolin-compatible extrinsics.

    Args:
        cam (dict): Must contain keys:
            - "position": [x, y, z]
            - "focal_point": [x, y, z]
            - "view_up": [x, y, z]
        device: torch device

    Returns:
        R (torch.Tensor): (1,3,3) world-to-camera rotation matrix
        t (torch.Tensor): (1,3) world-to-camera translation vector
    """

    # Extract fields
    pos = np.array(cam["position"], dtype=np.float32)
    focal = np.array(cam["focal_point"], dtype=np.float32)
    up = np.array(cam["view_up"], dtype=np.float32)

    # PyVista camera convention:
    # forward = (focal - position) -> looks along -Z
    forward = focal - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Camera-to-world (VTK)
    R_c2w = np.stack([right, up, -forward], axis=1)  # (3,3)

    # World-to-camera (Kaolin)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos

    # Convert to Kaolin (+Z forward)
    R_fix = np.diag([1, 1, -1])  # flip Z axis
    R_final = R_fix @ R_w2c
    t_final = R_fix @ t_w2c

    R = torch.from_numpy(R_final).float().unsqueeze(0)
    t = torch.from_numpy(t_final).float().unsqueeze(0)

    return R, t

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
    Returns:
        K (3x3), R_co (3x3), t_co (3,)
    - Reads PyVista camera and object pose directly.
    - Converts to Kaolin's +Z-forward convention.
    """

    # --- Intrinsics ---
    K = _kaolin_cam_to_K(cam)

    # --- Camera extrinsics from PyVista world -> Kaolin camera space ---
    R_cam, t_cam = camera_extrinsics_from_pyvista(cam)
    R_cam = R_cam.squeeze(0).numpy()
    t_cam = t_cam.squeeze(0).numpy()

    # --- Object pose from PyVista (world coordinates) ---
    R_obj, t_obj = _world_obj_to_obj2cam(clip)  # rotation & translation (world)

    # --- Convert object world pose into camera coordinates ---
    R_co = R_cam @ R_obj
    t_co = R_cam @ t_obj + t_cam

    # --- Sanity check: ensure object is in front of camera ---
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
