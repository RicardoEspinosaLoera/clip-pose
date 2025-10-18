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

def K_from_pyvista(plotter):
    cam = plotter.camera
    W, H = map(int, plotter.window_size)

    fov_deg = float(cam.GetViewAngle())
    use_h   = bool(cam.GetUseHorizontalViewAngle())  # False ⇒ vertical FOV (default)

    if use_h:
        fx = (W * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fy = fx * (H / W)
    else:
        fy = (H * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fy * (W / H)

    # principal point from WindowCenter (VTK y-up, image y-down)
    cx0, cy0 = (W - 1) * 0.5, (H - 1) * 0.5
    wcx, wcy = cam.GetWindowCenter()
    cx = cx0 + wcx * cx0
    cy = cy0 - wcy * cy0   # flip sign

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    return K


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def quat_wxyz_to_R(q):
    """Quaternion [w, x, y, z] → 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def kaolin_cam_to_K(cam, image_size=None, affine_xy=None):
    """
    Build a 3×3 intrinsic matrix from a PyVista-style camera dict.

    Args:
        cam: dict with fields like
             {
               "view_angle": float,            # degrees (VTK's ViewAngle)
               "use_horizontal_fov": bool,     # whether it's horizontal or vertical FOV
               "window_center": [wcx, wcy],    # normalized shift
               "position": [...],              # optional (ignored here)
               "focal_point": [...],            # optional
               "view_up": [...],                # optional
               "window_size": [W, H],          # pixel dimensions
             }
        image_size: optional (W, H); overrides cam["window_size"] if given
        affine_xy: optional (sx, sy, tx, ty); bake screenshot affine (PyVista compositor)
    """
    import numpy as np

    # --- Get size
    if image_size is not None:
        W, H = map(int, image_size)
    elif "window_size" in cam:
        W, H = map(int, cam["window_size"])
    else:
        raise ValueError("Missing window size for intrinsic computation.")

    # --- FOV
    fov_deg = float(cam.get("view_angle", 30.0))
    use_h = bool(cam.get("use_horizontal_fov", False))
    if use_h:
        fx = (W * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fy = fx * (H / W)
    else:
        fy = (H * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fy * (W / H)

    # --- Principal point from WindowCenter (VTK y-up → image y-down)
    cx0, cy0 = (W - 1) * 0.5, (H - 1) * 0.5
    wcx, wcy = cam.get("window_center", [0.0, 0.0])
    cx = cx0 + wcx * cx0
    cy = cy0 - wcy * cy0

    # --- Base intrinsics
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    # --- Optional 2D affine (screenshot correction)
    if affine_xy is not None:
        sx, sy, tx, ty = affine_xy
        A = np.array([[sx, 0.0, tx],
                      [0.0, sy, ty],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        K = A @ K

    return K

	
def sixd_to_rotmat(a):
    a1, a2 = a[..., :3], a[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)