# src/camera.py
import math
import numpy as np

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def _quat_wxyz_to_R(q):
    # q: [w, x, y, z]
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def _pyvista_cam_to_w2c(cam, H, W):
    """
    Build Kaolin-style world→camera:
      - +Z forward (visible points have z>0)
      - pixel origin top-left (we only build K; your renderer can flip v if needed)
    """
    C = np.array(cam['position'],    dtype=np.float32)
    F = np.array(cam['focal_point'], dtype=np.float32)
    U = np.array(cam['view_up'],     dtype=np.float32)

    # NOTE: +Z forward for Kaolin → use (C - F), not (F - C)
    z_cam = _normalize(C - F)
    x_cam = _normalize(np.cross(U, z_cam))
    y_cam = np.cross(z_cam, x_cam)

    R_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)     # columns are camera axes in world
    R_w2c = R_c2w.T.astype(np.float32).copy()
    t_w2c = (-R_w2c @ C).astype(np.float32)

    if all(k in cam for k in ('fx', 'fy', 'cx', 'cy')):
        fx, fy, cx, cy = float(cam['fx']), float(cam['fy']), float(cam['cx']), float(cam['cy'])
    else:
        # Fall back to vertical FOV if fx/fy not stored
        vdeg = float(cam.get('view_angle', 30.0))
        vFOV = math.radians(vdeg)
        fy = (H * 0.5) / math.tan(vFOV * 0.5)
        fx = fy
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5

    K = np.array([[fx, 0.,  cx],
                  [0.,  fy, cy],
                  [0.,  0.,  1.]], dtype=np.float32)
    return R_w2c, t_w2c, K

def _world_obj_to_obj2cam(clip, R_w2c, t_w2c):
    """
    clip: dict with pose in WORLD coords, e.g.:
      clip['pose_se3']['quaternion_wxyz']  (len 4)
      clip['pose_se3']['translation_m']    (len 3)
    Returns R_co, t_co (object→camera), tz>0 expected with the camera above.
    """
    if 'pose_se3' in clip:
        q = np.array(clip['pose_se3']['quaternion_wxyz'], dtype=np.float32)
        t_w = np.array(clip['pose_se3']['translation_m'], dtype=np.float32)
        R_wo = _quat_wxyz_to_R(q)
        R_co = (R_w2c @ R_wo).astype(np.float32)
        t_co = (R_w2c @ t_w + t_w2c).astype(np.float32)
        return R_co, t_co
    elif 'R_co' in clip and 't_co' in clip:
        # Already obj→cam in JSON
        return np.array(clip['R_co'], dtype=np.float32), np.array(clip['t_co'], dtype=np.float32)
    else:
        raise KeyError("clip must contain either 'pose_se3' (world pose) or 'R_co'/'t_co' (obj→cam).")

def _rescale_K(K, Hsrc, Wsrc, Hdst, Wdst):
    if (Hsrc, Wsrc) == (Hdst, Wdst):
        return K.astype(np.float32)
    sx = float(Wdst) / float(Wsrc)
    sy = float(Hdst) / float(Hsrc)
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2

def compose_camera_object(cam, clip, H, W, strict=True):
    """
    Adapts PyVista camera/object to Kaolin-friendly GT.
    Returns: K (3x3), R_co (3x3), t_co (3,)
    - Kaolin convention: +Z forward; tz>0 for visible objects.
    - K is for the provided (H, W).
    """
    R_w2c, t_w2c, K = _pyvista_cam_to_w2c(cam, H, W)
    R_co, t_co = _world_obj_to_obj2cam(clip, R_w2c, t_w2c)

    if strict and not (t_co[2] > 0.0):
        raise ValueError(f"compose_camera_object: tz<=0 (tz={t_co[2]:.6f}). "
                         "Check camera conversion or your source JSON.")
    return K.astype(np.float32), R_co.astype(np.float32), t_co.astype(np.float32)
