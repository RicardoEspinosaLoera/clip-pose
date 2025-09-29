import numpy as np
import torch

def look_at_Rt(position, focal, up):
    p = np.asarray(position, np.float32)
    f = np.asarray(focal, np.float32)
    u = np.asarray(up, np.float32)

    z = f - p; z = z / np.linalg.norm(z)           # forward
    x = np.cross(z, u); x = x / np.linalg.norm(x)  # right
    y = np.cross(x, z)                              # up (orthonormal)

    R_wc = np.stack([x, y, -z], axis=0)            # OpenCV convention
    t_wc = -R_wc @ p
    return R_wc.astype(np.float32), t_wc.astype(np.float32)

def quat_wxyz_to_R(q):
    w,x,y,z = q
    n = w*w+x*x+y*y+z*z
    s = 2.0/n
    R = np.array([
        [1-s*(y*y+z*z), s*(x*y - z*w), s*(x*z + y*w)],
        [s*(x*y + z*w), 1-s*(x*x+z*z), s*(y*z - x*w)],
        [s*(x*z - y*w), s*(y*z + x*w), 1-s*(x*x+y*y)]
    ], dtype=np.float32)
    return R

def compose_camera_object(json_cam, json_obj):
    fx, fy, cx, cy = json_cam['fx'], json_cam['fy'], json_cam['cx'], json_cam['cy']
    R_wc, t_wc = look_at_Rt(json_cam['position'], json_cam['focal_point'], json_cam['view_up'])
    R_wo = quat_wxyz_to_R(json_obj['pose_se3']['quaternion_wxyz'])
    t_wo = np.asarray(json_obj['pose_se3']['translation_m'], np.float32)

    R_co = R_wc @ R_wo
    t_co = R_wc @ t_wo + t_wc

    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    return K, R_co, t_co

def sixd_to_rotmat(a):  # Zhou et al.
    a1, a2 = a[..., :3], a[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (B,3,3)
