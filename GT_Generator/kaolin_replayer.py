# pip install kaolin trimesh imageio torch
import json, math
import numpy as np
import torch
import imageio.v2 as imageio
import trimesh

# -----------------------------
# Utilities
# -----------------------------
def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-12)

def quat_wxyz_to_R(q):
    """Quaternion [w, x, y, z] → 3x3 rotation (right-handed)."""
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=torch.float32)

def pyvista_camera_to_extrinsics(position, focal_point, view_up):
    """Rebuild camera frame from PyVista camera (world coords)."""
    C = torch.as_tensor(position, dtype=torch.float32)
    F = torch.as_tensor(focal_point, dtype=torch.float32)
    U = torch.as_tensor(view_up,    dtype=torch.float32)

    z = normalize(F - C)                           # forward (+Z)
    x = normalize(torch.linalg.cross(z, U))       # right
    y = torch.linalg.cross(x, z)                  # up

    R_cw = torch.stack([x, y, z], dim=0)          # rows are basis
    t_cw = -R_cw @ C
    return R_cw, t_cw

def intrinsics_from_size_or_K(H, W, fx=None, fy=None, cx=None, cy=None, vfov_deg=None, K_json=None):
    if K_json is not None:
        K = torch.tensor(K_json, dtype=torch.float32)
        return K
    if fx is None or fy is None:
        assert vfov_deg is not None, "Need fx/fy or vfov_deg."
        vfov = math.radians(vfov_deg)
        fy = 0.5 * H / math.tan(vfov * 0.5)
        fx = fy * (W / H)
    if cx is None: cx = (W - 1) * 0.5
    if cy is None: cy = (H - 1) * 0.5
    K = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], dtype=torch.float32)
    return K

def project_to_ndc(verts_cam, K, H, W):
    """Return (x_ndc,y_ndc) and camera-space z>0."""
    x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2].clamp_min(1e-6)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    x_ndc =  2.0 * (u / (W - 1.0)) - 1.0
    y_ndc = -2.0 * (v / (H - 1.0)) + 1.0    # flip to top-left origin
    return torch.stack([x_ndc, y_ndc], dim=-1), z

# -----------------------------
# Kaolin imports (after torch)
# -----------------------------
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.render.mesh import dibr_rasterization as dibr

# -----------------------------
# Core renderer
# -----------------------------
def render_like_pyvista(verts_obj, faces, R_ow, t_ow,
                        cam_position, cam_focal_point, cam_view_up,
                        H, W, K,
                        v_rgb=None, bg_color=(0.0, 0.0, 1.0)):
    device = verts_obj.device

    faces = faces.to(device).long()
    R_ow  = R_ow.to(device).float()
    t_ow  = t_ow.to(device).float()
    K     = K.to(device).float()

    # 1) Object -> World
    verts_world = (verts_obj @ R_ow.T) + t_ow

    # 2) World -> Camera (match PyVista)
    R_cw, t_cw = pyvista_camera_to_extrinsics(cam_position, cam_focal_point, cam_view_up)
    R_cw = R_cw.to(device); t_cw = t_cw.to(device)
    verts_cam = (verts_world @ R_cw.T) + t_cw   # z>0 in front

    # 3) Project to NDC (x,y) + keep camera-space z
    verts_ndc, verts_z = project_to_ndc(verts_cam, K, H, W)

    # 4) Per-face gather
    fvi_xy = index_vertices_by_faces(verts_ndc[None, ...], faces)      # (1,F,3,2)
    fvi_z  = index_vertices_by_faces(verts_z[None, ...].unsqueeze(-1), faces).squeeze(-1)  # (1,F,3)

    # 5) Features (RGB)
    if v_rgb is None:
        v_rgb = torch.ones_like(verts_obj, dtype=torch.float32, device=device) * 0.75
    face_features = index_vertices_by_faces(v_rgb[None, ...], faces)   # (1,F,3,3)

    # -------- 5) normals (neutral-ish) --------
    fvc_cam = index_vertices_by_faces(verts_cam[None, ...], faces)   # (F,3,3)
    
    e1 = fvc_cam[:, 1] - fvc_cam[:, 0]                       # (F,3)
    e2 = fvc_cam[:, 2] - fvc_cam[:, 0]                       # (F,3)
    n  = torch.linalg.cross(e1, e2)                          # (F,3)
    fnz = n[:, 2].unsqueeze(0).contiguous()                  # (1,F)

    # 6) Rasterize
    im_feat, im_alpha, _ = dibr(
        height=H, width=W,
        face_vertices_z=fvi_z,
        face_vertices_image=fvi_xy,
        face_features=face_features,
        face_normals_z=fnz
    )  # im_feat: (1,H,W,3), im_alpha: (1,H,W,1)

    # 7) Composite over background & gamma
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device).view(1,1,1,3).expand(1,H,W,3)
    alpha = im_alpha.clamp(0, 1)
    rgb = im_feat * alpha + bg * (1.0 - alpha)
    rgb = rgb.clamp(0, 1) ** (1/2.2)    # sRGB-ish

    return rgb[0]  # (H,W,3)

# -----------------------------
# JSON loader (robust to formats)
# -----------------------------
def load_example(json_path, device='cpu'):
    """
    Expects something like:
    {
      "camera": {
        "fx":..., "fy":..., "cx":..., "cy":..., "W":..., "H":...,
        "position":[...], "focal_point":[...], "view_up":[...]
        // or "vfov_deg": 30
        // or "K": [[...],[...],[...]]
      },
      "clip": {
        "quaternion_wxyz":[w,x,y,z],
        "translation_m":[tx,ty,tz]
      }
    }
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    cam = meta.get('camera', {})
    clip = meta.get('clip', {})

    W = int(cam.get('W') or cam.get('width') or cam.get('w') or 800)
    H = int(cam.get('H') or cam.get('height') or cam.get('h') or 544)

    # Intrinsics
    K_json = cam.get('K')
    fx = cam.get('fx'); fy = cam.get('fy')
    cx = cam.get('cx'); cy = cam.get('cy')
    vfov_deg = cam.get('vfov_deg')
    K = intrinsics_from_size_or_K(H, W, fx, fy, cx, cy, vfov_deg, K_json)

    # Camera pose (PyVista-style)
    cam_position    = cam.get('position')    or cam.get('pos') or [0., 0., -1.]
    cam_focal_point = cam.get('focal_point') or cam.get('target') or [0., 0., 0.]
    cam_view_up     = cam.get('view_up')     or cam.get('up') or [0., 1., 0.]

    # Object->World pose
    if 'quaternion_wxyz' in clip:
        R_ow = quat_wxyz_to_R(clip['quaternion_wxyz'])
    else:
        R_ow = torch.eye(3, dtype=torch.float32)
    t_ow = torch.tensor(clip.get('translation_m', [0., 0., 0.]), dtype=torch.float32)

    return R_ow.to(device), t_ow.to(device), K.to(device), (H, W), cam_position, cam_focal_point, cam_view_up

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- Load mesh
    m = trimesh.load("Item.obj", process=True)
    verts_np = np.asarray(m.vertices, dtype=np.float32)
    faces_np = np.asarray(m.faces, dtype=np.int64)

    # Vertex colors if available
    if hasattr(m, "visual") and hasattr(m.visual, "vertex_colors") and len(m.visual.vertex_colors) == len(verts_np):
        vc = np.asarray(m.visual.vertex_colors, dtype=np.float32)
        if vc.shape[-1] == 4: vc = vc[:, :3]
        v_rgb_np = (vc / 255.0).astype(np.float32)
    else:
        v_rgb_np = np.ones_like(verts_np, dtype=np.float32) * 0.75

    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int64,    device=device)
    v_rgb = torch.tensor(v_rgb_np, dtype=torch.float32,  device=device)

    # ---- Load example camera & poses
    R_ow, t_ow, K, (H, W), cam_position, cam_focal_point, cam_view_up = load_example("001.json", device=device)

    # ---- Render
    rgb = render_like_pyvista(
        verts, faces, R_ow, t_ow,
        cam_position, cam_focal_point, cam_view_up,
        H, W, K,
        v_rgb=v_rgb,
        bg_color=(0.0, 0.0, 1.0)
    )  # (H,W,3) float on device

    # ---- Save
    rgb_np = (rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # HxWx3
    imageio.imwrite("kaolin_render.png", rgb_np)
    print("Saved: kaolin_render.png")
