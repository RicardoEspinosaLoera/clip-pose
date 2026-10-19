import json
import numpy as np
import pyvista as pv
import torch
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rscipy
from kaolin.render.mesh import dibr_rasterization as dibr
from kaolin.ops.mesh import index_vertices_by_faces
import cv2

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def quat_wxyz_to_R(q):
    """Quaternion [w, x, y, z] → 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def object_to_camera_from_se3(clip_se3, device):
    q = clip_se3["quaternion_wxyz"]  # [w,x,y,z]
    t = np.array(clip_se3["translation_m"], np.float32)
    R = Rscipy.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)
    return (torch.from_numpy(R).to(device).unsqueeze(0),
            torch.from_numpy(t).to(device).unsqueeze(0))

def load_json_gt(path):
    with open(path, 'r') as f:
        gt = json.load(f)
    cam = gt["camera"]
    clip_world = gt["clip"]["pose_world"]  
    clip_se3 = gt["clip"]["pose_se3"]
    image = gt["image"]
    return cam, clip_world,clip_se3,image



def camera_extrinsics_from_pyvista(cam, device):
    pos = np.array(cam["position"], dtype=np.float32)
    focal = np.array(cam["focal_point"], dtype=np.float32)
    up = np.array(cam["view_up"], dtype=np.float32)

    # PyVista convention: camera looks along (focal - pos)
    forward = focal - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Camera-to-world → World-to-camera
    R_c2w = np.stack([right, up, -forward], axis=1)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos

    # Convert to Kaolin convention (+Z forward)
    R_fix = np.diag([1, 1, -1])
    R_w2c = R_fix @ R_w2c
    t_w2c = R_fix @ t_w2c

    return torch.tensor(R_w2c, dtype=torch.float32, device=device), torch.tensor(t_w2c, dtype=torch.float32, device=device)

def get_pyvista_camera_tuple(plotter):
    cam = plotter.camera
    pos   = np.array(cam.position, dtype=np.float32)
    focal = np.array(cam.focal_point, dtype=np.float32)
    vup   = np.array(cam.up, dtype=np.float32)  # already orthonormalized by VTK
    return (pos, focal, vup)
    
# -------------------------------------------------
# PyVista reconstruction
# -------------------------------------------------


def apply_intrinsics_to_pyvista_camera(plotter, fx, fy, cx, cy, W, H):
    """
    Make PyVista/VTK camera match Kaolin intrinsics:
      - Vertical FOV from fy (no extra zoom)
      - Principal point via WindowCenter
      - Full viewport, no AA, no padding
    """
    cam = plotter.camera

    # ---- Vertical FOV from fy (NumPy needs a Python float) ----
    fy_val   = float(fy)
    vfov_deg = np.degrees(2.0 * np.arctan((H * 0.5) / fy_val))

    cam.SetParallelProjection(False)          # perspective
    cam.SetUseHorizontalViewAngle(False)      # 'ViewAngle' = vertical FOV
    cam.SetViewAngle(vfov_deg)                # set vfov

    # ---- Principal point -> WindowCenter (normalized offsets) ----
    # VTK WindowCenter is relative to image center, in units of half-width/half-height.
    # +x right, +y up. Image coords usually have +y down, so we flip y.
    cx0 = (W - 1) * 0.5
    cy0 = (H - 1) * 0.5
    wcx = (float(cx) - cx0) / max(cx0, 1e-6)
    wcy = (cy0 - float(cy)) / max(cy0, 1e-6)  # flip: image y↓ -> VTK y↑
    cam.SetWindowCenter(wcx, wcy)

    # ---- Viewport/framebuffer hygiene ----
    try:
        plotter.set_viewport(0.0, 0.0, 1.0, 1.0)   # full viewport
    except Exception:
        pass

    try:
        plotter.enable_anti_aliasing(False)       # avoid half-pixel shifts
    except Exception:
        pass

    plotter.window_size = (int(W), int(H))
    plotter._theme.smooth_shading = True          # lighting only
    cam.clipping_range = (1e-3, 1e6)              # wide, won’t crop

    # IMPORTANT: after calling this, do NOT call reset_camera() or cam.Zoom()
    # and avoid add_background_image while debugging (can introduce letterboxing)

    return dict(vfov_deg=vfov_deg, window_center=(wcx, wcy))

def K_from_pyvista(plotter):
    cam = plotter.camera
    W, H = map(int, plotter.window_size)

    fov_deg = float(cam.GetViewAngle())
    #print("fov_deg",fov_deg)
    use_h   = bool(cam.GetUseHorizontalViewAngle())  # False ⇒ vertical FOV (default)
    #vprint("use_h",use_h)
    if use_h:
        fx = (W * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fy = fx * (H / W)
    else:
        fy = (H * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fy * (W / H)

    # principal point from WindowCenter (VTK y-up, image y-down)
    cx0, cy0 = (W - 1) * 0.5, (H - 1) * 0.5
    wcx, wcy = cam.GetWindowCenter()
    print("wcx, wcy",wcx, wcy)
    cx = cx0 + wcx * cx0
    cy = cy0 - wcy * cy0   # flip sign

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    # debug
    print(f"[PYVISTA→K] FOV(deg)={fov_deg:.4f} use_h={use_h}  fx={fx:.2f} fy={fy:.2f}  "
          f"cx={cx:.2f} cy={cy:.2f}  W×H={W}×{H}")
    return K


def render_pyvista(mesh_path, cam, clip_world,verts, image_size=(800, 544), bg="red"):
    W, H = image_size
    mesh = pv.read(mesh_path)

    # ----- object → world (from JSON) -----
    q = clip_world["quaternion_wxyz"]
    R = quat_wxyz_to_R(q)
    t = np.array(clip_world["translation_m"], dtype=np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = t
    mesh.transform(T, inplace=True)

    print("Pyvista --- R & t")
    print(R)
    print(t)

    # ----- plotter -----
    p = pv.Plotter(off_screen=True, window_size=(W, H))
    p.set_background(bg)

    # Add geometry WITHOUT resetting or auto-fitting the camera/frustum
    p.add_mesh(mesh, color="silver", smooth_shading=True, reset_camera=False)

    # ----- intrinsics (must match your K exactly) -----
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    apply_intrinsics_to_pyvista_camera(p, fx, fy, cx, cy, W, H)
    # ----- pose (extrinsics) -----
    p.camera.position    = tuple(cam["position"])
    p.camera.focal_point = tuple(cam["focal_point"])
    p.camera.up          = tuple(cam["view_up"])

    # Force a render, then capture pixels as-is (no scaling)
    p.render()
    #img = p.screenshot(return_img=True)  # uint8 [H,W,3], origin=top-left
    img = p.screenshot(None, return_img=True, window_size=(W, H), scale=1.0)
    K_pv = K_from_pyvista(p)


    ##########################
    # V_torch = verts
    # if isinstance(V_torch, np.ndarray):
    #     V_np = V_torch.astype(np.float32)
    # else:
    #     V_np = V_torch.detach().cpu().numpy().astype(np.float32)

    # # Sample N points
    # N = min(600, V_np.shape[0])
    # idx = np.random.choice(V_np.shape[0], size=N, replace=False)
    # P_obj = V_np[idx]                                  # (N,3) object-space

    # # object->world (Row, tow)
    # Row = R.astype(np.float32).reshape(3,3)
    # tow = t.astype(np.float32).reshape(1,3)
    # P_world = P_obj @ Row.T + tow                      # (N,3)

    # def kaolin_pose_from_pv_camera(pv_cam):
    #     C = np.array(pv_cam.GetPosition(),   np.float32)
    #     F = np.array(pv_cam.GetFocalPoint(), np.float32)
    #     U = np.array(pv_cam.GetViewUp(),     np.float32)

    #     f = F - C; f = f / (np.linalg.norm(f) + 1e-12)       # forward (+Z)
    #     r = np.cross(f, U); r = r / (np.linalg.norm(r) + 1e-12)
    #     u = np.cross(r, f)

    #     # y-down pixels later; for camera frame keep y-up here, we won’t use sign in Xc
    #     R_wc = np.stack([r, -u, f], axis=0).astype(np.float32)  # world->camera (y-down pixels if used directly)
    #     t_wc = (-R_wc @ C).astype(np.float32)
    #     return R_wc, t_wc

    # # world->camera from PyVista camera
    # R_wc, t_wc = kaolin_pose_from_pv_camera(p.camera)  # (3,3), (3,)
    # twc = t_wc.reshape(1,3)

    # def fx_fy_cx_cy_from_K(K):
    #     if isinstance(K, torch.Tensor):
    #         if K.ndim == 3:
    #             return float(K[0,0,0]), float(K[0,1,1]), float(K[0,0,2]), float(K[0,1,2])
    #         return float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    #     K = np.asarray(K)
    #     if K.ndim == 3:
    #         return float(K[0,0,0]), float(K[0,1,1]), float(K[0,0,2]), float(K[0,1,2])
    #     return float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    # # Kaolin-style camera coords and pixel projection (NO +0.5 here)
    # Xc = P_world @ R_wc.T + twc                        # (N,3)
    # xy = Xc[:, :2] / Xc[:, 2:3]
    # fx_pv, fy_pv, cx_pv, cy_pv = fx_fy_cx_cy_from_K(K_pv)
    # uK = fx_pv * xy[:, 0] + cx_pv
    # vK = fy_pv * xy[:, 1] + cy_pv

    # # VTK projection → bottom-left display coords → convert to TOP-LEFT pixels
    # uvV = vtk_project(P_world, p)                      # (N,2) bottom-left
    # uV = uvV[:, 0]
    # vV = (H - 1) - uvV[:, 1]                           # top-left

    # # Fit uV ≈ sx*uK + tx ; vV ≈ sy*vK + ty
    # A = np.c_[uK, np.ones_like(uK)]
    # sx, tx = np.linalg.lstsq(A, uV, rcond=None)[0]
    # A = np.c_[vK, np.ones_like(vK)]
    # sy, ty = np.linalg.lstsq(A, vV, rcond=None)[0]
    # print(f"[AFFINE] sx={sx:.6f}, tx={tx:.3f}, sy={sy:.6f}, ty={ty:.3f}")
    # # ======================================================================

    ##########################


    p.close()
    return img ,K_pv


import torch, numpy as np

def _faces_F3(faces):
    """Normalize faces to torch.long [F,3]. Handles PyVista packed arrays."""
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces)
    faces = faces.to(torch.long).contiguous()
    if faces.ndim == 1:
        # packed [3,i,j,k, 3,i,j,k, ...]
        if (faces.numel() % 4 == 0) and torch.all(faces.view(-1, 4)[:, 0] == 3):
            faces = faces.view(-1, 4)[:, 1:]
        else:
            assert faces.numel() % 3 == 0, "Flat faces must be multiple of 3."
            faces = faces.view(-1, 3)
    elif faces.ndim == 3 and faces.shape[0] == 1:
        faces = faces[0]
    assert faces.ndim == 2 and faces.shape[1] == 3, f"Got faces {tuple(faces.shape)}, want [F,3]."
    return faces

def gather_by_faces(vertices_features, faces):
    """
    vertices_features: [V,K] or [B,V,K] (torch/np)
    faces: [F,3] long
    returns: [B,F,3,K]
    """
    if isinstance(vertices_features, np.ndarray):
        vertices_features = torch.from_numpy(vertices_features)
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces)
    faces = faces.to(torch.long).contiguous()
    vf = vertices_features
    if vf.dim() == 2:
        vf = vf.unsqueeze(0)  # -> [1,V,K]
    return vf[:, faces, :]    # [B,F,3,K]


# -------------------------------------------------
# Functional Kaolin renderer
# -------------------------------------------------

def pixels_to_ndc(u, v, W, H, *, center_offset=0.5, use_wminus1=False, y_up=True):
    u = u + center_offset
    v = v + center_offset
    x = (u / float(W)) * 2.0 - 1.0
    y = (v / float(H)) * 2.0 - 1.0
    if y_up:
        y = -y
    return x, y


def world_to_camera_from_vtk(C_w, F_w, u_w):
    C, F, U = map(lambda x: np.asarray(x, dtype=np.float64), (C_w, F_w, u_w))

    z = F - C; z /= (np.linalg.norm(z)+1e-12)         # +Z forward
    x = np.cross(z, U); x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(x, z)

    R_wc = np.stack([x, -y, z], 0).astype(np.float32)  # world→cam
    t_wc = (-R_wc @ C).astype(np.float32)
    return R_wc, t_wc

    

def kaolin_pose_from_pyvista_camera(cam):
    C = np.array(cam.GetPosition(),   np.float32)
    F = np.array(cam.GetFocalPoint(), np.float32)
    U = np.array(cam.GetViewUp(),     np.float32)

    f = F - C; f /= (np.linalg.norm(f)+1e-12)      # forward (world, +Z forward)
    r = np.cross(f, U); r /= (np.linalg.norm(r)+1e-12)  # right
    u = np.cross(r, f)                              # up (y-up)
    R_wc = np.stack([r, -u, f], 0).astype(np.float32)   # y-down in image
    t_wc = (-R_wc @ C).astype(np.float32)
    return R_wc[None], t_wc[None]


# ---- small face gather (version-proof) ----
def gather_by_faces(vertices_features, faces_):
    vf = vertices_features
    if isinstance(vf, np.ndarray):
        vf = torch.from_numpy(vf).to(device)
    if vf.dim() == 2:  # [V,K] -> [1,V,K]
        vf = vf.unsqueeze(0)
    return vf[:, faces_, :]


def vtk_project(points_world, plotter):
    """
    points_world: any array-like of shape (N,3) (torch or numpy; any dtypes)
    returns: (N,2) numpy float64 pixel coords from the active VTK camera
    """
    # 1) Force clean (N,3) numpy array of float64 scalars.
    P = points_world
    if hasattr(P, "detach"):  # torch tensor
        P = P.detach().cpu().numpy()
    P = np.asarray(P)
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)  # ensure (N,3), no extra dims

    ren = plotter.renderer
    ren.ResetCameraClippingRange()

    uv = []
    for i in range(P.shape[0]):
        # 2) Extract plain Python floats (avoid length-1 arrays / matrices).
        x = float(np.ravel(P[i, 0])[0])
        y = float(np.ravel(P[i, 1])[0])
        z = float(np.ravel(P[i, 2])[0])

        ren.SetWorldPoint(x, y, z, 1.0)
        ren.WorldToDisplay()
        xd, yd, _ = ren.GetDisplayPoint()
        uv.append((xd, yd))

    return np.asarray(uv, dtype=np.float64)

# def render_mesh(
#     verts, faces, R, t, K, image_size,
#     v_rgb=None,
#     debug=False,
#     cam=None,                 # dict with fx,fy,cx,cy (only needed for debug RMS)
#     overlay_img=None,         # PyVista RGB for overlay plot (optional)
#     center_offset=0.5,        # try 0.5 (GL pixel centers) or 0.0 (pixel corners)
#     affine_xy=None,           # (sx, sy, tx, ty) to mimic PyVista’s screenshot (optional)
# ):
#     """
#     verts: [V,3] float (torch, on device)
#     faces: [F,3] long  (torch)
#     R,t:   [1,3,3], [1,3]  object->camera
#     K:     [1,3,3]
#     image_size: (H, W)

#     Returns:
#       rgb: [1,3,H,W], sil: [1,1,H,W], aux: dict with uv_pixels & bboxes (for debugging)
#     """
#     device = verts.device
#     H, W = int(image_size[0]), int(image_size[1])
#     B = R.shape[0]
#     #print(H,W)
#     #verts = verts.unsqueeze(0)

    
#     # -------- 1) Clip world/object -> camera --------
#     v_cam = torch.einsum('bij,vj->bvi', R, verts) + t[:, None, :]  # [B,V,3]
    
#     print("K for renderer:", K)
#     # -------- 2) project to pixels (y-down pixel convention) --------
#     Z  = v_cam[..., 2:3].clamp(min=1e-6)
#     xy = v_cam[..., :2] / Z

#     # -------- 2) prepare intrinsics for THIS raster size --------
#     # K may have been built for a different pixel grid (e.g., window_size vs framebuffer)
#     K_use = K.clone()

#     fx = K[:, 0, 0].view(-1, 1, 1)
#     fy = K[:, 1, 1].view(-1, 1, 1)
#     cx = K[:, 0, 2].view(-1, 1, 1)
#     cy = K[:, 1, 2].view(-1, 1, 1)
    
#     u =  fx * xy[..., 0:1] + cx 
#     v = fy * xy[..., 1:2] + cy    
#     uv_pixels = torch.cat([u, v], dim=-1)[0]     # [V,2] for diagnostics/overlay
#     v_img = torch.cat([u, v, Z], dim=-1)         # [B,V,3] (u,v,Zpix)

#     # -------- per-vertex color --------
#     if v_rgb is None:
#         v_rgb = torch.full((verts.shape[0], 3), 0.75, dtype=torch.float32, device=device)
#     else:
#         v_rgb = v_rgb.to(device).to(torch.float32)
#         if v_rgb.dim() == 1:  # [V] -> [V,3] (grayscale)
#             v_rgb = v_rgb.unsqueeze(-1).expand(-1, 3)
#     vfeat = v_rgb.unsqueeze(0).expand(B, -1, -1)  # [B,V,3]

#     # -------- 3) gather faces --------
#     fvcam = gather_by_faces(v_cam, faces)   # [B,F,3,3]
#     fvimg = gather_by_faces(v_img, faces)   # [B,F,3,3]
#     ffeat = gather_by_faces(vfeat, faces)   # [B,F,3,3]

#     # -------- 4) pixels -> NDC (y-down for your DIB-R build) --------
#     u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]           # [B,F,3]
    
#     # --- choose ONE convention (this matches your DIB-R call) ---
#     CENTER = center_offset          # try 0.5 (pixel centers) OR 0.0 (pixel corners), but be consistent
#     WMINUS1 = False        # True matches your earlier path; else False uses W/H
#     Y_UP = True          # you said your DIB-R path is y-down

#     if affine_xy is not None:
#         sx, sy, tx, ty = affine_xy
#         u_pix = u_pix * sx + tx
#         v_pix = v_pix * sy + ty

#     # forward
#     u_ndc, v_ndc = pixels_to_ndc(u_pix, v_pix, W, H,
#                                 center_offset=CENTER,   # try 0.5 then 0.0
#                                 use_wminus1=WMINUS1,    # try True then False
#                                 y_up=Y_UP)          # your Kaolin build looked y-down
                                

                            
#     face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)  # [B,F,3,2]
#     face_vertices_z  = fvcam[..., 2]                        # [B,F,3]

#     # -------- 5) normals (neutral-ish) --------
#     v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
#     n = torch.cross(v1 - v0, v2 - v0, dim=-1)
#     n = torch.nn.functional.normalize(n, dim=-1)
#     normals_z = n[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)

#     # -------- 6) rasterize --------
#     out = dibr(height=H, width=W,
#                face_vertices_z=face_vertices_z,
#                face_vertices_image=face_vertices_xy,
#                face_features=ffeat,
#                face_normals_z=normals_z)

#     rgb = out[0].permute(0, 3, 1, 2).clamp(0, 1)  # [B,3,H,W]
#     sil = out[1].unsqueeze(1).clamp(0, 1)         # [B,1,H,W]

#     return rgb, sil

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055).pow(2.4))

def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92*x, 1.055*x.pow(1/2.4)-0.055)

def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor):
    """
    Smooth vertex normals by area-weighted face accumulation.
    verts: [V,3] (cpu or cuda)
    faces: [F,3] long
    returns: [V,3] float
    """
    device = verts.device
    V = verts.shape[0]
    f = faces.to(torch.long)

    v0 = verts[f[:, 0]]
    v1 = verts[f[:, 1]]
    v2 = verts[f[:, 2]]

    # face normals (area weighted)
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F,3]

    # accumulate to vertices
    vn = torch.zeros((V, 3), device=device, dtype=verts.dtype)
    vn.index_add_(0, f[:, 0], fn)
    vn.index_add_(0, f[:, 1], fn)
    vn.index_add_(0, f[:, 2], fn)

    # normalize; avoid div by zero
    vn = torch.nn.functional.normalize(vn, dim=-1, eps=1e-12)
    return vn

def pixels_to_ndc_strict(u_pix, v_pix, W, H):
    # pixel_origin=top_left, pixel_center=0.5  →  NDC in [-1,1], y-up
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    sx = 2.0 / (W - 1)
    sy = 2.0 / (H - 1)
    x_ndc = -(u_pix - cx) * sx
    y_ndc = -(v_pix - cy) * sy
    return x_ndc, y_ndc

def choose_v_sign(Y_over_Z, fy, cy, H):
    """
    Decide whether v =  +fy*(Y/Z)+cy  or  v = -fy*(Y/Z)+cy
    by picking the one that keeps more vertices inside [0, H-1].
    """
    v_plus  =  fy * Y_over_Z + cy     # y-down camera
    v_minus = -fy * Y_over_Z + cy     # y-up camera

    def in_bounds(v):  # ratio inside the image vertically
        return ((v >= 0) & (v <= (H - 1))).float().mean()

    r_plus, r_minus = in_bounds(v_plus), in_bounds(v_minus)
    use_minus = r_minus >= r_plus      # tie → prefer minus (common with VTK)
    v = v_minus if use_minus else v_plus
    return v, use_minus

def choose_y_sign(Y_over_Z, fy, cy, H):
    v_plus  =  fy * Y_over_Z + cy
    v_minus = -fy * Y_over_Z + cy
    inb = lambda v: ((v >= 0) & (v <= (H - 1))).float().mean()
    return (v_minus if inb(v_minus) >= inb(v_plus) else v_plus)

def choose_axis_sign(val_over_Z, f, c, limit):  # works for X or Y
    v_plus  =  f * val_over_Z + c       # “no flip”
    v_minus = -f * val_over_Z + c       # “flip”
    inb = ((v_plus >= 0) & (v_plus <= (limit - 1))).float().mean()
    inb_m = ((v_minus >= 0) & (v_minus <= (limit - 1))).float().mean()
    use_minus = inb_m >= inb             # tie → prefer minus
    return (v_minus if use_minus else v_plus), use_minus

def project_pts(P, K):
    X, Y, Z = P[...,0], P[...,1], P[...,2].clamp_min(1e-6)
    fx, fy, cx, cy = K[:,0,0], K[:,1,1], K[:,0,2], K[:,1,2]
    u = fx.view(-1,1) * (X/Z).view(-1,1) + cx.view(-1,1)
    v = fy.view(-1,1) * (Y/Z).view(-1,1) + cy.view(-1,1)   # y-down pixels
    return u, v

def _probe_camera_axes(R, t, K, H, W):
    """
    Build 4 object-space points {O, O+dx, O+dy, O+dz}, transform by (R,t),
    project with "no-flip" pinhole (y-down pixel convention),
    and report the on-screen directions of +X and +Y.
    Returns: du_x, dv_x, du_y, dv_y  (floats)
    """
    device = R.device
    dtype  = R.dtype
    B = R.shape[0]

    # 4 small offsets in object frame (units don't matter, keep them tiny)
    d = torch.tensor(0.05, device=device, dtype=dtype)
    axes_obj = torch.stack([
        torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype),  # origin
        torch.tensor([d,   0.0, 0.0], device=device, dtype=dtype),  # +X
        torch.tensor([0.0, d,   0.0], device=device, dtype=dtype),  # +Y
        torch.tensor([0.0, 0.0, d  ], device=device, dtype=dtype),  # +Z
    ], dim=0)  # [4,3]

    # object -> camera
    axes_cam = torch.einsum('bij,vj->bvi', R, axes_obj) + t[:, None, :]  # [B,4,3]

    # pinhole (no sign flips here): y-down pixel convention
    X = axes_cam[..., 0]; Y = axes_cam[..., 1]; Z = axes_cam[..., 2].clamp_min(1e-6)
    fx = K[:, 0, 0].view(B, 1); fy = K[:, 1, 1].view(B, 1)
    cx = K[:, 0, 2].view(B, 1); cy = K[:, 1, 2].view(B, 1)

    u = fx * (X / Z) + cx    # [B,4]
    v = fy * (Y / Z) + cy    # [B,4]  (y-down pixels)

    # directions of +X and +Y on screen (avg over batch)
    du_x = (u[:, 1] - u[:, 0]).mean().item()
    dv_x = (v[:, 1] - v[:, 0]).mean().item()
    du_y = (u[:, 2] - u[:, 0]).mean().item()
    dv_y = (v[:, 2] - v[:, 0]).mean().item()

    print(f"probe dirs (pixels): +X→(du={du_x:.2f}, dv={dv_x:.2f}), "
          f"+Y→(du={du_y:.2f}, dv={dv_y:.2f})")
    # Expectations in OpenCV pixels: +X should move right ⇒ du_x>0, +Y should move down ⇒ dv_y>0
    return du_x, dv_x, du_y, dv_y


def render_mesh(
    verts, faces, R, t, K, image_size,
    v_rgb=None,
    debug=False,
    cam=None,                 # dict with fx,fy,cx,cy (only needed for debug RMS)
    overlay_img=None,         # PyVista RGB for overlay plot (optional)
    center_offset=0.5,        # try 0.5 (GL pixel centers) or 0.0 (pixel corners)
):

    
    


    B = R.shape[0]
    H, W = image_size
    device = verts.device
    R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

    # Prove camera access
    #du_x, dv_x, du_y, dv_y = _probe_camera_axes(R, t, K, H, W)


    # 1) object -> camera
    v_cam = torch.einsum('bij,vj->bvi', R, verts) + t[:, None, :]  # [B,V,3]
    
    Z = v_cam[..., 2:3]
    print("Z stats:", Z.min().item(), Z.median().item(), Z.max().item())
    neg_ratio = (Z <= 0).float().mean().item()
    if neg_ratio > 0:
        print(f"WARNING: {neg_ratio*100:.1f}% verts behind camera")
    Z = Z.clamp_min(1e-6)  # use this Z for division


    # 2) project
    Z  = v_cam[..., 2:3].clamp_min(1e-6)
    xy = v_cam[..., :2] / Z
    fx = K[:, 0, 0].view(-1, 1, 1);  fy = K[:, 1, 1].view(-1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1);  cy = K[:, 1, 2].view(-1, 1, 1)

    u = fx * xy[..., 0:1] + cx          # <-- NO sign flip
    v = -fy * xy[..., 1:2] + cy          # Y: flip (to match PyVista’s screen y)

    v_img = torch.cat([u, v, Z], dim=-1)  # [B,V,3]

    # 3) features: ALBEDO (sRGB) + NORMALS (camera space)
    # albedo as vertex color (your self.v_rgb is presumably sRGB in [0,1]):
    dtype  = verts.dtype
    v_rgb = torch.full((verts.shape[0], 3), 0.85, dtype=dtype, device=device)

    v_albedo_srgb = v_rgb.unsqueeze(0).expand(B, -1, -1).to(device)     # [B,V,3]
    v_albedo_lin  = srgb_to_linear(v_albedo_srgb.clamp(0,1))                 # linear for shading

    v_normals = _compute_vertex_normals(verts, faces)
    # transform vertex normals to camera space
    v_n_cam = torch.einsum('bij,vj->bvi', R, v_normals.to(device))      # [B,V,3]
    v_n_cam = torch.nn.functional.normalize(v_n_cam, dim=-1)

    # 4) gather by faces
    fvcam  = gather_by_faces(v_cam, faces)          # [B,F,3,3]
    fvimg  = gather_by_faces(v_img, faces)          # [B,F,3,3]
    falb   = gather_by_faces(v_albedo_lin, faces)   # [B,F,3,3]
    fnorm  = gather_by_faces(v_n_cam, faces)        # [B,F,3,3]

    # 5) pixels
    u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]
    
    x_ndc, y_ndc = pixels_to_ndc_strict(u_pix, v_pix, W, H) 

    face_vertices_xy = torch.stack([x_ndc, y_ndc], dim=-1)  # [B,F,3,2]
    
    face_vertices_z  = fvcam[..., 2]                        # [B,F,3]

    # 6) pack attributes: [albedo(3) | normal(3)]
    fattr = torch.cat([falb, fnorm], dim=-1)  # [B,F,3,6]


    t_cam = t.clone()
    u0 = fx * (t_cam[:,0:1] / t_cam[:,2:3]) + cx
    v0 = -fy * (t_cam[:,1:2] / t_cam[:,2:3]) + cy
    x0, y0 = pixels_to_ndc_strict(u0, v0, W, H)
    print("origin px:", float(u0), float(v0), "NDC:", float(x0), float(y0))
        

    # 7) rasterize attributes; use normals_z only for visibility heuristic
    out = dibr(height=H, width=W,
               face_vertices_z=face_vertices_z,
               face_vertices_image=face_vertices_xy,
               face_features=fattr,
               face_normals_z=fnorm[..., 2].abs())   # or your previous normals_z

    attr = out[0]         # [B,H,W,6] interpolated
    sil  = out[1].unsqueeze(1)  # [B,1,H,W]

    albedo_lin = attr[..., 0:3].permute(0,3,1,2).contiguous()   # [B,3,H,W]
    n_cam      = attr[..., 3:6]
    n_cam      = torch.nn.functional.normalize(n_cam, dim=-1)
    n_cam      = n_cam.permute(0,3,1,2).contiguous()            # [B,3,H,W]

    # 8) simple headlight shading (Phong-ish) in LINEAR space
    # view dir ~ -Z in camera space
    view_dir = torch.tensor([0,0,1.0], device=device).view(1,3,1,1).expand_as(n_cam)
    
    light_dir = view_dir  # headlight
    ndotl = torch.clamp((n_cam * light_dir).sum(1, keepdim=True), 0, 1)
    # Blinn-Phong specular
    h = torch.nn.functional.normalize(light_dir + view_dir, dim=1)
    ambient=0.10
    kd=1.0
    ks=1.0
    shininess=32
    spec = torch.clamp((n_cam * h).sum(1, keepdim=True), 0, 1).pow(shininess)

    rgb_linear = albedo_lin * (ambient + kd * ndotl) + ks * spec
    rgb_linear = rgb_linear * sil  # mask outside

    # 9) encode to sRGB for display/saving
    rgb = linear_to_srgb(rgb_linear.clamp(0,1))
    return rgb, sil

def pv_to_kaolin_extrinsics(R_pv, t_pv):
    """
    R_pv, t_pv express object->camera in PyVista camera frame.
    Returns R_k, t_k in Kaolin canonical frame (+Z forward, y-down).
    """
    # PyVista -> Kaolin (object→camera, +X right, +Y down, +Z forward)
    H = torch.tensor([[1.,0.,0.],
                    [0.,-1.,0.],
                    [0., 0.,-1.]], device=R_pv.device, dtype=R_pv.dtype)

    R_k = H @ R_pv                          # (3,3)
    t_k = H @ torch.as_tensor(t_pv, dtype=R_pv.dtype, device=R_pv.device)  # (3,)
    # If your renderer expects batch:
    R_k = R_k.unsqueeze(0)                  # (1,3,3)
    t_k = t_k.unsqueeze(0)                  # (1,3)

    return R_k, t_k

import json, torch, numpy as np

def quat_wxyz_to_R(q):
    """q = [w, x, y, z] -> 3x3 rotation (right-handed)."""
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    R = np.array([
        [ww+xx-yy-zz,   2*(x*y - w*z),  2*(x*z + w*y)],
        [2*(x*y + w*z), ww-xx+yy-zz,    2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x),  ww-xx-yy+zz  ]
    ], dtype=float)
    return R

def load_pose_for_kaolin(json_path, device="cpu", dtype=torch.float32):
    """
    Returns:
      R_co [1,3,3], t_co [1,3], K [1,3,3], (H,W)
    Such that: v_cam = R_co @ v_obj + t_co
    """
    with open(json_path, "r") as f:
        d = json.load(f)

    # --- image size
    H, W = int(d["image"]["height"]), int(d["image"]["width"])

    # --- intrinsics (assumes pixel_origin=top_left & pixel_center=0.5)
    fx, fy = float(d["camera"]["fx"]), float(d["camera"]["fy"])
    cx, cy = float(d["camera"]["cx"]), float(d["camera"]["cy"])
    K = torch.tensor([[[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]]], dtype=dtype, device=device)

    # --- pose (object -> camera) from JSON
    q_wxyz = d["clip"]["pose_se3"]["quaternion_wxyz"]
    t_m     = d["clip"]["pose_se3"]["translation_m"]   # meters
    R = quat_wxyz_to_R(q_wxyz)                         # numpy 3x3
    t = np.array(t_m, dtype=float)                     # numpy (3,)

    print("JSON --- R & t")
    print(R)
    print(t)
    print(K)

    # --- adapt to Kaolin (+Z forward) if needed
    # If the JSON stored -Z forward, flip to +Z via F = diag(1,1,-1).
    cam_forward = d.get("conventions", {}).get("cam_forward_axis", "+Z")
    if cam_forward.strip() == "-Z":
        F = np.diag([1.0, 1.0, -1.0]).astype(float)
        R = F @ R
        t = F @ t

    # (Optional) ensure det(R) ~ +1
    # If you want, you can re-orthonormalize here.

    # --- to torch
    R_co = torch.tensor(R, dtype=dtype, device=device).unsqueeze(0)  # [1,3,3]
    t_co = torch.tensor(t, dtype=dtype, device=device).unsqueeze(0)  # [1,3]

    return R_co, t_co, K, (H, W)

# Example:
# R, t, K, (H, W) = load_pose_for_kaolin("/mnt/data/001.json", device="cuda")
# v_cam = torch.einsum("bij,vj->bvi", R, verts) + t[:, None, :]


# -------------------------------------------------
# Main comparison
# -------------------------------------------------


if __name__ == "__main__":
    json_path = "002.json"
    mesh_path = "Item.obj"
    H, W = 544, 800

    cam, clip_world, clip_se3, image = load_json_gt(json_path)

    # Mesh in object space
    m = trimesh.load(mesh_path, process=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts = torch.tensor(m.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(m.faces, dtype=torch.int64, device=device)

    img_pv, k_pv= render_pyvista(mesh_path, cam, clip_world, m.vertices)

    #####################

    
    R, t, K, (H, W) = load_pose_for_kaolin(json_path, device="cuda")

    print("Kaolin --- R & t")
    print(R)
    print(t)
    print(K)

    # Call without affine:
    rgb, sil = render_mesh(verts, faces, R, t, K, (H, W), debug=True, cam=cam, overlay_img=img_pv)

   
    rgb_np = rgb[0].permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("PyVista"); plt.imshow(img_pv); plt.axis("off")
    plt.subplot(1,2,2); plt.title("Kaolin DIB-R"); plt.imshow(rgb_np, origin='upper'); plt.axis("off")
    plt.tight_layout(); plt.show()

    overlay = 0.5*img_pv.astype(float)/255 + 0.5*rgb_np
    plt.imshow(np.clip(overlay,0,1)); plt.title("PyVista ↔ Kaolin overlay"); plt.axis("off"); plt.show()

    