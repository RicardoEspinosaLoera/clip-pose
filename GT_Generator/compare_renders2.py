import sys
import os
import numpy as np
import torch
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.render.mesh import dibr_rasterization as dibr


# ---------- helpers from previous message stay the same ----------
# load_mesh, look_at_extrinsics_from_pyvista, intrinsics_from_pyvista,
# project_pixels, pack_faces_for_dibr, rasterize_silhouette, to_numpy_image
# (paste those here unchanged)

def load_mesh(path, device):
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = m.dump().sum()
    verts = torch.tensor(m.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(m.faces, dtype=torch.int64, device=device)
    return verts, faces

def maybe_enable_offscreen(plotter):
    """Make PyVista render off-screen if supported on this OS."""
    # On Linux, try to use xvfb/osmesa; on Windows/macOS just rely on off_screen=True.
    if sys.platform.startswith("linux"):
        # Prefer VTK-OSMesa if available; fall back to xvfb; otherwise continue silently.
        try:
            # Newer PyVista recommends OSMesa for headless rendering.
            # If your env has it, enabling off_screen is enough.
            pv.global_theme.off_screen = True
        except Exception:
            try:
                pv.start_xvfb()   # Linux only; safe to skip if not present
            except Exception:
                pass
    # On Windows/macOS, nothing special needed.

def intrinsics_from_pyvista(cam, W, H, device):
    """
    Use PyVista's 'focal_distance' and window size to form a pinhole K.
    If you already have fx,fy,cx,cy, just make K directly.
    """
    fx = cam.get("fx", None)
    fy = cam.get("fy", None)
    cx = cam.get("cx", W * 0.5)
    cy = cam.get("cy", H * 0.5)

    if fx is None or fy is None:
        # Derive from vertical FOV if provided; else assume square pixels and use a common fx.
        vfov = cam.get("vfov", None)  # radians
        if vfov is not None:
            fy = 0.5 * H / np.tan(0.5 * vfov)
            fx = fy * (W / H)
        else:
            # fallback (works fine as long as you're consistent across both sides)
            fx = fy = 1000.0

    K = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], dtype=torch.float32, device=device)
    return K

def look_at_extrinsics_from_pyvista(cam, device):
    """
    Produce world->camera (R, t) that EXACTLY matches PyVista camera,
    including roll from `view_up`.
    """
    pos   = np.asarray(cam["position"],    dtype=np.float32)
    focal = np.asarray(cam["focal_point"], dtype=np.float32)
    vup_r = np.asarray(cam["view_up"],     dtype=np.float32)

    fwd = focal - pos
    fwd /= np.linalg.norm(fwd) + 1e-12

    vup_r /= np.linalg.norm(vup_r) + 1e-12
    # Gram–Schmidt to preserve roll but enforce orthogonality
    up = vup_r - np.dot(vup_r, fwd) * fwd
    up /= np.linalg.norm(up) + 1e-12

    right = np.cross(fwd, up);  right /= np.linalg.norm(right) + 1e-12
    up    = np.cross(right, fwd)  # re-orthogonalize

    # camera-to-world columns = [right, up, forward]
    R_c2w = np.stack([right, up, fwd], axis=1)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos

    R = torch.tensor(R_w2c, dtype=torch.float32, device=device)
    t = torch.tensor(t_w2c, dtype=torch.float32, device=device)
    return R, t

def project_pixels(Xw, R, t, K):
    """
    World -> camera -> pixels (top-left origin).
    Returns:
      uv: [V,2] pixels (u right, v down), z: [V] depth (camera z>0 means in front)
    """
    Xc = (Xw @ R.T) + t          # [V,3]
    z  = Xc[:, 2].clamp(min=1e-6)
    xy = Xc[:, :2] / z.unsqueeze(1)
    uv = (xy @ K[:2, :2].T) + K[:2, 2]
    return uv, z


def _faces_F3(faces):
    # Normalize faces to torch.long [F,3]; handle packed PyVista faces.
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

def _index_by_faces_adaptive(vertices_features, faces):
    """
    Calls kaolin.ops.mesh.index_vertices_by_faces across versions:
    - Newer builds expect vertices_features: [B,V,K], faces: [B,F,3]
    - Older builds expect vertices_features: [V,K], faces: [F,3]
    Returns: (feat_faces, used_batched) where feat_faces is [B,F,3,K]
    """
    # Ensure tensor
    vf = vertices_features
    if isinstance(vf, np.ndarray):
        vf = torch.from_numpy(vf)
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces)

    vf = vf.contiguous()
    faces = faces.to(torch.long).contiguous()

    # Prepare batched versions
    vf_b = vf if vf.ndim == 3 else vf.unsqueeze(0)          # [B,V,K]
    f_b  = faces if faces.ndim == 3 else faces.unsqueeze(0)  # [B,F,3]

    # Try batched call first
    try:
        out = index_vertices_by_faces(vf_b, f_b)             # [B,F,3,K]
        return out, True
    except AssertionError:
        # Fall back to unbatched API
        vf_u = vf_b[0] if vf_b.ndim == 3 else vf_b           # [V,K]
        f_u  = f_b[0]  if f_b.ndim  == 3 else f_b            # [F,3]
        out_u = index_vertices_by_faces(vf_u, f_u)           # [F,3,K]
        return out_u.unsqueeze(0), False                     # -> [1,F,3,K]

def pack_faces_for_dibr(uv, z, faces):
    """
    Returns:
      face_vertices_image: [1, F, 3, 2]  (pixel coords, origin top-left)
      face_vertices_z:     [1, F, 3]
    Works across Kaolin versions.
    """
    uv = uv.to(torch.float32).contiguous()                    # [V,2]
    z1 = z.to(torch.float32).contiguous().unsqueeze(-1)       # [V,1]
    faces = _faces_F3(faces)                                  # [F,3]

    # Index UVs (K=2) and depths (K=1) via adaptive wrapper
    f_uv, _ = _index_by_faces_adaptive(uv, faces)             # [1,F,3,2] or -> forced to that
    f_zk, _ = _index_by_faces_adaptive(z1, faces)             # [1,F,3,1]
    f_z = f_zk.squeeze(-1)                                    # [1,F,3]

    return f_uv, f_z



def rasterize_silhouette(face_vertices_image, face_vertices_z, H, W):
    """
    Runs DIB-R to get a soft silhouette. API can vary slightly by Kaolin version;
    the first return is usually the soft mask / coverage.
    """
    # Some versions return (soft_mask, face_idx, weight, depth, ...) and some bundle features.
    out = dibr(face_vertices_z=face_vertices_z,
               face_vertices_image=face_vertices_image,
               image_size=(H, W))
    soft = out[0]  # [B,1,H,W] or [B,H,W]; keep it robust below
    if soft.dim() == 3:
        soft = soft.unsqueeze(1)
    return soft  # [B,1,H,W]


def to_numpy_image(t):
    t = t.detach().float().cpu()
    if t.dim() == 4:  # [B,C,H,W] -> first item
        t = t[0]
    if t.size(0) == 1:  # [1,H,W] -> [H,W]
        t = t[0]
    return t.numpy()

def main(
    mesh_path="Item.obj",
    W=800, H=544,
    bg_color=(0, 0, 255),
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # 1) Load mesh
    verts_w, faces = load_mesh(mesh_path, device)

    # 2) PyVista scene (reference RGB)
    plotter = pv.Plotter(off_screen=True, window_size=(W, H))
    maybe_enable_offscreen(plotter)
    plotter.set_background(bg_color)
    mesh_pv = pv.wrap(np.asarray(verts_w.cpu()))
    mesh_pv.faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64),
                               faces.cpu().numpy()]).ravel()
    plotter.add_mesh(mesh_pv, color="lightgray", smooth_shading=True)

    cam = {
        "position":    (120.0, -80.0,  60.0),
        "focal_point": (  0.0,   0.0,   0.0),
        "view_up":     ( 0.0,   0.0,   1.0),
        # Optionally: "fx": 975.0, "fy": 975.0, "cx": W/2, "cy": H/2,
        # or "vfov": np.deg2rad(30.0),
    }
    plotter.camera.position    = cam["position"]
    plotter.camera.focal_point = cam["focal_point"]
    plotter.camera.up          = cam["view_up"]

    img_pv = plotter.screenshot(return_img=True)   # [H,W,3] uint8

    # 3) Kaolin camera (match PyVista)
    R, t = look_at_extrinsics_from_pyvista(cam, device)
    K    = intrinsics_from_pyvista(cam, W, H, device)

    # 4) Project + pack
    uv, z = project_pixels(verts_w, R, t, K)           # pixels, y-down
    fimg, fz = pack_faces_for_dibr(uv, z, faces)

    # 5) Rasterize
    sil_soft = rasterize_silhouette(fimg, fz, H, W)    # [1,1,H,W]
    sil_np   = to_numpy_image(sil_soft)                # [H,W], 0..1

    # 6) Show
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(img_pv); ax1.set_title("PyVista"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(sil_np, cmap="gray", vmin=0, vmax=1); ax2.set_title("Kaolin DIB-R"); ax2.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    mesh_path = "Item.obj"
    main(mesh_path=mesh_path)
