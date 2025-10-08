import torch
from kaolin.render.mesh import dibr_rasterization as dibr
from kaolin.ops.mesh import index_vertices_by_faces

DEBUG_ONCE = {"done": False}

def pixels_to_ndc(x, y, H, W):
    # pixel center -> NDC in [-1, 1]; top-left origin to center origin
    x_ndc = (x + 0.5) / W * 2.0 - 1.0
    y_ndc = (y + 0.5) / H * 2.0 - 1.0
    return x_ndc, y_ndc

def project_pixels(verts_cam, K):
    Z  = verts_cam[..., 2:3].clamp(min=1e-6)
    xy = verts_cam[..., :2] / Z
    fx = K[:, 0, 0].view(-1, 1, 1)
    fy = K[:, 1, 1].view(-1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1)
    u = fx * xy[..., 0:1] + cx
    v = fy * xy[..., 1:2] + cy
    return torch.cat([u, v, Z], dim=-1)  # (B,V,3) [u,v,z]


class SoftMeshRenderer(torch.nn.Module):
    def __init__(self, verts, faces, per_vertex_rgb=None, negate_z=False, flip_v=False):
        super().__init__()
        self.register_buffer('verts', verts)                 # (V,3)
        self.register_buffer('faces', faces.long())
        #self.register_buffer('faces', faces.int())           # (F,3)
        if per_vertex_rgb is None:
            per_vertex_rgb = torch.ones_like(verts) * 0.75
        self.register_buffer('v_rgb', per_vertex_rgb)        # (V,3)
        self.negate_z = negate_z
        self.flip_v = flip_v   # flip pixel y (v) => v' = H-1 - v
        self.register_buffer('faces', faces.long())   # int64 for indexing


    def forward(self, R, t, K, image_size):
        """
        Forward render pass to match dataset ground truth
        Dataset convention (from PyVista):
        - World space: +X right, +Y up, +Z out of screen
        """
        B = R.shape[0]
        device = self.verts.device
        R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

        R_fix = torch.tensor([
            [1.,  0.,  0.],
            [0.,  1.,  0.],
            [0.,  0., -1.]
            ], device=device, dtype=R.dtype)

        R = R @ R_fix        # rotate into Kaolin frame
        t = (R_fix @ t.T).T  # transform translation accordingly

        # Scale vertices to better fit image
        scale = 0.15  # Increased from 0.1
        scaled_verts = self.verts * scale

        # Transform to camera space
        v_cam = torch.einsum('bij,vj->bvi', R, scaled_verts) + t[:, None, :]
        
        # Adjust Z-offset to center in frame
        #z_offset = torch.tensor([0., 0., 180.], device=device)[None, None, :]  # Increased from 100
        #v_cam = v_cam + z_offset

        # Center object in image plane
        H, W = int(image_size[0]), int(image_size[1])
        cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()
        fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
        
        xy_offset = torch.tensor([(W/2 - cx)/fx, (H/2 - cy)/fy, 0.], device=device)[None, None, :]
        v_cam = v_cam + xy_offset * v_cam[..., 2:3]  # Scale offset by depth

        # Project to image space
        v_img = project_pixels(v_cam, K)
        
        u, v = v_img[..., 0], v_img[..., 1]

        # Debug info
        visible = ((u >= 0) & (u < W) & (v >= 0) & (v < H) & (v_cam[..., 2] > 0))
        print(f"Translation: {t[0]}")
        print(f"Z range: {v_cam[...,2].min().item():.2f} to {v_cam[...,2].max().item():.2f}")
        print(f"Visible: {visible.float().mean().item()*100:.2f}%")

        # Prepare face buffers
        faces = self.faces
        fvcam = index_vertices_by_faces(v_cam, faces)
        fvimg = index_vertices_by_faces(v_img, faces)
        vfeat = self.v_rgb[None].expand(B, -1, -1)
        ffeat = index_vertices_by_faces(vfeat, faces)

        # ----------------------------------------------------
        # 5️⃣ Depth, screen coords, and NDC
        # ----------------------------------------------------
        face_vertices_z = fvcam[..., 2]  # (B,F,3)
        if getattr(self, "negate_z", False):
            face_vertices_z = -face_vertices_z

        u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]
        u_ndc = (u_pix + 0.5) / W * 2.0 - 1.0
        v_ndc = (v_pix + 0.5) / H * 2.0 - 1.0
        face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)  # (B,F,3,2)

        # ----------------------------------------------------
        # 6️⃣ Face normals (double-sided)
        # ----------------------------------------------------
        v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        normals_z = n[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)

        # ----------------------------------------------------
        # 7️⃣ Rasterization via Kaolin DIB-R
        # ----------------------------------------------------
        out = dibr(
            height=H,
            width=W,
            face_vertices_z=face_vertices_z,
            face_vertices_image=face_vertices_xy,
            face_features=ffeat,
            face_normals_z=normals_z
        )
        rgb = out[0].permute(0, 3, 1, 2).clamp(0, 1)  # (B,3,H,W)
        sil = out[1].unsqueeze(1).clamp(0, 1)         # (B,1,H,W)

        # ----------------------------------------------------
        # 8️⃣ Debug: check mean depth and validity
        # ----------------------------------------------------
        if not torch.isfinite(face_vertices_z).all():
            print("[WARN] Invalid z values (NaN/Inf) detected in renderer")

        print("mean z:", v_cam[...,2].mean().item())

        return rgb, sil