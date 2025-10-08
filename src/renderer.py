# src/renderer.py
import torch
from kaolin.render.mesh import dibr_rasterization as dibr
from kaolin.ops.mesh import index_vertices_by_faces

def pixels_to_ndc(x, y, H, W):
    """Convert pixel centers (x,y) to NDC in [-1,1]."""
    x_ndc = (x + 0.5) / W * 2.0 - 1.0
    y_ndc = (y + 0.5) / H * 2.0 - 1.0
    return x_ndc, y_ndc


def project_pixels(verts_cam, K):
    """Project 3D camera-space verts -> image (u,v,z)."""
    Z  = verts_cam[..., 2:3].clamp(min=1e-6)
    xy = verts_cam[..., :2] / Z
    fx, fy = K[:, 0, 0].view(-1, 1, 1), K[:, 1, 1].view(-1, 1, 1)
    cx, cy = K[:, 0, 2].view(-1, 1, 1), K[:, 1, 2].view(-1, 1, 1)
    u = fx * xy[..., 0:1] + cx
    v = fy * xy[..., 1:2] + cy
    return torch.cat([u, v, Z], dim=-1)   # (B,V,3)  [u,v,z]


class SoftMeshRenderer(torch.nn.Module):
    """
    Kaolin-style differentiable renderer.
      - Kaolin convention: +Z forward, tz > 0
      - Pixel origin: top-left (no v-flip)
    """
    def __init__(self, verts, faces, per_vertex_rgb=None):
        super().__init__()
        self.register_buffer('verts', verts.float())
        self.register_buffer('faces', faces.long())
        if per_vertex_rgb is None:
            per_vertex_rgb = torch.ones_like(verts) * 0.75
        self.register_buffer('v_rgb', per_vertex_rgb.float())

    # ------------------------------------------------------------------
    def forward(self, R, t, K, image_size):
        """
        Args:
            R: (B,3,3) object→camera rotation
            t: (B,3)   object→camera translation
            K: (B,3,3) intrinsics
            image_size: (H,W)
        Returns:
            rgb: (B,3,H,W)
            sil: (B,1,H,W)
        """
        B, H, W = R.shape[0], int(image_size[0]), int(image_size[1])
        device = self.verts.device
        R, t, K = R.to(device), t.to(device), K.to(device)

        # 1️⃣ Transform vertices to camera space
        v_cam = torch.einsum('bij,vj->bvi', R, self.verts) + t[:, None, :]

        # 2️⃣ Project to image plane
        v_img = project_pixels(v_cam, K)  # (B,V,3) [u,v,z]

        # 3️⃣ Keep only positive-z verts (in front of camera)
        z_mean = v_cam[..., 2].mean().item()
        if z_mean <= 0:
            print(f"[WARN] mean z ≤ 0: {z_mean:.4f}")

        # 4️⃣ Gather per-face data
        faces = self.faces
        fvcam = index_vertices_by_faces(v_cam, faces)  # (B,F,3,3)
        fvimg = index_vertices_by_faces(v_img, faces)
        vfeat = self.v_rgb[None].expand(B, -1, -1)
        ffeat = index_vertices_by_faces(vfeat, faces)

        # 5️⃣ Prepare DIB-R inputs (NDC coords)
        face_vertices_z = fvcam[..., 2]  # (B,F,3)
        u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]
        u_ndc, v_ndc = pixels_to_ndc(u_pix, v_pix, H, W)
        face_vertices_image = torch.stack([u_ndc, v_ndc], dim=-1)

        # 6️⃣ Face normals (for weighting)
        v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        normals_z = n[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)

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

        return rgb, sil
