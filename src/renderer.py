# src/renderer.py
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
    def __init__(self, verts, faces, per_vertex_rgb=None,
                 negate_z=False, flip_v=False, margin_px=32.0):
        super().__init__()
        # buffers (once, correct dtypes)
        self.register_buffer('verts', verts.float().contiguous())          # (V,3)
        self.register_buffer('faces', faces.long().contiguous())           # (F,3)
        if per_vertex_rgb is None:
            per_vertex_rgb = torch.ones_like(verts, dtype=torch.float32) * 0.75
        self.register_buffer('v_rgb', per_vertex_rgb.float().contiguous()) # (V,3)

        self.negate_z = bool(negate_z)
        self.flip_v   = bool(flip_v)   # v' = H-1 - v (pixel-origin swap)
        self.margin_px = float(margin_px)

    def _black(self, B, H, W, device, dtype):
        rgb0 = torch.zeros(B, 3, H, W, device=device, dtype=dtype)
        sil0 = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        return rgb0, sil0

    def forward(self, R, t, K, image_size):
        # ---- setup ----
        B = R.shape[0]
        device = self.verts.device
        R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()
        H, W = int(image_size[0]), int(image_size[1])

        # ---- world -> camera -> pixels (u,v,z) ----
        v_cam = torch.einsum('bij,vj->bvi', R, self.verts) + t[:, None, :]   # (B,V,3)
        v_img = project_pixels(v_cam, K)                                     # (B,V,3) [u,v,z]
        if self.flip_v:
            v_img[..., 1] = (H - 1) - v_img[..., 1]

        # ---- gather per-face data ----
        faces = self.faces
        fvcam = index_vertices_by_faces(v_cam, faces)     # (B,F,3,3)
        fvimg = index_vertices_by_faces(v_img, faces)     # (B,F,3,3)

        # per-vertex RGB features (DIB-R shades from these; fine to keep)
        vfeat = self.v_rgb[None].expand(B, -1, -1)        # (B,V,3)
        ffeat = index_vertices_by_faces(vfeat, faces)     # (B,F,3,3)

        # ---- depth & coarse pixel-space visibility ----
        face_vertices_z = fvcam[..., 2]                   # (B,F,3)
        if self.negate_z:
            face_vertices_z = -face_vertices_z

        u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]       # (B,F,3)
        xmin = u_pix.amin(dim=-1); xmax = u_pix.amax(dim=-1)  # (B,F)
        ymin = v_pix.amin(dim=-1); ymax = v_pix.amax(dim=-1)

        eps = 1e-6
        m = self.margin_px
        front = (face_vertices_z > eps).all(dim=-1)                    # (B,F)
        in_w  = (xmax >= -m) & (xmin <= W - 1 + m)
        in_h  = (ymax >= -m) & (ymin <= H - 1 + m)
        valid = front & in_w & in_h                                     # (B,F)
        has_valid = valid.any(dim=1)                                    # (B,)

        # ---- pixels -> NDC for rasterizer ----
        u_ndc = (u_pix + 0.5) / W * 2.0 - 1.0
        v_ndc = (v_pix + 0.5) / H * 2.0 - 1.0
        face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)          # (B,F,3,2)

        # ---- per-face normal z weight (double-sided) ----
        v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
        normals_z = n[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)     # (B,F,3)

        # ---- all invalid? return black safely ----
        if (~has_valid).all():
            return self._black(B, H, W, device, dtype=torch.float32)

        # ---- rasterize only valid samples; keep batch shape ----
        idx_good = torch.nonzero(has_valid, as_tuple=False).squeeze(1)

        fz_g   = face_vertices_z.index_select(0, idx_good)
        fxy_g  = face_vertices_xy.index_select(0, idx_good)
        ffeat_g= ffeat.index_select(0, idx_good)
        nz_g   = normals_z.index_select(0, idx_good)

        try:
            out = dibr(H, W, fz_g, fxy_g, ffeat_g, nz_g,
                       7000, 0.02, 30, 1000.0)
            rgb_g = out[0].permute(0, 3, 1, 2)           # (Bg,3,H,W)
            sil_g = out[1].unsqueeze(1)                  # (Bg,1,H,W)
        except Exception:
            rgb_g, sil_g = self._black(idx_good.numel(), H, W, device, torch.float32)

        rgb_g = torch.nan_to_num(rgb_g, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        sil_g = torch.nan_to_num(sil_g, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)

        # ---- stitch back to full batch ----
        rgb = torch.zeros(B, 3, H, W, device=device, dtype=rgb_g.dtype)
        sil = torch.zeros(B, 1, H, W, device=device, dtype=sil_g.dtype)
        rgb.index_copy_(0, idx_good, rgb_g)
        sil.index_copy_(0, idx_good, sil_g)

        return rgb, sil