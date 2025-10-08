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
        Forward render pass to match PyVista ground truth
        PyVista convention:
        - World space: +X right, +Y up, +Z out of screen
        - Camera looks along position→focal_point
        - View_up defines camera orientation
        """
        B = R.shape[0]
        device = self.verts.device
        R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

        # Match PyVista scale exactly
        scale = 0.15  # Same as GT generation
        scaled_verts = self.verts * scale

        # Transform to camera space using GT pose
        v_cam = torch.einsum('bij,vj->bvi', R, scaled_verts) + t[:, None, :]

        # Project to image space using GT camera intrinsics
        v_img = project_pixels(v_cam, K)

        # Debug info
        print(f"Camera matrix K:")
        print(f"fx, fy, cx, cy = {K[0,0,0].item():.4f} {K[0,1,1].item():.4f} {K[0,0,2].item():.4f} {K[0,1,2].item():.4f}")
        print(f"v_cam z range: {v_cam[...,2].min().item():.2f} to {v_cam[...,2].max().item():.2f}")
        print(f"mean z: {v_cam[...,2].mean().item():.2f}")

        # Rest of rendering pipeline...