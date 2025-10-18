import torch
from kaolin.render.mesh import dibr_rasterization as dibr
from kaolin.ops.mesh import index_vertices_by_faces


def gather_by_faces(vertices_features, faces):
    """
    vertices_features: [V,K] or [B,V,K] (torch/np)
    faces: [F,3] long
    returns: [B,F,3,K]
    """
    faces = faces.to(torch.long).contiguous()
    vf = vertices_features
    if vf.dim() == 2:
        vf = vf.unsqueeze(0)  # -> [1,V,K]
    return vf[:, faces, :]    # [B,F,3,K]


def pixels_to_ndc(u, v, W, H, *, center_offset=0.5, use_wminus1=False, y_up=True):
    u = u + center_offset
    v = v + center_offset
    x = (u / float(W)) * 2.0 - 1.0
    y = (v / float(H)) * 2.0 - 1.0
    if y_up:
        y = -y
    return x, y


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


    def forward(self, R, t, K, image_size):
        
        """
        Forward render pass to match dataset ground truth.

        Assumes R and t are already in Kaolin coordinate convention (+Z forward)
        as provided by compose_camera_object().

        Dataset convention (originally from PyVista):
        - World space: +X right, +Y up, +Z out of screen
        - Conversion to Kaolin is handled externally.
        """
        B = R.shape[0]
        H, W = image_size
        device = self.verts.device
        R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

        # -------- 1) Clip world/object -> camera --------
        v_cam = torch.einsum('bij,vj->bvi', R, self.verts) + t[:, None, :]  # [B,V,3]
        
        #print("K for renderer:", K)
        # -------- 2) project to pixels (y-down pixel convention) --------
        Z  = v_cam[..., 2:3].clamp(min=1e-6)
        xy = v_cam[..., :2] / Z

        # -------- 2) prepare intrinsics for THIS raster size --------
        # K may have been built for a different pixel grid (e.g., window_size vs framebuffer)

        fx = K[:, 0, 0].view(-1, 1, 1)
        fy = K[:, 1, 1].view(-1, 1, 1)
        cx = K[:, 0, 2].view(-1, 1, 1)
        cy = K[:, 1, 2].view(-1, 1, 1)
        
        u =  fx * xy[..., 0:1] + cx 
        v = fy * xy[..., 1:2] + cy    
        uv_pixels = torch.cat([u, v], dim=-1)[0]     # [V,2] for diagnostics/overlay
        v_img = torch.cat([u, v, Z], dim=-1)         # [B,V,3] (u,v,Zpix)

        # -------- per-vertex color --------
        vfeat = self.v_rgb.unsqueeze(0).expand(B, -1, -1)  # [B,V,3]

        # -------- 3) gather faces --------
        fvcam = gather_by_faces(v_cam, self.faces)   # [B,F,3,3]
        fvimg = gather_by_faces(v_img, self.faces)   # [B,F,3,3]
        ffeat = gather_by_faces(vfeat, self.faces)   # [B,F,3,3]

        # -------- 4) pixels -> NDC (y-down for your DIB-R build) --------
        u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]           # [B,F,3]
        
        # --- choose ONE convention (this matches your DIB-R call) ---
        CENTER = 0.5          # try 0.5 (pixel centers) OR 0.0 (pixel corners), but be consistent
        WMINUS1 = False        # True matches your earlier path; else False uses W/H
        Y_UP = False          # you said your DIB-R path is y-down

        sx, sy, tx, ty = (0.68, 1.0, 128.34, -0.5)
        u_pix = u_pix * sx + tx
        v_pix = v_pix * sy + ty

        # forward
        u_ndc, v_ndc = pixels_to_ndc(u_pix, v_pix, W, H,
                                    center_offset=CENTER,   # try 0.5 then 0.0
                                    use_wminus1=WMINUS1,    # try True then False
                                    y_up=Y_UP)          # your Kaolin build looked y-down
                                    
                                
        face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)  # [B,F,3,2]
        face_vertices_z  = fvcam[..., 2]                        # [B,F,3]

        # -------- 5) normals (neutral-ish) --------
        v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        normals_z = n[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)

        # -------- 6) rasterize --------
        out = dibr(height=H, width=W,
                face_vertices_z=face_vertices_z,
                face_vertices_image=face_vertices_xy,
                face_features=ffeat,
                face_normals_z=normals_z)

        rgb = out[0].permute(0, 3, 1, 2).clamp(0, 1)  # [B,3,H,W]
        sil = out[1].unsqueeze(1).clamp(0, 1)         # [B,1,H,W]

        return rgb, sil
