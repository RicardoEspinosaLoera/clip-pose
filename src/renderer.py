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


    def forward(self, R, t, K, new_image_size, image_size_orginal, scaled = False):
        
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
        Y_UP = True          # you said your DIB-R path is y-down


        sx, sy, tx, ty = (0.68, 1.0, 128.34, -0.5)

        if(scaled == True):
            sx = sx * (Ws / W)
            sy = sy * (Hs / H)
            tx = tx * (Ws / W)
            ty = ty * (Hs / H)
        
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

    # def forward(self, R, t, K, image_size):
    #     """
    #     Kaolin DIB-R forward with VTK-like Phong shading and sRGB output.
    #     - Assumes verts are in object space; R,t are object->camera in Kaolin (+Z forward).
    #     - Uses a headlight (at the camera) to mimic PyVista defaults.
    #     """
    #     # -------------------------- setup --------------------------
    #     B = R.shape[0]
    #     H, W = image_size
    #     device = self.verts.device
    #     R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

    #     # Shading params (tune to taste for PyVista parity)
    #     ambient   = 0.10
    #     kd        = 1.00
    #     ks        = 0.60
    #     shininess = 24.0
    #     exposure_ev = 0.0         # try 0.5~1.0 if PyVista looks brighter
    #     gamma_out  = 2.2          # linear -> sRGB

    #     eps = 1e-8

    #     # ------------------ 1) object -> camera --------------------
    #     # Per-batch camera-space vertex positions
    #     v_cam = torch.einsum('bij,vj->bvi', R, self.verts) + t[:, None, :]  # [B,V,3]

    #     # ------------------ 2) project to pixels -------------------
    #     Z  = v_cam[..., 2:3].clamp(min=1e-6)
    #     xy = v_cam[..., :2] / Z

    #     fx = K[:, 0, 0].view(-1, 1, 1)
    #     fy = K[:, 1, 1].view(-1, 1, 1)
    #     cx = K[:, 0, 2].view(-1, 1, 1)
    #     cy = K[:, 1, 2].view(-1, 1, 1)

    #     u = fx * xy[..., 0:1] + cx
    #     v = fy * xy[..., 1:2] + cy
    #     uv_pixels = torch.cat([u, v], dim=-1)[0]  # [V,2] (kept for diagnostics)
    #     v_img = torch.cat([u, v, Z], dim=-1)      # [B,V,3] (u,v,Zpix)

    #     # ---------------- 3) smooth vertex normals -----------------
    #     # Compute once per mesh, but we’ll do it here to keep the function self-contained.
    #     # Face normals in *object* space:
    #     V = self.verts
    #     F = self.faces
    #     v0o, v1o, v2o = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]   # [F,3]
    #     fn_obj = torch.cross(v1o - v0o, v2o - v0o, dim=-1)   # [F,3]

    #     # Accumulate to vertices (simple area-weighted average):
    #     vnorm_obj = torch.zeros_like(V)
    #     vnorm_obj.index_add_(0, F[:, 0], fn_obj)
    #     vnorm_obj.index_add_(0, F[:, 1], fn_obj)
    #     vnorm_obj.index_add_(0, F[:, 2], fn_obj)
    #     vnorm_obj = torch.nn.functional.normalize(vnorm_obj, dim=-1).clamp(-1, 1)  # [V,3]

    #     # Transform vertex normals to *camera* space per batch: n_cam = R * n_obj
    #     # (Ignore translation; normals are directions.)
    #     n_cam = torch.einsum('bij,vj->bvi', R, vnorm_obj)                # [B,V,3]
    #     n_cam = torch.nn.functional.normalize(n_cam, dim=-1)             # [B,V,3]

    #     # ---------------- 4) gather faces --------------------------
    #     fvcam = gather_by_faces(v_cam, self.faces)   # [B,F,3,3]  (camera-space positions)
    #     fvimg = gather_by_faces(v_img, self.faces)   # [B,F,3,3]  (u,v,Zpix)
    #     fncam = gather_by_faces(n_cam, self.faces)   # [B,F,3,3]  (camera-space normals)

    #     # Base vertex color (assume **linear** RGB)
    #     vfeat_rgb = self.v_rgb.unsqueeze(0).expand(B, -1, -1)  # [B,V,3]
    #     ffeat_rgb = gather_by_faces(vfeat_rgb, self.faces)     # [B,F,3,3]

    #     # ---------------- 5) pixels -> NDC -------------------------
    #     u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]  # [B,F,3]

    #     CENTER   = 0.5
    #     WMINUS1  = False
    #     Y_UP     = True

    #     # Optional affine to mimic PyVista’s screenshot quirks
    #     sx, sy, tx, ty = (0.68, 1.0, 128.34, -0.5)
    #     u_pix = u_pix * sx + tx
    #     v_pix = v_pix * sy + ty

    #     u_ndc, v_ndc = pixels_to_ndc(u_pix, v_pix, W, H,
    #                                 center_offset=CENTER,
    #                                 use_wminus1=WMINUS1,
    #                                 y_up=Y_UP)

    #     face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)  # [B,F,3,2]
    #     face_vertices_z  = fvcam[..., 2]                        # [B,F,3]

    #     # ---------------- 6) pack features for DIB-R ---------------
    #     # We want per-pixel base color, position (for view dir), and normal:
    #     #   feature = [rgb(3), P_cam(3), N_cam(3)]  -> C=9
    #     ffeat_p = fvcam  # camera-space positions per vertex
    #     ffeat_n = fncam  # camera-space normals per vertex (already unit length)
    #     ffeat = torch.cat([ffeat_rgb, ffeat_p, ffeat_n], dim=-1)  # [B,F,3,9]

    #     # For soft mask, Kaolin expects a "normals_z" like term; we can use |n·view| or |n_z|
    #     # Keep your previous choice (|nz|) for stability:
    #     v0, v1, v2 = fvcam[:, :, 0, :], fvcam[:, :, 1, :], fvcam[:, :, 2, :]
    #     n_face = torch.cross(v1 - v0, v2 - v0, dim=-1)
    #     n_face = torch.nn.functional.normalize(n_face, dim=-1)
    #     normals_z = n_face[..., 2].abs().unsqueeze(-1).expand(-1, -1, 3)

    #     # ---------------- 7) rasterize ------------------------------
    #     out = dibr(height=H, width=W,
    #             face_vertices_z=face_vertices_z,
    #             face_vertices_image=face_vertices_xy,
    #             face_features=ffeat,
    #             face_normals_z=normals_z)

    #     feat_img = out[0]                      # [B,H,W,C=9] interpolated features
    #     sil      = out[1].unsqueeze(1).clamp(0, 1)   # [B,1,H,W]

    #     # ---------------- 8) unpack & shade (Phong) ----------------
    #     # Unpack features
    #     base_rgb = feat_img[..., 0:3]                  # [B,H,W,3]  (assumed linear)
    #     P_cam    = feat_img[..., 3:6]                  # [B,H,W,3]
    #     N_cam    = feat_img[..., 6:9]                  # [B,H,W,3]

    #     # Normalize N, build view dir (headlight = light dir == view dir)
    #     N = torch.nn.functional.normalize(N_cam, dim=-1)
    #     V = torch.nn.functional.normalize(-P_cam, dim=-1)   # from point to camera
    #     L = V                                              # headlight

    #     # Diffuse
    #     diff = torch.clamp((N * L).sum(dim=-1, keepdim=True), 0.0, 1.0)  # [B,H,W,1]

    #     # Blinn-Phong specular
    #     Hvec = torch.nn.functional.normalize(L + V, dim=-1)
    #     spec = torch.clamp((N * Hvec).sum(dim=-1, keepdim=True), 0.0, 1.0)
    #     spec = torch.pow(spec + eps, shininess)

    #     # Combine (linear space)
    #     rgb_linear = ambient * base_rgb + kd * diff * base_rgb + ks * spec

    #     # Exposure (linear) and mask background
    #     rgb_linear = rgb_linear * (2.0 ** exposure_ev)
    #     rgb_linear = rgb_linear * sil.permute(0, 2, 3, 1)  # apply alpha in linear

    #     # ---------------- 9) linear -> sRGB & layout ----------------
    #     # Avoid negative/NaN, then gamma encode
    #     rgb_linear = torch.clamp(rgb_linear, 0.0, 1.0)
    #     rgb_srgb = torch.clamp(rgb_linear, 0.0, 1.0) ** (1.0 / gamma_out)

    #     # To NCHW
    #     rgb = rgb_srgb.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]

    #     return rgb, sil

