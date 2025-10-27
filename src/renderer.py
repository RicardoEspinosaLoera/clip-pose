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


# --- small helpers ------------------------------------------------------------
def srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92,
                       ((x + a) / (1 + a)).clamp(min=0) ** 2.4)

def linear_to_srgb(x):
    a = 0.055
    return torch.where(x <= 0.0031308, 12.92 * x,
                       (1 + a) * torch.clamp(x, 0) ** (1/2.4) - a)

def wrap_diffuse(N, L, k=0.2):
    # "wrapped" lambert: softer terminators
    L = torch.nn.functional.normalize(L, dim=-1)
    ndotl = (N * L).sum(dim=-1, keepdim=True)
    return torch.clamp((ndotl + k) / (1.0 + k), 0.0, 1.0)

def blinn_phong(N, V, L, kd, ks, shininess, wrap_k=None):
    L = torch.nn.functional.normalize(L, dim=-1)
    if wrap_k is None:
        diff = torch.clamp((N * L).sum(dim=-1, keepdim=True), 0.0, 1.0)
    else:
        diff = wrap_diffuse(N, L, wrap_k)
    H = torch.nn.functional.normalize(L + V, dim=-1)
    spec = torch.clamp((N * H).sum(dim=-1, keepdim=True), 0.0, 1.0) ** shininess
    return kd * diff, ks * spec

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


    def forward(self, R, t, K,image_size):
        
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

        #Normal affine vaues
        sx, sy, tx, ty = (0.68, 1.0, 128.34, -0.5)

        BASE_W, BASE_H = 800, 544

        tx = tx * (W / BASE_W)
        ty = ty * (H / BASE_H)

        # W/ 2 and H/2 to center, then scale to fill
        #sx, sy, tx, ty = (0.34, 0.5, 64.17, -0.25)
        
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
    #     Kaolin DIB-R forward with VTK-like light kit and correct color-space.
    #     - verts in object space; R,t are object->camera in Kaolin (+Z forward).
    #     - sRGB vertex colors are linearized for shading, then encoded back.
    #     """
    #     # -------------------------- setup --------------------------
    #     B = R.shape[0]
    #     H, W = image_size
    #     device = self.verts.device
    #     eps = 1e-8

    #     R, t, K = R.to(device).float(), t.to(device).float(), K.to(device).float()

    #     # Look-dev parameters (tweak to match PyVista)
    #     ambient        = 0.08
    #     kd             = 1.00
    #     ks             = 0.25
    #     shininess      = 100.0
    #     wrap_k         = 0.20          # 0 = Lambert, 0.2–0.4 = softer
    #     exposure_ev    = 5.00          # small lift in linear space
    #     wb_gain        = torch.tensor([1.06, 1.00, 0.94], device=device)  # warmer

    #     # PyVista framing quirk (affine on pixel coords)
    #     sx, sy, tx, ty = (0.68, 1.0, 128.34, -0.5)
    #     BASE_W, BASE_H = 800, 544

    #     tx = tx * (W / BASE_W)
    #     ty = ty * (H / BASE_H)

    #     # Rasterization coordinate options
    #     CENTER   = 0.5
    #     WMINUS1  = False
    #     Y_UP     = True

    #     # ------------------ 1) object -> camera --------------------
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

    #     v_img = torch.cat([u, v, Z], dim=-1)      # [B,V,3] (u,v,Zpix)

    #     # ---------------- 3) smooth vertex normals -----------------
    #     Vt = self.verts
    #     Ft = self.faces
    #     v0o, v1o, v2o = Vt[Ft[:, 0]], Vt[Ft[:, 1]], Vt[Ft[:, 2]]
    #     fn_obj = torch.cross(v1o - v0o, v2o - v0o, dim=-1)         # [F,3]

    #     vnorm_obj = torch.zeros_like(Vt)
    #     vnorm_obj.index_add_(0, Ft[:, 0], fn_obj)
    #     vnorm_obj.index_add_(0, Ft[:, 1], fn_obj)
    #     vnorm_obj.index_add_(0, Ft[:, 2], fn_obj)
    #     vnorm_obj = torch.nn.functional.normalize(vnorm_obj, dim=-1).clamp(-1, 1)

    #     n_cam = torch.einsum('bij,vj->bvi', R, vnorm_obj)
    #     n_cam = torch.nn.functional.normalize(n_cam, dim=-1)

    #     # ---------------- 4) gather faces --------------------------
    #     fvcam = gather_by_faces(v_cam, self.faces)   # [B,F,3,3]  (camera-space pos)
    #     fvimg = gather_by_faces(v_img, self.faces)   # [B,F,3,3]  (u,v,Zpix)
    #     fncam = gather_by_faces(n_cam, self.faces)   # [B,F,3,3]  (camera-space n)

    #     # Base vertex color (sRGB -> linear)
    #     vfeat_rgb = self.v_rgb.clamp(0,1)
    #     vfeat_rgb_lin = srgb_to_linear(vfeat_rgb)
    #     vfeat_rgb_lin = vfeat_rgb_lin.unsqueeze(0).expand(B, -1, -1)  # [B,V,3]
    #     ffeat_rgb_lin = gather_by_faces(vfeat_rgb_lin, self.faces)    # [B,F,3,3]

    #     # ---------------- 5) pixels -> NDC -------------------------
    #     u_pix, v_pix = fvimg[..., 0], fvimg[..., 1]  # [B,F,3]
    #     u_pix = u_pix * sx + tx
    #     v_pix = v_pix * sy + ty

    #     u_ndc, v_ndc = pixels_to_ndc(u_pix, v_pix, W, H,
    #                                 center_offset=CENTER,
    #                                 use_wminus1=WMINUS1,
    #                                 y_up=Y_UP)
    #     face_vertices_xy = torch.stack([u_ndc, v_ndc], dim=-1)  # [B,F,3,2]
    #     face_vertices_z  = fvcam[..., 2]                        # [B,F,3]

    #     # ---------------- 6) pack features for DIB-R ---------------
    #     ffeat_p = fvcam
    #     ffeat_n = fncam
    #     ffeat   = torch.cat([ffeat_rgb_lin, ffeat_p, ffeat_n], dim=-1)  # [B,F,3,9]

    #     # soft mask term (use |n_face.z|)
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
    #     feat_img = out[0]                            # [B,H,W,9]
    #     sil      = out[1].unsqueeze(1).clamp(0, 1)   # [B,1,H,W]

    #     # ---------------- 8) unpack & shade (light kit) ------------
    #     base_rgb_lin = feat_img[..., 0:3]            # in linear space
    #     P_cam        = feat_img[..., 3:6]
    #     N_cam        = feat_img[..., 6:9]

    #     # normalize
    #     N = torch.nn.functional.normalize(N_cam, dim=-1)
    #     Vv = torch.nn.functional.normalize(-P_cam, dim=-1)

    #     # Light directions in camera space (approx VTK light kit)
    #     L_head = Vv  # headlight == view dir
    #     L_key  = torch.tensor([0.35, -0.25, -0.90], device=device).view(1,1,1,3).expand_as(P_cam)
    #     L_fill = torch.tensor([-0.90, 0.10, -0.35], device=device).view(1,1,1,3).expand_as(P_cam)
    #     L_rim  = torch.tensor([0.00,  0.20,  1.00], device=device).view(1,1,1,3).expand_as(P_cam)

    #     I_head, I_key, I_fill, I_rim = 0.6, 0.9, 0.25, 0.35

    #     d1, s1 = blinn_phong(N, Vv, L_head, kd, ks, shininess, wrap_k)
    #     d2, s2 = blinn_phong(N, Vv, L_key,  kd, ks, shininess, wrap_k)
    #     d3, s3 = blinn_phong(N, Vv, L_fill, kd*0.8, ks*0.5, shininess*0.8, wrap_k)
    #     d4, s4 = blinn_phong(N, Vv, L_rim,  kd*0.6, ks*1.2, shininess*1.5, wrap_k)

    #     light_sum = (I_head*(d1 + s1) +
    #                 I_key *(d2 + s2) +
    #                 I_fill*(d3 + s3) +
    #                 I_rim *(d4 + s4))

    #     rgb_linear = ambient * base_rgb_lin + base_rgb_lin * light_sum

    #     # ---------------- 9) exposure, WB, mask, encode -------------
    #     rgb_linear = rgb_linear * (2.0 ** exposure_ev)
    #     rgb_linear = rgb_linear * wb_gain.view(1,1,1,3)

    #     # apply silhouette (linear)
    #     rgb_linear = rgb_linear * sil.permute(0, 2, 3, 1)

    #     # clamp & encode to sRGB
    #     rgb_linear = torch.clamp(rgb_linear, 0.0, 1.0)
    #     rgb_srgb   = linear_to_srgb(rgb_linear).clamp(0, 1)

    #     # NCHW
    #     rgb = rgb_srgb.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
    #     return rgb, sil
