# ---------- FAST Δ CALIBRATION (one-time, cached) ----------
import math
import torch
import torch.nn.functional as F

_RAD2DEG = 57.29577951308232

def _ensure_nchw(x):
    if x.ndim == 2:                    # H,W
        x = x[None, None, ...]
    elif x.ndim == 3:
        if x.shape[0] in (1,3):        # C,H,W
            x = x[None, ...]
        elif x.shape[-1] in (1,3):     # H,W,C
            x = x.permute(2,0,1).unsqueeze(0)
        else:
            raise ValueError(f"Unknown 3D shape {x.shape}")
    elif x.ndim == 4 and x.shape[-1] in (1,3):  # B,H,W,C
        x = x.permute(0,3,1,2)
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {x.shape}")
    return x

@torch.no_grad()
def composite(rgb, bg, sil):
    """
    rgb: rendered clip (B,3,Hr,Wr) in [0,1]
    bg:  your real image (B,3,H,W) in [0,1]
    sil: silhouette alpha (B,1,*,*) in [0,1] or {0,255}
    returns: (B,3,H,W)
    """
    rgb = _ensure_nchw(rgb).float().clamp(0,1)
    bg  = _ensure_nchw(bg).float().clamp(0,1)
    sil = _ensure_nchw(sil).float()
    if sil.shape[1] != 1:   # keep 1-channel alpha
        sil = sil[:, :1, ...]
    if sil.max() > 1.5:     # 0/255 → 0/1
        sil = sil / 255.0

    H, W = bg.shape[-2:]
    if rgb.shape[-2:] != (H, W):
        rgb = F.interpolate(rgb, size=(H, W), mode='bilinear', align_corners=False)
    if sil.shape[-2:] != (H, W):
        sil = F.interpolate(sil, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)

    # stats (debug)
    smin, sme, smax = sil.min().item(), sil.mean().item(), sil.max().item()
    r_in  = rgb[sil.expand_as(rgb) > 0.5].mean().item() if (sil > 0.5).any() else float('nan')
    #print(f"[composite] sil min/mean/max: {smin:.4f}/{sme:.4f}/{smax:.4f} | rgb_mean_inside: {r_in:.4f}")

    return sil * rgb + (1.0 - sil) * bg

@torch.no_grad()
def overlay_mask_on_image(img, sil, color=(1,1,1), alpha=0.9, hard=False, outline_px=0, thr=0.5):
    img = _ensure_nchw(img).float().clamp(0,1)
    sil = _ensure_nchw(sil).float()
    if sil.shape[1] != 1: sil = sil[:, :1, ...]
    if sil.max() > 1.5: sil = sil/255.0
    H, W = img.shape[-2:]
    if sil.shape[-2:] != (H, W):
        sil = F.interpolate(sil, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)

    if hard:
        m = (sil > thr).float()
    else:
        m = sil.clamp(0,1)

    B = img.size(0)
    color_t = torch.tensor(color, dtype=img.dtype, device=img.device).view(1,3,1,1).expand(B,-1,H,W)

    if outline_px > 0:
        k = outline_px
        dil = F.max_pool2d(m, kernel_size=2*k+1, stride=1, padding=k)
        ero = -F.max_pool2d(-m, kernel_size=2*k+1, stride=1, padding=k)
        edge = (dil - ero > 1e-3).float()
        return torch.where(edge>0, color_t, img)

    a = (alpha * m).expand(-1,3,-1,-1)
    return img * (1.0 - a) + color_t * a


#_DELTA = None  # {'R':(3,3), 't':(3,), 's':float}

@torch.no_grad()
def _downsize_hw(H, W, max_side=128, min_side=48):
    f = min(max_side / max(H, W), 1.0)
    h = max(min_side, int(round(H * f)))
    w = max(min_side, int(round(W * f)))
    return h, w

@torch.no_grad()
def sample_mesh_points(verts, faces, n=1500):
    """Area-weighted sampling of N points on a triangle mesh."""
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)  # (F,)
    prob = (areas / areas.sum()).clamp_min(1e-12)
    idx = torch.multinomial(prob, n, replacement=True)  # (n,)
    f0, f1, f2 = v0[idx], v1[idx], v2[idx]
    u = torch.rand(n, 1, device=verts.device)
    v = torch.rand(n, 1, device=verts.device)
    swap = (u + v > 1.0).float()
    u = u * (1 - swap) + (1 - u) * swap
    v = v * (1 - swap) + (1 - v) * swap
    pts = f0 + u * (f1 - f0) + v * (f2 - f0)  # (n,3)
    return pts


@torch.no_grad()
def _rodrigues_from_euler(yaw_deg, pitch_deg, roll_deg, device, dtype):
    # yaw (z), pitch (y), roll (x) in degrees -> Rodrigues rotation matrix
    def Rz(a): 
        c, s = math.cos(a), math.sin(a)
        M = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], device=device, dtype=dtype); return M
    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        M = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], device=device, dtype=dtype); return M
    def Rx(a):
        c, s = math.cos(a), math.sin(a)
        M = torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], device=device, dtype=dtype); return M
    rz = Rz(math.radians(yaw_deg))
    ry = Ry(math.radians(pitch_deg))
    rx = Rx(math.radians(roll_deg))
    return (rz @ ry @ rx)  # (3,3)

@torch.no_grad()
def _compose_with_delta(R, t, RΔ, tΔ, s):
    B = R.size(0)
    RΔB = RΔ.unsqueeze(0).expand(B,3,3)
    tΔB = tΔ.unsqueeze(0).expand(B,3)
    Rr  = torch.einsum('bij,bjk->bik', R, RΔB)
    tr  = s*t + torch.einsum('bij,bj->bi', R, s*tΔB)
    return Rr, tr

@torch.no_grad()
def _render_iou(renderer, R, t, K, M_ref, image_size, flip_v=False):
    _, sil = renderer(R, t, K, image_size=image_size)
    sil = sil if sil.ndim == 4 else sil.unsqueeze(1)
    sil = sil.float().clamp(0,1)
    if flip_v: sil = torch.flip(sil, [2])
    if sil.shape[-2:] != M_ref.shape[-2:]:
        sil = F.interpolate(sil, size=M_ref.shape[-2:], mode='bilinear', align_corners=False).clamp(0,1)
    A = (sil > 0.5).float(); B = (M_ref > 0.5).float()
    inter = (A*B).sum(dim=(1,2,3))
    union = (A+B - A*B).sum(dim=(1,2,3)).clamp_min(1)
    return (inter/union).mean().item()

@torch.no_grad()
def _ensure_nchw(x):
    if x.ndim == 2: x = x[None,None,...]
    elif x.ndim == 3:
        if x.shape[0] in (1,3): x = x[None,...]
        elif x.shape[-1] in (1,3): x = x.permute(2,0,1).unsqueeze(0)
    elif x.ndim == 4 and x.shape[-1] in (1,3): x = x.permute(0,3,1,2)
    return x

@torch.no_grad()
def _iou_bin(A, B, thr=0.5):
    A = (A > thr).float(); B = (B > thr).float()
    inter = (A*B).sum(dim=(1,2,3))
    union = (A+B - A*B).sum(dim=(1,2,3)).clamp_min(1)
    return (inter/union).mean().item()

@torch.no_grad()
def probe_alignment(renderer, R_gt, t_gt, K, M, image_size):
    """Try invert/flip_v/halfpx and cache best settings."""
    M = _ensure_nchw(M).float();  M = M/255.0 if M.max()>1.5 else M
    Hm, Wm = M.shape[-2:]
    results = []
    for invert in (False, True):
        if invert:
            R_ = R_gt.transpose(1,2)
            t_ = -torch.einsum('bij,bj->bi', R_, t_gt)  # inverse extrinsics
        else:
            R_, t_ = R_gt, t_gt
        for halfpx in (0.0, -0.5):
            K_ = K.clone()
            K_[:,0,2] += halfpx; K_[:,1,2] += halfpx
            _, sil = renderer(R_, t_, K_, image_size=image_size)
            sil = _ensure_nchw(sil).float().clamp(0,1)
            if sil.shape[-2:] != (Hm, Wm):
                sil = F.interpolate(sil, size=(Hm, Wm), mode='bilinear', align_corners=False).clamp(0,1)
            for flip_v in (False, True):
                sil2 = torch.flip(sil, [2]) if flip_v else sil
                iou = _iou_bin(sil2, M)
                results.append({'invert':invert, 'halfpx':halfpx, 'flip_v':flip_v, 'IoU':iou})
    best = max(results, key=lambda d: d['IoU'])
    print(f"[probe] BEST → invert={best['invert']} halfpx={best['halfpx']} flip_v={best['flip_v']}  IoU={best['IoU']:.3f}")
    return {'invert':best['invert'], 'halfpx':best['halfpx'], 'flip_v':best['flip_v']}

@torch.no_grad()
def gt_pose_iou(M, R_gt, t_gt, K, renderer, image_size):
    """Render at image_size, resize to mask size, compare IoU."""
    M = _ensure_nchw(M).float();  M = M/255.0 if M.max()>1.5 else M
    Hm, Wm = M.shape[-2:]
    _, sil = renderer(R_gt, t_gt, K, image_size=image_size)
    sil = _ensure_nchw(sil).float().clamp(0,1)
    if sil.shape[-2:] != (image_size[0], image_size[1]):
        sil = F.interpolate(sil, size=image_size, mode='bilinear', align_corners=False).clamp(0,1)
    if sil.shape[-2:] != (Hm, Wm):
        sil = F.interpolate(sil, size=(Hm,Wm), mode='bilinear', align_corners=False).clamp(0,1)
    return _iou_bin(sil, M)
# ---------------------------------------------------------------------------
def _dice_loss(p, g, eps=1e-6): 
    # p,g: (B,1,H,W) in [0,1] 
    inter = (p * g).sum(dim=(1,2,3)) 
    denom = (p + g).sum(dim=(1,2,3)) 
    dice = 1. - (2*inter + eps) / (denom + eps) 
    return dice.mean() 
    
def _iou_loss(p, g, eps=1e-6): 
    inter = (p * g).sum(dim=(1,2,3)) 
    union = (p + g - p*g).sum(dim=(1,2,3)) 
    iou = 1. - (inter + eps) / (union + eps) 
    return iou.mean()

def _sobel_grad(x): # x: (B,1,H,W) 
    kx = torch.tensor([[-1.,0.,1.], [-2.,0.,2.], [-1.,0.,1.]], device=x.device, dtype=x.dtype).view(1,1,3,3) 
    ky = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3) 
    gx = F.conv2d(x, kx, padding=1) 
    gy = F.conv2d(x, ky, padding=1) 
    return torch.sqrt(gx*gx + gy*gy + 1e-8)


def _so3_relative_angle(R1, R2, eps=1e-6):
    # θ = atan2(||vee(R)^skew||/2, (tr(R)−1)/2), where R = R1^T R2
    R = torch.einsum('bij,bjk->bik', R1.transpose(1,2), R2)
    tr = R[:, 0,0] + R[:, 1,1] + R[:, 2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    v = torch.stack([R[:,2,1]-R[:,1,2],
                     R[:,0,2]-R[:,2,0],
                     R[:,1,0]-R[:,0,1]], dim=-1)  # (B,3)
    sin = (0.5 * torch.linalg.norm(v, dim=-1)).clamp(0, 1.0 - eps)
    return torch.atan2(sin, cos)  # (B,) radians

# ---------- stable rotation loss ----------
def project_to_so3(R):
    # R: (B,3,3)
    U, _, Vt = torch.linalg.svd(R)
    Rproj = U @ Vt
    # ensure det=+1 (proper rotation)
    det = torch.det(Rproj).unsqueeze(-1).unsqueeze(-1)
    fix = torch.diag_embed(Rproj.new_tensor([1., 1., -1.]))
    Rproj = torch.where(det < 0, U @ fix @ Vt, Rproj)
    return Rproj

def _so3_angle(R1, R2, eps=1e-6):
    R = torch.einsum('bij,bjk->bik', R1.transpose(1,2), R2)
    tr = R[:,0,0] + R[:,1,1] + R[:,2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0+eps, 1.0-eps)
    v = torch.stack([R[:,2,1]-R[:,1,2],
                     R[:,0,2]-R[:,2,0],
                     R[:,1,0]-R[:,0,1]], dim=-1)
    sin = (0.5 * torch.linalg.norm(v, dim=-1)).clamp(0, 1.0-eps)
    return torch.atan2(sin, cos)   # (B,)

    
def rot_geodesic_loss(R_pred, R_gt, project=True, eps=1e-7):
    if project:
        R_pred = project_to_so3(R_pred)
    Rt = torch.einsum('bij,bjk->bik', R_pred.transpose(1,2), R_gt)
    tr = Rt[:,0,0] + Rt[:,1,1] + Rt[:,2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    L = torch.acos(cos).mean()
    return torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- rendering safety helpers ----------
@torch.no_grad()
def _sanitize_t(R, t, z_min=1e-2, z_max=None):
    t = t.clone()
    t[:, 2] = torch.nn.functional.softplus(t[:, 2]) + z_min  # force z>0
    if z_max is not None:
        t[:, 2] = torch.clamp(t[:, 2], max=z_max)
    return R, t

@torch.no_grad()
def _apply_alignment(R, t, K, align):
    """Apply cached alignment tweaks for rendering only."""
    inv, flip, hpx = align['invert'], align['flip_v'], align['halfpx']
    if inv:
        Rr = R.transpose(1,2)
        tr = -torch.einsum('bij,bj->bi', Rr, t)
    else:
        Rr, tr = R, t
    K_r = K.clone()
    K_r[:,0,2] += hpx; K_r[:,1,2] += hpx
    return Rr, tr, K_r, flip

@torch.no_grad()
def _render_safe(renderer, R, t, K, image_size, flip_v=False):
    """Never crash: return zeros if Kaolin fails."""
    Hs, Ws = image_size
    B = R.shape[0]
    try:
        rgb, sil = renderer(R, t, K, image_size=(Hs, Ws))
    except Exception as e:
        device = R.device
        print(f"[render-safe] fallback: {type(e).__name__}: {e}")
        rgb = torch.zeros(B, 3, Hs, Ws, device=device)
        sil = torch.zeros(B, 1, Hs, Ws, device=device)
        return rgb, sil
    sil = sil if sil.ndim == 4 else sil.unsqueeze(1)
    sil = sil.float().clamp(0,1)
    if flip_v:
        sil = torch.flip(sil, [2])
    rgb = rgb.float().clamp(0,1)
    return rgb, sil

def normalized_t_loss(t_pred, t_gt, D_obj=None, eps=1e-6):
    # Both in meters; normalize by ||t_gt|| so the scale is tame
    denom = torch.linalg.norm(t_gt, dim=1, keepdim=True).clamp_min(eps)
    dt = (t_pred - t_gt) / denom
    L = F.smooth_l1_loss(dt, torch.zeros_like(dt), beta=0.1, reduction='mean')
    return torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

import torch
import torch.nn.functional as F

# --- helper 0: (optional) rescale intrinsics if you render at a different size ---
def rescale_K(K, old_H, old_W, new_H, new_W):
    if (new_H == old_H) and (new_W == old_W):
        return K
    K_ = K.clone().float()
    sx, sy = new_W / float(old_W), new_H / float(old_H)
    K_[:, 0, 0] *= sx;  K_[:, 1, 1] *= sy
    K_[:, 0, 2] *= sx;  K_[:, 1, 2] *= sy
    return K_

# --- helper 1: your FOV fit (unchanged) ---
def fit_mesh_in_fov(verts, R, K, H, W, fill=0.9):
    """
    Returns t so v_cam = R @ v + t:
      - projects the mesh center to (cx,cy)
      - chooses a depth so the mesh fits within the frame with margin `fill`
    """
    device = verts.device
    R = R.to(device).float(); K = K.to(device).float()

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    c_world    = verts.mean(dim=0)              # (3,)
    v_centered = verts - c_world                # (V,3)
    v_cam_rot  = (R @ v_centered.t()).t()       # (V,3)

    rx = v_cam_rot[:,0].abs().max()
    ry = v_cam_rot[:,1].abs().max()

    half_w = torch.minimum(cx, (W - 1 - cx))
    half_h = torch.minimum(cy, (H - 1 - cy))

    eps = torch.tensor(1e-6, device=device)
    need_zx = fx * rx / torch.maximum(fill * half_w, eps)
    need_zy = fy * ry / torch.maximum(fill * half_h, eps)
    z_cam   = torch.maximum(need_zx, need_zy).clamp_min(1e-2)

    Rc = R @ c_world
    t  = torch.tensor([0.0, 0.0, z_cam], device=device) - Rc
    return t

# --- helper 2: batched anchor translation ---
@torch.no_grad()
def fit_batch_in_fov(renderer, R, K, H, W, fill=0.9):
    device = renderer.verts.device
    R = R.to(device).float(); K = K.to(device).float()
    t_list = [fit_mesh_in_fov(renderer.verts, R[b], K[b], H, W, fill) for b in range(R.shape[0])]
    return torch.stack(t_list, 0)  # (B,3)

def cam2obj_to_world2cam(R_co, t_co):
    # R_co : camera -> object; t_co : camera origin in object coords
    R_oc = R_co.transpose(-1, -2)                     # inverse
    t_oc = -(R_oc @ t_co.unsqueeze(-1)).squeeze(-1)   # inverse
    return R_oc, t_oc

def so3_reg(R):
    I = torch.eye(3, device=R.device).unsqueeze(0)
    RtR = R.transpose(1,2) @ R
    ortho = (RtR - I).pow(2).mean()
    det_pen = (torch.det(R) - 1.0).pow(2).mean()
    return ortho + 0.1 * det_pen

def pose_loss2(
    R_pred, t_pred, R_gt, t_gt, D_obj,
    M, K, image_size, renderer, BG,
    λR=0.5, λt=0.5,
    λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.0,
    mask_downsample=1, z_min=1e-2, z_max=None,
    make_vis=True,
    anchor_fill=0.85,       # NEW: margin used by FOV-fit
    anchor_alpha=0.0        # NEW: blend between anchor and prediction (0..1)
):
    # --- base pose losses (as before) ---
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    H, W = image_size
    Hs, Ws = (H//mask_downsample, W//mask_downsample) if mask_downsample>1 else (H, W)

    # Downsample GT mask / BG to match render size
    M_use  = F.interpolate(M.float(),  size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else M.float()
    BG_use = F.interpolate(BG.float(), size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else BG.float()

    # Make K consistent with render size
    K_use = rescale_K(K, H, W, Hs, Ws)

    # ---- anchor for visibility (per-batch) ----
    # Enforce tz>0 on the prediction used for rendering
    #t_pred_render = t_pred.clone()
    #t_pred_render[:, 2] = F.softplus(t_pred_render[:, 2]) + 1e-2

    t_pred_render = t_pred.clone()
    tz = F.softplus(t_pred_render[:, 2:3]) + 1e-2      # (B,1)
    t_pred_render = torch.cat([t_pred_render[:, :2], tz], dim=1)

    R_pred = project_to_so3(R_pred)
    t_anchor = fit_batch_in_fov(renderer, R_pred.detach(), K_use, Hs, Ws, fill=anchor_fill)
    # Blend: alpha=0 => all anchor (guaranteed visible), alpha=1 => all t_pred_render
    t_render = (1.0 - anchor_alpha) * t_anchor + anchor_alpha * t_pred_render

    R_pred_w2c, t_pred_w2c = cam2obj_to_world2cam(R_pred, t_render)
    #R_gt_w2c,  t_gt_w2c    = cam2obj_to_world2cam(R_gt,  t_gt)
    rgb_hat, sil_hat = renderer(R_pred_w2c, t_pred_w2c, K_use, image_size=(Hs, Ws))  # ✅

    #rgb_hat, sil_hat = renderer(R_pred_w2c, t_render, K_use, image_size=(Hs,Ws))
    # ---- differentiable render (Kaolin DIB-R) ----
    #rgb_hat, sil_hat = renderer(R_pred, t_render, K_use, image_size=(Hs, Ws))
    sil_hat = sil_hat.float().clamp(0,1)  # (B,1,Hs,Ws)

    # ---- silhouette loss (optional) ----
    L_mask = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    bce_val = torch.tensor(0., device=R_pred.device)
    dice_val = torch.tensor(0., device=R_pred.device)
    edge_val = torch.tensor(0., device=R_pred.device)

    has_fg = (M_use.sum(dim=(1,2,3)) > 10).float().view(-1,1,1,1)
    sil_eff = sil_hat * has_fg
    M_eff   = M_use   * has_fg

    if λmask > 0.0 and has_fg.any():
        bce_val  = F.binary_cross_entropy(sil_eff.clamp(1e-6,1-1e-6), M_eff)
        dice_val = _dice_loss(sil_eff, M_eff)
        if λedge > 0.0:
            gp = _sobel_grad(sil_eff); gg = _sobel_grad(M_eff)
            edge_val = F.l1_loss(gp, gg)
        L_mask = λbce*bce_val + λdice*dice_val + λedge*edge_val

    # ---- totals ----
    loss = λR*L_R + λt*L_T + λmask*L_mask
    
    logs = {
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach(),
        'mask_bce': bce_val.detach(),
        'mask_dice': dice_val.detach(),
        'mask_edge': edge_val.detach(),
        'anchor_alpha': torch.tensor(anchor_alpha, device=R_pred.device)
    }

    if not make_vis:
        return loss, logs

    # ---- visuals (downsampled or upsample back) ----
    I_comp  = composite(rgb_hat, BG_use, sil_eff)
    overlay = overlay_mask_on_image(BG_use, sil_eff, color=(0,1,0), alpha=0.6, outline_px=2)

    if mask_downsample > 1:
        I_comp  = F.interpolate(I_comp,  size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)
        overlay = F.interpolate(overlay, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)

    return loss, logs, I_comp, overlay
