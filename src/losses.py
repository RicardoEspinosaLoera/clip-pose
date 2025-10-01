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
def _project_to_so3(R):
    U, _, Vt = torch.linalg.svd(R)
    Rproj = U @ Vt
    det = torch.det(Rproj).unsqueeze(-1).unsqueeze(-1)
    Vt_fix = torch.where(det < 0,
                         torch.cat([Vt[..., :2, :], -Vt[..., 2:3, :]], dim=-2),
                         Vt)
    return U @ Vt_fix

def _so3_angle(R1, R2, eps=1e-6):
    R = torch.einsum('bij,bjk->bik', R1.transpose(1,2), R2)
    tr = R[:,0,0] + R[:,1,1] + R[:,2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0+eps, 1.0-eps)
    v = torch.stack([R[:,2,1]-R[:,1,2],
                     R[:,0,2]-R[:,2,0],
                     R[:,1,0]-R[:,0,1]], dim=-1)
    sin = (0.5 * torch.linalg.norm(v, dim=-1)).clamp(0, 1.0-eps)
    return torch.atan2(sin, cos)   # (B,)

def rot_geodesic_loss(R_pred, R_gt, project=True, mode='chordal'):
    if project:
        R_pred = _project_to_so3(R_pred)
    if mode == 'geodesic':
        return _so3_angle(R_pred, R_gt).mean()
    # chordal = 1 - cosθ
    Rt = torch.einsum('bij,bjk->bik', R_pred.transpose(1,2), R_gt)
    tr = Rt[:,0,0] + Rt[:,1,1] + Rt[:,2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-0.999999, 0.999999)
    return (1.0 - cos).mean()


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

def normalized_t_loss(t_pred, t_gt, D_obj, eps=1e-8): 
    return (torch.linalg.norm(t_pred - t_gt, dim=1) / (D_obj + eps)).mean()

def pose_loss2(
    R_pred, t_pred, R_gt, t_gt, D_obj,
    M, K, image_size, renderer, BG,
    λR=0.5, λt=0.5,
    λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.0,  # set λedge=0 for speed
    mask_downsample=2,z_min=1e-2, z_max=None,
    make_vis=True                              # turn off visuals to speed up
):
    # base pose losses (your originals)
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    # alignment probe (cached)
    #global _ALIGN, _DELTA

    #if _ALIGN is None:
    #    _ALIGN = probe_alignment(renderer, R_gt, t_gt, K, M, image_size)
    #inv, flip, hpx = _ALIGN['invert'], _ALIGN['flip_v'], _ALIGN['halfpx']

    # extrinsics for rendering (don’t change what the numeric losses see)
    #if inv:
    #    Rr_pred = R_pred.transpose(1,2)
    #    tr_pred = -torch.einsum('bij,bj->bi', Rr_pred, t_pred)
    #    Rr_gt   = R_gt.transpose(1,2)
    #    tr_gt   = -torch.einsum('bij,bj->bi', Rr_gt, t_gt)
    #else:
    #    Rr_pred, tr_pred = R_pred, t_pred
    #    Rr_gt,   tr_gt   = R_gt,   t_gt

    # calibrate Δ fast (cached)
    #K_r = K.clone(); K_r[:,0,2] += hpx; K_r[:,1,2] += hpx
    #if _DELTA is None:
    #    _DELTA = calibrate_delta_fast(renderer, Rr_gt, tr_gt, K_r, M, image_size,
    #                                  flip_v=flip, halfpx=hpx, D_obj=float(D_obj))
    #RΔ, tΔ, sΔ = _DELTA['R'], _DELTA['t'], _DELTA['s']

    Rr_pred_s, tr_pred_s = _sanitize_t(R_pred, t_pred, z_min=z_min, z_max=z_max)
    # render at training resolution (downsample for speed)
    H, W = image_size
    Hs, Ws = (H//mask_downsample, W//mask_downsample) if mask_downsample>1 else (H, W)

    M_use  = F.interpolate(M.float(),  size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else M.float()
    BG_use = F.interpolate(BG.float(), size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else BG.float()

    #Rr_pred_eff, tr_pred_eff = _compose_with_delta(R_pred, t_pred, RΔ, tΔ, sΔ)
    rgb_hat, sil_hat = _render_safe(renderer, Rr_pred_s, tr_pred_s, K, (Hs, Ws))
    #rgb_hat, sil_hat = renderer(R_pred, t_pred, K, image_size=(Hs, Ws))
    sil_hat = _ensure_nchw(sil_hat).float().clamp(0,1)
    #if flip: sil_hat = torch.flip(sil_hat, [2])

    # silhouette loss
    L_mask = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    bce_val = torch.tensor(0., device=R_pred.device)
    dice_val = torch.tensor(0., device=R_pred.device)
    edge_val = torch.tensor(0., device=R_pred.device)

    has_fg = (M_use.sum(dim=(1,2,3)) > 10).float().view(-1,1,1,1)
    sil_eff = sil_hat * has_fg
    M_eff   = M_use   * has_fg

    if λmask > 0.0 and has_fg.any():
        bce_val  = F.binary_cross_entropy(sil_eff, M_eff)
        dice_val = _dice_loss(sil_eff, M_eff)
        if λedge > 0.0:
            gp = _sobel_grad(sil_eff); gg = _sobel_grad(M_eff)
            edge_val = F.l1_loss(gp, gg)
        L_mask = λbce*bce_val + λdice*dice_val + λedge*edge_val

    # GT IoU (low-cost, once in a while you can compute full-res outside)
    #Rr_gt_eff, tr_gt_eff = _compose_with_delta(Rr_gt, tr_gt, RΔ, tΔ, sΔ)
    #gt_iou = gt_pose_iou(M, Rr_gt_eff, tr_gt_eff, K_r, renderer, image_size=(H, W))

    # totals
    loss = λR*L_R + λt*L_T + λmask*L_mask
    logs = {
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach(),
        'mask_bce': bce_val.detach(),
        'mask_dice': dice_val.detach(),
        'mask_edge': edge_val.detach(),
    }

    if not make_vis:
        return loss, logs  # fastest path

    # Optional visuals (downsampled or upsample back if you want)
    I_comp  = composite(rgb_hat, BG_use, sil_eff)
    overlay = overlay_mask_on_image(BG_use, sil_eff, color=(0,1,0), alpha=0.6, outline_px=2)
    #overlay = overlay_mask_on_image(overlay, M_eff,  color=(1,0,0), alpha=0.6, outline_px=2)

    I_comp  = F.interpolate(I_comp,  size=(H, W), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else I_comp
    overlay = F.interpolate(overlay, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else overlay
    return loss, logs, I_comp, overlay
