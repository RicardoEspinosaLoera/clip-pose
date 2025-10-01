import torch
import torch.nn.functional as F
from math import pi

# -------- basic image losses --------
def robust_l1(x, y, eps=1e-3):
    return torch.sqrt((x - y)**2 + eps**2).mean()

def composite(rgb, bg, sil):
    # rgb/bg: (B,3,H,W), sil: (B,1,H,W)
    return sil * rgb + (1.0 - sil) * bg

# -------- pose losses & helpers --------
def rot_geodesic_loss(R_pred, R_gt, eps=1e-7):
    """Geodesic rotation loss in radians. R_*: (B,3,3)"""
    Rt = torch.einsum('bij,bjk->bik', R_pred.transpose(1,2), R_gt)  # R_p^T R_g
    tr = Rt[:, 0,0] + Rt[:, 1,1] + Rt[:, 2,2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()

def transform_pts(R, t, P):
    """R:(B,3,3), t:(B,3), P:(N,3) -> (B,N,3)"""
    return torch.einsum('bij,nj->bni', R, P) + t[:, None, :]

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
    print(f"[composite] sil min/mean/max: {smin:.4f}/{sme:.4f}/{smax:.4f} | rgb_mean_inside: {r_in:.4f}")

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

def add_loss(Rp, tp, Rg, tg, P_obj, symmetric=False):
    """ADD (or ADD-S if symmetric=True); returns mean over batch (in mesh units)."""
    Pp = transform_pts(Rp, tp, P_obj)  # (B,N,3)
    Pg = transform_pts(Rg, tg, P_obj)  # (B,N,3)
    if symmetric:
        d = torch.cdist(Pp, Pg).min(dim=-1).values.mean(dim=1)  # (B,)
    else:
        d = torch.linalg.norm(Pp - Pg, dim=-1).mean(dim=1)      # (B,)
    return d.mean()

def total_loss(
    I_comp, I_gt, S_pred, S_gt,
    R_pred, t_pred, R_gt, t_gt,
    K, P_obj, D_obj,
    λs=0, λp=0, λadd=0.5, λR=0.25, λt=0.25, symmetric=False
):
    """Returns total scalar loss and a dict of detached components."""
    L_s = F.binary_cross_entropy(S_pred, S_gt)
    L_p = robust_l1(I_comp, I_gt)
    L_add = add_loss(R_pred, t_pred, R_gt, t_gt, P_obj, symmetric=symmetric)
    L_addn = L_add / max(D_obj, 1e-6)
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    loss = λs*L_s + λp*L_p + λadd*L_addn + λR*L_R + λt*L_T
    logs = {
        'sil': L_s.detach(),
        'photo': L_p.detach(),
        'ADDn': L_addn.detach(),
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach()
    }
    return loss, logs

def rot_geodesic_loss(R_pred, R_gt, eps=1e-6):
    R_delta = R_gt.transpose(-1,-2) @ R_pred
    cos = ((R_delta.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1.0) * 0.5).clamp(-1+eps, 1-eps)
    return torch.acos(cos).mean()

def normalized_t_loss(t_pred, t_gt, D_obj, eps=1e-8):
    return (torch.linalg.norm(t_pred - t_gt, dim=1) / (D_obj + eps)).mean()

def pose_loss(R_pred, t_pred, R_gt, t_gt, D_obj, λR=0.5, λt=0.5):
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)
    loss = λR*L_R + λt*L_T
    logs = {'rot_rad': L_R.detach(), 'trans_n': L_T.detach()}
    return loss, logs

def has_foreground(sil, thr=0.01):
    # sil: (B,1,H,W)
    return (sil.max(dim=-1)[0].max(dim=-1)[0].max(dim=1)[0] > thr)  # (B,)

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

def _sobel_grad(x):
    # x: (B,1,H,W)
    kx = torch.tensor([[-1.,0.,1.],
                       [-2.,0.,2.],
                       [-1.,0.,1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1.,-2.,-1.],
                       [ 0., 0., 0.],
                       [ 1., 2., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-8)

@torch.no_grad()
def gt_pose_iou(I, M, R_gt, t_gt, K, renderer):
    rgb_gt, sil_gt = renderer(R_gt, t_gt, K, image_size=I.shape[-2:])
    if sil_gt.ndim == 3: sil_gt = sil_gt.unsqueeze(1)
    sil_b = (sil_gt > 0.5).float()
    M_b   = (M > 0.5).float()
    inter = (sil_b * M_b).sum(dim=(1,2,3))
    union = (sil_b + M_b - sil_b*M_b).sum(dim=(1,2,3)).clamp_min(1)
    iou = (inter / union).mean().item()
    print(f"[check] GT IoU: {iou:.3f}")
    return iou

# ---- put these at module scope ---------------------------------------------

_ALIGN = None  # cached alignment settings

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
@torch.no_grad()
def probe_units_scale(renderer, R_gt, t_gt, K, M, image_size, flip_v=False, halfpx=-0.5):
    M = _ensure_nchw(M).float();  M = M/255.0 if M.max()>1.5 else M
    H, W = image_size
    K_r = K.clone(); K_r[:,0,2] += halfpx; K_r[:,1,2] += halfpx

    # Try orders of magnitude; refine once you see a peak
    scales = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1, 3, 10, 30, 100, 300, 1e3]
    best = (0.0, None)
    for s in scales:
        _, sil = renderer(R_gt, t_gt * s, K_r, image_size=(H, W))
        sil = _ensure_nchw(sil).float().clamp(0,1)
        if flip_v: sil = torch.flip(sil, [2])
        if sil.shape[-2:] != M.shape[-2:]:
            sil = F.interpolate(sil, size=M.shape[-2:], mode='bilinear', align_corners=False).clamp(0,1)
        iou = _iou_bin(sil, M)
        print(f"[units] s={s:g}  IoU={iou:.3f}")
        if iou > best[0]:
            best = (iou, s)
    print(f"[units] BEST scale s={best[1]}  IoU={best[0]:.3f}")
    return best[1] or 1.0

import torch, torch.nn.functional as F
from math import pi

_DELTA = None  # {'R':(3,3), 't':(3,), 's':float}

def _rodrigues(r):
    # r: (...,3)
    theta = torch.clamp(torch.linalg.norm(r, dim=-1, keepdim=True), min=1e-8)
    k = r / theta
    K = torch.zeros(r.shape[:-1]+(3,3), device=r.device, dtype=r.dtype)
    K[...,0,1] = -k[...,2]; K[...,0,2] =  k[...,1]
    K[...,1,0] =  k[...,2]; K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]; K[...,2,1] =  k[...,0]
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(K)
    return I + torch.sin(theta)[...,None]*K + (1-torch.cos(theta))[...,None]*(K@K)

@torch.no_grad()
def _resize_to(img, size):
    x = img
    if x.shape[-2:] != size:
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False).clamp(0,1)
    return x

def _compose_with_delta(R, t, RΔ, tΔ, s):
    # R,t: (B,3,3),(B,3), RΔ:(3,3), tΔ:(3,), s: scalar
    B = R.size(0)
    RΔB = RΔ.unsqueeze(0).expand(B,3,3)
    tΔB = tΔ.unsqueeze(0).expand(B,3)
    Rr  = torch.einsum('bij,bjk->bik', R, RΔB)
    tr  = s*t + torch.einsum('bij,bj->bi', R, s*tΔB)
    return Rr, tr

def calibrate_delta(renderer, R_gt, t_gt, K, M, image_size, flip_v=False, halfpx=-0.5, 
                    steps=300, lr=5e-2, use_scale=True, max_rot_deg=30):
    """
    Optimize Δ = (RΔ, tΔ, s) to maximize IoU on GT masks across a small batch.
    """
    device = R_gt.device
    B = R_gt.size(0)
    H, W = image_size
    K_r = K.clone(); K_r[:,0,2] += halfpx; K_r[:,1,2] += halfpx
    M = _ensure_nchw(M.float()); M = M/255.0 if M.max()>1.5 else M
    M = _resize_to(M, (H, W))

    # Params: small rotation, translation in object frame, log-scale
    rvec = torch.zeros(3, device=device, requires_grad=True)
    tΔ   = torch.zeros(3, device=device, requires_grad=True)
    log_s = torch.zeros(1, device=device, requires_grad=True) if use_scale else torch.zeros(1, device=device, requires_grad=False)

    opt = torch.optim.Adam([rvec, tΔ, log_s], lr=lr)
    best = {'IoU': -1.0, 'r': None, 't': None, 's': None}

    for it in range(steps):
        opt.zero_grad()
        s = torch.exp(log_s)[0] if use_scale else torch.tensor(1.0, device=device)
        # limit rotation to ±max_rot_deg to keep it stable
        r_clamped = rvec.clamp(-max_rot_deg*pi/180, max_rot_deg*pi/180)
        RΔ = _rodrigues(r_clamped.unsqueeze(0))[0]   # (3,3)

        Rr, tr = _compose_with_delta(R_gt, t_gt, RΔ, tΔ, s)
        _, sil = renderer(Rr, tr, K_r, image_size=(H, W))
        sil = _ensure_nchw(sil.float()).clamp(0,1)
        if flip_v: sil = torch.flip(sil, [2])
        sil = _resize_to(sil, (H, W))

        # IoU loss (maximize IoU => minimize 1-IoU)
        A = (sil > 0.5).float(); Bm = (M > 0.5).float()
        inter = (A*Bm).sum(dim=(1,2,3))
        union = (A+Bm - A*Bm).sum(dim=(1,2,3)).clamp_min(1)
        iou   = (inter/union).mean()
        loss  = (1 - iou)
        loss.backward()
        opt.step()

        if iou.item() > best['IoU']:
            best = {'IoU': iou.item(), 'r': rvec.detach().clone(), 't': tΔ.detach().clone(), 's': float(torch.exp(log_s).item())}

        if (it+1) % 50 == 0:
            print(f"[Δcal] step {it+1}/{steps}  IoU={iou.item():.3f}  s={float(torch.exp(log_s).item()):.3f}  tΔ={tΔ.tolist()}  r={rvec.tolist()}")

    # finalize
    RΔ_best = _rodrigues(best['r'].unsqueeze(0))[0].detach()
    tΔ_best = best['t'].detach()
    s_best  = best['s'] if use_scale else 1.0
    print(f"[Δcal] BEST  IoU={best['IoU']:.3f}  s={s_best:.3f}  tΔ={tΔ_best.tolist()}")
    return {'R': RΔ_best, 't': tΔ_best, 's': s_best}


_ALIGN = None
_TSCALE = None

def pose_loss2(
    R_pred, t_pred, R_gt, t_gt, D_obj,
    M, K, image_size, renderer, BG,
    λR=0.5, λt=0.5,
    λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.1,
    mask_downsample=1
):
    # ---------------- base pose losses ----------------
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    # ---------------- alignment probe (run once) ------
    global _ALIGN, _TSCALE

    if _ALIGN is None:
        _ALIGN = probe_alignment(renderer, R_gt, t_gt, K, M, image_size)
    inv  = _ALIGN['invert']
    flip = _ALIGN['flip_v']
    hpx  = _ALIGN['halfpx']


    # Prepare extrinsics for rendering only (do NOT change what pose losses see)
    if inv:
        Rr_pred = R_pred.transpose(1,2)
        tr_pred = -torch.einsum('bij,bj->bi', Rr_pred, t_pred)
        Rr_gt   = R_gt.transpose(1,2)
        tr_gt   = -torch.einsum('bij,bj->bi', Rr_gt, t_gt)
    else:
        Rr_pred, tr_pred = R_pred, t_pred
        Rr_gt,   tr_gt   = R_gt,   t_gt

    K_r = K.clone(); K_r[:,0,2] += hpx; K_r[:,1,2] += hpx

    global _DELTA
    if _DELTA is None:
        _DELTA = calibrate_delta(
            renderer, Rr_gt, tr_gt, K_r, M, image_size=(H, W),
            flip_v=flip, halfpx=hpx, steps=300, lr=5e-2, use_scale=True
        )

    # --- NEW: units/scale probe (run once)
    #if _TSCALE is None:
    #    _TSCALE = probe_units_scale(renderer, Rr_gt, tr_gt, K_r, M, image_size, flip_v=flip, halfpx=hpx)
    #s = _TSCALE

    # ---------------- silhouette term -----------------
    L_mask = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    bce_val = torch.tensor(0., device=R_pred.device)
    dice_val = torch.tensor(0., device=R_pred.device)
    edge_val = torch.tensor(0., device=R_pred.device)
    iou_val  = torch.tensor(0., device=R_pred.device)

    H, W = image_size
    Hs, Ws = (H//mask_downsample, W//mask_downsample) if mask_downsample>1 else (H, W)

    M_use  = F.interpolate(M.float(),  size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else M.float()
    BG_use = F.interpolate(BG.float(), size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else BG.float()

    #rgb_hat, sil_hat = renderer(Rr_pred, tr_pred * s, K_r, image_size=(Hs, Ws))
    #sil_hat = _ensure_nchw(sil_hat).float().clamp(0,1)

    #if flip:
    #    sil_hat = torch.flip(sil_hat, [2])

    # apply Δ to PRED for rendering/loss/vis
    RΔ, tΔ, sΔ = _DELTA['R'], _DELTA['t'], _DELTA['s']
    Rr_pred_eff, tr_pred_eff = _compose_with_delta(Rr_pred, tr_pred, RΔ, tΔ, sΔ)

    # render at training resolution (Hs, Ws)
    rgb_hat, sil_hat = renderer(Rr_pred_eff, tr_pred_eff, K_r, image_size=(Hs, Ws))
    sil_hat = _ensure_nchw(sil_hat).float().clamp(0,1)
    if flip: sil_hat = torch.flip(sil_hat, [2])

    # GT IoU check using Δ as well (should jump up)
    Rr_gt_eff, tr_gt_eff = _compose_with_delta(Rr_gt, tr_gt, RΔ, tΔ, sΔ)
    iou_gt = gt_pose_iou(M, Rr_gt_eff, tr_gt_eff, K_r, renderer, image_size=(H, W))
    print(f"[check] GT IoU (Δ-applied): {iou_gt:.3f}")

    has_fg = (M_use.sum(dim=(1,2,3)) > 10).float().view(-1,1,1,1)
    sil_eff = sil_hat * has_fg
    M_eff   = M_use   * has_fg

    if λmask > 0.0 and has_fg.any():
        bce_val  = F.binary_cross_entropy(sil_eff, M_eff)
        dice_val = _dice_loss(sil_eff, M_eff)
        iou_val  = _iou_loss(sil_eff, M_eff)
        if λedge > 0.0:
            gp = _sobel_grad(sil_eff); gg = _sobel_grad(M_eff)
            edge_val = F.l1_loss(gp, gg)
        L_mask = λbce*bce_val + λdice*dice_val + λedge*edge_val

    # --------------- visualization -------------------
    # composite of rendered RGB over BG at (Hs,Ws)
    I_comp  = composite(rgb_hat, BG_use, sil_eff)

    # overlay (pred outline in green, GT outline in red) at (Hs,Ws)
    overlay = overlay_mask_on_image(BG_use, sil_eff, color=(0,1,0), alpha=0.6, outline_px=2)
    overlay = overlay_mask_on_image(overlay, M_eff,  color=(1,0,0), alpha=0.6, outline_px=2)

    # If you prefer full-res visuals for logging, uncomment:
    I_comp  = F.interpolate(I_comp,  size=(H,W), mode='bilinear', align_corners=False).clamp(0,1)
    overlay = F.interpolate(overlay, size=(H,W), mode='bilinear', align_corners=False).clamp(0,1)

    # Correct GT IoU check (mask only; render at image_size then resize inside)
    #iou_gt = gt_pose_iou(M, Rr_gt, tr_gt, K, renderer, image_size=(H, W))
    #print(f"[check] GT IoU: {iou_gt:.3f}")

    # --------------- total & logs --------------------
    loss = λR*L_R + λt*L_T + λmask*L_mask
    logs = {
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach(),
        'mask_bce': bce_val.detach(),
        'mask_dice': dice_val.detach(),
        'mask_edge': edge_val.detach(),
        'mask_iou': (1. - iou_val).detach(),
        'sil_mean': (M.float().mean().detach()),
        'gt_iou': torch.tensor(iou_gt, device=R_pred.device)
    }
    return loss, logs, I_comp, overlay
