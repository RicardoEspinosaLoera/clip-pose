import torch
import torch.nn.functional as F

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

def pose_loss2(
    R_pred, t_pred, R_gt, t_gt, D_obj,
    M, K, image_size, renderer, BG,
    λR=0.5, λt=0.5,
    λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.1,
    mask_downsample=1
):
    """
    R_pred,t_pred: (B,3,3), (B,3)
    R_gt,t_gt:     (B,3,3), (B,3)
    D_obj:         normalization scalar(s) for your transl. loss
    M:             GT mask (B,1,H,W) in {0,1}
    K:             intrinsics (B,3,3)
    image_size:    (H,W)
    renderer:      callable -> (rgb_pred, sil_pred) with sil in [0,1]
    λ*:            weights; set λmask=0 to disable silhouette loss
    mask_downsample: integer factor to downscale mask & render for speed
    """

    

    # --- base pose terms (your existing) ---
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    # --- silhouette term ---
    L_mask = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    bce_val = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    dice_val = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    edge_val = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)
    iou_val = torch.tensor(0., device=R_pred.device, dtype=L_R.dtype)

    if λmask > 0.0:
        # Render at (possibly) lower resolution for speed
        H, W = image_size
        if mask_downsample > 1:
            Hs, Ws = H // mask_downsample, W // mask_downsample
            M_use = F.interpolate(M, size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1)
            BG = F.interpolate(BG, size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1)
            rgb_hat, sil_hat = renderer(R_pred, t_pred, K, image_size=(Hs, Ws))
        else:
            M_use = M
            rgb_hat, sil_hat = renderer(R_pred, t_pred, K, image_size=(H, W))
        
        # Ensure shape (B,1,H,W) & range [0,1]
        if sil_hat.ndim == 3:  # (B,H,W)
            sil_hat = sil_hat.unsqueeze(1)
        sil_hat = sil_hat.clamp(0, 1)

        # Compute mask only when GT has any foreground (avoids degenerate grads)
        has_fg = (M_use.sum(dim=(1,2,3)) > 10).float().view(-1,1,1,1)
        sil_eff = sil_hat * has_fg
        M_eff   = M_use   * has_fg

        # BCE + Dice; optional IoU for logging; edge with Sobel
        if (has_fg.sum() > 0):
            bce_val  = F.binary_cross_entropy(sil_eff, M_eff)
            dice_val = _dice_loss(sil_eff, M_eff)
            iou_val  = _iou_loss(sil_eff, M_eff)  # for logs (not added unless you prefer IoU to Dice)

            if λedge > 0.0:
                gp = _sobel_grad(sil_eff)
                gg = _sobel_grad(M_eff)
                edge_val = F.l1_loss(gp, gg)

            L_mask = λbce*bce_val + λdice*dice_val + λedge*edge_val
        else:
            # No foreground in batch; keep L_mask=0
            pass

    # --- total ---
    loss = λR*L_R + λt*L_T + λmask*L_mask

    B, _, H, W = sil_eff.shape
    img = torch.zeros(B, 3, H, W, device=sil_hat.device)  # black background
    overlay = overlay_mask_on_image(
        img, sil_eff,
        color=(1,1,1),   # white fill
        alpha=1.0,       # opaque
        hard=True,       # crisp edges like your sample
        thr=0.5
    )

    I_comp = composite(rgb_hat, BG, sil_eff)
    gt_pose_iou(rgb_hat, M_eff, R_gt, t_gt, K, renderer)

    logs = {
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach(),
        'mask_bce': bce_val.detach(),
        'mask_dice': dice_val.detach(),
        'mask_edge': edge_val.detach(),
        'mask_iou': (1. - iou_val).detach(),  # IoU (not loss): higher is better
        'sil_mean': (M.float().mean().detach())
    }
    return loss, logs, I_comp, overlay