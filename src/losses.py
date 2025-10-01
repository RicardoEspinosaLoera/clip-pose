# ---------- FAST Δ CALIBRATION (one-time, cached) ----------
import math
import torch
import torch.nn.functional as F

_DELTA = None  # {'R':(3,3), 't':(3,), 's':float}

@torch.no_grad()
def _downsize_hw(H, W, max_side=128, min_side=48):
    f = min(max_side / max(H, W), 1.0)
    h = max(min_side, int(round(H * f)))
    w = max(min_side, int(round(W * f)))
    return h, w

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
def calibrate_delta_fast(renderer, R_gt, t_gt, K, M, image_size, flip_v=False, halfpx=-0.5, D_obj=1.0):
    """
    Fast, low-res coordinate search for Δ=(RΔ,tΔ,s).
    Returns dict with 'R','t','s'. Runs ONCE; cache globally.
    """
    device, dtype = R_gt.device, R_gt.dtype
    H, W = image_size
    # downsize everything for calibration (keeps aspect)
    h, w = _downsize_hw(H, W, max_side=128, min_side=48)
    K_r = K.clone()
    K_r[:,0,2] += halfpx; K_r[:,1,2] += halfpx
    M_ref = _ensure_nchw(M.float())
    if M_ref.max() > 1.5: M_ref = M_ref/255.0
    M_ref = F.interpolate(M_ref, size=(h,w), mode='bilinear', align_corners=False).clamp(0,1)

    # init
    best = {'IoU': -1.0, 'yaw':0.0, 'pitch':0.0, 'roll':0.0,
            'tx':0.0, 'ty':0.0, 'tz':0.0, 's':1.0}

    # 0) quick scale sweep
    scale_candidates = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for s in scale_candidates:
        RΔ = torch.eye(3, device=device, dtype=dtype)
        tΔ = torch.tensor([0.,0.,0.], device=device, dtype=dtype)
        Rr, tr = _compose_with_delta(R_gt, t_gt, RΔ, tΔ, s)
        iou = _render_iou(renderer, Rr, tr, K_r, M_ref, image_size=(h,w), flip_v=flip_v)
        if iou > best['IoU']:
            best.update({'IoU':iou, 's':s})

    # 1) coordinate descent (few passes, small grids)
    rot_steps = [30.0, 15.0, 7.5]          # degrees
    trans_steps = [0.25, 0.10, 0.05]       # fractions of D_obj along object axes
    improve_eps = 1e-3

    for pass_id in range(2):  # two coarse-to-fine passes
        improved = False

        # rotations
        for step in rot_steps:
            for axis, key in zip([(0,0,1),(0,1,0),(1,0,0)], ['yaw','pitch','roll']):
                best_local = (best['IoU'], best[key])
                for delta in (-step, 0.0, step):
                    yaw,pitch,roll = best['yaw'],best['pitch'],best['roll']
                    if key=='yaw':   yaw += delta
                    if key=='pitch': pitch += delta
                    if key=='roll':  roll += delta
                    RΔ = _rodrigues_from_euler(yaw, pitch, roll, device, dtype)
                    tΔ = torch.tensor([best['tx'],best['ty'],best['tz']], device=device, dtype=dtype)
                    Rr, tr = _compose_with_delta(R_gt, t_gt, RΔ, tΔ, best['s'])
                    iou = _render_iou(renderer, Rr, tr, K_r, M_ref, image_size=(h,w), flip_v=flip_v)
                    if iou > best['IoU'] + improve_eps:
                        best.update({'IoU':iou,'yaw':yaw,'pitch':pitch,'roll':roll})
                        improved = True
                # slight early stop per-axis
                if best['IoU'] > best_local[0] + improve_eps:
                    continue

        # translations
        for frac in trans_steps:
            step = frac * float(D_obj)
            for axis_i, key in enumerate(['tx','ty','tz']):
                best_local = (best['IoU'], best[key])
                for delta in (-step, 0.0, step):
                    tx,ty,tz = best['tx'],best['ty'],best['tz']
                    if key=='tx': tx += delta
                    if key=='ty': ty += delta
                    if key=='tz': tz += delta
                    RΔ = _rodrigues_from_euler(best['yaw'], best['pitch'], best['roll'], device, dtype)
                    tΔ = torch.tensor([tx,ty,tz], device=device, dtype=dtype)
                    Rr, tr = _compose_with_delta(R_gt, t_gt, RΔ, tΔ, best['s'])
                    iou = _render_iou(renderer, Rr, tr, K_r, M_ref, image_size=(h,w), flip_v=flip_v)
                    if iou > best['IoU'] + improve_eps:
                        best.update({'IoU':iou,'tx':tx,'ty':ty,'tz':tz})
                        improved = True
                if best['IoU'] > best_local[0] + improve_eps:
                    continue

        if not improved:
            break  # converged

    print(f"[Δcal-fast] IoU={best['IoU']:.3f}  s={best['s']:.3f}  "
          f"rpy=({best['roll']:.1f},{best['pitch']:.1f},{best['yaw']:.1f})  "
          f"tΔ=({best['tx']:.3f},{best['ty']:.3f},{best['tz']:.3f})")

    # pack
    RΔ_best = _rodrigues_from_euler(best['yaw'], best['pitch'], best['roll'], device, dtype)
    tΔ_best = torch.tensor([best['tx'],best['ty'],best['tz']], device=device, dtype=dtype)
    return {'R': RΔ_best, 't': tΔ_best, 's': best['s']}


# ---------- UPDATED pose_loss2 (uses fast calibration; optional visuals) ----------
def pose_loss2(
    R_pred, t_pred, R_gt, t_gt, D_obj,
    M, K, image_size, renderer, BG,
    λR=0.5, λt=0.5,
    λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.0,  # set λedge=0 for speed
    mask_downsample=2,
    make_vis=False                              # turn off visuals to speed up
):
    # base pose losses (your originals)
    L_R = rot_geodesic_loss(R_pred, R_gt)
    L_T = normalized_t_loss(t_pred, t_gt, D_obj)

    # alignment probe (cached)
    global _ALIGN, _DELTA
    if _ALIGN is None:
        _ALIGN = probe_alignment(renderer, R_gt, t_gt, K, M, image_size)
    inv, flip, hpx = _ALIGN['invert'], _ALIGN['flip_v'], _ALIGN['halfpx']

    # extrinsics for rendering (don’t change what the numeric losses see)
    if inv:
        Rr_pred = R_pred.transpose(1,2)
        tr_pred = -torch.einsum('bij,bj->bi', Rr_pred, t_pred)
        Rr_gt   = R_gt.transpose(1,2)
        tr_gt   = -torch.einsum('bij,bj->bi', Rr_gt, t_gt)
    else:
        Rr_pred, tr_pred = R_pred, t_pred
        Rr_gt,   tr_gt   = R_gt,   t_gt

    # calibrate Δ fast (cached)
    K_r = K.clone(); K_r[:,0,2] += hpx; K_r[:,1,2] += hpx
    if _DELTA is None:
        _DELTA = calibrate_delta_fast(renderer, Rr_gt, tr_gt, K_r, M, image_size,
                                      flip_v=flip, halfpx=hpx, D_obj=float(D_obj))
    RΔ, tΔ, sΔ = _DELTA['R'], _DELTA['t'], _DELTA['s']

    # render at training resolution (downsample for speed)
    H, W = image_size
    Hs, Ws = (H//mask_downsample, W//mask_downsample) if mask_downsample>1 else (H, W)

    M_use  = F.interpolate(M.float(),  size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else M.float()
    BG_use = F.interpolate(BG.float(), size=(Hs, Ws), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else BG.float()

    Rr_pred_eff, tr_pred_eff = _compose_with_delta(Rr_pred, tr_pred, RΔ, tΔ, sΔ)
    rgb_hat, sil_hat = renderer(Rr_pred_eff, tr_pred_eff, K_r, image_size=(Hs, Ws))
    sil_hat = _ensure_nchw(sil_hat).float().clamp(0,1)
    if flip: sil_hat = torch.flip(sil_hat, [2])

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
    Rr_gt_eff, tr_gt_eff = _compose_with_delta(Rr_gt, tr_gt, RΔ, tΔ, sΔ)
    gt_iou = gt_pose_iou(M, Rr_gt_eff, tr_gt_eff, K_r, renderer, image_size=(H, W))

    # totals
    loss = λR*L_R + λt*L_T + λmask*L_mask
    logs = {
        'rot_rad': L_R.detach(),
        'trans_n': L_T.detach(),
        'mask_bce': bce_val.detach(),
        'mask_dice': dice_val.detach(),
        'mask_edge': edge_val.detach(),
        'gt_iou': torch.tensor(gt_iou, device=R_pred.device)
    }

    if not make_vis:
        return loss, logs  # fastest path

    # Optional visuals (downsampled or upsample back if you want)
    I_comp  = composite(rgb_hat, BG_use, sil_eff)
    overlay = overlay_mask_on_image(BG_use, sil_eff, color=(0,1,0), alpha=0.6, outline_px=2)
    overlay = overlay_mask_on_image(overlay, M_eff,  color=(1,0,0), alpha=0.6, outline_px=2)

    I_comp  = F.interpolate(I_comp,  size=(H, W), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else I_comp
    overlay = F.interpolate(overlay, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1) if mask_downsample>1 else overlay
    return loss, logs, I_comp, overlay
