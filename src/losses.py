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

