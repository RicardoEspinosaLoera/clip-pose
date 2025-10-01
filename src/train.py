import os, yaml, torch, random, tqdm, torchvision.utils as vutils
import numpy as np
from torch.utils.data import DataLoader
from src.datasets import TripletDataset
from src.models import Regressor
from src.camera import sixd_to_rotmat
from src.renderer import SoftMeshRenderer
from src.losses import (
    sample_mesh_points, pose_loss2
)
import wandb
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import math

@torch.no_grad()
def save_or_log_overlay(I, I_comp, sil, M, out_dir, tag, step, to_wandb=False):
    os.makedirs(out_dir, exist_ok=True)
    # I, I_comp: (B,3,H,W); sil, M: (B,1,H,W)

    #print("I "+str(I[0].shape))
    #print("I_comp "+str(I_comp[0].shape))
    #print("I_sil "+str(sil[0].shape))
    #print("I_M "+str(M[0].shape))

    #H, W = I.shape[-2], I.shape[-1]               # spatial dims only
    #I_comp_r = F.interpolate(I_comp, size=(H, W), mode='bilinear', align_corners=False).clamp(0, 1)
    #I_sil_r = F.interpolate(sil, size=(H, W), mode='bilinear', align_corners=False).clamp(0,1)

    grid = vutils.make_grid([
        I[0].detach().cpu(),
        I_comp[0].detach().cpu(),
        sil[0].detach().cpu(),
        M[0].detach().cpu().repeat(3,1,1)
    ], nrow=4, normalize=True, scale_each=True)
    #path = os.path.join(out_dir, f"{tag}_{step:06d}.png")
    #vutils.save_image(grid, path)
    if to_wandb and wandb is not None:
        wandb.log({f"{tag}/overlay": wandb.Image(grid)})

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

@torch.no_grad()
def fit_batch(renderer, R, K, H, W, fill=0.9):
    """Compute anchor t per item that guarantees visibility for the current R,K."""
    device = renderer.verts.device
    R = R.to(device).float(); K = K.to(device).float()
    return torch.stack([fit_mesh_in_fov(renderer.verts, R[b], K[b], H, W, fill) for b in range(R.shape[0])], 0)

def ramp_alpha(step, warmup=500, ramp=4000):
    if step <= warmup:
        return 0.0
    t = (step - warmup) / max(ramp, 1)
    t = min(max(t, 0.0), 1.0)
    # cosine ease-in-out using torch only
    return float(0.5 - 0.5 * torch.cos(torch.tensor(t) * torch.pi))

def overlay_mask_on_image(
    img,        # (B,3,H,W) float in [0,1]   (your picture)
    sil,        # (B,1,H,W) float in [0,1]   (Kaolin silhouette)
    color=(1.0, 1.0, 1.0),  # overlay color for the mask (e.g., white)
    alpha=0.9,              # fill opacity (0..1) for inside the mask
    hard=False,             # True => hard threshold, False => soft edges
    outline_px=0,           # >0 to draw outline only (px width). 0 disables outline.
    thr=0.5                 # threshold if hard=True
):
    """
    Returns: (B,3,H,W) float in [0,1] with the mask drawn on top of img.
    """
    #assert img.dim()==4 and sil.dim()==4 and img.size(-2:)==sil.size(-2:)

    if hard:
        m = (sil > thr).float()           # hard binary
    else:
        m = sil.clamp(0, 1)               # soft alpha

    B, _, H, W = img.shape
    color_t = torch.tensor(color, dtype=img.dtype, device=img.device).view(1,3,1,1).expand(B,-1,H,W)

    if outline_px > 0:
        # 1-px outline: dilate - erode (morphological edge)
        k = outline_px
        dil = F.max_pool2d(m, kernel_size=2*k+1, stride=1, padding=k)
        ero = -F.max_pool2d(-m, kernel_size=2*k+1, stride=1, padding=k)
        edge = (dil - ero).clamp(0,1)
        edge = (edge > 0.01).float()      # make it crisp

        # Draw outline with full opacity, keep image elsewhere
        return torch.where(edge>0, color_t, img)

    # Filled overlay (soft or hard)
    # alpha_map = alpha * m  (broadcast to 3 channels)
    a = (alpha * m).expand(-1, 3, -1, -1)
    out = img * (1.0 - a) + color_t * a
    return out

@torch.no_grad()
def push_mesh_in_front(renderer, R, t, z_target: float = 1.0):
    """
    Ensures all vertices satisfy z >= z_target in camera coords
    by shifting along +Z in camera space.
    """
    device = renderer.verts.device
    # Current camera-space verts with the provided R,t
    v_cam = torch.einsum('bij,vj->bvi', R.to(device), renderer.verts) + t[:, None, :].to(device)
    zmin  = v_cam[..., 2].amin(dim=1)                # (B,)
    # If anything is behind (zmin < z_target), shift forward along +Z
    dz    = (-zmin + z_target).clamp_min(0.0)        # (B,)
    t_fix = t.clone().to(device)
    t_fix[:, 2] += dz
    return t_fix

def make_train_transform():
    return T.Compose([
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.RandomAutocontrast(p=0.2),
        AddGaussianNoise(0.02),           # custom below (kept small)
        T.RandomAdjustSharpness(1.5, p=0.2),
    ])

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.03): super().__init__(); self.sigma = sigma
    def forward(self, x):
        if self.sigma <= 0: return x
        noise = torch.randn_like(x) * self.sigma
        return (x + noise).clamp(0, 1)


def load_mesh(path, scale=1.0):
    import trimesh, torch, numpy as np
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        # merge all geometries into one watertight-ish mesh
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    if (m.faces is None) or (m.faces.shape[1] != 3):
        m = m.triangulate()
    if not m.is_winding_consistent:
        m.fix_normals()

    V = torch.tensor(m.vertices, dtype=torch.float32) * float(scale)   # (V,3)
    F = torch.tensor(m.faces.astype(np.int64), dtype=torch.long)       # (F,3)
    return V, F


def rot_angles_rad(R_pred, R_gt, eps=1e-6):
    R_delta = R_gt.transpose(-1, -2) @ R_pred         # (B,3,3)
    tr = R_delta.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # (B,)
    cos = ((tr - 1.0) * 0.5).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos)  # (B,)

import torch

def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Nearest-rotation (polar) projection. R: (B,3,3) -> (B,3,3) in SO(3).
    Differentiable (through SVD).
    """
    # SVD
    U, S, Vh = torch.linalg.svd(R)          # U @ Vh is orthogonal but det could be -1
    Rhat = U @ Vh

    # Build batched correction D = diag(1,1,sign), sign = +1 unless det(Rhat)<0
    B = R.shape[0]
    device, dtype = R.device, R.dtype

    det = torch.det(Rhat)                   # (B,)
    D = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  # (B,3,3)
    D[:, 2, 2] = torch.where(det < 0, torch.tensor(-1.0, device=device, dtype=dtype),
                                   torch.tensor( 1.0, device=device, dtype=dtype))

    # Projected rotation
    Rproj = U @ D @ Vh
    return Rproj


def run_epoch(model, renderer, loader, device, cfg, P_obj, D_obj, verts, mode,
              optimizer=None, wb_logger=None):
    is_train = (mode == 'train')
    model.train(is_train)

    # weights (kept here even if you only use pose loss)
    λR = cfg['loss'].get('lambda_rot', 0.5)
    λt = cfg['loss'].get('lambda_trans', 0.5)

    totals = {'loss': 0.0, 'rot_rad': 0.0, 'trans_n': 0.0}
    count = 0
    pbar = tqdm.tqdm(loader, desc=f"{mode}")

    # anchor/blend config
    fill = cfg.get('render', {}).get('fit_fill', 0.85)                 # NEW ⟶ margin
    warmup = cfg.get('train_io', {}).get('anchor_warmup_steps', 500)   # NEW ⟶ schedule
    ramp   = cfg.get('train_io', {}).get('anchor_ramp_steps', 4000)    # NEW

    for step, batch in enumerate(pbar, 1):
        I    = batch['image'].to(device)
        R_gt = batch['R_co'].to(device)
        t_gt = batch['t_co'].to(device)
        BG = batch['bg'].to(device)
        K  = batch['K'].to(device)
        M  = batch['mask'].to(device)

        r6, t_pred = model(I)
        R_pred = sixd_to_rotmat(r6)
        R_pred = project_to_so3(R_pred) 
        #R_gt = sixd_to_rotmat(R_gt)
        #R_gt = project_to_so3(R_gt)
        H, W = I.shape[-2], I.shape[-1]

        #Rendering
        #with torch.no_grad():
        """
        t_pred_render = t_pred.clone()
        t_pred_render[:, 2] = torch.nn.functional.softplus(t_pred_render[:, 2]) + 1e-2  

        # Compute an anchor t that guarantees the mesh is in-FOV for this R,K
        t_anchor = fit_batch(renderer, R_pred.detach(), K, H, W, fill=fill)             

        # Blend: early = mostly anchor (visible), later = network prediction
        alpha = ramp_alpha(step, warmup=warmup, ramp=ramp)                               
        t_fixed = (1.0 - alpha) * t_anchor + alpha * t_pred_render                       

        #diagnose_visibility(renderer, R_pred, t_fixed, K, H, W)

    
        rgb, sil = renderer(R_pred, t_fixed, K, image_size=(H, W))
        B, _, H, W = sil.shape
        img = torch.zeros(B, 3, H, W, device=sil.device)  # black background

        """
        
        loss, logs, I_comp, overlay = pose_loss2(R_pred, t_pred, R_gt, t_gt, D_obj, M, K, (H, W), renderer,BG, λR=0.5, λt=0.5, λmask=1.0, λbce=1.0, λdice=0.5, λedge=0.1, mask_downsample=2)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg['optim'].get('grad_clip_norm', 1.0))
            optimizer.step()

        # --- batch metrics (for logging & running means) ---
        with torch.no_grad():
            # rotation in degrees (mean over batch)
            ang_rad = rot_angles_rad(R_pred, R_gt)            # (B,)
            batch_Rdeg = float(ang_rad.mean().item() * 180.0 / np.pi)

            # normalized translation (mean over batch)
            tn = torch.linalg.norm(t_pred - t_gt, dim=1) / (D_obj + 1e-8)  # (B,)
            batch_Tn = float(tn.mean().item())

        bsz = I.size(0)
        totals['loss']    += float(loss.item()) * bsz
        totals['rot_rad'] += float(ang_rad.mean().item()) * bsz
        totals['trans_n'] += float(tn.mean().item()) * bsz

        count += bsz

        pbar.set_postfix({
            'L':  f"{totals['loss']/count:.3f}",
            'R°': f"{(totals['rot_rad']/count)*180/np.pi:.2f}",
            'Tn': f"{totals['trans_n']/count:.3f}",
        })

        # ---- per-batch logging via the provided callback ONLY ----
        if wb_logger is not None and (step % cfg['train_io']['log_interval'] == 0):
            wb_logger({
                f"{mode}/loss": float(loss.item()),
                f"{mode}/R_deg": batch_Rdeg,
                f"{mode}/Tn": batch_Tn,
                f"{mode}/Tn": batch_Tn,
            })
            save_or_log_overlay(I, I_comp, overlay, M, os.path.join(cfg['train_io']['out_dir'], 'val_vis'), 'val', step,
                                to_wandb=(wandb is not None and cfg['wandb']['enabled']))

    # ---- epoch averages ----
    avg = {
        'loss':    totals['loss'] / max(count, 1),
        'rot_rad': totals['rot_rad'] / max(count, 1),
        'R_deg':   (totals['rot_rad'] / max(count, 1)) * 180.0 / np.pi,
        'trans_n': totals['trans_n'] / max(count, 1),
    }

    return avg

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def main(cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get('seed', 42))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(cfg['train_io']['out_dir'], exist_ok=True)

    # ---- wandb ----
    wandb.init(project=cfg['wandb']['project'],
               name=cfg['wandb']['run_name'],
               config=cfg,
               mode=('offline' if not cfg['wandb'].get('enabled', True) else 'online'))
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*",   step_metric="global_step")

    # ---- data ----
    train_tf = make_train_transform()
    ds_tr = TripletDataset(cfg['data']['train_root'], train=True,  transform=train_tf)
    ds_va = TripletDataset(cfg['data']['val_root'],   train=False, transform=None)

    tr = DataLoader(ds_tr, batch_size=cfg['optim']['batch_size'], shuffle=True,
                    num_workers=4, pin_memory=True)
    va = DataLoader(ds_va, batch_size=cfg['optim']['batch_size'], shuffle=False,
                    num_workers=4, pin_memory=True)

    # ---- mesh & renderer ----
    SCALE = cfg.get('mesh', {}).get('scale', 2.0)
    verts, faces = load_mesh('./meshes/Item.obj', scale=SCALE)
    faces = faces.long()
    verts, faces = verts.to(device), faces.to(device)
    renderer = SoftMeshRenderer(verts, faces).to(device)

    with torch.no_grad():
        samp = sample_mesh_points(verts, faces, n=4096).to(device)
        D_obj = torch.cdist(samp[None], samp[None]).amax().item()
    P_obj = sample_mesh_points(verts, faces, n=cfg.get('eval', {}).get('add_points', 1500)).to(device)

    # ---- model/optim ----
    model = Regressor().to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg['optim']['lr'],
                            weight_decay=cfg['optim']['weight_decay'])

    best_key = cfg.get('train_io', {}).get('model_select_key', 'loss')  # 'ADDn' if you use it
    best_val = float('inf')
    val_every = cfg['train_io']['val_interval_epochs']

    # ---- global step for ALL logs ----
    global_step = 0
    def wb_log(d):
        nonlocal global_step
        if cfg['wandb'].get('enabled', True):
            dd = dict(d)
            dd["global_step"] = global_step
            wandb.log(dd, step=global_step)
        global_step += 1

    try:
        for epoch in range(cfg['optim']['epochs']):
            # train
            train_stats = run_epoch(model, renderer, tr, device, cfg, P_obj, D_obj,verts,
                                    mode='train', optimizer=opt, wb_logger=wb_log)

            # val (every N)
            val_stats = None
            if (epoch % val_every) == 0:
                val_stats = run_epoch(model, renderer, va, device, cfg, P_obj, D_obj,verts,
                                      mode='val', optimizer=None, wb_logger=wb_log)

            # epoch summary (use SAME global_step)
            if cfg['wandb'].get('enabled', True):
                log_dict = {f"train/epoch_{k}": v for k, v in train_stats.items()}
                if val_stats:
                    log_dict.update({f"val/epoch_{k}": v for k, v in val_stats.items()})
                log_dict["epoch"] = epoch
                log_dict["global_step"] = global_step
                wandb.log(log_dict, step=global_step)

            # checkpoint (works even if ADDn isn't present)
            if val_stats:
                score = val_stats.get(best_key, val_stats.get('loss', float('inf')))
                if score < best_val:
                    best_val = score
                    ckpt_path = os.path.join(cfg['train_io']['out_dir'], f"best_epoch{epoch:03d}.pth")
                    torch.save(model.state_dict(), ckpt_path)

    finally:
        wandb.finish()



if __name__ == "__main__":
    main()
