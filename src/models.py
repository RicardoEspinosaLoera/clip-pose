import torch, torch.nn as nn
import torchvision.models as tv
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            backbone = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = tv.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.neck = nn.Sequential(           # small MLP head helps
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.head_rot = nn.Linear(512, 6)     # 6D rotation
        self.head_t   = nn.Linear(512, 3)     # tx, ty, log(z/D)  (predict normalized)
    
    def forward(self, x, D_obj=None):
        f = self.backbone(x)
        f = self.neck(f)

        r6 = self.head_rot(f)
        t3 = self.head_t(f)

        txn, tyn, logzn = t3[..., 0], t3[..., 1], t3[..., 2]

        # POSITIVE, STABLE DEPTH  (softplus is safer than exp)
        zn = torch.nn.functional.softplus(logzn).clamp_min(1e-6)  # >0
        t_norm = torch.stack([txn, tyn, zn], -1)                  # (B,3)

        if D_obj is None:
            return r6, t_norm
        else:
            if not torch.is_tensor(D_obj):
                D_obj = torch.tensor(D_obj, device=t3.device, dtype=t3.dtype)
            D_obj = D_obj.to(device=t3.device, dtype=t3.dtype)

            B = x.shape[0]
            if D_obj.dim() == 0:
                D_obj = D_obj.expand(B)     # scalar -> (B,)
            elif D_obj.dim() == 1 and D_obj.shape[0] != B:
                D_obj = D_obj.reshape(1).expand(B)

            t = t_norm * D_obj.view(-1, 1)  # meters
            return r6, t