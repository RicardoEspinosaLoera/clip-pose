import torch, torch.nn as nn
import torchvision.models as tv
import torch.nn.functional as F
from .camera import sixd_to_rotmat

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

        # translation: predict normalized tx,ty,logz; then un-normalize by D_obj
        txn, tyn, logzn = t3[...,0], t3[...,1], t3[...,2]
        zn = torch.exp(logzn).clamp(min=1e-6)     # positive normalized depth
        t_norm = torch.stack([txn, tyn, zn], -1)  # (B,3)

        if D_obj is None:
            # return normalized if diameter not given (e.g., for loss code to handle)
            return r6, t_norm
        else:
            D = D_obj.view(-1,1)                  # (B,1)
            t = t_norm * D                        # meters
            return r6, t
