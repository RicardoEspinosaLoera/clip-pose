import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("Please `pip install timm` to use the DINOv3 backbone.") from e


class DinoV3Regressor(nn.Module):
    """
    DINOv3-based regressor.
    - Uses timm to load a DINOv3 ViT backbone.
    - Global pooled features -> small MLP neck -> rot(6D), t(3)
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        neck_hidden: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Build DINOv3 backbone; num_classes=0 + global_pool='avg' => returns pooled features (B, C)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # Fallback: many timm ViTs expose embed_dim
            feat_dim = getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            raise RuntimeError("Could not infer DINOv3 feature dimension from backbone.")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Neck: small MLP head (same spirit as your ResNet version)
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Heads
        self.head_rot = nn.Linear(neck_hidden, 6)  # 6D rotation
        self.head_t   = nn.Linear(neck_hidden, 3)  # tx, ty, log(z/D) normalized

    def forward(self, x, D_obj=None):
        """
        x: (B,3,H,W) preprocessed to match the DINOv3 backbone's expected input.
           (Use timm's transform config at training time.)
        D_obj: scalar or (B,) object diameter/range for de-normalizing t if provided.
        """
        f = self.backbone(x)          # (B, C)
        f = self.neck(f)              # (B, H)

        r6 = self.head_rot(f)         # (B, 6)
        t3 = self.head_t(f)           # (B, 3)

        txn, tyn, logzn = t3[..., 0], t3[..., 1], t3[..., 2]
        # Positive, stable depth (softplus safer than exp)
        zn = F.softplus(logzn).clamp_min(1e-6)   # > 0
        t_norm = torch.stack([txn, tyn, zn], dim=-1)  # (B, 3)

        if D_obj is None:
            return r6, t_norm
        else:
            if not torch.is_tensor(D_obj):
                D_obj = torch.tensor(D_obj, device=t3.device, dtype=t3.dtype)
            D_obj = D_obj.to(device=t3.device, dtype=t3.dtype)

            B = x.shape[0]
            if D_obj.dim() == 0:
                D_obj = D_obj.expand(B)      # scalar -> (B,)
            elif D_obj.dim() == 1 and D_obj.shape[0] != B:
                D_obj = D_obj.reshape(1).expand(B)

            t = t_norm * D_obj.view(-1, 1)   # meters
            return r6, t
