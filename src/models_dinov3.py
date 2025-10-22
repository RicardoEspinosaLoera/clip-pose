import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DinoV3Regressor(nn.Module):
    """
    DINOv3-based regressor with partial unfreezing:
      - Freeze entire ViT backbone
      - Unfreeze last `unfreeze_last_blocks` transformer blocks + final norm
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        neck_hidden: int = 512,
        dropout: float = 0.1,
        # --- freezing controls ---
        freeze_backbone: bool = False,       # if True: freeze everything (no partial unfreeze)
        unfreeze_last_blocks: int = 1,       # ignored if freeze_backbone=True
        unfreeze_final_norm: bool = True,    # usually True
    ):
        super().__init__()

        # 1) Backbone (global pooled features)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            raise RuntimeError("Could not infer DINOv3 feature dim (num_features/embed_dim).")

        # 2) Freeze policy
        self._apply_freeze_policy(freeze_backbone, unfreeze_last_blocks, unfreeze_final_norm)

        # 3) Neck + Heads
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head_rot = nn.Linear(neck_hidden, 6)   # 6D rotation
        self.head_t   = nn.Linear(neck_hidden, 3)   # tx, ty, log(z/D)

    # ---- freeze helpers ------------------------------------------------------
    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_k_blocks(self, k: int, unfreeze_final_norm: bool):
        """
        Assumes a timm ViT with .blocks: ModuleList of transformer blocks, and .norm final layer.
        """
        if not hasattr(self.backbone, "blocks"):
            # Fallback: if model has no .blocks, skip (some conv backbones)
            return
        blocks = self.backbone.blocks
        k = max(0, min(k, len(blocks)))
        for i in range(len(blocks) - k, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True

        if unfreeze_final_norm and hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def _apply_freeze_policy(self, freeze_backbone: bool, unfreeze_last_blocks: int, unfreeze_final_norm: bool):
        # start by freezing everything
        self._freeze_all()

        if not freeze_backbone:
            # then unfreeze last K transformer blocks (+ final norm)
            self._unfreeze_last_k_blocks(unfreeze_last_blocks, unfreeze_final_norm)

    # -------------------------------------------------------------------------
    def forward(self, x, D_obj=None):
        f = self.backbone(x)          # (B, C)
        f = self.neck(f)              # (B, H)

        r6 = self.head_rot(f)
        t3 = self.head_t(f)

        txn, tyn, logzn = t3[..., 0], t3[..., 1], t3[..., 2]
        zn = F.softplus(logzn).clamp_min(1e-6)      # positive, stable depth
        t_norm = torch.stack([txn, tyn, zn], dim=-1)

        if D_obj is None:
            return r6, t_norm
        else:
            if not torch.is_tensor(D_obj):
                D_obj = torch.tensor(D_obj, device=t3.device, dtype=t3.dtype)
            D_obj = D_obj.to(device=t3.device, dtype=t3.dtype)

            B = x.shape[0]
            if D_obj.dim() == 0:
                D_obj = D_obj.expand(B)
            elif D_obj.dim() == 1 and D_obj.shape[0] != B:
                D_obj = D_obj.reshape(1).expand(B)

            t = t_norm * D_obj.view(-1, 1)
            return r6, t
