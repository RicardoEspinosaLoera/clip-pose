import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.LoRALinear import (
    _lora_wrap_linear, add_lora_to_vit_blocks
)

class DinoV3RegressorLoRA(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        neck_hidden: int = 512,
        dropout: float = 0.1,
        # freeze / unfreeze
        freeze_backbone: bool = True,        # default: freeze; LoRA carries learning
        unfreeze_last_blocks: int = 0,       # optional extra unfreeze on top of LoRA
        unfreeze_final_norm: bool = True,
        # LoRA
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_last_blocks: int = 6,           # put adapters only on the last K blocks
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            raise RuntimeError("Could not infer DINOv3 feature dim.")

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optional: partial unfreeze in addition to LoRA
        if not freeze_backbone and hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
            k = max(0, min(unfreeze_last_blocks, len(blocks)))
            for i in range(len(blocks)-k, len(blocks)):
                for p in blocks[i].parameters():
                    p.requires_grad = True
            if unfreeze_final_norm and hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        # LoRA adapters on last K blocks
        if use_lora and lora_rank > 0:
            add_lora_to_vit_blocks(
                self.backbone,
                last_k=lora_last_blocks,
                r=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # Neck + heads (trainable)
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head_rot = nn.Linear(neck_hidden, 6)
        self.head_t   = nn.Linear(neck_hidden, 3)

    def forward(self, x, D_obj=None):
        f = self.backbone(x)
        f = self.neck(f)
        r6 = self.head_rot(f)
        t3 = self.head_t(f)

        txn, tyn, logzn = t3[..., 0], t3[..., 1], t3[..., 2]
        zn = F.softplus(logzn).clamp_min(1e-6)
        t_norm = torch.stack([txn, tyn, zn], dim=-1)

        if D_obj is None:
            return r6, t_norm
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
