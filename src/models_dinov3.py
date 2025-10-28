import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.LoRALinear import add_lora_to_vit_blocks

class DinoV3RegressorLoRA(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        neck_hidden: int = 512,
        dropout: float = 0.1,
        # freeze / unfreeze
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        unfreeze_final_norm: bool = True,
        # LoRA
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_last_blocks: int = 6,
        lora_include_mlp: bool = False,   # NEW
        # pooling
        pool_mode: str = "avg",           # "avg" | "cls" | "avg+cls"
        t_head_scale: float = 0.1,        # gentle early training for t
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.t_head_scale = float(t_head_scale)

        # IMPORTANT: no global_pool here so we can pool tokens ourselves
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feat_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            raise RuntimeError("Could not infer DINOv3 feature dim.")

        # Neck + heads (define early so we can init later)
        in_dim = feat_dim * (2 if pool_mode == "avg+cls" else 1)
        self.neck = nn.Sequential(
            nn.Linear(in_dim, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head_rot = nn.Linear(neck_hidden, 6)
        self.head_t   = nn.Linear(neck_hidden, 3)

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optional partial unfreeze
        if not freeze_backbone and hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
            k = max(0, min(unfreeze_last_blocks, len(blocks)))
            for i in range(len(blocks) - k, len(blocks)):
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
                include_mlp=lora_include_mlp,
            )

        # init heads lightly
        self._init_heads()

        # quick print of trainable params
        self._print_trainable_summary()

    # ---------- utilities ----------
    def _init_heads(self):
        def _init_lin(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.neck.apply(_init_lin)
        self.head_rot.apply(_init_lin)
        self.head_t.apply(_init_lin)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] from backbone.forward_features(x)
        """
        assert tokens.ndim == 3, "Expected tokens [B,N,C]"
        cls_tok, patch_tok = tokens[:, 0], tokens[:, 1:]  # [B,C], [B,N-1,C]
        if self.pool_mode == "avg":
            return patch_tok.mean(dim=1)
        elif self.pool_mode == "cls":
            return cls_tok
        elif self.pool_mode == "avg+cls":
            return torch.cat([patch_tok.mean(dim=1), cls_tok], dim=-1)
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

    def _print_trainable_summary(self):
        total, trainable, lora, heads = 0, 0, 0, 0
        for n, p in self.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                if "lora_" in n.lower():
                    lora += num
                if n.startswith("neck.") or n.startswith("head_"):
                    heads += num
        print(f"[DINOv3-LoRA] params: total={total/1e6:.2f}M  "
              f"trainable={trainable/1e6:.2f}M  (LoRA={lora/1e6:.2f}M, heads={heads/1e6:.2f}M)")

    # ---------- forward ----------
    def forward(self, x, D_obj=None):
        tokens = self.backbone.forward_features(x)     # [B,N,C]
        f = self._pool_tokens(tokens)
        f = self.neck(f)

        r6 = self.head_rot(f)
        t3 = self.t_head_scale * self.head_t(f)       # gentle start

        txn, tyn, logzn = t3.unbind(dim=-1)
        zn = F.softplus(logzn).clamp_min(1e-6)
        t_norm = torch.stack([txn, tyn, zn], dim=-1)

        if D_obj is None:
            return r6, t_norm

        if not torch.is_tensor(D_obj):
            D_obj = torch.tensor(D_obj, device=t3.device, dtype=t3.dtype)
        else:
            D_obj = D_obj.to(device=t3.device, dtype=t3.dtype)

        B = x.shape[0]
        if D_obj.ndim == 0:
            D_obj = D_obj.expand(B)
        elif D_obj.ndim == 1 and D_obj.shape[0] != B:
            D_obj = D_obj.reshape(1).expand(B)

        t = t_norm * D_obj[:, None]
        return r6, t

# -------- helper for optimizer param groups --------
def trainable_param_groups(model: nn.Module, base_lr=1e-5, lora_lr=1e-4, wd=0.05):
    heads, lora, stray = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n.lower():
            lora.append(p)
        elif n.startswith("neck.") or n.startswith("head_"):
            heads.append(p)
        else:
            stray.append(p)  # shouldn't really happen; just in case

    # If there are stray gradients, freeze them to be safe
    for p in stray:
        p.requires_grad = False

    assert len(lora) > 0, "No LoRA params found — check add_lora_to_vit_blocks() names."
    assert len(heads) > 0, "Heads must be trainable."

    return [
        {"params": heads, "lr": base_lr, "weight_decay": wd},
        {"params": lora,  "lr": lora_lr, "weight_decay": 0.0},
    ]
