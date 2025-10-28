import torch, torch.nn as nn, torch.nn.functional as F
import math

def _lora_wrap_linear(module: nn.Module, attr: str, r: int, alpha: float, dropout: float):
    lin = getattr(module, attr)
    assert isinstance(lin, nn.Linear), f"{attr} is not nn.Linear"
    lora_lin = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
    setattr(module, attr, lora_lin)

def add_lora_to_vit_blocks(
    backbone: nn.Module,
    last_k: int,
    r: int,
    alpha: float,
    dropout: float,
    include_mlp: bool = False,   # NEW: optionally add LoRA to MLP fc1/fc2
):
    """
    Add LoRA to attention (qkv + proj) in the last_k blocks of a timm ViT.
    Optionally also add to MLP (fc1, fc2).
    """
    if not hasattr(backbone, "blocks"):
        raise RuntimeError("Backbone has no .blocks; not a ViT-like model.")

    blocks = backbone.blocks
    k = max(0, min(last_k, len(blocks)))
    for i in range(len(blocks) - k, len(blocks)):
        attn = blocks[i].attn
        _lora_wrap_linear(attn, "qkv",  r=r, alpha=alpha, dropout=dropout)
        _lora_wrap_linear(attn, "proj", r=r, alpha=alpha, dropout=dropout)
        if include_mlp and hasattr(blocks[i], "mlp"):
            mlp = blocks[i].mlp
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                _lora_wrap_linear(mlp, "fc1", r=r, alpha=alpha, dropout=dropout)
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                _lora_wrap_linear(mlp, "fc2", r=r, alpha=alpha, dropout=dropout)

class LoRALinear(nn.Linear):
    """
    LoRA wrapper around a frozen base Linear.
    y = xW^T + b + scale * (lora_up(lora_down(drop(x))))
    """
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 16.0, dropout: float = 0.0):
        assert isinstance(base_linear, nn.Linear)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        super().__init__(in_f, out_f, bias=(base_linear.bias is not None))

        with torch.no_grad():
            self.weight.copy_(base_linear.weight)
            if self.bias is not None and base_linear.bias is not None:
                self.bias.copy_(base_linear.bias)

        # freeze base weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA params
        self.r = int(r)
        self.scaling = float(alpha) / float(r) if r and r > 0 else 0.0
        self.lora_down = nn.Linear(in_f, r, bias=False) if r and r > 0 else None
        self.lora_up   = nn.Linear(r, out_f, bias=False) if r and r > 0 else None
        self.lora_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r and self.r > 0:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.r and self.r > 0:
            y = y + self.scaling * self.lora_up(self.lora_down(self.lora_drop(x)))
        return y
