import torch, torch.nn as nn, torch.nn.functional as F
import math

def _lora_wrap_linear(module: nn.Module, attr: str, r: int, alpha: float, dropout: float):
    """
    Replace module.<attr> (must be nn.Linear) with LoRALinear, preserving weights.
    """
    lin = getattr(module, attr)
    assert isinstance(lin, nn.Linear), f"{attr} is not nn.Linear"
    lora_lin = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
    setattr(module, attr, lora_lin)

def add_lora_to_vit_blocks(backbone: nn.Module, last_k: int, r: int, alpha: float, dropout: float):
    """
    timm ViT has backbone.blocks[i].attn with .qkv and .proj linears.
    Add LoRA to qkv and proj in the last_k blocks.
    """
    if not hasattr(backbone, "blocks"):
        raise RuntimeError("Backbone has no .blocks; not a ViT-like model.")

    blocks = backbone.blocks
    k = max(0, min(last_k, len(blocks)))
    for i in range(len(blocks) - k, len(blocks)):
        attn = blocks[i].attn
        # timm ViT attention uses fused qkv and a proj layer
        _lora_wrap_linear(attn, "qkv",  r=r, alpha=alpha, dropout=dropout)
        _lora_wrap_linear(attn, "proj", r=r, alpha=alpha, dropout=dropout)


class LoRALinear(nn.Linear):
    """
    LoRA wrapper around a frozen base Linear.
    y = x W^T + b + scale * x A^T B^T  (implemented as B(A(x)))
    """
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float=16.0, dropout: float=0.0):
        assert isinstance(base_linear, nn.Linear)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        super().__init__(in_f, out_f, bias=base_linear.bias is not None)

        # copy the pretrained weights/bias
        with torch.no_grad():
            self.weight.copy_(base_linear.weight)
            if self.bias is not None and base_linear.bias is not None:
                self.bias.copy_(base_linear.bias)

        # freeze base weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA adapters
        self.r = r
        self.scaling = alpha / float(r) if r > 0 else 0.0
        self.lora_down = nn.Linear(in_f, r, bias=False) if r > 0 else None
        self.lora_up   = nn.Linear(r, out_f, bias=False) if r > 0 else None
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # init LoRA (common practice)
        if r > 0:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.r and self.r > 0:
            y = y + self.scaling * self.lora_up(self.lora_down(self.lora_drop(x)))
        return y
