"""
model.py — M1 keypoint heatmap network (SimpleBaseline: ResNet-34 + 3 deconvs) and M2 direct
6-DoF regression head (translation + 6D rotation) on the same backbone.
"""

import torch
import torch.nn as nn
import torchvision


class Backbone(nn.Module):
    def __init__(self, arch="resnet34", pretrained=True):
        super().__init__()
        weights = {"resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
                   "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                   "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2}[arch] if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)
        self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
        self.out_ch = 2048 if arch == "resnet50" else 512

    def forward(self, x):
        return self.body(x)                       # stride 32


class KeypointNet(nn.Module):
    """Input (B,3,512,512) -> heatmaps (B,K,128,128)."""

    def __init__(self, n_kp=8, arch="resnet34", pretrained=True, deconv_ch=256):
        super().__init__()
        self.backbone = Backbone(arch, pretrained)
        layers, ch = [], self.backbone.out_ch
        for _ in range(3):
            layers += [nn.ConvTranspose2d(ch, deconv_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(deconv_ch), nn.ReLU(inplace=True)]
            ch = deconv_ch
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch, n_kp, 1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.deconv(self.backbone(x)))


def rot6d_to_matrix(x):
    """(B,6) -> (B,3,3) via Gram-Schmidt (Zhou et al. 2019)."""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = nn.functional.normalize(a1, dim=1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


class DirectPoseNet(nn.Module):
    """Input crop + K_crop (flattened) -> translation (3) and 6D rotation (3x3)."""

    def __init__(self, arch="resnet34", pretrained=True):
        super().__init__()
        self.backbone = Backbone(arch, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.backbone.out_ch + 4, 512), nn.ReLU(inplace=True), nn.Linear(512, 9))

    def forward(self, x, K_crop):
        f = self.pool(self.backbone(x)).flatten(1)
        kvec = torch.stack([K_crop[:, 0, 0], K_crop[:, 1, 1], K_crop[:, 0, 2], K_crop[:, 1, 2]], 1) / 1000.0
        out = self.fc(torch.cat([f, kvec], 1))
        t = out[:, :3] * torch.tensor([10.0, 10.0, 100.0], device=out.device) + torch.tensor([0.0, 0.0, 100.0], device=out.device)
        R = rot6d_to_matrix(out[:, 3:])
        return R, t


def heatmap_loss(pred, target, vis):
    """MSE over visible keypoints only; vis (B,K)."""
    w = vis[:, :, None, None]
    return ((pred - target) ** 2 * w).sum() / (w.sum() * pred.shape[-1] * pred.shape[-2] + 1e-6)


def geodesic_sym(R_pred, R_gt, R_sym):
    """Rotation error in degrees, min over the clip's symmetry; all (B,3,3)."""
    def ang(Ra, Rb):
        tr = (Ra.transpose(1, 2) @ Rb).diagonal(dim1=1, dim2=2).sum(1)
        return torch.rad2deg(torch.acos(torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)))
    return torch.minimum(ang(R_pred, R_gt), ang(R_pred, R_gt @ R_sym))
