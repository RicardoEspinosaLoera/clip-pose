# src/data_transforms.py
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

def build_backbone_transform(backbone, to_size_from_cfg=True):
    """
    Returns a callable that takes a float tensor image [C,H,W] in [0,1] and:
      - resizes to cfg.input_size[-1] (if to_size_from_cfg=True)
      - normalizes with cfg.mean/std
    """
    cfg = backbone.pretrained_cfg
    size = cfg.get("input_size", (3, 224, 224))[-1]
    mean = cfg.get("mean", (0.5, 0.5, 0.5))
    std  = cfg.get("std",  (0.5, 0.5, 0.5))
    interp = cfg.get("interpolation", "bicubic").upper()
    interp = getattr(T.InterpolationMode, interp, T.InterpolationMode.BICUBIC)

    def _transform(x):
        # x: [C,H,W], float in [0,1]
        if to_size_from_cfg and (x.shape[-1] != size or x.shape[-2] != size):
            x = TF.resize(x, [size, size], interpolation=interp, antialias=True)
        x = TF.normalize(x, mean=mean, std=std)
        return x
    return _transform
