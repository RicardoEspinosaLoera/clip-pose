"""
hadepth_infer.py — single-image inference wrapper for HADepth
(highlight-aware endoscopic monocular depth, DepthAnything ViT-B + DoRA).

The repo (external/HADepth) ships only evaluation code, so this adapts its model
construction + forward into a clean `hadepth_disparity(bgr)` call returning a
relative disparity map in [0,1] (higher = closer), the same convention as the
Depth Anything path in estimate_bg_depth.py — so it is a drop-in disparity source
for building the tissue height field.

Weights: HADEPTH_WEIGHTS env or the default Downloads path.
"""

import os
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")   # force standard attention (no xformers dep)

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HAD_ROOT = os.path.normpath(os.path.join(HERE, "..", "external", "HADepth"))
WEIGHTS = os.environ.get("HADEPTH_WEIGHTS", r"C:\Users\ricar\Downloads\depth_model.pth")

# checkpoint config (inferred from external/HADepth/options.py + the state_dict):
#   lora_type=dora, r=4, residual_block_indexes=[2,5,8,11], include_cls_token=True,
#   image_shape=(224,280); dataset feeds RGB/255 (no ImageNet normalisation).
_CFG = dict(backbone_size="base", r=4, lora_type="dora", image_shape=(224, 280),
            pretrained_path=None, residual_block_indexes=[2, 5, 8, 11], include_cls_token=True)
_IN_HW = (256, 320)   # dataset resize (H, W) before the model's internal 224x280 interp

_STATE = {}


def _model():
    if "m" in _STATE:
        return _STATE["m"], _STATE["dev"]
    if HAD_ROOT not in sys.path:
        sys.path.insert(0, HAD_ROOT)
    import models.hadepth as hadepth_mod
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = hadepth_mod.hadepth(**_CFG)
    ck = torch.load(WEIGHTS, map_location="cpu")
    md = m.state_dict()
    used = {k: v for k, v in ck.items() if k in md and v.shape == md[k].shape}
    res = m.load_state_dict(used, strict=False)
    missing_lora = [k for k in res.missing_keys if "lora" in k.lower()]
    print(f"[hadepth] loaded {len(used)}/{len(md)} params from checkpoint "
          f"(missing {len(res.missing_keys)}, of which {len(missing_lora)} lora)")
    m.eval().to(dev)
    _STATE["m"], _STATE["dev"] = m, dev
    return m, dev


def hadepth_disparity(bgr):
    """Relative disparity in [0,1] (higher = closer) at the input resolution."""
    m, dev = _model()
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (_IN_HW[1], _IN_HW[0]))
    t = (torch.from_numpy(rgb.astype(np.float32) / 255.0)
         .permute(2, 0, 1)[None].to(dev))
    with torch.no_grad():
        out = m(t)
    disp = out[("disp", 0)][0, 0].float().cpu().numpy()
    disp = cv2.resize(disp, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(disp.min()), float(disp.max())
    return ((disp - lo) / (hi - lo + 1e-9)).astype(np.float32)


if __name__ == "__main__":
    import glob
    p = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(HERE, "Data", "noclips_bank", "*.png")))[0]
    d = hadepth_disparity(cv2.imread(p))
    print("disparity", d.shape, "range", float(d.min()), float(d.max()))
