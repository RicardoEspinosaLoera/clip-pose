"""
masks.py — editable-region construction and the pose-preserving recomposite.

The editable region is the clip plus a feathered contact ring; everything outside
it is copied verbatim from the original render so the pose ground truth cannot be
disturbed.
"""

import cv2
import numpy as np


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def build_editable_mask(clip_mask_path, contact_mask_path=None,
                        dilate_px=21, feather_px=9, size=None):
    """
    Return:
      hard_mask  uint8 {0,255}  -- binary editable region (for the inpaint pipeline)
      alpha      float32 [0,1]  -- feathered matte (for the final recomposite)

    Uses the precomputed contact mask if available, else dilates the clip mask.
    """
    if contact_mask_path is not None:
        region = load_gray(contact_mask_path)
    elif dilate_px and dilate_px > 0:
        clip = load_gray(clip_mask_path)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        region = cv2.dilate((clip > 127).astype(np.uint8) * 255, k)
    else:                                   # dilate_px <= 0 -> exact clip silhouette
        clip = load_gray(clip_mask_path)
        region = (clip > 127).astype(np.uint8) * 255

    if size is not None and (region.shape[1], region.shape[0]) != size:
        region = cv2.resize(region, size, interpolation=cv2.INTER_NEAREST)

    hard_mask = (region > 127).astype(np.uint8) * 255

    # feathered alpha for seamless blending at the border
    if feather_px > 0:
        ksize = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(hard_mask.astype(np.float32), (ksize, ksize), 0) / 255.0
    else:
        alpha = hard_mask.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    return hard_mask, alpha


def recomposite(original_bgr, generated_bgr, alpha):
    """
    out = alpha * generated + (1 - alpha) * original

    With alpha == 0 outside the editable region, the output equals the original
    pixel-for-pixel there -> the clip's pose ground truth is preserved by
    construction (identical to the paper's mask-composite property).
    """
    a = alpha[..., None]
    out = a * generated_bgr.astype(np.float32) + (1.0 - a) * original_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
