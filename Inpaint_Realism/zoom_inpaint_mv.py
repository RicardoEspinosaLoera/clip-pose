"""
zoom_inpaint_mv.py — depth-consistent zoomed SDXL/ControlNet realism for the
multi-view 3-D dataset (scene_XXXX/images|depth|masks + transforms.json).

Same zoom idea as zoom_inpaint.py (crop a window around the clip, upscale, masked
ControlNet-depth inpaint with the MACS+clip combined LoRA, recomposite), but:
  * reads the multi-view output format, and
  * uses OUR DENSE SCENE DEPTH (depth/vNN.npy, clip + tissue) as the ControlNet-depth
    control over the whole crop -- so the restyle is consistent with the real depth
    everywhere in the zoom area, instead of the old mid-gray-surroundings hack.

The clip's projected silhouette is preserved (pose GT stays valid); the tissue around
it in the crop is restyled to coherent MACS, pinned by the depth control.

Usage:
  python zoom_inpaint_mv.py --data-dir ../GT_Generator/_decide --lora lora/combined_lora --limit 1
"""

import os
import glob
import json
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from pipeline_build import build_pipeline
# reuse the validated crop / blend helpers
from zoom_inpaint import clip_bbox, square_window, context_alpha


def dense_depth_control(depth_crop, work):
    """Our dense scene depth (metric, larger=farther) -> ControlNet-depth control
    (near = bright), normalised over the crop. NaN (no geometry) -> far/dark."""
    d = depth_crop.astype(np.float32)
    valid = np.isfinite(d)
    n = np.zeros_like(d)
    if valid.any():
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
        n[valid] = 1.0 - (d[valid] - lo) / (hi - lo + 1e-9)      # near -> bright
    g = cv2.resize((n * 255).astype(np.uint8), (work, work), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))


def iter_views(data_dir, limit, start):
    scenes = sorted(glob.glob(os.path.join(data_dir, "scene_*")))
    items = []
    for sc in scenes:
        for img in sorted(glob.glob(os.path.join(sc, "images", "v*.png"))):
            v = os.path.splitext(os.path.basename(img))[0]
            items.append((sc, v))
    items = items[start:]
    if limit:
        items = items[:limit]
    return items


def parse_args():
    p = argparse.ArgumentParser(description="Depth-consistent zoom inpaint (multi-view)")
    cfg = Config()
    here = os.path.dirname(__file__)
    p.add_argument("--data-dir", default=os.path.join(here, "..", "GT_Generator", "_decide"))
    p.add_argument("--out-dir", default="out_mv_real")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--pad", type=float, default=3.0)
    p.add_argument("--work", type=int, default=768)
    p.add_argument("--strength", type=float, default=0.55)
    p.add_argument("--cn-scale", type=float, default=0.85, help="depth control strength (pin geometry)")
    p.add_argument("--control-end", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--guidance", type=float, default=cfg.guidance_scale)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--lora", default=None)
    p.add_argument("--lora-scale", type=float, default=cfg.lora_scale)
    p.add_argument("--prompt", default=("MACS intraoperative microsurgical photograph through a "
                                        "surgical microscope, a titanium aneurysm clip clamping a vessel, "
                                        "polished blue titanium, sharp specular highlights, sharp focus"))
    p.add_argument("--neg-prompt", default=cfg.negative_prompt)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config()
    cfg.lora_path = a.lora
    cfg.lora_scale = a.lora_scale
    cfg.low_vram = True

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = a.data_dir if os.path.isabs(a.data_dir) else os.path.normpath(os.path.join(here, a.data_dir))
    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(here, a.out_dir)

    items = iter_views(data_dir, a.limit, a.start_index)
    if not items:
        print(f"No views under {data_dir}")
        return
    print(f"Zoom-inpaint MV | {len(items)} views | pad {a.pad} work {a.work} "
          f"strength {a.strength} cn {a.cn_scale} | out: {out_dir}")
    pipe = build_pipeline(cfg)

    t0 = time.time()
    for i, (sc, v) in enumerate(items):
        scene = os.path.basename(sc)
        render = cv2.imread(os.path.join(sc, "images", f"{v}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(sc, "masks", f"{v}.png"), cv2.IMREAD_GRAYSCALE)
        depth = np.load(os.path.join(sc, "depth", f"{v}.npy"))
        if render is None or mask is None:
            print(f"[SKIP] {scene}/{v}: missing render/mask"); continue
        H, W = render.shape[:2]
        bbox = clip_bbox(mask)
        if bbox is None:
            print(f"[SKIP] {scene}/{v}: empty clip mask"); continue

        x0, y0, side = square_window(bbox, a.pad, W, H)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        crop, mask_crop, depth_crop = render[sl], mask[sl], depth[sl]

        crop_up = cv2.resize(crop, (a.work, a.work), interpolation=cv2.INTER_CUBIC)
        hard = np.full((a.work, a.work), 255, np.uint8)                # restyle the whole crop
        control = dense_depth_control(depth_crop, a.work)              # OUR dense depth pins it

        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        gen.manual_seed(a.seed + i)
        result = pipe(
            prompt=a.prompt, negative_prompt=a.neg_prompt,
            image=Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)),
            mask_image=Image.fromarray(hard), control_image=control,
            num_inference_steps=a.steps, guidance_scale=a.guidance, strength=a.strength,
            controlnet_conditioning_scale=a.cn_scale, control_guidance_end=a.control_end,
            generator=gen, height=a.work, width=a.work).images[0]
        res_down = cv2.resize(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR),
                              (side, side), interpolation=cv2.INTER_AREA)

        alpha = context_alpha(mask_crop, side)[..., None]
        blended = (alpha * res_down.astype(np.float32) + (1 - alpha) * crop.astype(np.float32)).astype(np.uint8)
        out_full = render.copy(); out_full[sl] = blended

        od = os.path.join(out_dir, scene, "images")
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, f"{v}.png"), out_full)
        # side-by-side crop triptych: before | depth-control | after
        ctrl = cv2.resize(cv2.cvtColor(np.asarray(control), cv2.COLOR_RGB2BGR), (a.work, a.work))
        tri = np.concatenate([cv2.resize(crop, (a.work, a.work)), ctrl,
                              cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)], axis=1)
        cv2.imwrite(os.path.join(out_dir, scene, f"{v}_crop.png"), tri)
        print(f"[{i+1}/{len(items)}] {scene}/{v}  window {side}px  ({(time.time()-t0)/(i+1):.1f}s/img avg)")

    dt = time.time() - t0
    print(f"\nDone. {len(items)} images in {dt:.1f}s ({dt/max(len(items),1):.1f}s/img)")


if __name__ == "__main__":
    main()
