"""
zoom_inpaint.py — pose-exact, realistic clip via ZOOMED masked inpainting.

The clip occupies <1% of a full GT frame, so a full-frame inpaint either leaves it
untouched (low strength) or dissolves it into tissue (high strength): SDXL simply
has too few pixels on the clip to render real metal.

This script fixes that by working on a CROP around the clip, upscaled so the clip
fills ~1/3 of the canvas. There SDXL (with the MACS+clip combined LoRA + the clip
depth as a volume cue) can render genuine titanium detail. The result is pasted
back at the exact crop location, and only the clip's true silhouette is changed —
so the projected 6-DoF pose is preserved by construction (the GT stays valid).

Pipeline per sample:
  1. crop a square window around the clip mask (--pad x the clip bbox)
  2. upscale crop + mask + clip-depth to --work px
  3. masked ControlNet-depth inpaint of the clip silhouette (combined LoRA)
  4. downscale back, recomposite only inside the (feathered) silhouette
  5. carry the pose JSON forward unchanged; verify clip edge shift

Usage:
  python zoom_inpaint.py --lora lora\combined_lora --limit 4
  python zoom_inpaint.py --pad 3.0 --work 768 --strength 0.65 --cn-scale 0.7
"""

import os
import json
import glob
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from pipeline_build import build_pipeline
from metrics import clip_shift_robust


def clip_bbox(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def square_window(bbox, pad, W, H):
    """Square crop centered on the clip bbox, side = pad*max(w,h), clamped to image."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = int(round(max(w, h) * pad))
    side = max(32, min(side, min(W, H)))           # never larger than the frame
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, W - side))                 # keep the window inside the frame
    y0 = max(0, min(y0, H - side))
    return x0, y0, side


def build_masks(mask_crop, dilate_px, feather_px):
    """hard mask for the pipeline (slightly dilated) + feathered alpha for blending."""
    m = (mask_crop > 127).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hard = cv2.dilate(m, k)
    else:
        hard = m
    if feather_px > 0:
        alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), feather_px) / 255.0
    else:
        alpha = (hard > 127).astype(np.float32)
    return hard, np.clip(alpha, 0.0, 1.0)


def depth_control(depth_crop, work):
    """clip depth crop -> RGB control, min-max normalized so the clip uses full range."""
    g = depth_crop
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif g.max() > g.min():
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.resize(g, (work, work), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))


def depth_control_context(depth_crop, mask_crop, work, bg_value=110, near_lo=150, near_hi=255):
    """Clip = bright NEAR band, surroundings = neutral mid-gray (not void).

    For --restyle-context: lets the depth ControlNet pin the clip's pose/volume while
    the mid-gray surroundings read as 'a surface at medium distance', so the LoRA
    restyles the tissue around the clip into a realistic MACS scene instead of the
    dark void a black background would force."""
    clip = mask_crop > 127
    out = np.full(depth_crop.shape, bg_value, dtype=np.uint8)
    if clip.any():
        vals = depth_crop[clip].astype(np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        norm = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        out[clip] = (near_lo + norm * (near_hi - near_lo)).astype(np.uint8)
    out = cv2.resize(out, (work, work), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB))


def context_alpha(mask_crop, side, border_frac=0.12):
    """Recomposite alpha for --restyle-context: 1 over the whole crop but feathered
    to 0 at the crop border (hides the seam with the untouched outer image), and
    forced to 1 over the clip silhouette (the clip is always fully restyled) -- EXCEPT
    the clip's INTERNAL holes (e.g. the coil/spring loops), which are set to 0 so they
    keep the original clean tissue that shows through the clip. Without this, SDXL
    repaints the whole crop and fills those small enclosed holes with a tissue-coloured
    BLOB (the clip is too small/blurry to resolve the thin coil)."""
    border = max(4, int(side * border_frac))
    win = np.ones((side, side), np.float32)
    win[:border] = 0; win[-border:] = 0; win[:, :border] = 0; win[:, -border:] = 0
    win = cv2.GaussianBlur(win, (0, 0), border / 2.0)
    clip = cv2.GaussianBlur((mask_crop > 127).astype(np.float32) * 255, (0, 0), 2.0) / 255.0
    a = np.clip(np.maximum(win, clip), 0.0, 1.0)
    # find the clip's enclosed holes: flood the background from a background corner; any
    # background NOT reached is an enclosed hole. Feather + zero the alpha there.
    mm = (mask_crop > 127).astype(np.uint8) * 255
    seed = next(((x, y) for (x, y) in [(0, 0), (side - 1, 0), (0, side - 1), (side - 1, side - 1)]
                 if mm[y, x] == 0), (0, 0))
    ff = mm.copy()
    cv2.floodFill(ff, np.zeros((side + 2, side + 2), np.uint8), seed, 255)
    holes = ff == 0
    if holes.any():
        holes = cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        keep = cv2.GaussianBlur((~holes).astype(np.float32), (0, 0), 1.0)   # 1 outside holes, 0 inside
        a = a * keep
    return a


def parse_args():
    p = argparse.ArgumentParser(description="Zoomed pose-exact realistic clip inpainting")
    cfg = Config()
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__),
                                                      "..", "GT_Generator", "out_rand_depth"))
    p.add_argument("--out-dir", default="out_clip_zoom")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--random", type=int, default=0,
                   help="diffuse N RANDOM samples (seeded) instead of a sequential slice")
    p.add_argument("--pad", type=float, default=3.0, help="crop side = pad x clip bbox")
    p.add_argument("--work", type=int, default=768, help="upscaled working resolution")
    p.add_argument("--strength", type=float, default=0.65, help="inpaint strength in the crop")
    p.add_argument("--cn-scale", type=float, default=0.7, help="depth control strength")
    p.add_argument("--control-end", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--guidance", type=float, default=cfg.guidance_scale)
    p.add_argument("--dilate", type=int, default=2, help="dilate the silhouette (work px)")
    p.add_argument("--feather", type=int, default=3, help="feather for seamless blend (work px)")
    p.add_argument("--restyle-context", action="store_true",
                   help="also restyle the tissue AROUND the clip inside the zoom crop "
                        "(repaints the whole crop; surroundings depth = mid-gray)")
    p.add_argument("--bg-depth", type=int, default=110,
                   help="mid-gray surroundings depth for --restyle-context")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-sbs", action="store_true")
    p.add_argument("--no-verify", action="store_true")
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
    os.makedirs(out_dir, exist_ok=True)

    all_jsons = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.json")))
    if a.random and a.random > 0:
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(all_jsons), size=min(a.random, len(all_jsons)), replace=False)
        jsons = [all_jsons[i] for i in sorted(pick)]
    else:
        jsons = all_jsons[a.start_index:]
        if a.limit is not None:
            jsons = jsons[:a.limit]
    if not jsons:
        print(f"No samples in {data_dir}")
        return

    print(f"Zoom-inpaint | {len(jsons)} samples | pad {a.pad} work {a.work} "
          f"strength {a.strength} cn {a.cn_scale} | out: {out_dir}")
    pipe = build_pipeline(cfg)

    shifts, t0 = [], time.time()
    for i, jp in enumerate(jsons):
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp) as f:
            meta = json.load(f)
        img = meta.get("image", {})

        render = cv2.imread(os.path.join(data_dir, img.get("file_rgb", f"{stem}.png")), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(data_dir, img.get("file_mask", f"{stem}_mask.png")), cv2.IMREAD_GRAYSCALE)
        depth_name = img.get("file_depth_clip", f"{stem}_depth_clip.png")
        depth = cv2.imread(os.path.join(data_dir, depth_name), cv2.IMREAD_UNCHANGED)
        if render is None or mask is None or depth is None:
            print(f"[SKIP] {stem}: missing render/mask/depth")
            continue
        H, W = render.shape[:2]
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        bbox = clip_bbox(mask)
        if bbox is None:
            print(f"[SKIP] {stem}: empty clip mask")
            continue
        x0, y0, side = square_window(bbox, a.pad, W, H)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))

        crop = render[sl]
        mask_crop = mask[sl]
        depth_crop = depth[sl]

        crop_up = cv2.resize(crop, (a.work, a.work), interpolation=cv2.INTER_CUBIC)
        mask_up = cv2.resize(mask_crop, (a.work, a.work), interpolation=cv2.INTER_NEAREST)
        if a.restyle_context:
            # repaint the WHOLE crop (clip + surrounding tissue)
            hard = np.full((a.work, a.work), 255, np.uint8)
            control = depth_control_context(depth_crop, mask_crop, a.work, bg_value=a.bg_depth)
        else:
            hard, _ = build_masks(mask_up, a.dilate, a.feather)
            control = depth_control(depth_crop, a.work)

        init_pil = Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(hard)

        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        gen.manual_seed(a.seed + i)
        result = pipe(
            prompt=a.prompt, negative_prompt=a.neg_prompt,
            image=init_pil, mask_image=mask_pil, control_image=control,
            num_inference_steps=a.steps, guidance_scale=a.guidance, strength=a.strength,
            controlnet_conditioning_scale=a.cn_scale, control_guidance_end=a.control_end,
            generator=gen, height=a.work, width=a.work).images[0]
        res_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

        # downscale back to crop size and recomposite
        res_down = cv2.resize(res_bgr, (side, side), interpolation=cv2.INTER_AREA)
        if a.restyle_context:
            # blend the whole restyled crop (feathered at the border) + full clip
            alpha = context_alpha(mask_crop, side)
        else:
            # recomposite ONLY inside the clip silhouette
            _, alpha = build_masks(mask_crop, dilate_px=0, feather_px=max(1, a.feather // 2))
        a3 = alpha[..., None]
        blended = (a3 * res_down.astype(np.float32) + (1 - a3) * crop.astype(np.float32)).astype(np.uint8)

        out_full = render.copy()
        out_full[sl] = blended
        out_png = os.path.join(out_dir, f"{stem}_real.png")
        cv2.imwrite(out_png, out_full)

        meta.setdefault("image", {})["file_real"] = os.path.basename(out_png)
        meta["zoom_inpaint"] = {"pad": a.pad, "work": a.work, "strength": a.strength,
                                "cn_scale": a.cn_scale, "window": [x0, y0, side], "seed": a.seed + i}
        shift = None
        if not a.no_verify:
            shift = clip_shift_robust(render, out_full, mask)
            shifts.append(shift)
            meta["zoom_inpaint"]["clip_shift_px"] = round(shift, 4)
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if not a.no_sbs:
            ctrl_bgr = cv2.cvtColor(np.asarray(control), cv2.COLOR_RGB2BGR)
            ctrl_full = cv2.resize(ctrl_bgr, (side, side))
            strip = render.copy()
            cv2.rectangle(strip, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            # crop-level before/after/control triptych for a close look at the clip
            tri = np.concatenate([cv2.resize(crop, (a.work, a.work)),
                                  ctrl_bgr if ctrl_bgr.shape[0] == a.work else cv2.resize(ctrl_bgr, (a.work, a.work)),
                                  res_bgr], axis=1)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_sbs.png"), strip)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_crop.png"), tri)

        msg = f"[{i+1}/{len(jsons)}] {stem}_real.png  window {side}px"
        if shift is not None:
            msg += f" | clip shift {shift:.3f} px"
        print(msg)

    dt = time.time() - t0
    print(f"\nDone. {len(jsons)} images in {dt:.1f}s ({dt/max(len(jsons),1):.1f}s/img)")
    if shifts:
        arr = np.array(shifts)
        print(f"Clip displacement: mean {arr.mean():.3f} px | max {arr.max():.3f} px "
              f"| >1px: {(arr > 1.0).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
