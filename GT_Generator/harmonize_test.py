"""
harmonize_test.py — does a light diffusion pass make the composite look like ONE photo?

Hypothesis: the remaining "pasted" look is not colour or shape but texture/noise/lighting
coherence, which img2img is good at. Unlike the earlier silhouette-inpaint + shape-bound
LoRA (which could only recolour, or broke the shape), this re-synthesises a CROP around the
clip (clip + tissue) at LOW strength with the structure pinned by two ControlNets:
  * canny of the current composite (every edge, clip and tissue alike)
  * clip depth on mid-gray (volume cue for the clip)
No LoRA. SD1.5 for speed (a 5000-sample run must be feasible on the 8 GB laptop GPU).
The result is pasted back with a feathered crop border; the clip displacement is measured
with the robust template metric so the GT validity can be checked.

Outputs: <out>/NNN_h{strength}.png (full frames), <out>/_harmonize_montage.png
         (before | strength A | strength B | real crop), shift stats on stdout.

Usage
  python harmonize_test.py --src out_pilot_v3_r2 --out _harm_test --limit 12 --strengths 0.3,0.45
"""

import os
import sys
import json
import glob
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism")))
from restyle_appearance import zoom_cell, real_crop_cell
from metrics import clip_shift_robust

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"
CN_CANNY = "lllyasviel/control_v11p_sd15_canny"
PROMPT = ("intraoperative microsurgery photograph through a surgical microscope, a titanium aneurysm clip "
          "on a blood vessel, wet glistening brain tissue, specular highlights, shallow depth of field, "
          "video still, photorealistic")
NEG = "cartoon, illustration, cgi, 3d render, plastic, smooth, clean, painting, blurry, text, watermark"


CN_TILE = "lllyasviel/control_v11f1e_sd15_tile"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
SD15_CN = {"depth": CN_DEPTH, "canny": CN_CANNY, "tile": CN_TILE}
SDXL_CN = {"depth": SDXL_DEPTH, "canny": SDXL_CANNY}


def build_pipe(model="sd15", controls=("depth", "canny"), lora=None, lora_scale=0.8):
    from diffusers import ControlNetModel, UniPCMultistepScheduler
    dtype = torch.float16
    if model == "sd15":
        from diffusers import StableDiffusionControlNetImg2ImgPipeline as Pipe
        cns = [ControlNetModel.from_pretrained(SD15_CN[c], torch_dtype=dtype) for c in controls]
        pipe = Pipe.from_pretrained(SD15, controlnet=cns, torch_dtype=dtype, safety_checker=None)
    else:
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as Pipe, AutoencoderKL
        cns = [ControlNetModel.from_pretrained(SDXL_CN[c], torch_dtype=dtype) for c in controls]
        vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=dtype)
        pipe = Pipe.from_pretrained(SDXL, controlnet=cns, vae=vae, torch_dtype=dtype)
    if lora:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    for fn in ("enable_vae_tiling", "enable_xformers_memory_efficient_attention"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    pipe.set_progress_bar_config(disable=True)
    return pipe


def square_window(mask, pad, W, H, min_side=256):
    ys, xs = np.where(mask > 127)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = int(max(min_side, pad * max(xs.max() - xs.min(), ys.max() - ys.min())))
    side = min(side, min(W, H))
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))
    return x0, y0, side


def depth_control(depth_crop, mask_crop, work, bg_value=110, near_lo=150, near_hi=255):
    clip = mask_crop > 127
    out = np.full(depth_crop.shape, bg_value, dtype=np.uint8)
    if clip.any():
        vals = depth_crop[clip].astype(np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        norm = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        out[clip] = (near_lo + norm * (near_hi - near_lo)).astype(np.uint8)
    out = cv2.resize(out, (work, work), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB))


def border_alpha(side, frac=0.12):
    b = max(4, int(side * frac))
    win = np.ones((side, side), np.float32)
    win[:b] = 0; win[-b:] = 0; win[:, :b] = 0; win[:, -b:] = 0
    return cv2.GaussianBlur(win, (0, 0), b / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_pilot_v3_r2")
    ap.add_argument("--out", default="_harm_test")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--strengths", default="0.3,0.45")
    ap.add_argument("--work", type=int, default=512)
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--cn", default="0.8,0.7", help="conditioning scales: depth,canny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="sd15", choices=["sd15", "sdxl"])
    ap.add_argument("--controls", default="depth,canny", help="comma list of depth,canny,tile (tile = sd15 only)")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--tag", default="h")
    a = ap.parse_args()
    controls = [c for c in a.controls.split(",") if c]
    src = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    strengths = [float(s) for s in a.strengths.split(",")]
    cn_scales = [float(s) for s in a.cn.split(",")]
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(src, "[0-9]*.json")))[:a.limit]

    print(f"loading {a.model} + {controls} ControlNets (first run may download weights)...", flush=True)
    pipe = build_pipe(a.model, controls, a.lora, a.lora_scale)
    prompt = (a.trigger + ", " + PROMPT) if a.trigger else PROMPT
    rng = np.random.default_rng(a.seed + 7)
    rows, shifts, t0 = [], {s: [] for s in strengths}, time.time()
    for i, stem in enumerate(stems):
        meta = json.load(open(os.path.join(src, f"{stem}.json")))
        img = cv2.imread(os.path.join(src, f"{stem}.png"))
        mask = cv2.imread(os.path.join(src, meta["image"].get("file_mask", f"{stem}_mask.png")), 0)
        depth = cv2.imread(os.path.join(src, meta["image"].get("file_depth_clip", f"{stem}_depth_clip.png")), cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        H, W = img.shape[:2]
        x0, y0, side = square_window(mask, a.pad, W, H)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        crop, mcrop, dcrop = img[sl], mask[sl], depth[sl]
        crop_up = cv2.resize(crop, (a.work, a.work), interpolation=cv2.INTER_CUBIC)
        edges = cv2.Canny(cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY), 60, 160)
        ctrl_canny = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        ctrl_depth = depth_control(dcrop, mcrop, a.work)
        init = Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB))
        ctrl_map = {"depth": ctrl_depth, "canny": ctrl_canny, "tile": init}
        ctrls = [ctrl_map[c] for c in controls]
        scales = (cn_scales + [cn_scales[-1]] * len(controls))[:len(controls)]
        cells = [zoom_cell(img, mask, 256, pad=3.0, label=f"{stem} before")]
        for s in strengths:
            gen = torch.Generator(device="cuda").manual_seed(a.seed + i)
            res = pipe(prompt=prompt, negative_prompt=NEG, image=init, control_image=ctrls,
                       strength=s, num_inference_steps=a.steps, guidance_scale=a.guidance,
                       controlnet_conditioning_scale=scales, generator=gen).images[0]
            res_bgr = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
            res_down = cv2.resize(res_bgr, (side, side), interpolation=cv2.INTER_AREA)
            al = border_alpha(side)[..., None]
            full = img.copy()
            full[sl] = (al * res_down.astype(np.float32) + (1 - al) * crop.astype(np.float32)).astype(np.uint8)
            tag = f"{a.tag}{int(round(s * 100)):03d}"
            cv2.imwrite(os.path.join(out, f"{stem}_{tag}.png"), full)
            sh = clip_shift_robust(img, full, mask)
            shifts[s].append(sh)
            cells.append(zoom_cell(full, mask, 256, pad=3.0, label=f"{stem} strength {s:.2f} shift {sh:.1f}px"))
        cells.append(real_crop_cell(rng, 256))
        rows.append(np.hstack(cells))
        print(f"[{i+1}/{len(stems)}] {stem} window {side}px | shifts " +
              ", ".join(f"{s:.2f}:{shifts[s][-1]:.1f}px" for s in strengths), flush=True)
    grid = np.vstack(rows)
    cv2.imwrite(os.path.join(out, f"_harmonize_montage_{a.tag}.png"), grid)
    dt = time.time() - t0
    print(f"\n{len(stems)} samples x {len(strengths)} strengths in {dt:.0f}s "
          f"({dt / max(1, len(stems) * len(strengths)):.1f} s per image)")
    for s in strengths:
        arr = np.array(shifts[s])
        print(f"strength {s:.2f}: clip shift median {np.median(arr):.2f}px max {arr.max():.2f}px | >2px: {(arr > 2).sum()}/{len(arr)}")
    print("montage ->", os.path.join(out, f"_harmonize_montage_{a.tag}.png"))


if __name__ == "__main__":
    main()
