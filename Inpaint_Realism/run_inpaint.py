"""
run_inpaint.py — masked ControlNet-depth inpainting over the depth-aware dataset.

For each sample produced by `GT_Generator/render_clip_poses_depth.py`
(+ `estimate_bg_depth.py`):

  1. Load the original render, the editable contact mask, and the geometry
     control image (fused depth, falling back to clip depth or normals).
  2. Run SDXL/SD1.5 ControlNet-depth inpainting restricted to the contact mask.
  3. Hard-recomposite outside the mask from the original render, so the 6-DoF
     pose ground truth is preserved by construction.
  4. Carry the pose JSON forward unchanged (clip never moves) and, optionally,
     verify the clip displacement with the edge phase-correlation metric.

Usage
-----
  python run_inpaint.py                         # SDXL, all samples
  python run_inpaint.py --backend sd15 --limit 5
  python run_inpaint.py --start-index 2000 --limit 2000     # resumable chunk
"""

import os
import json
import glob
import time
import shutil
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from pipeline_build import build_pipeline
from masks import build_editable_mask, recomposite
from metrics import clip_edge_shift


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Masked ControlNet-depth inpainting")
    p.add_argument("--backend", choices=["sdxl", "sd15"], default=cfg.backend)
    p.add_argument("--data-dir", default=cfg.data_dir)
    p.add_argument("--out-dir", default=cfg.out_dir)
    p.add_argument("--limit", type=int, default=cfg.limit)
    p.add_argument("--start-index", type=int, default=cfg.start_index)
    p.add_argument("--steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--cn-scale", type=float, default=cfg.controlnet_conditioning_scale)
    p.add_argument("--strength", type=float, default=cfg.strength)
    p.add_argument("--guidance", type=float, default=cfg.guidance_scale)
    p.add_argument("--dilate", type=int, default=cfg.dilate_px)
    p.add_argument("--feather", type=int, default=cfg.feather_px)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-sbs", action="store_true")
    p.add_argument("--prompt", default=cfg.prompt)
    p.add_argument("--lora", default=cfg.lora_path, help="dir/.safetensors with the clip LoRA")
    p.add_argument("--lora-scale", type=float, default=cfg.lora_scale)
    p.add_argument("--lora-trigger", default=cfg.lora_trigger)
    p.add_argument("--init-dir", default=None,
                   help="refine the clip on top of scenes here (NNN_gen.png) instead of the raw render")
    args = p.parse_args()

    cfg.backend = args.backend
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.limit = args.limit
    cfg.start_index = args.start_index
    cfg.num_inference_steps = args.steps
    cfg.controlnet_conditioning_scale = args.cn_scale
    cfg.strength = args.strength
    cfg.guidance_scale = args.guidance
    cfg.dilate_px = args.dilate
    cfg.feather_px = args.feather
    cfg.seed = args.seed
    cfg.verify_shift = not args.no_verify
    cfg.save_side_by_side = not args.no_sbs
    cfg.prompt = args.prompt
    cfg.lora_path = args.lora
    cfg.lora_scale = args.lora_scale
    cfg.lora_trigger = args.lora_trigger
    cfg.init_dir = args.init_dir
    if cfg.init_dir and not os.path.isabs(cfg.init_dir):
        cfg.init_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.init_dir)
    # prepend the LoRA trigger phrase so the adapter is actually invoked
    if cfg.lora_trigger and cfg.lora_trigger.lower() not in cfg.prompt.lower():
        cfg.prompt = f"{cfg.lora_trigger}, {cfg.prompt}"
    return cfg


def resolve(d, fname):
    return os.path.join(d, fname) if fname else None


def pick_control_path(d, meta, sources):
    """Return (path, kind) for the first available geometry control source."""
    img = meta.get("image", {})
    for key in sources:
        fname = img.get(key)
        if fname and os.path.exists(os.path.join(d, fname)):
            return os.path.join(d, fname), key
    return None, None


def load_control_image(path, kind, size):
    """Load a geometry control as an RGB PIL image (closer = brighter for depth)."""
    if kind == "file_normal":
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g.dtype != np.uint8:
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(rgb)


def make_side_by_side(orig_bgr, control_pil, result_bgr):
    ctrl = cv2.cvtColor(np.asarray(control_pil), cv2.COLOR_RGB2BGR)
    ctrl = cv2.resize(ctrl, (orig_bgr.shape[1], orig_bgr.shape[0]))
    return np.concatenate([orig_bgr, ctrl, result_bgr], axis=1)


def main():
    cfg = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(here, cfg.data_dir)
    out_dir = cfg.out_dir if os.path.isabs(cfg.out_dir) else os.path.join(here, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.json")))
    jsons = jsons[cfg.start_index:]
    if cfg.limit is not None:
        jsons = jsons[:cfg.limit]
    if not jsons:
        print(f"No samples found in {data_dir}")
        return

    print(f"Backend: {cfg.backend} | samples: {len(jsons)} | out: {out_dir}")
    print("Loading pipeline (first run downloads model weights)...")
    pipe = build_pipeline(cfg)

    shifts = []
    t_start = time.time()
    for i, jp in enumerate(jsons):
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp) as f:
            meta = json.load(f)
        img = meta.get("image", {})

        rgb_path  = resolve(data_dir, img.get("file_rgb", f"{stem}.png"))
        # optional: refine on top of an externally generated scene (e.g. a
        # generate_scene output) instead of the raw render. The clip region is
        # repainted toward MACS realism while everything else = that scene.
        if getattr(cfg, "init_dir", None):
            for cand in (f"{stem}_gen.png", f"{stem}_real.png", f"{stem}.png"):
                p = os.path.join(cfg.init_dir, cand)
                if os.path.exists(p):
                    rgb_path = p
                    break
        mask_path = resolve(data_dir, img.get("file_mask", f"{stem}_mask.png"))
        contact_path = resolve(data_dir, img.get("file_contact_mask"))
        if contact_path and not os.path.exists(contact_path):
            contact_path = None

        orig_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if orig_bgr is None:
            print(f"[SKIP] {stem}: missing {rgb_path}")
            continue
        h, w = orig_bgr.shape[:2]
        size = (w, h)

        ctrl_path, ctrl_kind = pick_control_path(data_dir, meta, cfg.control_sources)
        if ctrl_path is None:
            print(f"[SKIP] {stem}: no control image found")
            continue
        control_pil = load_control_image(ctrl_path, ctrl_kind, size)

        hard_mask, alpha = build_editable_mask(
            mask_path, contact_path, cfg.dilate_px, cfg.feather_px, size=size)

        init_pil = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(hard_mask)

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(cfg.seed + i)

        kw = dict(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=init_pil,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
            controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
            generator=generator,
            height=h, width=w,
        )
        # control timing args (supported by both controlnet inpaint pipelines)
        kw["control_guidance_start"] = cfg.control_guidance_start
        kw["control_guidance_end"] = cfg.control_guidance_end

        result = pipe(**kw).images[0]
        gen_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, size)

        # pose-preserving recomposite: outside the mask == original render
        final_bgr = recomposite(orig_bgr, gen_bgr, alpha)

        out_png = os.path.join(out_dir, f"{stem}{cfg.out_suffix}.png")
        cv2.imwrite(out_png, final_bgr)

        # carry the pose JSON forward unchanged (the clip never moved)
        out_json = os.path.join(out_dir, f"{stem}.json")
        meta.setdefault("image", {})["file_real"] = os.path.basename(out_png)
        meta["inpaint"] = {
            "backend": cfg.backend, "steps": cfg.num_inference_steps,
            "controlnet_conditioning_scale": cfg.controlnet_conditioning_scale,
            "strength": cfg.strength, "control_source": ctrl_kind,
            "guidance_scale": cfg.guidance_scale, "seed": cfg.seed + i,
        }

        shift = None
        if cfg.verify_shift:
            clip_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            shift = clip_edge_shift(orig_bgr, final_bgr, clip_mask)
            shifts.append(shift)
            meta["inpaint"]["clip_shift_px"] = round(shift, 4)

        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)

        if cfg.save_side_by_side:
            sbs = make_side_by_side(orig_bgr, control_pil, final_bgr)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_sbs.png"), sbs)

        msg = f"[{i+1}/{len(jsons)}] {stem}{cfg.out_suffix}.png"
        if shift is not None:
            msg += f" | clip shift {shift:.3f} px"
        print(msg)

    dt = time.time() - t_start
    n = len(jsons)
    print(f"\nDone. {n} images in {dt:.1f}s ({dt/max(n,1):.1f}s/img)")
    if shifts:
        arr = np.array(shifts)
        print(f"Clip displacement: mean {arr.mean():.3f} px | max {arr.max():.3f} px "
              f"| >1px: {(arr > 1.0).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
