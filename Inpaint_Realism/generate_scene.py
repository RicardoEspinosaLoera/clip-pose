"""
generate_scene.py — generate a BRAND-NEW realistic surgical image (background +
clip) with the clip held at its exact 6-DoF pose.

Unlike run_inpaint.py (which keeps the real backplate and only repaints the clip),
this generates the WHOLE image from noise with a text-to-image ControlNet, using
the fused depth as the geometry condition. Because background and clip are produced
in a single pass, their lighting/shadows are coherent by construction — which is
exactly what the composite pipeline struggled with.

Pose preservation:
  * The depth ControlNet pins the clip's silhouette/orientation in its region.
  * `--lock-clip` additionally recomposites the exact-pose clip (from the relit
    render) over the generated background, GUARANTEEING the pose by construction
    (the model's clip is replaced by the geometrically exact one). Recommended when
    the pose ground truth must stay perfectly valid.

Reuses the SDXL base + depth ControlNet already downloaded (no new weights).

Usage:
  python generate_scene.py --limit 4
  python generate_scene.py --lock-clip --lora lora\lora_out --lora-trigger MACSCLIP
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
from masks import build_editable_mask, recomposite
from metrics import clip_edge_shift

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CANNY_CN = "diffusers/controlnet-canny-sdxl-1.0"   # blank bg = free scene (unlike depth)


def build_txt2img_pipeline(cfg: Config):
    """SDXL text-to-image + ControlNet (depth or canny), with 8 GB-VRAM offloading."""
    preset = cfg.preset()
    dtype = torch.float16
    from diffusers import (StableDiffusionXLControlNetPipeline, ControlNetModel,
                           AutoencoderKL)
    cn_model = CANNY_CN if getattr(cfg, "control_type", "depth") == "canny" else preset.controlnet_model
    controlnet = ControlNetModel.from_pretrained(cn_model, torch_dtype=dtype)
    kwargs = dict(controlnet=controlnet, torch_dtype=dtype)
    if preset.vae_model:
        kwargs["vae"] = AutoencoderKL.from_pretrained(preset.vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(SDXL_BASE, **kwargs)

    if cfg.lora_path:
        pipe.load_lora_weights(cfg.lora_path)
        pipe.fuse_lora(lora_scale=cfg.lora_scale)
        pipe.unload_lora_weights()
        print(f"Fused LoRA from {cfg.lora_path} (scale {cfg.lora_scale})")

    if cfg.low_vram and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        for fn in ("enable_vae_tiling", "enable_xformers_memory_efficient_attention"):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_depth_control(path, size):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.resize(g, size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def load_clip_depth_gray(depth_clip_path, mask_path, size, bg_value=110,
                         near_lo=150, near_hi=255):
    """Clip depth on a NEUTRAL mid-gray background (not black).

    A depth ControlNet reads black as 'void / infinitely far' and generates a dark
    empty scene around the clip. Filling the background with a constant mid-gray
    ('a surface at medium distance') lets the LoRA invent a free MACS scene there,
    while the clip's own depth values (remapped to a bright NEAR band) still give
    the model a SOLID volume cue at the right 6-DoF pose. Shape may drift; pose is
    conditioned, which is what we want."""
    g = cv2.imread(depth_clip_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if g is None or m is None:
        return None
    g = cv2.resize(g, size, interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    out = np.full(g.shape, bg_value, dtype=np.uint8)         # neutral surface
    clip = m > 127
    if clip.any():
        vals = g[clip].astype(np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            norm = (vals - lo) / (hi - lo)                   # keep the volume gradient
        else:
            norm = np.ones_like(vals)
        out[clip] = (near_lo + norm * (near_hi - near_lo)).astype(np.uint8)  # bright = near
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB))


def load_canny_control(render_path, mask_path, size):
    """Canny edges of the clip ONLY (blank background -> free scene).

    Pins the clip silhouette/orientation while leaving the background unconstrained,
    so the model invents a realistic MACS scene around a generated MACS clip."""
    g = cv2.imread(render_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if g is None or m is None:
        return None
    g = cv2.resize(g, size); m = cv2.resize(m, size)
    edges = cv2.Canny(g, 60, 160)
    # add the silhouette outline so a small clip is a strong-enough signal
    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges, cnts, -1, 255, 2)
    edges[m <= 127] = 0                                  # keep only clip edges
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # thicken
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Full-scene generation with pose-locked clip")
    p.add_argument("--backend", choices=["sdxl"], default="sdxl")
    p.add_argument("--data-dir", default=cfg.data_dir)
    p.add_argument("--out-dir", default="out_scene")
    p.add_argument("--limit", type=int, default=cfg.limit)
    p.add_argument("--start-index", type=int, default=cfg.start_index)
    p.add_argument("--steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--control", choices=["depth", "canny"], default="depth",
                   help="canny = blank bg -> free realistic scene + clip pinned by edges (some drift)")
    p.add_argument("--cn-scale", type=float, default=0.9,
                   help="control strength; higher = clip pose held more strictly")
    p.add_argument("--clip-depth-gray", action="store_true",
                   help="depth mode: use ONLY the clip depth on a mid-gray background "
                        "(conditions the clip pose/volume, leaves the scene free)")
    p.add_argument("--depth-bg", type=int, default=110,
                   help="mid-gray background value for --clip-depth-gray (0=void..255=near)")
    p.add_argument("--control-end", type=float, default=1.0,
                   help="release ControlNet after this fraction of steps "
                        "(e.g. 0.5 = pose set early, clip+scene refined freely after)")
    p.add_argument("--guidance", type=float, default=cfg.guidance_scale)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--variants", type=int, default=1,
                   help="how many different scenes to generate per pose (augmentation)")
    p.add_argument("--lock-clip", action="store_true",
                   help="recomposite the exact relit clip to guarantee the pose")
    p.add_argument("--relit-suffix", default="_relit",
                   help="suffix of the relit render used by --lock-clip")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-sbs", action="store_true")
    p.add_argument("--prompt", default=("a photorealistic intraoperative microsurgical photograph "
                                        "taken through a surgical microscope, wet glistening brain "
                                        "tissue, red blood vessels, cerebrospinal fluid, specular "
                                        "highlights, shallow depth of field, real surgery footage"))
    # scene generation WANTS blood/red/tissue, so the negative must NOT suppress
    # them (unlike the clip-inpaint negative). It suppresses the CGI/illustration
    # look that generic SDXL falls into.
    p.add_argument("--neg-prompt", default=("illustration, drawing, cartoon, 3d render, cgi, "
                                            "anatomical diagram, clean, plastic, synthetic, "
                                            "painting, stylized, blurry, low quality, text, watermark"))
    p.add_argument("--lora", default=cfg.lora_path)
    p.add_argument("--lora-scale", type=float, default=cfg.lora_scale)
    p.add_argument("--lora-trigger", default=cfg.lora_trigger)
    args = p.parse_args()

    cfg.backend = args.backend
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.limit = args.limit
    cfg.start_index = args.start_index
    cfg.num_inference_steps = args.steps
    cfg.control_type = args.control
    cfg.controlnet_conditioning_scale = args.cn_scale
    cfg.guidance_scale = args.guidance
    cfg.seed = args.seed
    cfg.negative_prompt = args.neg_prompt
    cfg.verify_shift = not args.no_verify
    cfg.save_side_by_side = not args.no_sbs
    cfg.prompt = args.prompt
    cfg.lora_path = args.lora
    cfg.lora_scale = args.lora_scale
    cfg.lora_trigger = args.lora_trigger
    if cfg.lora_trigger and cfg.lora_trigger.lower() not in cfg.prompt.lower():
        cfg.prompt = f"{cfg.lora_trigger}, {cfg.prompt}"
    cfg._variants = args.variants
    cfg._lock_clip = args.lock_clip
    cfg._relit_suffix = args.relit_suffix
    cfg._clip_depth_gray = args.clip_depth_gray
    cfg._depth_bg = args.depth_bg
    cfg._control_end = args.control_end
    return cfg


def main():
    cfg = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(here, cfg.data_dir)
    out_dir = cfg.out_dir if os.path.isabs(cfg.out_dir) else os.path.join(here, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.json")))[cfg.start_index:]
    if cfg.limit is not None:
        jsons = jsons[:cfg.limit]
    if not jsons:
        print(f"No samples in {data_dir}")
        return

    print(f"Full-scene generation | {len(jsons)} poses x {cfg._variants} variant(s) "
          f"| lock-clip={cfg._lock_clip} | out: {out_dir}")
    print("Loading pipeline (first run may download SDXL base)...")
    pipe = build_txt2img_pipeline(cfg)

    shifts, t_start = [], time.time()
    for i, jp in enumerate(jsons):
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp) as f:
            meta = json.load(f)
        img = meta.get("image", {})

        rgb_path = os.path.join(data_dir, img.get("file_rgb", f"{stem}.png"))
        orig_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        h, w = orig_bgr.shape[:2]
        size = (w, h)
        mask_path = os.path.join(data_dir, img.get("file_mask", f"{stem}_mask.png"))

        if cfg.control_type == "canny":
            control = load_canny_control(rgb_path, mask_path, size)
        elif getattr(cfg, "_clip_depth_gray", False):
            depth_name = img.get("file_depth_clip", f"{stem}_depth_clip.png")
            control = load_clip_depth_gray(os.path.join(data_dir, depth_name), mask_path,
                                           size, bg_value=cfg._depth_bg)
        else:
            ctrl_name = img.get("file_depth_fused") or img.get("file_depth_clip")
            control = load_depth_control(os.path.join(data_dir, ctrl_name), size) if ctrl_name else None
        if control is None:
            print(f"[SKIP] {stem}: control image unreadable")
            continue

        for v in range(cfg._variants):
            gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            gen.manual_seed(cfg.seed + i * 100 + v)
            result = pipe(
                prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
                image=control, controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
                control_guidance_end=getattr(cfg, "_control_end", 1.0),
                num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale,
                generator=gen, height=h, width=w).images[0]
            scene_bgr = cv2.resize(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR), size)

            # hybrid: paste the geometrically exact clip over the generated scene
            if cfg._lock_clip:
                relit_path = os.path.join(data_dir, f"{stem}{cfg._relit_suffix}.png")
                clip_src = cv2.imread(relit_path, cv2.IMREAD_COLOR)
                if clip_src is None:                       # fall back to the raw render
                    clip_src = orig_bgr
                _, alpha = build_editable_mask(mask_path, None, dilate_px=0, feather_px=2, size=size)
                final_bgr = recomposite(scene_bgr, clip_src, alpha)
            else:
                final_bgr = scene_bgr

            tag = f"{stem}_gen" if cfg._variants == 1 else f"{stem}_gen{v}"
            cv2.imwrite(os.path.join(out_dir, f"{tag}.png"), final_bgr)

            out_meta = dict(meta)
            out_meta.setdefault("image", {})["file_real"] = f"{tag}.png"
            out_meta["generation"] = {"mode": "full_scene", "lock_clip": cfg._lock_clip,
                                      "cn_scale": cfg.controlnet_conditioning_scale,
                                      "variant": v, "seed": cfg.seed + i * 100 + v}
            shift = None
            if cfg.verify_shift:
                clip_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                shift = clip_edge_shift(orig_bgr, final_bgr, clip_mask)
                shifts.append(shift)
                out_meta["generation"]["clip_shift_px"] = round(shift, 4)
            with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
                json.dump(out_meta, f, indent=2)

            if cfg.save_side_by_side:
                ctrl_bgr = cv2.cvtColor(np.asarray(control), cv2.COLOR_RGB2BGR)
                sbs = np.concatenate([orig_bgr, cv2.resize(ctrl_bgr, size), final_bgr], axis=1)
                cv2.imwrite(os.path.join(out_dir, f"{tag}_sbs.png"), sbs)

            msg = f"[{i+1}/{len(jsons)}] {tag}.png"
            if shift is not None:
                msg += f" | clip shift {shift:.3f} px"
            print(msg)

    dt = time.time() - t_start
    n = len(jsons) * cfg._variants
    print(f"\nDone. {n} images in {dt:.1f}s ({dt/max(n,1):.1f}s/img)")
    if shifts:
        arr = np.array(shifts)
        print(f"Clip displacement: mean {arr.mean():.3f} px | max {arr.max():.3f} px "
              f"| >1px: {(arr > 1.0).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
