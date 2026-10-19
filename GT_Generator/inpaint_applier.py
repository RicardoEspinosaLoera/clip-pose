"""
inpaint_applier.py — re-paint ONLY the clip applier with diffusion (the clip stays untouched).

The rendered applier is a clean CG rod; SDXL inpainting can turn that region into a real-looking
instrument. The mask covers the applier (dilated) MINUS the clip mask (dilated), so every clip
pixel is copied back verbatim: the 6-DoF GT, the clip mask and the visible mask stay valid by
construction. Structure inside the mask is held by an SDXL canny ControlNet computed on the
current composite; the MACS domain LoRA (SDXL) pushes the texture toward real footage.

Inputs : a dir produced by restyle_appearance.py with NNN_mask_applier.png (samples without an
         applier are skipped). Outputs: NNN.png rewritten (clean copy kept as NNN_preinpaint.png
         with --keep-clean), JSON gains `applier_inpaint`; montage before | after | real.

Usage
  python inpaint_applier.py --src <dir> --limit 8 --strength 0.85 --montage <dir>/_applier_inpaint.png
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
from restyle_appearance import zoom_cell, real_crop_cell

SDXL_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
LORA = os.path.normpath(os.path.join(HERE, "..", "Inpaint_Realism", "lora", "macs_lora"))
TRIGGER = "MACS intraoperative microsurgical photograph through a surgical microscope"
PROMPT = (TRIGGER + ", a dark metallic surgical clip applier shaft holding the clip, brushed steel with soft "
          "specular highlights, slightly out of focus, wet tissue around, photorealistic video still")
NEG = "cartoon, illustration, cgi, 3d render, plastic, toy, clean, sharp edges, text, watermark, extra instruments"


def build_pipe(lora_scale):
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
    dtype = torch.float16
    cn = ControlNetModel.from_pretrained(SDXL_CANNY, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(SDXL_INPAINT, controlnet=cn, vae=vae,
                                                                      torch_dtype=dtype)
    if lora_scale > 0 and os.path.exists(LORA):
        pipe.load_lora_weights(LORA)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
    pipe.enable_model_cpu_offload()
    for fn in ("enable_vae_tiling", "enable_xformers_memory_efficient_attention"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strength", type=float, default=0.70)
    ap.add_argument("--cn", type=float, default=0.0, help="canny scale; 0 = off (edges forced sharp lines on a blurred rod)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--lora-scale", type=float, default=0.5)
    ap.add_argument("--no-reblur", action="store_true", help="skip re-applying the render's defocus profile to the repainted rod")
    ap.add_argument("--window", type=int, default=704, help="max side of the crop sent to SDXL (speed); 0 = full frame")
    ap.add_argument("--dilate", type=int, default=7, help="grow the applier mask (px) so its soft edge is repainted too")
    ap.add_argument("--keep-clean", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage", default=None)
    ap.add_argument("--blend", type=float, default=0.7, help="repaint weight over the render inside the region (keeps body)")
    ap.add_argument("--protect", type=float, default=1.2, help="keep the render within this many clip-diagonals of the clip (junction)")
    ap.add_argument("--min-body", type=float, default=0.80, help="revert if the rod gets this much closer to the backplate (see-through)")
    ap.add_argument("--max-texture", type=float, default=1.35, help="revert if luminance structure along the rod grows more than this")
    ap.add_argument("--max-odd", type=float, default=0.005,
                    help="revert to the render when more than this fraction of the region turns an odd (non-tissue) colour")
    a = ap.parse_args()
    src = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    stems = [os.path.basename(p)[:-len("_mask_applier.png")]
             for p in sorted(glob.glob(os.path.join(src, "[0-9]*_mask_applier.png")))]
    if a.limit:
        stems = stems[:a.limit]
    if not stems:
        print("no samples with an applier mask in", src)
        return
    print(f"{len(stems)} samples with applier | loading SDXL inpaint + canny CN + MACS LoRA...", flush=True)
    pipe = build_pipe(a.lora_scale)
    rows, t0 = [], time.time()
    rng_real = np.random.default_rng(a.seed + 5)
    for i, stem in enumerate(stems):
        jp = os.path.join(src, f"{stem}.json")
        meta = json.load(open(jp))
        img = cv2.imread(os.path.join(src, f"{stem}.png"))
        clip = cv2.imread(os.path.join(src, meta["image"].get("file_mask", f"{stem}_mask.png")), 0) > 127
        amask = cv2.imread(os.path.join(src, f"{stem}_mask_applier.png"), 0) > 127
        H, W = img.shape[:2]
        k = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8)
        region = (cv2.dilate(amask.astype(np.uint8), k) > 0) & ~(cv2.dilate(clip.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        # protect the junction: the tip/grip zone is in focus and must stay the solid render (the
        # repaint kept inventing light/dark segments there); only the shaft farther out is repainted
        ys_c, xs_c = np.nonzero(clip)
        cyc, cxc = ys_c.mean(), xs_c.mean()
        diag = float(np.hypot(np.ptp(xs_c), np.ptp(ys_c)))
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.hypot(xx - cxc, yy - cyc)
        region = region & (dist >= a.protect * diag)
        near = region & (dist < (a.protect + 1.5) * diag)        # where banding shows up if it does
        if region.sum() < 400:
            continue
        # crop window around the repaint region (+ margin), multiple of 8, capped for speed
        ry, rx = np.nonzero(region)
        if a.window and a.window > 0:
            mrg = 48
            x0, x1 = max(0, rx.min() - mrg), min(W, rx.max() + mrg + 1)
            y0, y1 = max(0, ry.min() - mrg), min(H, ry.max() + mrg + 1)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        crop = img[y0:y1, x0:x1]
        mcrop = (region[y0:y1, x0:x1].astype(np.uint8) * 255)
        ch, cw = crop.shape[:2]
        scale = min(1.0, a.window / max(ch, cw)) if a.window else 1.0
        tw, th = max(8, int(round(cw * scale / 8)) * 8), max(8, int(round(ch * scale / 8)) * 8)
        init_np = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        mask_np = cv2.resize(mcrop, (tw, th), interpolation=cv2.INTER_NEAREST)
        edges = cv2.Canny(cv2.cvtColor(init_np, cv2.COLOR_BGR2GRAY), 50, 140)
        init = Image.fromarray(cv2.cvtColor(init_np, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_np)
        ctrl = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        gen = torch.Generator(device="cuda").manual_seed(a.seed + i)
        res = pipe(prompt=PROMPT, negative_prompt=NEG, image=init, mask_image=mask_pil, control_image=ctrl,
                   strength=a.strength, num_inference_steps=a.steps, guidance_scale=a.guidance,
                   controlnet_conditioning_scale=a.cn, generator=gen, height=th, width=tw).images[0]
        res_c = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (cw, ch), interpolation=cv2.INTER_CUBIC)
        res = img.copy()
        res[y0:y1, x0:x1] = res_c
        # re-apply the render's defocus profile so the repainted rod keeps the finish but not the
        # sharp edges/texture SDXL invents: sigma grows from the clip toward the frame border
        if not a.no_reblur:
            ap_info = meta.get("appearance", {}).get("applier") or {}
            s_near, s_far = float(ap_info.get("blur_near", 3.0)), float(ap_info.get("blur_far", 12.0))
            cyc, cxc = np.nonzero(clip)[0].mean(), np.nonzero(clip)[1].mean()
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            wfar = (np.clip(np.hypot(xx - cxc, yy - cyc) / (0.6 * np.hypot(W, H)), 0, 1) ** 1.2).astype(np.float32)
            rn = cv2.GaussianBlur(res, (0, 0), s_near * 0.8).astype(np.float32)
            rf = cv2.GaussianBlur(res, (0, 0), s_far * 0.8).astype(np.float32)
            res = np.clip(rn * (1 - wfar)[..., None] + rf * wfar[..., None], 0, 255).astype(np.uint8)
        # recomposite: ONLY the (feathered) applier region changes; clip pixels are verbatim
        alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.0) * a.blend
        alpha[clip] = 0.0
        out = alpha[..., None] * res.astype(np.float32) + (1 - alpha[..., None]) * img.astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)
        # hallucination guard: diffusion sometimes paints coloured blobs / blue tints on the rod.
        # Count region pixels that became saturated in a NON-tissue hue (not red/orange) and did
        # not look like that in the render; above a small fraction -> keep the render for this sample.
        def odd_colour(im):
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            return (sat > 100) & (v > 90) & (h > 22) & (h < 160)
        odd = float((odd_colour(out) & ~odd_colour(img) & region).sum() / max(region.sum(), 1))
        # structural guards: the repaint must keep the rod's BODY (not drift toward the clip-free
        # backplate = see-through) and must not add strong structure along it (bands/segments)
        bgp = os.path.join(src, meta["image"].get("file_bg", f"{stem}_bg.png"))
        bgim = cv2.imread(bgp)
        body_ratio, tex_ratio = 1.0, 1.0
        if bgim is not None and bgim.shape == img.shape:
            d_ren = np.abs(img.astype(np.float32) - bgim.astype(np.float32)).mean(-1)[region].mean()
            d_rep = np.abs(out.astype(np.float32) - bgim.astype(np.float32)).mean(-1)[region].mean()
            body_ratio = float(d_rep / max(d_ren, 1e-3))
        def lum_std(im, reg):
            return float(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)[reg].std())
        tex_ratio = lum_std(out, region) / max(lum_std(img, region), 1.0)
        if near.sum() > 300:                                    # local check near the junction
            tex_ratio = max(tex_ratio, lum_std(out, near) / max(lum_std(img, near), 1.0))
        reasons = []
        if odd > a.max_odd: reasons.append(f"odd colour {odd*100:.1f}%")
        if body_ratio < a.min_body: reasons.append(f"see-through {body_ratio:.2f}")
        if tex_ratio > a.max_texture: reasons.append(f"banding x{tex_ratio:.2f}")
        reverted = bool(reasons)
        if reverted:
            out = img
        changed_in_clip = int(np.abs(out.astype(int) - img.astype(int))[clip].max())
        if a.keep_clean:
            cv2.imwrite(os.path.join(src, f"{stem}_preinpaint.png"), img)
        cv2.imwrite(os.path.join(src, f"{stem}.png"), out)
        meta["applier_inpaint"] = dict(model="sdxl-inpaint+macs_lora", strength=a.strength, cn=a.cn,
                                       lora_scale=a.lora_scale, guidance=a.guidance, reblur=not a.no_reblur,
                                       window=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                                       region_px=int(region.sum()), max_change_in_clip=changed_in_clip,
                                       odd_colour_frac=round(odd, 4), body_ratio=round(body_ratio, 3),
                                       texture_ratio=round(tex_ratio, 3), blend=a.blend, protect=a.protect,
                                       reverted=bool(reverted), revert_reasons=reasons)
        json.dump(meta, open(jp, "w"), indent=2)
        if a.montage:
            m8 = clip.astype(np.uint8) * 255
            rows.append(np.hstack([zoom_cell(img, m8, 300, pad=6.0, label=f"{stem} antes"),
                                   zoom_cell(out, m8, 300, pad=6.0, label=f"{stem} " + ("render (revertido)" if reverted else "colocador repintado")),
                                   real_crop_cell(rng_real, 300)]))
        print(f"[{i+1}/{len(stems)}] {stem} region {int(region.sum())} px | odd {odd*100:.1f}% body {body_ratio:.2f} tex x{tex_ratio:.2f}"
              f"{' -> REVERTED (' + ', '.join(reasons) + ')' if reverted else ''} | clip change {changed_in_clip}", flush=True)
    dt = time.time() - t0
    print(f"\n{len(stems)} samples in {dt:.0f}s ({dt / max(1, len(stems)):.1f} s per image)")
    if a.montage and rows:
        cv2.imwrite(a.montage, np.vstack(rows))
        print("montage ->", a.montage)


if __name__ == "__main__":
    main()
