"""
mv_warp_realism.py — multi-view-consistent realism by DIFFUSE-ONE-VIEW + GEOMETRIC WARP.

Per-view SDXL is inconsistent across views. Instead:
  1. SDXL-restyle the CLIP in the reference view v00 (MACS combined LoRA + our depth as
     ControlNet control) -> a realistic titanium clip on the real tissue. ONE SDXL pass.
  2. WARP that realistic clip into every other view using our per-view depth + camera
     poses (exact geometric correspondence), compositing onto each view's already-
     consistent real tissue. Disocclusions fall back to the original render.

Result: a realistic clip that is IDENTICAL-in-appearance and geometrically consistent
across all 16 views, from a single diffusion pass (~6 min/scene, no OOM).

Usage:
  python mv_warp_realism.py --scene ../GT_Generator/_warptest/scene_0000 --lora lora/combined_lora
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from pipeline_build import build_pipeline

# EXACT combined_lora trigger (the caption it was trained on) -> keep as a fixed PREFIX so
# the LoRA activates; extra property tokens are appended after it to steer appearance.
TRIGGER = ("MACS intraoperative microsurgical photograph through a surgical microscope, "
           "a titanium aneurysm clip clamping a vessel")
from zoom_inpaint import (clip_bbox, square_window, build_masks,
                          depth_control, depth_control_context, context_alpha)


def tone_clip(bgr, mask, gain=0.85, hi_gamma=1.2, tint=(1.22, 1.02, 0.72)):
    """Give the clip a blue-anodized-titanium MATTE look: mild darken (gain<1),
    highlight rolloff (hi_gamma>1 = less shiny), and a BGR colour tint pushing it toward
    blue titanium (boost blue, cut red) without going too dark. Feathered on the mask."""
    out = bgr.astype(np.float32)
    alpha = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.2)[..., None]
    x = np.clip(out / 255.0, 0, 1)
    toned = np.power(x, hi_gamma) * gain * np.asarray(tint, np.float32)[None, None, :]
    toned = np.clip(toned * 255.0, 0, 255)
    return np.clip(alpha * toned + (1 - alpha) * out, 0, 255).astype(np.uint8)


def depth_control_clip(depth_crop, mask_crop, work, near_lo=0.18):
    """Clip-only ControlNet depth that MATCHES zoom_inpaint's file_depth_clip: a BRIGHT clip
    silhouette on a PURE BLACK background, near=brightest (standard depth convention).

    zoom_inpaint fed the ControlNet an 8/16-bit clip-only depth PNG (background 0, only the
    clip has values) -> the net renders the clip as a crisp standalone object. The multi-view
    scenes only carry a full-scene float .npy (near=small value), so here we mask it to the
    clip and INVERT polarity (near -> bright) with black elsewhere, reproducing that control."""
    clip = mask_crop > 127
    out = np.zeros(depth_crop.shape, np.float32)                 # black background
    if clip.any():
        vals = depth_crop[clip].astype(np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        norm = (hi - vals) / (hi - lo) if hi > lo else np.ones_like(vals)   # near(small)->1 bright
        out[clip] = near_lo + (1.0 - near_lo) * norm            # clip in [near_lo,1], bg 0
    out = cv2.resize((out * 255).astype(np.uint8), (work, work), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB))


def shade_clip(bgr, mask, depth, orig=None, light=(-0.4, -0.5, 0.85), amb=0.5, diff=0.5,
               ao=0.7, spec_knock=0.4, edge=0.55, sharpen=0.5):
    """Bake DEPTH-derived shading + STRUCTURAL EDGES onto the clip so it isn't a flat metal
    blob and keeps the detail SDXL smooths away:
      * surface normals from clip depth -> diffuse form (darker back-faces),
      * local depth-recess term -> ambient occlusion (ring concavity, gap between jaws),
      * isolated-highlight knock-down -> tames the 'disconnected' bright specular,
      * EDGE restoration: dark crevice lines along the clip's real structural edges, taken
        from the ORIGINAL grey render (Canny) + depth discontinuities -> brings back the coil
        loops / jaw / box-lock lines the inpaint loses (stricter texture & shadows),
      * mild unsharp -> crisper metal.
    Multiplicative + mean-normalised so metal texture/brightness are preserved. Feathered."""
    clip = mask > 127
    if not clip.any():
        return bgr
    d = depth.astype(np.float32)
    gx = np.clip(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) * 0.15, -4, 4)
    gy = np.clip(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) * 0.15, -4, 4)
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1); nx = -gx * inv; ny = -gy * inv; nz = inv
    L = np.array(light, np.float32); L /= np.linalg.norm(L)
    ndotl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    dblur = cv2.GaussianBlur(d, (0, 0), 9); recess = np.clip((d - dblur) / 6.0, 0, 1)
    shade = (amb + diff * ndotl) * (1 - ao * recess)
    m = shade[clip].mean() + 1e-6
    factor = np.clip(shade / m, 0.2, 1.7)[..., None]
    out = bgr.astype(np.float32).copy()
    if spec_knock > 0:                                        # tame the 'disconnected' hot blob
        hot = (out.max(2) > 200) & clip
        out[hot] = out[hot] * (1 - spec_knock)
    out = np.clip(out * factor, 0, 255)
    if edge > 0:                                              # restore structural crevice edges
        E = np.zeros(clip.shape, np.float32)
        if orig is not None:
            E = np.maximum(E, (cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), 40, 120) > 0).astype(np.float32))
        dn = np.zeros_like(d)
        dn[clip] = (d[clip] - d[clip].min()) / (np.ptp(d[clip]) + 1e-6)
        E = np.maximum(E, (cv2.Canny((dn * 255).astype(np.uint8), 30, 90) > 0).astype(np.float32))
        E[~clip] = 0
        E = cv2.GaussianBlur(cv2.dilate(E, np.ones((2, 2), np.uint8), 1), (0, 0), 0.8)
        out = out * (1 - edge * E[..., None])
    if sharpen > 0:                                           # crisper metal micro-contrast
        bl = cv2.GaussianBlur(out, (0, 0), 1.5)
        ca = cv2.GaussianBlur(clip.astype(np.float32), (0, 0), 0.8)[..., None]
        out = ca * np.clip(out + sharpen * (out - bl), 0, 255) + (1 - ca) * out
    a = cv2.GaussianBlur(clip.astype(np.float32), (0, 0), 1.0)[..., None]
    return (a * out + (1 - a) * bgr.astype(np.float32)).astype(np.uint8)


def contact_shadow(render, mask, depth, strength=0.4, elev_ref=15.0,
                   light_img=(0.5, 0.7), max_shift=13, blur=13):
    """Cast a TIGHT contact shadow of the clip onto the tissue so it doesn't float, hugging
    the clip base. Shadow = clip silhouette shifted along the (image-space) light direction by
    an amount scaling with the clip's elevation above surrounding tissue (median depth gap),
    lightly blurred, applied only to tissue (never the clip). Per-view from that view's
    depth+mask -> geometrically plausible as the camera orbits."""
    clip = mask > 127
    if not clip.any():
        return render
    ring = cv2.dilate(clip.astype(np.uint8), np.ones((21, 21), np.uint8), 1).astype(bool) & ~clip
    elev = max(0.0, float(np.median(depth[ring]) - np.median(depth[clip]))) if ring.any() else elev_ref
    sh = int(np.clip(elev / elev_ref * max_shift, 3, max_shift))
    dx, dy = int(light_img[0] * sh), int(light_img[1] * sh)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(clip.astype(np.float32), M, (render.shape[1], render.shape[0]))
    shad = cv2.GaussianBlur(shifted, (0, 0), blur / 3.0) * (~clip); shad = np.clip(shad, 0, 1)
    return np.clip(render.astype(np.float32) * (1 - strength * shad)[..., None], 0, 255).astype(np.uint8)


def restyle_v0(pipe, scene, a):
    """SDXL CLIP-ONLY restyle of view 0's zoom crop -> a realistic clip on the REAL tissue.
    Only the clip silhouette is repainted (surrounding tissue is left untouched, same as every
    other view), then depth-driven shading + edges are baked onto the clip. Returns
    (v0_real full frame, alpha0 = clip silhouette, feathered)."""
    render = cv2.imread(os.path.join(scene, "images", "v00.png"))
    mask = cv2.imread(os.path.join(scene, "masks", "v00.png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(os.path.join(scene, "depth", "v00.npy"))
    H, W = render.shape[:2]
    x0, y0, side = square_window(clip_bbox(mask), a.pad, W, H)
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    crop, mask_crop, depth_crop = render[sl], mask[sl], depth[sl]

    crop_up = cv2.resize(crop, (a.work, a.work), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(mask_crop, (a.work, a.work), interpolation=cv2.INTER_NEAREST)
    # CLIP-ONLY: inpaint just the (dilated) clip silhouette; depth control = BRIGHT clip on
    # BLACK (matches zoom_inpaint's file_depth_clip). Feeding the full-scene .npy here (the old
    # bug) gave the ControlNet a whole-crop gradient with the clip dark -> wrong clip.
    hard, _ = build_masks(mask_up, a.dilate, a.feather)
    control = depth_control_clip(depth_crop, mask_crop, a.work)

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(a.seed)
    result = pipe(prompt=a.prompt, negative_prompt=a.neg_prompt,
                  image=Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)),
                  mask_image=Image.fromarray(hard), control_image=control,
                  num_inference_steps=a.steps, guidance_scale=a.guidance, strength=a.strength,
                  controlnet_conditioning_scale=a.cn_scale, control_guidance_end=0.9,
                  generator=gen, height=a.work, width=a.work).images[0]
    res_down = cv2.resize(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR), (side, side), interpolation=cv2.INTER_AREA)
    # strip the SDXL tissue HALO: the inpaint (dilated mask) restyles a ring of tissue around
    # the clip; outside the exact clip silhouette keep the RAW crop so ONLY the clip is from SDXL
    # -> v00's near-clip tissue is identical to the warped views (no v00-only halo).
    res_down[mask_crop <= 127] = crop[mask_crop <= 127]
    # composite ONLY the clip silhouette (feathered) -> tissue around the clip is untouched
    _, clip_alpha = build_masks(mask_crop, dilate_px=0, feather_px=max(1, a.feather // 2))
    a3 = clip_alpha[..., None]
    v0 = render.copy()
    v0[sl] = (a3 * res_down.astype(np.float32) + (1 - a3) * crop.astype(np.float32)).astype(np.uint8)
    alpha0 = np.zeros((H, W), np.float32); alpha0[sl] = clip_alpha
    # optional blue-matte clip tone (OFF by default = exact zoom_inpaint look).
    if getattr(a, "clip_tone", False):
        v0 = tone_clip(v0, mask, gain=a.clip_gain, hi_gamma=a.clip_gamma)
    # optional depth self-shading + AO + edges on the clip (OFF by default = raw out_zoom_artic)
    if getattr(a, "shade", False):
        v0 = shade_clip(v0, mask, depth, orig=render, ao=a.shade_ao, spec_knock=a.spec_knock,
                        edge=a.edge_strength, sharpen=a.sharpen)
    # CONSISTENCY GUARANTEE: the SDXL inpaint (dilated mask) restyles a thin ring of tissue
    # around the clip that the warped views (v01..) don't have -> v00 would look different.
    # Re-blend the styled clip over the RAW render with the SAME feathered clip mask the warp
    # uses (sigma 1.3), so outside the clip v00 is exactly the raw tissue, identical to v01..v15.
    cf = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.3)[..., None]
    v0 = (cf * v0.astype(np.float32) + (1 - cf) * render.astype(np.float32)).astype(np.uint8)
    return v0, alpha0


def warp_region(v0_real, alpha0, mask0, depth0, T, fr0, frj, depth_j, mask_j, render_j, occ_tol=2.5):
    """CLIP-ONLY warp (consistent, ghost-free). Transfers ONLY the realistic clip from v0
    into view j; the tissue in view j stays its own real rendered tissue (already consistent
    across views), so nothing can smear onto it. Each view-j CLIP pixel (mask_j) is back-
    projected via its depth and reprojected into v0 (u0,v0); it takes v0's colour only where
    the reprojection lands on v0's clip (mask0) AND the depth matches (|reproj_d-depth0|<=
    occ_tol) = same clip surface. Clip pixels the warp can't reach (self-occluded between the
    two views) are inpainted from the covered clip. Composited feathered inside mask_j only,
    so tissue is never touched. (Restyled SDXL tissue is NOT warped: through our approximate
    monocular-depth height-field it speckles/ghosts; v00 keeps the full restyle, side views
    keep real tissue.)"""
    fx, fy, cx, cy = T["fl_x"], T["fl_y"], T["cx"], T["cy"]
    c2w_j = np.array(frj["transform_matrix"]); w2c0 = np.linalg.inv(np.array(fr0["transform_matrix"]))
    H, W = render_j.shape[:2]
    clip = mask_j > 127
    if not clip.any():
        return render_j.copy()
    ys, xs = np.where(clip)
    d = depth_j[ys, xs].astype(np.float64)
    ok = np.isfinite(d); dd = np.where(ok, d, 1.0)
    xc = (xs - cx) / fx * dd; yc = -(ys - cy) / fy * dd; zc = -dd
    P0 = w2c0 @ (c2w_j @ np.stack([xc, yc, zc, np.ones_like(dd)], 0))
    X, Y, Z = P0[0], P0[1], P0[2]
    front = Z < -1e-6
    u0 = np.rint(fx * (X / np.where(front, -Z, 1)) + cx).astype(int)
    v0 = np.rint(cy - fy * (Y / np.where(front, -Z, 1))).astype(int)
    inb = ok & front & (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H)
    reproj_d = -Z
    good = inb.copy()
    good[inb] &= (np.abs(reproj_d[inb] - depth0[v0[inb], u0[inb]]) <= occ_tol) \
        & (mask0[v0[inb], u0[inb]] > 127)                   # lands on v0's clip, same surface
    warped = render_j.copy()
    warped[ys[good], xs[good]] = v0_real[v0[good], u0[good]]
    # clip pixels the warp couldn't reach (self-occlusion) -> inpaint from the covered clip
    gap = np.zeros((H, W), np.uint8); gap[ys[~good], xs[~good]] = 255
    if gap.any():
        warped = cv2.inpaint(warped, gap, 3, cv2.INPAINT_TELEA)
    # composite ONLY inside view j's clip (feathered) -> tissue untouched, grey clip replaced
    alpha = cv2.GaussianBlur(clip.astype(np.float32), (0, 0), 1.3)[..., None]
    return (alpha * warped.astype(np.float32) + (1 - alpha) * render_j.astype(np.float32)).astype(np.uint8)


def process_scene(pipe, scene, a, out_dir):
    """Restyle v0's clip once (SDXL), warp it to every view, write to out_dir/images."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    T = json.load(open(os.path.join(scene, "transforms.json")))
    fr0 = T["frames"][0]
    v0_real, alpha0 = restyle_v0(pipe, scene, a)
    depth0 = np.load(os.path.join(scene, "depth", "v00.npy"))
    mask0 = cv2.imread(os.path.join(scene, "masks", "v00.png"), cv2.IMREAD_GRAYSCALE)
    if getattr(a, "shade", False):                                # optional: ground v0's clip with a cast shadow
        v0_real = contact_shadow(v0_real, mask0, depth0, strength=a.shadow_strength)
    cv2.imwrite(os.path.join(out_dir, "images", "v00.png"), v0_real)
    for j, frj in enumerate(T["frames"]):
        if j == 0:
            continue
        v = f"v{j:02d}"
        depth_j = np.load(os.path.join(scene, "depth", f"{v}.npy"))
        mask_j = cv2.imread(os.path.join(scene, "masks", f"{v}.png"), cv2.IMREAD_GRAYSCALE)
        render_j = cv2.imread(os.path.join(scene, "images", f"{v}.png"))
        out = warp_region(v0_real, alpha0, mask0, depth0, T, fr0, frj, depth_j, mask_j, render_j)
        if getattr(a, "shade", False):                           # optional per-view contact shadow
            out = contact_shadow(out, mask_j, depth_j, strength=a.shadow_strength)
        cv2.imwrite(os.path.join(out_dir, "images", f"{v}.png"), out)


def parse_args():
    cfg = Config(); here = os.path.dirname(__file__)
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="single scene dir")
    p.add_argument("--data-dir", default=None, help="dataset dir of scene_* (loops all)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--lora", default="lora/combined_lora")
    p.add_argument("--lora-scale", type=float, default=cfg.lora_scale)
    p.add_argument("--pad", type=float, default=2.6)          # EXACT out_zoom_artic recipe
    p.add_argument("--work", type=int, default=768)
    p.add_argument("--strength", type=float, default=0.5)     # 0.5 keeps small/thin clips SOLID
                                                              # (0.6+ dissolves them into translucent teal)
    p.add_argument("--cn-scale", type=float, default=0.85)    # EXACT out_zoom_artic recipe
    p.add_argument("--bg-depth", type=int, default=110, help="mid-gray surroundings depth (restyle-context)")
    p.add_argument("--shade", action="store_true",
                   help="ENABLE depth clip shading + AO + contact shadow (OFF = exact out_zoom_artic, "
                        "clip-only, no tissue modification)")
    p.add_argument("--shade-ao", type=float, default=0.7, help="ambient-occlusion strength on the clip")
    p.add_argument("--spec-knock", type=float, default=0.4, help="knock down isolated bright specular blob")
    p.add_argument("--edge-strength", type=float, default=0.0, help="restore structural clip edges (0=off)")
    p.add_argument("--sharpen", type=float, default=0.5, help="clip micro-contrast / crispness")
    p.add_argument("--shadow-strength", type=float, default=0.4, help="cast contact-shadow darkness on tissue")
    p.add_argument("--clip-tone", action="store_true",
                   help="apply blue-matte clip tone (OFF = exact zoom_inpaint polished look)")
    p.add_argument("--clip-gain", type=float, default=0.85, help="clip brightness (<1 darker), needs --clip-tone")
    p.add_argument("--clip-gamma", type=float, default=1.2, help="highlight rolloff (>1 = less shiny), needs --clip-tone")
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--feather", type=int, default=3)
    p.add_argument("--steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--guidance", type=float, default=cfg.guidance_scale)
    p.add_argument("--seed", type=int, default=23)
    # prompt = TRIGGER (fixed prefix, activates the LoRA) + these extra property tokens.
    #p.add_argument("--extra", default=("brushed matte titanium clip, soft muted reflections, "
    #                                   "smooth rounded shape, blue-grey metal, glistening wet vessel, "
    #                                   "crisp focus, fine detail"),
    #               help="property tokens appended after the LoRA trigger")
    # match zoom_inpaint.py's exact prompt (TRIGGER + polished-titanium property tokens)
    p.add_argument("--prompt", default=(TRIGGER + ", polished blue titanium, "
                                        "sharp specular highlights, sharp focus"),
                   help="full prompt override (skips TRIGGER+extra)")
    p.add_argument("--neg-prompt", default=cfg.negative_prompt)
    return p.parse_args()


def main():
    import glob, time
    a = parse_args()
    if a.prompt is None:
        extra = getattr(a, "extra", None)          # --extra may be commented out -> trigger only
        a.prompt = TRIGGER + (", " + extra if extra else "")
    print(f"prompt: {a.prompt}")
    cfg = Config(); cfg.lora_path = a.lora; cfg.lora_scale = a.lora_scale; cfg.low_vram = True
    here = os.path.dirname(os.path.abspath(__file__))

    if a.data_dir:
        dd = a.data_dir if os.path.isabs(a.data_dir) else os.path.normpath(os.path.join(here, a.data_dir))
        scenes = sorted(glob.glob(os.path.join(dd, "scene_*")))
        if a.limit:
            scenes = scenes[:a.limit]
        out_root = a.out_dir or (dd.rstrip("/\\") + "_real")
    else:
        scenes = [a.scene if os.path.isabs(a.scene) else os.path.normpath(os.path.join(here, a.scene))]
        out_root = None

    print(f"diffuse-one-view + warp | {len(scenes)} scene(s) | lora {a.lora}")
    pipe = build_pipeline(cfg)
    t0 = time.time()
    for i, scene in enumerate(scenes):
        out_dir = (a.out_dir or (scene + "_real")) if out_root is None \
            else os.path.join(out_root, os.path.basename(scene))
        process_scene(pipe, scene, a, out_dir)
        print(f"[{i+1}/{len(scenes)}] {os.path.basename(scene)} -> {out_dir}/images "
              f"({(time.time()-t0)/(i+1):.0f}s/scene avg)")
    print(f"Done. {len(scenes)} scenes in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
