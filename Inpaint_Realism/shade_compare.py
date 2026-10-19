"""masked1200@0.8 (and clipsonly@0.8) — PLAIN inpaint vs the depth-based shading pass (--shade).

The SDXL generation is identical for both variants; shade_clip is a cheap CPU post-pass that
bakes depth-derived diffuse shading + ambient-occlusion + specular-knockdown + micro-sharpen
onto the clip (edge restoration off, matching mv_warp_realism's --shade defaults). So we
generate each config ONCE and emit plain + shaded, then measure redness and montage them.
"""
import os, numpy as np, cv2, torch
from PIL import Image
from config import Config
from pipeline_build import build_pipeline
from zoom_inpaint import clip_bbox, square_window, build_masks
from mv_warp_realism import depth_control_clip, shade_clip
from lora_redness_sweep import redness, label  # reuse metric + labeller

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "GT_Generator", "src_mv10new", "scene_0000")
OUT = os.path.join(HERE, "shade_compare_out"); os.makedirs(OUT, exist_ok=True)
PROMPT = "MACSCLIP, a titanium aneurysm clip, surgical microscope photo, polished blue titanium, sharp specular highlights, sharp focus"
WORK, PAD, STRENGTH, CN, SEED = 768, 2.6, 0.5, 0.85, 23
# --shade defaults from mv_warp_realism argparse: ao=0.7, spec_knock=0.4, edge=0.0, sharpen=0.5
SHADE = dict(ao=0.7, spec_knock=0.4, edge=0.0, sharpen=0.5)
CONFIGS = [("lora/lora_clipmasked1200", "masked1200", 0.8),
           ("lora/lora_out", "clipsonly", 0.8)]


def gen_crop(pipe, neg):
    render = cv2.imread(os.path.join(SCENE, "images", "v00.png"))
    mask = cv2.imread(os.path.join(SCENE, "masks", "v00.png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(os.path.join(SCENE, "depth", "v00.npy"))
    H, W = render.shape[:2]
    x0, y0, side = square_window(clip_bbox(mask), PAD, W, H)
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    crop, mask_crop, depth_crop = render[sl], mask[sl], depth[sl]
    crop_up = cv2.resize(crop, (WORK, WORK), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(mask_crop, (WORK, WORK), interpolation=cv2.INTER_NEAREST)
    hard, _ = build_masks(mask_up, 2, 3)
    control = depth_control_clip(depth_crop, mask_crop, WORK)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)
    result = pipe(prompt=PROMPT, negative_prompt=neg,
                  image=Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)),
                  mask_image=Image.fromarray(hard), control_image=control,
                  num_inference_steps=25, guidance_scale=7.0, strength=STRENGTH,
                  controlnet_conditioning_scale=CN, control_guidance_end=0.9,
                  generator=gen, height=WORK, width=WORK).images[0]
    styled = cv2.resize(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR), (side, side), interpolation=cv2.INTER_AREA)
    styled[mask_crop <= 127] = crop[mask_crop <= 127]
    return styled, mask_crop, depth_crop, crop


def main():
    cfg = Config(); cfg.lora_path = None; cfg.low_vram = True
    neg = cfg.negative_prompt
    print("Building pipeline...", flush=True)
    pipe = build_pipeline(cfg)
    for d, lab, _ in CONFIGS:
        pipe.load_lora_weights(os.path.join(HERE, d), adapter_name=lab)
        print(f"  loaded {lab}", flush=True)
    rows = []
    for d, lab, s in CONFIGS:
        pipe.set_adapters([lab], adapter_weights=[s])
        styled, mask_crop, depth_crop, raw = gen_crop(pipe, neg)
        shaded = shade_clip(styled, mask_crop, depth_crop, orig=raw, **SHADE)
        rp, _ = redness(styled, mask_crop)
        rs, _ = redness(shaded, mask_crop)
        cv2.imwrite(os.path.join(OUT, f"{lab}_plain.png"), styled)
        cv2.imwrite(os.path.join(OUT, f"{lab}_shaded.png"), shaded)
        print(f"  {lab}@{s}: plain red {rp:.4f} | shaded red {rs:.4f}", flush=True)
        cell_p = label(cv2.resize(styled, (320, 320)), f"{lab}@{s} PLAIN", f"red {rp:.3f}")
        cell_s = label(cv2.resize(shaded, (320, 320)), f"{lab}@{s} +SHADE", f"red {rs:.3f}")
        rows.append(np.hstack([cell_p, cell_s]))
    cv2.imwrite(os.path.join(OUT, "_SHADE_MONTAGE.png"), np.vstack(rows))
    print(f"Montage -> {os.path.join(OUT, '_SHADE_MONTAGE.png')}", flush=True)


if __name__ == "__main__":
    main()
