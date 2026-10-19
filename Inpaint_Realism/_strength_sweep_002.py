"""Inpaint ONE sample (002) at several denoise strengths (SDXL loaded once, fixed seed),
using the hole-protected --restyle-context path from zoom_inpaint. Montage the spring zoom."""
import os, numpy as np, cv2, torch
from PIL import Image
from config import Config
from pipeline_build import build_pipeline
from zoom_inpaint import clip_bbox, square_window, depth_control_context, context_alpha

DD = "../GT_Generator/out_pose100"; STEM = "002"; SEED = 24
WORK, PAD, CN, CEND = 768, 3.0, 0.85, 0.9
STRENGTHS = [0.20, 0.25, 0.30, 0.35, 0.40]
PROMPT = ("MACSCLIP, a titanium aneurysm clip, surgical microscope photo, polished blue "
          "titanium, sharp specular highlights, sharp focus")

cfg = Config(); cfg.lora_path = "lora/lora_out"; cfg.lora_scale = 0.8; cfg.low_vram = True
neg = cfg.negative_prompt
print("building pipeline (lora_out @0.8 fused)...", flush=True)
pipe = build_pipeline(cfg)

render = cv2.imread(f"{DD}/{STEM}.png"); mask = cv2.imread(f"{DD}/{STEM}_mask.png", 0)
depth = cv2.imread(f"{DD}/{STEM}_depth_clip.png", cv2.IMREAD_UNCHANGED)
if depth.ndim == 3:
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
H, W = render.shape[:2]
x0, y0, side = square_window(clip_bbox(mask), PAD, W, H)
sl = (slice(y0, y0 + side), slice(x0, x0 + side))
crop, mask_crop, depth_crop = render[sl], mask[sl], depth[sl]
crop_up = cv2.resize(crop, (WORK, WORK), interpolation=cv2.INTER_CUBIC)
control = depth_control_context(depth_crop, mask_crop, WORK, bg_value=110)
hard = Image.fromarray(np.full((WORK, WORK), 255, np.uint8))
init = Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB))
alpha = context_alpha(mask_crop, side)[..., None]     # hole-protected

# zoom window on the clip for the montage
ys, xs = np.where(mask > 127); pad = 14
zx0, zx1 = max(0, xs.min() - pad), xs.max() + pad; zy0, zy1 = max(0, ys.min() - pad), ys.max() + pad
def zoom(im, t):
    c = cv2.resize(im[zy0:zy1, zx0:zx1], (380, 380), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(c, (0, 0), (380, 22), (0, 0, 0), -1)
    cv2.putText(c, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); return c

cells = [zoom(render, f"{STEM} RENDER")]
for s in STRENGTHS:
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    res = pipe(prompt=PROMPT, negative_prompt=neg, image=init, mask_image=hard, control_image=control,
               num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale,
               strength=s, controlnet_conditioning_scale=CN, control_guidance_end=CEND,
               generator=gen, height=WORK, width=WORK).images[0]
    res_down = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (side, side), interpolation=cv2.INTER_AREA)
    blended = (alpha * res_down.astype(np.float32) + (1 - alpha) * crop.astype(np.float32)).astype(np.uint8)
    full = render.copy(); full[sl] = blended
    cv2.imwrite(f"_sw_{STEM}_s{int(s*100):02d}.png", full)
    cells.append(zoom(full, f"strength {s:.2f}"))
    print(f"  strength {s:.2f} done", flush=True)
cv2.imwrite("_002_strength_sweep.png", np.hstack(cells))
print("montage -> _002_strength_sweep.png", flush=True)
