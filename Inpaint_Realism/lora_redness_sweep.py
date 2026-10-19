"""LoRA-scale sweep + redness comparison for the clip inpaint.

Goal: check whether the MASKED-loss LoRA (trained only on the SAM-segmented clip pixels)
removes the reddish tissue-colour contamination on the clip, vs the earlier clips-only LoRA.

It loads the SDXL ControlNet-inpaint pipeline ONCE, attaches every LoRA as a named PEFT
adapter, and for each (adapter, scale) runs the EXACT clip-only restyle used in production
(mv_warp_realism.restyle_v0 logic) on view v00 of one scene. For each result it measures a
redness score inside the clip silhouette and writes a labelled montage.

redness = mean over clip pixels of max(0, R - max(G,B)) / 255   (0 = neutral/blue metal,
higher = more tissue-red bleed). Lower is better.

Run AFTER training (needs the GPU free):
  python lora_redness_sweep.py
"""
import os, argparse, gc, numpy as np, cv2, torch
from PIL import Image
from config import Config
from pipeline_build import build_pipeline
from zoom_inpaint import clip_bbox, square_window, build_masks
from mv_warp_realism import depth_control_clip

HERE = os.path.dirname(os.path.abspath(__file__))

# adapter dir (relative to Inpaint_Realism) -> label. All these share ONE trigger, so the
# prompt is identical across them and only the learned weights differ (fair comparison).
SHARED_TRIGGER = "MACSCLIP, a titanium aneurysm clip, surgical microscope photo"
PROMPT_SUFFIX = ", polished blue titanium, sharp specular highlights, sharp focus"


def redness(bgr_crop, clip_mask):
    """Mean positive (R - max(G,B)) over the clip silhouette, normalised to [0,1].
    Also returns the fraction of clip pixels that read reddish (R noticeably > G and B)."""
    m = clip_mask > 127
    if m.sum() < 20:
        return 0.0, 0.0
    f = bgr_crop.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    red_excess = np.clip(r - np.maximum(g, b), 0, None)
    mean_red = float(red_excess[m].mean()) / 255.0
    frac_red = float((((r > g + 8) & (r > b + 8))[m]).mean())
    return mean_red, frac_red


def restyle_v0_scaled(pipe, scene, prompt, neg, strength, cn_scale, pad, work, seed,
                      dilate=2, feather=3):
    """restyle_v0's clip-only inpaint (verbatim logic) returning the styled CROP + mask_crop
    so redness is measured on the clip region only. LoRA scale is set on the pipe via
    set_adapters BEFORE calling this."""
    render = cv2.imread(os.path.join(scene, "images", "v00.png"))
    mask = cv2.imread(os.path.join(scene, "masks", "v00.png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(os.path.join(scene, "depth", "v00.npy"))
    H, W = render.shape[:2]
    x0, y0, side = square_window(clip_bbox(mask), pad, W, H)
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    crop, mask_crop, depth_crop = render[sl], mask[sl], depth[sl]
    crop_up = cv2.resize(crop, (work, work), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(mask_crop, (work, work), interpolation=cv2.INTER_NEAREST)
    hard, _ = build_masks(mask_up, dilate, feather)
    control = depth_control_clip(depth_crop, mask_crop, work)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    result = pipe(prompt=prompt, negative_prompt=neg,
                  image=Image.fromarray(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)),
                  mask_image=Image.fromarray(hard), control_image=control,
                  num_inference_steps=25, guidance_scale=7.0, strength=strength,
                  controlnet_conditioning_scale=cn_scale, control_guidance_end=0.9,
                  generator=gen, height=work, width=work).images[0]
    res_down = cv2.resize(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR), (side, side),
                          interpolation=cv2.INTER_AREA)
    res_down[mask_crop <= 127] = crop[mask_crop <= 127]           # keep only clip from SDXL
    return res_down, mask_crop, crop


def label(img, txt, sub=""):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(HERE, "..", "GT_Generator", "src_mv10new", "scene_0000"))
    ap.add_argument("--out", default=os.path.join(HERE, "lora_redness_sweep_out"))
    ap.add_argument("--scales", default="0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--cn", type=float, default=0.85)
    ap.add_argument("--pad", type=float, default=2.6)
    ap.add_argument("--work", type=int, default=768)
    ap.add_argument("--seed", type=int, default=23)
    # adapters: dir=label ; each is compared across the scale sweep
    ap.add_argument("--adapters", default="lora/lora_clipmasked1200=masked1200,lora/lora_out=clipsonly,lora/lora_clipmasked=masked800")
    ap.add_argument("--fused", action="store_true",
                    help="robust fallback: rebuild+fuse the pipeline per config (proven 8GB path) "
                         "instead of hot-swapping PEFT adapters on one loaded pipe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scales = [float(s) for s in a.scales.split(",")]
    adapters = []
    for tok in a.adapters.split(","):
        d, _, lab = tok.partition("=")
        adapters.append((d.strip(), (lab or d).strip()))
    adapters = [(d, l) for d, l in adapters if os.path.isdir(os.path.join(HERE, d))]
    prompt = SHARED_TRIGGER + PROMPT_SUFFIX
    neg = Config().negative_prompt

    rows = []; scores = {}; raw_ref = [None]; names = []

    def run_one(pipe, lab, s):
        styled, mask_crop, raw = restyle_v0_scaled(pipe, a.scene, prompt, neg, a.strength,
                                                    a.cn, a.pad, a.work, a.seed)
        mr, fr = redness(styled, mask_crop)
        scores[(lab, s)] = (mr, fr)
        if raw_ref[0] is None:
            raw_ref[0] = label(cv2.resize(raw, (256, 256)), "RAW render", "input")
        cv2.imwrite(os.path.join(a.out, f"{lab}_s{s}.png"), styled)
        print(f"  {lab} scale {s}: redness {mr:.4f}  frac_red {fr:.3f}", flush=True)
        return label(cv2.resize(styled, (256, 256)), f"{lab} @{s}", f"red {mr:.3f}  fr {fr:.2f}")

    if a.fused:
        # ---- ROBUST: one build+fuse per (adapter,scale). Slow (~90s reload each) but bulletproof.
        for d, lab in adapters:
            names.append(lab); cells = []
            for s in scales:
                cfg = Config(); cfg.lora_path = os.path.join(HERE, d); cfg.lora_scale = s; cfg.low_vram = True
                print(f"Building+fusing {lab} @ {s}...", flush=True)
                pipe = build_pipeline(cfg)
                cells.append(run_one(pipe, lab, s))
                del pipe; gc.collect(); torch.cuda.empty_cache()
            rows.append(np.hstack(cells))
    else:
        # ---- FAST: load all LoRAs as named adapters on ONE pipe, switch by set_adapters.
        cfg = Config(); cfg.lora_path = None; cfg.low_vram = True   # no fused LoRA baked in
        print("Building pipeline (no fused LoRA)...", flush=True)
        pipe = build_pipeline(cfg)
        for d, lab in adapters:
            pipe.load_lora_weights(os.path.join(HERE, d), adapter_name=lab)
            names.append(lab); print(f"  loaded adapter '{lab}' <- {d}", flush=True)
        for d, lab in adapters:
            cells = []
            for s in scales:
                pipe.set_adapters([lab], adapter_weights=[s])
                cells.append(run_one(pipe, lab, s))
            rows.append(np.hstack(cells))
    # pad raw_ref row to width
    W = rows[0].shape[1]
    ref_row = np.zeros((256, W, 3), np.uint8); ref_row[:, :256] = raw_ref[0]
    montage = np.vstack([ref_row] + rows)
    cv2.imwrite(os.path.join(a.out, "_MONTAGE.png"), montage)

    # summary table
    print("\n=== REDNESS SUMMARY (lower = less tissue-red bleed on clip) ===", flush=True)
    for lab in names:
        line = "  ".join(f"s{s}:{scores[(lab,s)][0]:.4f}" for s in scales)
        best = min(scales, key=lambda s: scores[(lab, s)][0])
        print(f"  {lab:12s} {line}   | min@{best}", flush=True)
    # head-to-head at each scale
    if "masked1200" in names and "clipsonly" in names:
        print("\n  masked1200 vs clipsonly (redness delta, negative = masked better):", flush=True)
        for s in scales:
            dlt = scores[("masked1200", s)][0] - scores[("clipsonly", s)][0]
            print(f"    @{s}: {dlt:+.4f}", flush=True)
    print(f"\nMontage -> {os.path.join(a.out, '_MONTAGE.png')}", flush=True)


if __name__ == "__main__":
    main()
