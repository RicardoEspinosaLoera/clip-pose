"""
train_lora_sdxl.py — train an SDXL LoRA on real clip crops, tuned for 8 GB VRAM.

The LoRA teaches SDXL the *specific* appearance of a real titanium aneurysm clip
under a surgical microscope (MACS colour palette, specular reflections), which a
generic SDXL does not know. It is trained on the SDXL base text-to-image UNet and
loads cleanly into the SDXL *inpaint* pipeline used by `run_inpaint.py`, because
LoRA only touches the attention layers, which are identical in both UNets.

Low-VRAM strategy:
  * Pre-encode every crop to a VAE latent and pre-compute the (single, shared)
    text embeddings ONCE, then free the VAE and both text encoders from memory.
  * Only the bf16 UNet + small fp32 LoRA params remain on the GPU.
  * Gradient checkpointing keeps activation memory low.

Usage:
  python train_lora_sdxl.py --crops crops --out lora_out --steps 1200 \
      --instance-prompt "MACSCLIP, a titanium aneurysm clip, surgical microscope photo"
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"


def list_crops(crops_dir):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(crops_dir, ext)))
    if not paths:
        raise RuntimeError(f"No crops found in {crops_dir}")
    return sorted(paths)


def load_image(path, size):
    """Load ONE image as a (1,3,H,W) [-1,1] tensor (keeps RAM minimal for big sets)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = (np.asarray(im).astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]   # (1,3,H,W)


def encode_prompt(prompt, tokenizers, text_encoders, device):
    """SDXL dual-encoder prompt embedding. Returns (prompt_embeds, pooled)."""
    embeds_list, pooled = [], None
    for tok, te in zip(tokenizers, text_encoders):
        ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        out = te(ids, output_hidden_states=True)
        pooled = out[0]                       # pooled from the LAST (te2) encoder
        embeds_list.append(out.hidden_states[-2])
    return torch.concat(embeds_list, dim=-1), pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--masks", default="",
                    help="dir of per-crop clip masks (same stem). If set, the diffusion loss is "
                         "computed ONLY on the clip pixels -> LoRA learns the clip, not the tissue.")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--instance-prompt",
                    default="MACSCLIP, a titanium aneurysm clip, surgical microscope photo")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable gradient checkpointing (~2x faster; fits at 512px on 8GB)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    crops_dir = args.crops if os.path.isabs(args.crops) else os.path.join(here, args.crops)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print("Listing crops...")
    paths = list_crops(crops_dir)
    print(f"{len(paths)} crops at {args.size}px")

    # ---- 1) pre-encode latents one image at a time (then free VAE) ----
    # Latents are cached to disk so a re-run (e.g. after a crash) skips the slow
    # ~30-40 min VAE pass entirely. Cache is keyed by crops dir + size + count.
    lat_cache = f"{crops_dir.rstrip(os.sep)}_latents_{args.size}.pt"
    latents = None
    if os.path.exists(lat_cache):
        try:
            cached = torch.load(lat_cache)
            if cached.shape[0] == len(paths):
                latents = cached.to(torch.bfloat16)
                print(f"Loaded cached latents {tuple(latents.shape)} from {lat_cache}")
            else:
                print(f"Cache count {cached.shape[0]} != {len(paths)} crops; re-encoding.")
        except Exception as e:
            print(f"Could not load latent cache ({e}); re-encoding.")
    if latents is None:
        print("Encoding latents...")
        vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=torch.float32).to(device).eval()
        buf = []
        with torch.no_grad():
            for n_i, p in enumerate(paths):
                x = load_image(p, args.size).to(device, torch.float32)
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                buf.append(lat.cpu())
                if (n_i + 1) % 200 == 0:
                    print(f"  encoded {n_i + 1}/{len(paths)}")
        latents = torch.cat(buf).to(torch.bfloat16)         # (N,4,h,w) on CPU
        del vae
        torch.cuda.empty_cache()
        try:
            torch.save(latents, lat_cache)
            print(f"Cached latents to {lat_cache}")
        except Exception as e:
            print(f"Warning: could not cache latents ({e})")

    # ---- 1b) mask latents: downsample each clip mask to latent res for the MASKED loss ----
    mask_lat = None
    if args.masks:
        masks_dir = args.masks if os.path.isabs(args.masks) else os.path.join(here, args.masks)
        h = args.size // 8
        mlat_cache = f"{crops_dir.rstrip(os.sep)}_maskslat_{args.size}.pt"
        if os.path.exists(mlat_cache):
            c = torch.load(mlat_cache)
            if c.shape[0] == len(paths):
                mask_lat = c
                print(f"Loaded cached mask latents {tuple(mask_lat.shape)}")
        if mask_lat is None:
            print("Building mask latents...")
            buf = []
            for p in paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                mp = os.path.join(masks_dir, stem + ".png")
                m = None
                if os.path.exists(mp):
                    mm = Image.open(mp).convert("L").resize((h, h), Image.BILINEAR)
                    m = (np.asarray(mm) > 100).astype(np.float32)
                if m is None or m.sum() < 1.0:
                    m = np.ones((h, h), np.float32)          # empty/missing -> full supervision
                buf.append(torch.from_numpy(m)[None])
            mask_lat = torch.stack(buf)                       # (N,1,h,w)
            torch.save(mask_lat, mlat_cache)
        nz = (mask_lat.flatten(1).sum(1) > 0.9 * (args.size // 8) ** 2).sum().item()
        print(f"mask latents {tuple(mask_lat.shape)}  ({len(paths)-nz} masked / {nz} full-supervision)")

    # ---- 2) pre-compute the shared text embeddings (then free text encoders) ----
    print("Encoding prompt...")
    tok1 = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer", use_fast=False)
    tok2 = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer_2", use_fast=False)
    te1 = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(BASE, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        prompt_embeds, pooled = encode_prompt(args.instance_prompt, [tok1, tok2], [te1, te2], device)
    prompt_embeds = prompt_embeds.to(torch.bfloat16)
    pooled = pooled.to(torch.bfloat16)
    del te1, te2, tok1, tok2
    torch.cuda.empty_cache()

    # ---- 3) UNet + LoRA ----
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=torch.bfloat16).to(device)
    unet.requires_grad_(False)
    if not args.no_checkpoint:
        unet.enable_gradient_checkpointing()
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank,
                      init_lora_weights="gaussian",
                      target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet.add_adapter(lora)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()                              # train LoRA in fp32
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    noise_sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    # constant SDXL conditioning for square crops
    add_time_ids = torch.tensor([[args.size, args.size, 0, 0, args.size, args.size]],
                                dtype=torch.bfloat16, device=device)

    N = latents.shape[0]
    print("Training...")
    step, accum = 0, 0
    opt.zero_grad()
    while step < args.steps:
        idx = torch.randint(0, N, (args.batch,))
        lat = latents[idx].to(device, torch.bfloat16)
        noise = torch.randn_like(lat)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (args.batch,), device=device).long()
        noisy = noise_sched.add_noise(lat, noise, t)

        b = args.batch
        added = {"text_embeds": pooled.repeat(b, 1),
                 "time_ids": add_time_ids.repeat(b, 1)}
        # autocast lets the fp32 LoRA params run with the bf16 base UNet without a
        # dtype clash; gradients still accumulate in fp32 on the LoRA weights.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(noisy, t, encoder_hidden_states=prompt_embeds.repeat(b, 1, 1),
                        added_cond_kwargs=added).sample
        if mask_lat is not None:
            # MASKED diffusion loss: supervise only the clip pixels (mask=1) so the LoRA learns
            # the clip metal, not the surrounding tissue (removes the reddish-tissue contamination).
            m = mask_lat[idx].to(device).float()                 # (b,1,h,w)
            per = F.mse_loss(pred.float(), noise.float(), reduction="none")   # (b,4,h,w)
            loss = (per * m).sum() / (m.sum() * per.shape[1] + 1e-8) / args.grad_accum
        else:
            loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum
        loss.backward()
        accum += 1

        if accum == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            opt.zero_grad()
            accum = 0
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"step {step}/{args.steps}  loss {loss.item()*args.grad_accum:.4f}")

    # ---- 4) save LoRA (loadable via pipe.load_lora_weights) ----
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(out_dir, unet_lora_layers=unet_lora)
    with open(os.path.join(out_dir, "trigger.txt"), "w") as f:
        f.write(args.instance_prompt + "\n")
    print(f"Saved LoRA to {out_dir} (pytorch_lora_weights.safetensors)")
    print(f"Trigger phrase: {args.instance_prompt}")


if __name__ == "__main__":
    main()
