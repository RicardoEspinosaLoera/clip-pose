# Clip appearance LoRA (optional, recommended)

A generic SDXL knows "metal clip" but not the *specific* look of a real titanium
aneurysm clip under a MACS surgical microscope. This small LoRA teaches it that
appearance, so the inpainted clips match real intraoperative footage — which
directly improves how well the downstream pose model transfers to real data.

You only need **~50–200 good real clip crops**, not the whole MACS dataset.

## 1. Extract clip crops from real frames
```powershell
conda activate clip_pose
cd Inpaint_Realism\lora
# default: semantic GroundingDINO detector (~700 MB download on first run)
python extract_clip_crops.py --frames <dir_of_real_MACS_frames> --out crops --size 768 --debug
```
Two detectors are available:
- `--detector gdino` (default): zero-shot **open-vocabulary** detection — localises
  the clip by meaning via `--text-prompt "a surgical metal clip. an aneurysm clip."`.
  Robust to white gauze, specular tissue, and reflective clips that defeat colour
  rules. Tune `--box-threshold` / `--text-threshold` if it over/under-detects.
- `--detector heuristic`: colour-only (achromatic chroma + morphology), no download;
  a fast fallback, less reliable.

Both write the confidence into each filename (`clip_NNNN_sSS.png`), drop near-
duplicate consecutive frames via perceptual hash, keep the top `--topk` regions per
frame, and record `crops.jsonl`. `--debug` saves boxed frames to `crops/_debug/`.

Detection is good but not perfect (clip vs. clip-applier/forceps can blur) — **sort
by the score in the filename, skim the lowest, and delete any non-clip crop** before
training. Quality > quantity; ~50–200 clean crops is plenty.

## 2. Train the LoRA (tuned for 8 GB VRAM)
```powershell
python train_lora_sdxl.py --crops crops --out lora_out --steps 1200 ^
    --instance-prompt "MACSCLIP, a titanium aneurysm clip, surgical microscope photo"
```
- Caches latents + text embeddings, then frees the VAE/text encoders, so only the
  bf16 UNet + fp32 LoRA stay on the GPU. ~768px fits 8 GB; drop to `--size 512` if
  you hit OOM.
- Output: `lora_out/pytorch_lora_weights.safetensors` (+ `trigger.txt`).
- "MACSCLIP" is your **trigger phrase** — include it in the inpainting prompt.

## 3. Use it in the inpainting run
```powershell
cd ..
python run_inpaint.py --backend sdxl ^
    --lora lora\lora_out ^
    --lora-scale 0.8 ^
    --lora-trigger MACSCLIP
```
The LoRA is loaded and **fused** into the SDXL inpaint UNet (LoRA only touches
attention layers, which are identical between the base and inpaint UNets, so this
is safe). The trigger phrase is auto-prepended to the prompt.

## Tuning
- `--lora-scale` 0.6–1.0: higher = stronger clip-appearance match (too high can
  over-stylise). Start at 0.8.
- If the clip looks over-saturated/burned, lower the scale or retrain with more
  varied crops.
- Pose is unaffected: the mask-composite + depth ControlNet still lock the clip;
  the LoRA only changes *appearance* inside the editable region.

## Note
The LoRA is SDXL-specific. It will not load into the `sd15` backend (train a
separate SD1.5 LoRA if you switch backends).
