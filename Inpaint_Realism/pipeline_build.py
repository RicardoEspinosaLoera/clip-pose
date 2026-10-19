"""
pipeline_build.py — construct the ControlNet-depth inpainting pipeline.

Handles both the SDXL and SD1.5 backends and the 8 GB-VRAM memory strategy
(fp16, model-cpu-offload, VAE tiling).
"""

import torch

from config import Config


def build_pipeline(cfg: Config):
    preset = cfg.preset()
    dtype = torch.float16

    from diffusers import ControlNetModel
    controlnet = ControlNetModel.from_pretrained(preset.controlnet_model, torch_dtype=dtype)

    kwargs = dict(controlnet=controlnet, torch_dtype=dtype)

    if preset.is_sdxl:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL
        if preset.vae_model:
            kwargs["vae"] = AutoencoderKL.from_pretrained(preset.vae_model, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            preset.base_model, **kwargs)
    else:
        from diffusers import StableDiffusionControlNetInpaintPipeline
        kwargs["safety_checker"] = None
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            preset.base_model, **kwargs)

    # Optional appearance LoRA: load and fuse before offloading so the merged
    # weights are baked in. LoRA only affects attention layers, so an SDXL-base
    # LoRA fuses cleanly into the SDXL inpaint UNet.
    if cfg.lora_path:
        pipe.load_lora_weights(cfg.lora_path)
        pipe.fuse_lora(lora_scale=cfg.lora_scale)
        pipe.unload_lora_weights()
        print(f"Fused LoRA from {cfg.lora_path} (scale {cfg.lora_scale})")

    # Memory strategy for laptop-class 8 GB GPUs.
    if cfg.low_vram and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()           # stream modules to GPU on demand
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.set_progress_bar_config(disable=True)
    return pipe
