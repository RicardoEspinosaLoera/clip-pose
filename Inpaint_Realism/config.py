"""
config.py — configuration for the masked ControlNet-depth inpainting stage.

Two backends are supported:

  sdxl  (realism-first)  : SDXL inpaint UNet + SDXL depth ControlNet.
                           Best fidelity, ~10-20 s/image on an 8 GB laptop GPU
                           (requires model-cpu-offload), ~9 GB of weights.
  sd15  (throughput)     : SD1.5 inpaint + SD1.5 depth ControlNet.
                           Lighter (~4 GB), ~2-5 s/image, fits 8 GB comfortably.
                           Lower fidelity than SDXL but much faster for bulk runs.

Everything outside the editable (contact) mask is hard-recomposited from the
original render, so the 6-DoF pose ground truth in the JSON stays valid by
construction; the clip geometry inside the mask is pinned by the depth condition.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackendPreset:
    name: str
    base_model: str
    controlnet_model: str
    vae_model: Optional[str]          # fp16-safe VAE for SDXL; None for SD1.5
    pipeline_cls: str                 # diffusers class name
    is_sdxl: bool


PRESETS = {
    "sdxl": BackendPreset(
        name="sdxl",
        base_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_model="diffusers/controlnet-depth-sdxl-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        pipeline_cls="StableDiffusionXLControlNetInpaintPipeline",
        is_sdxl=True,
    ),
    "sd15": BackendPreset(
        name="sd15",
        base_model="runwayml/stable-diffusion-inpainting",
        controlnet_model="lllyasviel/control_v11f1p_sd15_depth",
        vae_model=None,
        pipeline_cls="StableDiffusionControlNetInpaintPipeline",
        is_sdxl=False,
    ),
}


@dataclass
class Config:
    # --- backend ---
    backend: str = "sdxl"

    # --- optional appearance LoRA (trained by lora/train_lora_sdxl.py) ---
    lora_path: Optional[str] = None       # dir or .safetensors with the clip LoRA
    lora_scale: float = 0.8               # blend strength of the LoRA
    lora_trigger: str = ""                # phrase prepended to the prompt (e.g. "MACSCLIP")

    # --- prompts ---
    # The editable mask IS the clip, so the prompt must describe ONLY the clip
    # (metal), not the surrounding tissue/blood — otherwise the model paints tissue
    # over the clip. Tissue realism comes from the preserved background, not here.
    prompt: str = (
        "a polished titanium aneurysm clip, smooth metallic surface, bright sharp "
        "specular highlights, brushed stainless steel, surgical microscope close-up, "
        "sharp focus, high detail, photorealistic"
    )
    negative_prompt: str = (
        "blood, bloody, red, tissue covering the clip, organic texture on metal, flesh, "
        "rust, corrosion, gauze, cartoon, illustration, drawing, cgi, plastic, matte, "
        "blurry, low quality, deformed clip, duplicated clip, extra clip, text, watermark"
    )

    # --- diffusion params ---
    num_inference_steps: int = 30
    guidance_scale: float = 6.5
    strength: float = 1.0                 # full regeneration inside the mask
    controlnet_conditioning_scale: float = 0.85   # how hard the depth pins geometry
    control_guidance_start: float = 0.0
    control_guidance_end: float = 0.9
    seed: int = 1234

    # --- editable region (contact mask) ---
    # If a *_contact_mask.png exists it is used directly; otherwise the binary
    # clip mask is dilated by `dilate_px` to form the locked-clip + contact ring.
    dilate_px: int = 21
    feather_px: int = 9                   # gaussian feather for seamless blending

    # --- control signal ---
    # priority order of geometry conditions to look for, first found wins
    control_sources: tuple = ("file_depth_fused", "file_depth_clip", "file_normal")

    # --- io ---
    data_dir: str = "../GT_Generator/out_rand_depth"
    out_dir: str = "out_real"
    out_suffix: str = "_real"
    save_side_by_side: bool = True        # also save an original|control|result strip

    # --- runtime ---
    limit: Optional[int] = None           # cap number of samples
    start_index: int = 0                  # resume from this sample index
    verify_shift: bool = True             # measure clip displacement after inpaint
    low_vram: bool = True                 # enable model-cpu-offload + vae tiling

    def preset(self) -> BackendPreset:
        return PRESETS[self.backend]
