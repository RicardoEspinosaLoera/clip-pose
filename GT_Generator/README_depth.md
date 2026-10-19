# Depth-aware GT generation

New version of the synthetic generator that adds a **clip depth pass** and a
**background depth estimate**, so the downstream diffusion inpainter can render a
realistic clip *and* a plausible clip–tissue contact while the 6-DoF pose ground
truth stays exactly valid.

## Files

| Script | Role |
|---|---|
| `render_clip_poses_depth.py` | Recreated generator. Same pose sampler & JSON as `render_clip_poses.py`, **plus** a metric clip depth pass and a derived normal map. Runs head-less (batch). |
| `estimate_bg_depth.py` | Post-process: monocular **Depth Anything V2** depth of the tissue backplate, **fused** with the metric clip depth into a ControlNet-ready depth map. |

## Per-sample outputs (`out_rand_depth/`)

Original (unchanged convention):
- `NNN.png`, `NNN_bg.png`, `NNN_mask.png`, `NNN.json`

Added by the generator:
- `NNN_depth_clip.npy` — float32 **metric** depth (world units), `NaN` outside the clip. This is exact render ground truth.
- `NNN_depth_clip.png` — 16-bit viz (closer = brighter, background = 0).
- `NNN_normal.png` — 8-bit camera-frame normal map (derived from the clip depth).

Added by the fusion step:
- `NNN_depth_bg.npy` / `NNN_depth_bg.png` — relative tissue depth (disparity, closer = brighter).
- `NNN_depth_fused.png` — **ControlNet-depth input**: tissue depth with the clip pasted at/just-in-front-of the local tissue level. Use this to condition the inpainter.
- `NNN_contact_mask.png` — dilated clip region = the free "contact" zone (locked clip + editable annulus for tissue deformation).

The `NNN.json` is extended with `image.file_depth_*` / `image.file_normal` and a
`depth` block recording units & convention.

## How to run

In the **same conda env** you used for the original generator (PyVista + VTK):
```powershell
python render_clip_poses_depth.py        # set N / paths in the CONFIG block
```

Then, in an env with `torch` + `transformers` (can be a different env):
```powershell
python estimate_bg_depth.py --dir out_rand_depth
# optional: --model depth-anything/Depth-Anything-V2-Large-hf   (best quality)
```

## Convention to verify on first run

PyVista's `get_image_depth()` returns camera-space z (negative in front of the
camera) with `NaN` for the backplate. The generator negates this to a positive
distance. Open one `NNN_depth_clip.png`: the clip should be **brighter where it is
closer to the camera** (e.g. the jaw tips nearest the lens). If your VTK build
reports the opposite sign, flip the negation in `render_clip_depth()` and the
inversion in `save_depth_artifacts()`.

## How this feeds the realism stage

`NNN_depth_fused.png` (+ Canny of the render) is the geometry condition for an
SDXL-inpaint + ControlNet pass restricted to `NNN_contact_mask.png`:
- inside the **clip silhouette** the depth/edge condition pins the 6-DoF pose;
- the **contact annulus** (ring around the clip) is free to repaint tissue
  indentation / vessel compression at the correct depth;
- everything outside the mask is copied from `NNN.png` unchanged, so the pose
  ground truth in `NNN.json` remains valid by construction.
