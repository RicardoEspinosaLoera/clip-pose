# Plan — More realistic GT_Generator overlays (poses + scenes)

Goal: a **robust synthetic dataset for clip pose estimation AND depth prediction**. So the
overlays from [render_clip_poses_depth.py](GT_Generator/render_clip_poses_depth.py) must look
like real MACS footage across varied **poses** and **scenes**, AND ship dense, metric,
image‑consistent **depth GT** — all with valid 6‑DoF clip pose. Ordered by value; every phase
is independently shippable and testable.

**Dual‑goal consequence (decided):** a clip over a flat 2‑D backplate cannot give consistent
scene‑depth GT (the RGB has real tissue parallax the flat depth lacks). So **Phase E (3‑D
textured surface) is the core renderer, not optional** — it is the only path that yields clip
pose GT + dense metric depth GT + real occlusion at once. Phases A/B feed it (bank = textures,
wider poses unchanged); Phase C shrinks (occlusion becomes geometric).

---

## Phase 0 — Eval harness (½ day, do first)

We can't judge "more real" by eye on 5000 images. Build a tiny scorer so every later phase
is measurable.

- `GT_Generator/eval_overlays.py`: sample N overlays + N real MACS frames, report
  - pose spread: histograms of tilt, roll, model‑up dot, on‑screen area, distance.
  - scene spread: # unique backplates used, pHash cluster count.
  - realism proxy: FID/KID of overlays vs a real‑MACS reference set (torch‑fidelity, CPU ok
    on the 800×544 crops), plus a contact‑sheet PNG (`_eval_grid.png`) for eyeballing.
- Baseline the current `out_rand_depth` so all deltas are relative.

**Done when:** one command prints pose/scene stats + FID and writes a contact sheet.

---

## Phase A — Real scene bank (1 day) — biggest win

Today all 5000 samples draw from **30** backplates in `Data/noclips`
([BG_DIR](GT_Generator/render_clip_poses_depth.py#L51)); `augment_backplate` can only
jitter, not add scene content. Replace with 1–3k harvested **clip‑free** real frames.

- New `Inpaint_Realism/lora/build_backplate_bank.py` (reuses existing tooling):
  - diverse‑frame selection = `sample_macs_frames.py` logic (`--balanced` round‑robin across
    vids, pHash dedup, `--path-contains "\1\"` for label‑1).
  - **reject frames that already contain a clip/instrument** by running the
    `extract_clip_crops.detect_gdino` + `metal_gate` path and dropping any frame with a
    clip score ≥ ~0.35 (we want empty tissue to place OUR clip on).
  - drop too‑dark / blurry frames (mean‑V and Laplacian‑var gates).
  - keep native aspect (renderer resizes to 800×544 in `on_new_pose`), output → `Data/noclips_bank`.
- Point `BG_DIR` at `noclips_bank` (env‑overridable; keep 30‑set as fallback).
- Keep `augment_backplate` but soften it (it was compensating for 30 images) — less rotate/zoom,
  keep photometric jitter.

**Recipe (draft):**
`--path-contains "\1\" --balanced --stride 5 --dedup-hamming 12 --max 2000 --reject-clip-score 0.35`

**Validation:** Phase‑0 scene‑spread jumps from 30 → ~2000 unique; FID drops. Skim a contact
sheet for any leaked clips.

---

## Phase B — Wider, realistic pose envelope (1 day)

Poses are boxed by the 14 near‑top‑down `SAMPLES` plus hard caps
`MAX_TILT_DEG=15`, `MAX_ROLL_DEG=10`, `MIN_UP_DOT` (≤30° off world‑up)
([lines 67‑69](GT_Generator/render_clip_poses_depth.py#L67-L69)). Estimator won't generalize
past top‑down.

- Make the caps env‑overridable (`MAX_TILT_DEG`, `MAX_ROLL_DEG`, `MIN_UP_DEG`).
- Decouple orientation from the 14 references: sample clip orientation over a controlled
  camera‑facing hemisphere — full 360° roll, tilt up to ~45–60° for a configurable
  `STEEP_FRAC` of samples, near‑top‑down for the rest. Keep translation/scale from the data
  bounds but widen `PAD_T` and the distance range.
- Reuse the existing safety net unchanged:
  [sample_visible_pose_guaranteed](GT_Generator/render_clip_poses_depth.py#L282) still
  guarantees in‑frame + minimum on‑screen size after the wider sampling.
- Keep jaw articulation as‑is (already randomized).

**Validation:** Phase‑0 tilt/roll histograms broaden to target; visibility‑guarantee reject
rate stays low (tune `STEEP_FRAC`); GT re‑projection still lands on the clip (mask vs depth
silhouette agree).

---

## Phase C — Compositing realism (2–3 days, independent sub‑items)

The clip is a fully‑visible, hard‑edged, single‑material paste. Four independent upgrades:

1. **Partial occlusion** — after compositing, draw sampled foreground tissue/vessel/cotton
   over part of the clip (from the backplate's dark‑vessel / low‑V regions). Keep full 6‑DoF
   GT; update the **visible** `*_mask.png` to the un‑occluded pixels. Adds the occluded
   examples the set currently has zero of. Config: `OCCLUDE_PROB`, max coverage cap.
2. **Material randomization** — per sample, pick a clip family (blue‑anodized titanium /
   bright silver / muted titanium), jitter roughness/specular, optional thin wet/blood‑film
   specular streak. Extends the single fixed
   [CLIP_MAT](GT_Generator/render_clip_poses_depth.py#L383).
3. **Microscope lighting** — add an off‑axis specular hotspot + vignette instead of a single
   flat headlight ([line 79](GT_Generator/render_clip_poses_depth.py#L79) /
   [warm_light_color](GT_Generator/render_clip_poses_depth.py#L441)).
4. **Seam kill** — feather the silhouette, add slight boundary defocus, and white‑balance‑match
   the clip to the backplate to remove the hard paste edge (currently only
   [contact_shadow](GT_Generator/render_clip_poses_depth.py#L473) grounds it).

**Validation:** contact sheet + FID per sub‑item; confirm mask/GT consistency after occlusion.

---

## Phase D — Push through SDXL where it counts, decide scale (½ day + runs)

The improved overlay is a better **init** for
[zoom_inpaint.py](Inpaint_Realism/zoom_inpaint.py) (real realism, ~6 min/img on 8 GB).

- Re‑run a small set through `zoom_inpaint` to confirm the better init reduces drift / needs
  lower strength.
- Scale decision: bulk via improved overlay (+relight, fast, local) and SDXL on a subset, or
  cloud 24 GB GPU for full SDXL. (Open question below.)

---

## Phase E — Depth-useful GT: 3D textured tissue surface (2–3 days) — optional but high value

Today's depth GT is **true & metric only on the clip** (VTK z-buffer,
`NNN_depth_clip.npy`); tissue has no geometry. [estimate_bg_depth.py](GT_Generator/estimate_bg_depth.py)
fuses a Depth-Anything guess for tissue, but that pairing is a pseudo-label: the RGB
backplate is a real photographed 3-D scene while its paired depth is a flat plane + monocular
guess with a fusion seam → physically **inconsistent** image/depth pairs, which train a depth
net on partly-wrong cues.

Fix: replace the flat backplate (`add_background_image`) with a **displaced height-field mesh**
textured with the MACS frame, so VTK renders RGB **and** depth from the *same* geometry.

- One-time Depth-Anything pass per backplate → height field (seed only; the rendered depth is
  then exact w.r.t. the geometry that made the pixels).
- Build a subdivided plane mesh, displace along view axis by the height field, apply the frame
  as texture, render RGB + z-buffer together → dense metric depth for **clip + tissue**,
  mutually consistent, no estimator in the GT.
- Composite the clip in true 3-D: real occlusion / parallax / contact shadow become geometry
  (supersedes Phase C's fake 2-D occlusion).
- Emit `NNN_depth_scene.npy` (dense metric) + valid-everywhere mask; extend `eval_overlays.py`
  with a depth-consistency check.

**Tiers of depth supervision this unlocks:**
| Tier | Source | GT quality | Effort |
|------|--------|-----------|--------|
| 1 | clip z-buffer (exists) | true metric, clip only | 0 |
| 2 | Depth-Anything fused (exists) | pseudo, inconsistent pairs | 0 |
| 3 | **3-D textured surface (this phase)** | **true dense metric, consistent** | 2–3 d |

**Validation:** rendered depth reproject-consistent with RGB; clip sits on the surface (no
float/intersect); a monocular depth net overfits a tiny split to low error (sanity that pairs
are learnable).

## Suggested order & effort

| Phase | What | Effort | Answers |
|------|------|--------|---------|
| 0 | Eval harness | ½ d | measurable |
| A | Real scene bank | 1 d | **scenes** |
| B | Pose envelope | 1 d | **poses** |
| C | Compositing realism | 2–3 d | look |
| D | SDXL + scale | ½ d + runs | throughput |
| E | 3-D surface = true dense depth GT | 2–3 d | **depth prediction** |

**Revised order for the dual (pose+depth) goal:** 0 → A → **E** → B → C → D.
Phase E is now the critical path (depth GT); A feeds it textures and B/C layer onto it.
Do E early so pose (B) and realism (C) are built on the 3‑D renderer, not the 2‑D one.

**Progress (this session):**
- Phase 0 harness built + baseline captured (poses confirmed ≤15° tilt / ≤10° roll /
  up_dot≥0.87 = near‑top‑down; clip ~1% area; clips render too dark because PBR metal has no
  env map). See [eval_overlays.py](GT_Generator/eval_overlays.py).
- Phase A harvester built + validated on one video (clean clip‑free backplates, heuristic
  reject works). See [build_backplate_bank.py](Inpaint_Realism/lora/build_backplate_bank.py).
- **Phase E PROTOTYPE VALIDATED** — [render_scene3d_depth.py](GT_Generator/render_scene3d_depth.py):
  Depth‑Anything height field → back‑projected textured tissue surface → clip on top →
  **dense 100%‑coverage metric depth + realistic RGB + clip mask** in one render. Dual goal
  proven. Blemishes to fix in integration: (a) bilateral‑smooth the disparity to kill
  Depth‑Anything streak artifacts; (b) drop the clip onto the surface (raycast contact) so it
  rests instead of floating ~10u above.

- **Phase E BATCH GENERATOR built + smoke-tested (5 samples)** —
  [render_scene3d_batch.py](GT_Generator/render_scene3d_batch.py): reuses the flat path's pose
  sampler + GT writers, swaps in the 3-D surface, emits the backward-compatible superset
  (old files + `NNN_depth_scene.npy/.png`, 100% coverage). Output contract: keep-same+add-depth.
  Pilot size: 200.
  - Refinements identified from the 5-sample check: (a) **contact-drop** — raycast the surface
    under the clip + `plotter.enable_shadows()` so the clip rests & casts a real shadow instead
    of floating; (b) **clip brightness** — add a PBR environment map (metal renders dark in some
    orientations without one).

- **Phase A harvest DONE** — `Data/noclips_bank` = **953** clip-free real backplates across all
  16 videos (32× the old 30). Heuristic reject; loosen gates/stride for more.
- **TARGET REFOCUSED (user): multi-view 3-D reconstruction (NeRF / MVS / 3DGS).** Key scientific
  point: for multi-view, GT = the geometry that generated the images, which is EXACT here (every
  view renders the same known meshes; tissue depth per view IS the mesh, not an estimate).
  Cross-view consistency is exact; the caveats are realism (small-baseline regime, DA 2.5-D
  tissue shape, baked tissue shading), not GT correctness.
- **Phase E-MV BUILT + VALIDATED** — [render_scene3d_multiview.py](GT_Generator/render_scene3d_multiview.py):
  per scene, fixes tissue+clip in world and orbits K posed cameras (small baseline), world-fixed
  lights (clip speculars are view-dependent), emits NeRF/instant-ngp `transforms.json` +
  per-view exact metric depth (.npy) + masks. Smoke test 2 scenes × 8 views, 100% depth coverage.
  **Cross-view reprojection IoU 0.70–0.81** (view0 clip → other views vs their masks) ⇒ exported
  OpenGL c2w + intrinsics + depth are mutually consistent = reconstructable. Convention verified.

- **MV PILOT generated:** `out_mv` = **20 scenes × 16 views = 320 posed views**, all 100% depth
  coverage. NeRF-format, per-view exact metric depth + masks.
- **RECONSTRUCTION DEMONSTRATED (not just claimed):**
  [mv_fuse_check.py](GT_Generator/mv_fuse_check.py) unprojects all views' depth via the exported
  c2w+intrinsics into ONE world point cloud (`scene/fused.ply` + `fused_views.png`). 8 views fuse
  into a single coherent surface + 3-D clip (no smearing) ⇒ the data supports MVS/NeRF/3DGS.
  Known cosmetic artifact: the 0.65 overscan skirt shows as flat flaps in the *global* fused cloud
  (irrelevant to per-view training; crop to valid frame if a clean global cloud is wanted).

- **Phase B DONE (pose + baseline widening):** env-overridable pose envelope in
  render_clip_poses_depth.py (`POSE_PERTURB_MAX_DEG`/`POSE_STEEP_FRAC`/`POSE_STEEP_MAX_DEG`/
  `MIN_UP_DEG`/`MAX_TILT_DEG`); a random orientation perturbation is composed onto the sampled
  near-top-down reference pose. Flat path unchanged (defaults 0); the 3-D paths opt in via
  `os.environ.setdefault` (perturb 35°, 25% steep to 65°, MIN_UP_DEG 80). Multi-view orbit
  baseline now **per-scene** from [az 8–28°, el 6–22°]. Measured widening: clip tilt-from-vertical
  2–30° (mean 18°) → **10–69° (mean 38°)**, up_dot 0.87–1.0 → **0.36–0.99**; depth coverage still
  ~100%; poses look plausible (clip on tissue).
- **Wide-baseline coverage fix:** first Phase-B regen exposed edge holes (down to 88% depth
  coverage) on scenes whose orbit baseline hit az~26° — a single backplate has no content beyond
  its frame, so wide orbits open border holes. Fix: capped baseline to az[8,15]/el[6,13]
  (more microscope-realistic) + **adaptive over-scan** sized per scene from baseline vs half-FOV
  (`B.OVERSCAN = clip(1.25*tan(maxang+3)/tan(fov_half), 0.6, 1.1)`). `out_mv` regenerated clean.
  (Fundamental single-image limit; truly wide baselines need multi-frame content or CG tissue.)

- **Bank refined:** `refine_backplate_bank.py` quarantined 336 clip-containing frames (gdino) →
  `Data/noclips_bank` now **617 clean** (+ `_flagged_clip/` for review). Blue clips leaked past the
  achromatic harvest heuristic; blue-blob alone can't separate clips from blue vessel loops.
- **Camera–clip distance cap:** `sample_visible_pose_guaranteed` now also caps MAX apparent size
  (zoom out when too close) — env `MAX_AREA_FRAC`(0.10)/`MAX_DIAG_FRAC`(0.42) for 3-D paths, off for
  flat path. Max clip mask-area 4.3%→2.2%, median 1.3% unchanged (only too-close clips trimmed).

- **Contour/border artifact fixed:** orbited views showed streaky/dark border wedges because the
  over-scan used CLAMPED edge texels. Fix in `build_surface`: un-clamped texcoords +
  `vtkTexture.MirroredRepeat` wrap → off-frame over-scan shows a MIRROR of the interior tissue
  (base frame unaffected). Adaptive over-scan capped <1.0 for valid reflect.
- **LATENT INTRINSICS BUG FOUND + FIXED (important):** `intrinsics_from_vtk` returned
  `fx = fy*(w/h)`, but VTK renders SQUARE pixels (vertical view-angle + aspect widen the H-FOV,
  focal unchanged) → true `fx = fy`. The wrong fx was ~1.47× off and hidden by texcoord clamping;
  the reprojection-IoU/fusion checks only proved SELF-consistency, not agreement with the rendered
  pixels. Every previously-exported `fx` (flat `out_rand_depth` JSON + all 3-D `transforms.json`)
  is wrong and would distort a real NeRF/MVS/COLMAP fit on the actual images. Fixed at root
  (`fx = fy`). Old data: patch `fx:=fy` in JSONs (images/depth/pose are fine) OR regenerate.

**Remaining:** (0) regenerate `out_mv` (applies clean bank + size cap + border fix + fx fix)
→ (1) [optional] load a scene into real NeRF/3DGS/COLMAP (heavy install; fusion already confirms
consistency) →
(2) load one scene into a NeRF/3DGS or COLMAP/MVS to confirm end-to-end →
(3) Phase B pose/baseline widening → (4) scale + scientific-validation plan (SCARED / synthetic-GT
subset for bias). Realism follow-ups: PBR env-map (some clip orientations still dark),
view-dependent wet-tissue shading.

## Open questions
1. Regenerate all 5000 in place, or a fresh `out_rand_depth_v2` for A/B comparison? (I'd keep both.)
2. Backplate bank target size — 2000 enough, or push to 3–5k?
3. Pose realism target: do you have reference clip poses steeper than the current top‑down set,
   or should `STEEP_FRAC`/tilt ceiling be chosen from real MACS footage in Phase 0?
