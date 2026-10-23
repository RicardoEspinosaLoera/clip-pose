"""
make_bundle.py — pack everything the DGX needs into one tar (run on the Windows laptop).

Contents (relative to the bundle root RealAC_Generator/):
  GT_Generator/   scripts, meshes (Item.obj legacy; clip_lazic_std.obj + .parts.json = MACS L-Clip), keypoints*.json,
                  Data/noclips_bank (600 real backplates)
  pose_train/     training / evaluation code
  Inpaint_Realism/lora/clean_clips2 (+_masks)   real clip crops for the domain-gap metric
  real_eval/      selected real frames (+ annotations if present)
  dgx/            this folder
  models/hf       GroundingDINO tiny (HF cache layout)      models/torch  ImageNet ResNet-18/34
  --with-repaint  adds SDXL inpainting + VAE + canny ControlNet (~22 GB) and the MACS LoRA for variant A4
                  (prefer `huggingface-cli download` on the DGX when it has internet)
Usage: python dgx/make_bundle.py [--out C:/Users/ricar/realac_bundle] [--with-repaint] [--dry-run]
"""

import os
import sys
import time
import tarfile
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(HERE, ".."))
HF_HUB = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
TORCH_CK = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
ARC = "RealAC_Generator"


def skip(rel):
    base = os.path.basename(rel)
    if base == "__pycache__" or base.endswith((".pyc", ".lock", ".log")):
        return True
    if "GT_Generator" in rel.split(os.sep)[:1] and base.startswith("_"):      # debug images / scratch dirs
        return True
    if rel.startswith(os.path.join("GT_Generator", "out")):                    # generated sets (regenerated on the DGX)
        return True
    return False


def walk(src, arc_prefix, out, filt=None):
    n, size = 0, 0
    if os.path.isfile(src):
        out.append((src, arc_prefix))
        return 1, os.path.getsize(src)
    for dp, dns, fns in os.walk(src):
        rel = os.path.relpath(dp, PROJ) if dp.startswith(PROJ) else os.path.relpath(dp, os.path.dirname(src))
        dns[:] = [d for d in dns if not skip(os.path.join(rel, d))]
        for fn in fns:
            r = os.path.join(rel, fn)
            if skip(r) or (filt and not filt(fn)):
                continue
            p = os.path.join(dp, fn)
            out.append((p, os.path.join(arc_prefix, os.path.relpath(p, src)).replace("\\", "/")))
            n += 1
            size += os.path.getsize(p)
    return n, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "realac_bundle"))
    ap.add_argument("--with-repaint", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    groups = [
        ("GT_Generator (code+mesh)", os.path.join(PROJ, "GT_Generator"), f"{ARC}/GT_Generator",
         lambda fn: fn.endswith((".py", ".obj", ".json", ".md"))),
        ("GT_Generator/Data/noclips_bank", os.path.join(PROJ, "GT_Generator", "Data", "noclips_bank"), f"{ARC}/GT_Generator/Data/noclips_bank", None),
        ("pose_train", os.path.join(PROJ, "pose_train"), f"{ARC}/pose_train", lambda fn: fn.endswith(".py")),
        ("dgx", HERE, f"{ARC}/dgx", None),
        ("real clip crops", os.path.join(PROJ, "Inpaint_Realism", "lora", "clean_clips2"), f"{ARC}/Inpaint_Realism/lora/clean_clips2", None),
        ("real clip masks", os.path.join(PROJ, "Inpaint_Realism", "lora", "clean_clips2_masks"), f"{ARC}/Inpaint_Realism/lora/clean_clips2_masks", None),
        ("real_eval", os.path.join(PROJ, "real_eval"), f"{ARC}/real_eval", None),
        ("GroundingDINO (HF cache)", os.path.join(HF_HUB, "models--IDEA-Research--grounding-dino-tiny"), f"{ARC}/models/hf/hub/models--IDEA-Research--grounding-dino-tiny", None),
        ("ResNet-34 ImageNet", os.path.join(TORCH_CK, "resnet34-b627a593.pth"), f"{ARC}/models/torch/hub/checkpoints/resnet34-b627a593.pth", None),
        ("ResNet-18 ImageNet", os.path.join(TORCH_CK, "resnet18-f37072fd.pth"), f"{ARC}/models/torch/hub/checkpoints/resnet18-f37072fd.pth", None),
    ]
    if a.with_repaint:
        for repo in ("models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1", "models--madebyollin--sdxl-vae-fp16-fix",
                     "models--diffusers--controlnet-canny-sdxl-1.0"):
            groups.append((repo, os.path.join(HF_HUB, repo), f"{ARC}/models/hf/hub/{repo}", None))
        groups.append(("MACS LoRA", os.path.join(PROJ, "Inpaint_Realism", "lora", "macs_lora"), f"{ARC}/Inpaint_Realism/lora/macs_lora", None))
    # the generator-side walk filters GT_Generator by extension but must not descend into Data (handled as its own group)
    files, total = [], 0
    for label, src, arc, filt in groups:
        if not os.path.exists(src):
            print(f"  MISSING  {label}: {src}")
            continue
        before = len(files)
        if label.startswith("GT_Generator (code"):
            n, size = 0, 0
            for fn in os.listdir(src):
                p = os.path.join(src, fn)
                if os.path.isfile(p) and filt(fn) and not fn.startswith("_"):
                    files.append((p, f"{arc}/{fn}"))
                    n += 1
                    size += os.path.getsize(p)
        else:
            n, size = walk(src, arc, files, filt)
        total += size
        print(f"  {size / 1e6:9.1f} MB  {n:6d} files  {label}")
    print(f"  {total / 1e9:9.2f} GB  {len(files):6d} files  TOTAL")
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    tag = time.strftime("%Y%m%d")
    out = os.path.join(a.out, f"realac_dgx_bundle_{tag}{'_repaint' if a.with_repaint else ''}.tar")
    t0 = time.time()
    with tarfile.open(out, "w", dereference=True) as tar:
        for i, (p, arc) in enumerate(files):
            tar.add(p, arcname=arc, recursive=False)
            if i % 500 == 0:
                print(f"\r  {i}/{len(files)}", end="", flush=True)
    with open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(arc for _, arc in files))
    print(f"\n-> {out} ({os.path.getsize(out) / 1e9:.2f} GB, {time.time() - t0:.0f} s); manifest {os.path.join(a.out, 'MANIFEST.txt')}")
    print("copy with:  scp", out, "user@dgx:/raid/    then on the DGX:  tar xf", os.path.basename(out))


if __name__ == "__main__":
    main()
