#!/usr/bin/env python3
"""
Visualize a single JSON + triplet (render, bg, mask).
- Overlays mask on render
- Draws reprojected mesh silhouette (using JSON pose + intrinsics)
"""
import argparse, json, imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.camera import compose_camera_object
from src.renderer import SoftMeshRenderer, project_pixels
from src.train import load_mesh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="Path stem without extension, e.g. data/train/094")
    ap.add_argument("--mesh", default="meshes/clip_mesh.obj")
    args = ap.parse_args()

    with open(args.stem + ".json") as f:
        meta = json.load(f)

    I = imageio.imread(args.stem + ".png")
    BG = imageio.imread(args.stem + "_bg.png")
    M = imageio.imread(args.stem + "_mask.png")

    K, R_co, t_co = compose_camera_object(meta["camera"], meta["clip"])

    verts, faces = load_mesh(args.mesh)
    renderer = SoftMeshRenderer(verts, faces, image_size=I.shape[0])
    with torch.no_grad():
        rgb, sil = renderer(torch.from_numpy(R_co)[None],
                            torch.from_numpy(t_co)[None],
                            torch.from_numpy(K)[None])
    I_rend = rgb.squeeze(0).permute(1,2,0).numpy()

    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(I); axs[0].set_title("Render PNG")
    axs[1].imshow(M, cmap="gray"); axs[1].set_title("Mask")
    axs[2].imshow(0.5*I.astype(np.float32)/255. + 0.5*I_rend); axs[2].set_title("Overlay")
    for ax in axs: ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
