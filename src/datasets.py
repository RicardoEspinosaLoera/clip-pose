# src/datasets.py
import json, os
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from .camera import kaolin_cam_to_K, world_to_camera_from_vtk, quat_wxyz_to_R
import numpy as np

class TripletDataset(Dataset):
    """
    Dataset loader for Kaolin-style ground truth:
      - JSON has 'camera' (fx, fy, cx, cy)
      - JSON has 'clip.pose_se3.rotation' and 'translation_m' in camera coords
      - Images are (rgb, bg, mask) triplets
    """

    def __init__(self, root, train=True, transform=None, strict_tz=True):
        self.root = root
        self.items = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')])
        self.train = train
        self.transform = transform  # torchvision-style
        self.strict_tz = strict_tz

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        jpath = self.items[idx]
        stem = os.path.splitext(jpath)[0]
        ipath, bpath, mpath = stem + '.png', stem + '_bg.png', stem + '_mask.png'

        # --- Load images ---
        I_np  = imageio.imread(ipath)       # (H, W, 3)
        BG_np = imageio.imread(bpath)       # (H, W, 3)
        M_np  = imageio.imread(mpath)       # (H, W)

        H, W = I_np.shape[:2]

        # --- Load metadata ---
        with open(jpath, 'r') as f:
            meta = json.load(f)

        try:
            cam = meta['camera']
            clip_world = meta['clip']['pose_world']
            clip_se3 = meta['clip']['pose_se3']

            R_wc_np, t_wc_np = world_to_camera_from_vtk(cam["position"], cam["focal_point"], cam["view_up"])
            R_wc = torch.from_numpy(R_wc_np).float()
            t_wc = torch.from_numpy(t_wc_np).float()

            q = np.asarray(clip_world["quaternion_wxyz"], dtype=np.float32)
            R_ow = torch.from_numpy(quat_wxyz_to_R(q)).float()       # [3,3]
            t_ow = torch.tensor(clip_world["translation_m"], dtype=torch.float32)  # [3]

            # --- 3) Compose Object → Camera
            R_oc = torch.matmul(R_wc, R_ow)                        # [1,3,3]
            t_oc = torch.matmul(R_wc, t_ow) + t_wc      # [1,3]

            K = kaolin_cam_to_K(cam)


        except Exception as e:
            raise RuntimeError(f"[{os.path.basename(jpath)}] compose_camera_object failed: {e}")

        # --- To tensors ---
        I_t  = torch.from_numpy(I_np).permute(2, 0, 1).float() / 255.0
        BG_t = torch.from_numpy(BG_np).permute(2, 0, 1).float() / 255.0
        if M_np.ndim == 3:
            M_np = M_np[..., 0]
        M_t = torch.from_numpy((M_np > 0).astype('float32')).unsqueeze(0)

        # --- Optional augmentations ---
        if self.train and self.transform is not None:
            I_t = self.transform(I_t)
            BG_t = self.transform(BG_t)

        # --- Pack final sample ---
        sample = {
            'image': I_t,
            'bg': BG_t,
            'mask': M_t,
            'K': torch.from_numpy(K).float(),
            'R_co': R_oc,
            't_co': t_oc,
            'stem': os.path.basename(stem),
            #'cam': meta['camera']
        }

        return sample
