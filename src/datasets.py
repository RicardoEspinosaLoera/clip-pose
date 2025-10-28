# src/datasets.py
import json, os
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from .camera import kaolin_cam_to_K, world_to_camera_from_vtk, quat_wxyz_to_R
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, Union

def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _from_uint8(img_u8):
    return (img_u8.astype(np.float32)) / 255.0

def scale_K(K, src_size, dst_size):
    
    H, W   = src_size
    H2, W2 = dst_size
    if (H != H2 and W != W2):
        sx = W2 / W
        sy = H2 / H
        K_out = K.copy()
        if K_out.ndim == 2:  # [3,3]
            K_out[0,0] *= sx  # fx
            K_out[1,1] *= sy  # fy
            K_out[0,2] *= sx  # cx
            K_out[1,2] *= sy  # cy
            K_out[0,1] *= sx  # skew (u scales with width)
        else:                 # [B,3,3]
            K_out[:,0,0] *= sx
            K_out[:,1,1] *= sy
            K_out[:,0,2] *= sx
            K_out[:,1,2] *= sy
            K_out[:,0,1] *= sx
    else: 
        K_out = K
    return K_out

class TripletDataset(Dataset):
    """
    Kaolin-style GT:
      - JSON 'camera' (fx,fy,cx,cy + VTK extrinsics fields)
      - JSON clip.pose_* (world pose); we compose to camera here
      - Files: stem.png, stem_bg.png, stem_mask.png

    New:
      - downsample: integer factor or explicit out_size=(Hs,Ws)
      - normalize_from_backbone: callable(img[C,H,W]->img_norm) e.g., from timm cfg
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,                      # optional extra transform after resizing (expects [C,H,W] float [0,1])
        strict_tz: bool = True,                # kept for compatibility (unused)
        downsample: Union[int, Tuple[int,int]] = 1,
        out_size: Optional[Tuple[int,int]] = None,
        normalize_from_backbone = None,        # callable built via build_backbone_transform(backbone)
        return_d_obj: bool = False,            # if your JSON has object diameter in meters
        d_obj_json_path: Tuple[str,...] = ("object", "diameter_m"),
    ):
        self.root = root
        self.items = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')])
        self.train = train
        self.transform = transform
        self.strict_tz = strict_tz

        self.normalize_from_backbone = normalize_from_backbone
        self.return_d_obj = return_d_obj
        self.d_obj_json_path = d_obj_json_path

        self.downsample = downsample
        self.out_size = out_size

    def __len__(self):
        return len(self.items)

    def _decide_size(self, H, W):
        if self.out_size is not None:
            Hs, Ws = self.out_size
        elif isinstance(self.downsample, int) and self.downsample > 1:
            Hs, Ws = int(H / self.downsample), int(W / self.downsample)
        else:
            Hs, Ws = H, W
        return Hs, Ws

    def __getitem__(self, idx):
        jpath = self.items[idx]
        stem = os.path.splitext(jpath)[0]
        ipath, bpath, mpath = stem + '.png', stem + '_bg.png', stem + '_mask.png'

        # --- Load images (RGB uint8) ---
        I_np  = imageio.imread(ipath)       # (H,W,3)
        BG_np = imageio.imread(bpath)       # (H,W,3)
        M_np  = imageio.imread(mpath)       # (H,W) or (H,W,1/3)

        H, W = I_np.shape[:2]
        Hs, Ws = self._decide_size(H, W)
        #Hs, Ws = (224,224)

        # --- Resize ---
        # cv2.resize expects (width, height)
        I_use  = cv2.resize(I_np,  (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        BG_use = cv2.resize(BG_np, (Ws, Hs), interpolation=cv2.INTER_LINEAR)

        if M_np.ndim == 3:
            M_np = M_np[..., 0]
        M_use  = cv2.resize(M_np,  (Ws, Hs), interpolation=cv2.INTER_NEAREST)

        # --- Load metadata / compose O->C ---
        with open(jpath, 'r') as f:
            meta = json.load(f)

        try:
            cam = meta['camera']
            clip_world = meta['clip']['pose_world']
            clip_se3 = meta['clip']['pose_se3']

            R_wc, t_wc = world_to_camera_from_vtk(cam["position"], cam["focal_point"], cam["view_up"])

            q = np.asarray(clip_world["quaternion_wxyz"], dtype=np.float32)
            R_ow = torch.from_numpy(quat_wxyz_to_R(q)).float()       # [3,3]
            t_ow = torch.tensor(clip_world["translation_m"], dtype=torch.float32)  # [3]

            # --- 3) Compose Object → Camera
            R_oc = torch.matmul(R_wc, R_ow)                        # [1,3,3]
            t_oc = torch.matmul(R_wc, t_ow) + t_wc      # [1,3]

            K = kaolin_cam_to_K(cam, image_size=(W, H))
            K_use = scale_K(K, (H, W), (Hs, Ws))
        except Exception as e:
            raise RuntimeError(f"[{os.path.basename(jpath)}] compose_camera_object failed: {e}")

        # --- To tensors float[0,1] ---
        I_t  = torch.from_numpy(I_use).permute(2,0,1).float() / 255.0
        BG_t = torch.from_numpy(BG_use).permute(2,0,1).float() / 255.0
        M_t  = torch.from_numpy((M_use > 0).astype('float32')).unsqueeze(0)

        # --- Optional user transforms (e.g., color jitter) BEFORE normalization ---
        if self.train and self.transform is not None:
            I_t  = self.transform(I_t)
            BG_t = self.transform(BG_t)

        # --- Backbone-specific normalization / final resize ---
        if self.normalize_from_backbone is not None:
            I_t  = self.normalize_from_backbone(I_t)
            BG_t = self.normalize_from_backbone(BG_t)

        sample = {
            'image': I_t,              # [3,h,w] (possibly resized & normalized)
            'bg': BG_t,                # same shape / norm as image
            'mask': M_t,               # [1,h,w], binary float
            'K': torch.from_numpy(K_use).float(),  # [3,3]
            'R_co': R_oc,              # torch [3,3]
            't_co': t_oc,              # torch [3]
            'stem': os.path.basename(stem),
        }


        return sample
