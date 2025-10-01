# src/datasets.py
import json, os
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from .camera import compose_camera_object, _rescale_K  # re-use rescale_K if you downsample later

def _to_3ch(img_np):
    # Ensures HxWx3 (handles grayscale / RGBA)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    if img_np.shape[-1] == 4:  # RGBA -> RGB
        img_np = img_np[..., :3]
    return img_np

class TripletDataset(Dataset):
    def __init__(self, root, train=True, transform=None, strict_tz=True):
        self.root = root
        self.items = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')])
        self.train = train
        self.transform = transform  # torchvision-style; expects CHW float in [0,1]
        self.strict_tz = strict_tz

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        jpath = self.items[idx]
        stem = os.path.splitext(jpath)[0]
        ipath, bpath, mpath = stem + '.png', stem + '_bg.png', stem + '_mask.png'

        # Load images first to get (H, W)
        I_np  = imageio.imread(ipath)            # H,W,3(/4)
        BG_np = imageio.imread(bpath)
        M_np  = imageio.imread(mpath)            # H,W (0/255 or 0/1)

        H, W = int(I_np.shape[0]), int(I_np.shape[1])

        with open(jpath, 'r') as f:
            meta = json.load(f)

        # Compose Kaolin-friendly GT from PyVista JSON (obj→cam, tz>0)
        try:
            K, R_co, t_co = compose_camera_object(meta['camera'], meta['clip'], H, W, strict=self.strict_tz)
        except Exception as e:
            # Attach filename to help debugging
            raise RuntimeError(f"[{os.path.basename(jpath)}] compose_camera_object failed: {e}")

        # Convert to tensors in [0,1]
        I_t  = torch.from_numpy(I_np).permute(2, 0, 1).float() / 255.0
        BG_t = torch.from_numpy(BG_np).permute(2, 0, 1).float() / 255.0
        if M_np.ndim == 3:
            M_np = M_np[..., 0]
        M_t  = torch.from_numpy((M_np > 0).astype('float32')).unsqueeze(0)

        # Optional photometric augmentation (train only), applied consistently to I & BG
        if self.train and self.transform is not None:
            I_t  = self.transform(I_t)
            BG_t = self.transform(BG_t)

        # Pack sample
        sample = {
            'image': I_t, 'bg': BG_t, 'mask': M_t,
            'K': torch.from_numpy(K).float(),         # (3,3)
            'R_co': torch.from_numpy(R_co).float(),   # (3,3)
            't_co': torch.from_numpy(t_co).float(),   # (3,)
            'stem': os.path.basename(stem),
        }
        return sample
