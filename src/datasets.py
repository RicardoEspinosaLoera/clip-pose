import json, os, glob, imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from .camera import compose_camera_object

class TripletDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.items = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')])
        self.train = train
        self.transform = transform  # torchvision-style or custom (expects tensors in [0,1])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        jpath = self.items[idx]
        stem = os.path.splitext(jpath)[0]
        ipath, bpath, mpath = stem + '.png', stem + '_bg.png', stem + '_mask.png'

        with open(jpath, 'r') as f:
            meta = json.load(f)
        K, R_co, t_co = compose_camera_object(meta['camera'], meta['clip'])

        # load -> torch float tensors in [0,1]
        I  = torch.from_numpy(imageio.imread(ipath)).permute(2, 0, 1).float() / 255.0
        BG = torch.from_numpy(imageio.imread(bpath)).permute(2, 0, 1).float() / 255.0
        M  = torch.from_numpy((imageio.imread(mpath) > 0).astype('float32')).unsqueeze(0)

        # ---- photometric aug (train only), label-safe ----
        if self.train and self.transform is not None:
            # if you might re-enable photo/sil losses later, augment I & BG consistently
            I  = self.transform(I)
            BG = self.transform(BG)
            # optional masked occluder on I (doesn't change labels)
            I  = I

        sample = {
            'image': I, 'bg': BG, 'mask': M,
            'K': torch.from_numpy(K).float(),
            'R_co': torch.from_numpy(R_co).float(),
            't_co': torch.from_numpy(t_co).float(),
            'stem': os.path.basename(stem)
        }
        return sample
