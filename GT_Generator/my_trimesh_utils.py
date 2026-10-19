import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import math

def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + z*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

# Load R from JSON
meta = json.load(open('001.json'))
q = np.array(meta['clip']['pose_se3']['quaternion_wxyz'], dtype=np.float32)
R_co = _quat_wxyz_to_R(q)
R_gt = torch.tensor(R_co)

R_effective = R_gt.clone()   # JSON rotation
# no extra rotation matrix now
R_delta = R.from_matrix((R_effective.T @ R_gt).numpy())
print("Rotation difference (degrees):", R_delta.magnitude()*180/np.pi)
