"""
dataset.py — synthetic clip-pose dataset for the keypoint (M1) and direct-regression (M2) models.

Each sample (from GT_Generator/render_clip_poses_depth.py) gives: NNN.png, NNN_mask.png,
NNN_mask_visible.png (tool-aware), NNN.json (pose_se3 in the OpenCV camera frame, intrinsics,
jaw_angle_deg, tool_held, scene.backplate). Keypoints are vertex indices of Item.obj
(GT_Generator/keypoints.json) articulated with the sample's jaw angle, projected with the GT.

A sample is a square crop around the clip bbox (margin x1.5, jitter in training) resized to
`crop` px; intrinsics are transformed accordingly (K_crop = A @ K) so PnP on crop-space
keypoints returns the pose in the ORIGINAL camera frame.
"""

import os
import json
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as scipyR

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.normpath(os.path.join(HERE, "..", "GT_Generator"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ClipModel:
    """Item.obj + keypoint indices, articulated per jaw angle (cached in 0.5 deg bins)."""

    def __init__(self):
        import sys
        import pyvista as pv
        sys.path.insert(0, GT_DIR)
        from articulate_clip import articulate, pca_frame
        self._articulate = articulate
        self.mesh = pv.read(os.path.join(GT_DIR, "Item.obj"))
        self.V0 = np.asarray(self.mesh.points, np.float64)
        kp = json.load(open(os.path.join(GT_DIR, "keypoints.json")))
        self.names = kp["names"]
        self.idx = np.asarray(kp["indices"], int)
        self.sym_pairs = kp.get("symmetric_pairs", [])
        c, e = pca_frame(self.V0)
        self.centre, self.axes = c, e                  # e[:,0] long axis
        self._cache = {}

    def verts(self, jaw):
        key = round(float(jaw) * 2) / 2.0
        if key not in self._cache:
            m = self._articulate(self.mesh, key, hollow=False) if key > 0 else self.mesh
            self._cache[key] = np.asarray(m.points, np.float64)
        return self._cache[key]

    def keypoints3d(self, jaw):
        return self.verts(jaw)[self.idx]

    def symmetry_rotation(self):
        """180 deg about the long axis: swaps the two blades (the clip's near-symmetry)."""
        a = self.axes[:, 0]
        return scipyR.from_rotvec(np.pi * a).as_matrix()


def pose_from_json(meta):
    q = meta["clip"]["pose_se3"]["quaternion_wxyz"]
    R = scipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(meta["clip"]["pose_se3"]["translation_m"], np.float64)
    c = meta["camera"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
    return R, t, K


def project(P, R, t, K):
    X = (R @ P.T).T + t
    u = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]
    v = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
    return np.stack([u, v], 1), X[:, 2]


def read_split(path):
    return [l.strip() for l in open(path) if l.strip()]


class ClipPoseDataset(Dataset):
    def __init__(self, data_dir, stems, clip_model, crop=512, heat=128, sigma=2.0, train=True,
                 margin=1.5, jitter=0.10, scale_jitter=0.10, photometric=True, image_suffix=""):
        self.d = data_dir
        self.image_suffix = image_suffix                 # "_flat" = ablation A0 (flat render, no applier)
        self.stems = list(stems)
        self.cm = clip_model
        self.crop, self.heat, self.sigma = crop, heat, sigma
        self.train, self.margin, self.jitter, self.scale_jitter = train, margin, jitter, scale_jitter
        self.photometric = photometric and train
        self.K = len(clip_model.idx)

    def __len__(self):
        return len(self.stems)

    def _window(self, mask, rng):
        ys, xs = np.nonzero(mask > 127)
        H, W = mask.shape
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        side = self.margin * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        if self.train:
            side *= 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            cx += rng.uniform(-self.jitter, self.jitter) * side
            cy += rng.uniform(-self.jitter, self.jitter) * side
        side = float(max(64.0, side))
        return cx, cy, side

    def __getitem__(self, i):
        stem = self.stems[i]
        rng = np.random.default_rng()
        meta = json.load(open(os.path.join(self.d, f"{stem}.json")))
        img = cv2.imread(os.path.join(self.d, f"{stem}{self.image_suffix}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.d, meta["image"].get("file_mask", f"{stem}_mask.png")), 0)
        vp = os.path.join(self.d, meta["image"].get("file_mask_visible", f"{stem}_mask_visible.png"))
        vis_mask = cv2.imread(vp, 0) if (os.path.exists(vp) and not self.image_suffix) else mask   # flat: no applier
        H, W = mask.shape
        jaw = float(meta["clip"].get("jaw_angle_deg", 0.0))
        R, t, K = pose_from_json(meta)
        P3 = self.cm.keypoints3d(jaw)
        uv, z = project(P3, R, t, K)

        cx, cy, side = self._window(mask, rng)
        s = self.crop / side
        A = np.array([[s, 0, self.crop / 2.0 - s * cx], [0, s, self.crop / 2.0 - s * cy]], np.float64)
        crop = cv2.warpAffine(img, A, (self.crop, self.crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        K_crop = np.array([[s * K[0, 0], 0, s * K[0, 2] + A[0, 2]], [0, s * K[1, 1], s * K[1, 2] + A[1, 2]], [0, 0, 1]], np.float64)
        kp = uv * s + A[:, 2][None, :]

        # visibility: in the original frame, in front of the camera, and not under the tool
        vis = np.zeros(self.K, np.float32)
        vm = cv2.dilate((vis_mask > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for k in range(self.K):
            ui, vi = int(round(uv[k, 0])), int(round(uv[k, 1]))
            if z[k] > 0 and 0 <= ui < W and 0 <= vi < H and vm[vi, ui]:
                vis[k] = 1.0
        inside = (kp[:, 0] >= 0) & (kp[:, 0] < self.crop) & (kp[:, 1] >= 0) & (kp[:, 1] < self.crop)
        vis *= inside.astype(np.float32)

        if self.photometric:
            crop = self._photometric(crop, rng)

        hm = np.zeros((self.K, self.heat, self.heat), np.float32)
        r = self.heat / self.crop
        gy, gx = np.mgrid[0:self.heat, 0:self.heat].astype(np.float32)
        for k in range(self.K):
            if vis[k] > 0:
                hm[k] = np.exp(-((gx - kp[k, 0] * r) ** 2 + (gy - kp[k, 1] * r) ** 2) / (2 * self.sigma ** 2))

        x = (crop[..., ::-1].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return dict(image=torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                    heatmaps=torch.from_numpy(hm), vis=torch.from_numpy(vis),
                    kp=torch.from_numpy(kp.astype(np.float32)), K_crop=torch.from_numpy(K_crop.astype(np.float32)),
                    R=torch.from_numpy(R.astype(np.float32)), t=torch.from_numpy(t.astype(np.float32)),
                    jaw=torch.tensor(jaw, dtype=torch.float32), tool=torch.tensor(float(bool(meta["clip"].get("tool_held")))),
                    scale=torch.tensor(s, dtype=torch.float32), K=torch.from_numpy(K.astype(np.float32)),
                    bbox_diag=torch.tensor(float(np.hypot(np.ptp(np.nonzero(mask > 127)[1]), np.ptp(np.nonzero(mask > 127)[0])))),
                    stem=stem)

    @staticmethod
    def _photometric(bgr, rng):
        im = bgr.astype(np.float32)
        im = im * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)                      # brightness / offset
        m = im.mean()
        im = (im - m) * rng.uniform(0.8, 1.2) + m                                   # contrast
        hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        if rng.random() < 0.5:
            im = cv2.GaussianBlur(im, (0, 0), float(rng.uniform(0.3, 1.5)))
        if rng.random() < 0.5:
            ok, buf = cv2.imencode(".jpg", np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))])
            if ok:
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
        im = im + rng.normal(0, rng.uniform(0, 3), im.shape)
        return np.clip(im, 0, 255).astype(np.uint8)


def decode_heatmaps(hm):
    """hm: (K, h, w) numpy -> (K, 2) sub-pixel peaks in heatmap px and (K,) confidences."""
    Kk, h, w = hm.shape
    pts = np.zeros((Kk, 2), np.float64)
    conf = np.zeros(Kk, np.float64)
    for k in range(Kk):
        m = hm[k]
        iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
        conf[k] = float(m[iy, ix])
        dx = dy = 0.0
        # quadratic peak refinement: x0 = ix - 0.5 (f+ - f-) / (f+ - 2 f0 + f-); the denominator is
        # NEGATIVE at a maximum (an earlier max(den, 1e-6) clamp made every offset saturate at +/-0.5)
        if 0 < ix < w - 1:
            den = float(m[iy, ix + 1] + m[iy, ix - 1] - 2 * m[iy, ix])
            if den < -1e-9:
                dx = float(np.clip(-0.5 * (m[iy, ix + 1] - m[iy, ix - 1]) / den, -0.5, 0.5))
        if 0 < iy < h - 1:
            den = float(m[iy + 1, ix] + m[iy - 1, ix] - 2 * m[iy, ix])
            if den < -1e-9:
                dy = float(np.clip(-0.5 * (m[iy + 1, ix] - m[iy - 1, ix]) / den, -0.5, 0.5))
        pts[k] = (ix + dx, iy + dy)
    return pts, conf


def pose_pnp(kp_crop, conf, P3, K_crop, min_conf=0.2, reproj=8.0):
    """PnP-RANSAC on confident keypoints. Returns (R, t, inliers) or None."""
    sel = np.flatnonzero(conf >= min_conf)
    if len(sel) < 4:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(P3[sel].astype(np.float64), kp_crop[sel].astype(np.float64), K_crop.astype(np.float64),
                                             None, iterationsCount=200, reprojectionError=reproj, confidence=0.99,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 4:
        return None
    ok2, rvec, tvec = cv2.solvePnP(P3[sel][inl[:, 0]].astype(np.float64), kp_crop[sel][inl[:, 0]].astype(np.float64),
                                   K_crop.astype(np.float64), None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rvec)[0]
    return R, tvec.reshape(3), sel[inl[:, 0]]


def pose_pnp_jawsearch(kp_crop, conf, clip_model, K_crop, jaws=(0.0, 6.0, 12.0, 18.0), **kw):
    """At inference the jaw angle is unknown: solve PnP for a few articulations and keep the one
    with the lowest reprojection residual. Returns (R, t, inliers, jaw) or None."""
    best = None
    for j in jaws:
        P3 = clip_model.keypoints3d(j)
        res = pose_pnp(kp_crop, conf, P3, K_crop, **kw)
        if res is None:
            continue
        R, t, inl = res
        uv, _ = project(P3[inl], R, t, K_crop)
        resid = float(np.linalg.norm(uv - kp_crop[inl], axis=1).mean()) - 0.05 * len(inl)
        if best is None or resid < best[0]:
            best = (resid, R, t, inl, j)
    return None if best is None else best[1:]


def list_stems(data_dir):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(data_dir, "[0-9]*.json")))
