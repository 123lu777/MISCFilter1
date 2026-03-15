"""
data/dataset_blade.py
---------------------
Dataset classes for wind turbine blade deblurring with optical flow labels.

DataLoaderBladeFlowTrain:
    Training dataset that returns (sharp, blur, flow_label, region_weights)
    tuples.  Optical flow can be pre-computed (fast) or computed on-the-fly.

DataLoaderBladeFlowVal:
    Validation dataset (centre-crop, no augmentation).
"""

import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

# Allow imports from the project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optical_flow_utils import (
    compute_farneback_flow,
    flow_to_psf_label,
    flow_to_region_weights,
)


def _is_image(filename):
    return any(filename.lower().endswith(ext)
               for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp'])


def _load_flow(flow_dir, basename):
    """Try to load a pre-computed .npy flow file.  Returns None if not found."""
    if flow_dir is None:
        return None
    npy_path = os.path.join(flow_dir, basename + '.npy')
    if os.path.isfile(npy_path):
        return np.load(npy_path).astype(np.float32)
    return None


def _compute_flow_from_images(blur_path, sharp_path):
    """Compute Farneback optical flow between a blur–sharp pair on the fly."""
    blur_bgr  = cv2.imread(blur_path)
    sharp_bgr = cv2.imread(sharp_path)
    if blur_bgr is None or sharp_bgr is None:
        return None
    # Resize to the same size if needed
    if blur_bgr.shape[:2] != sharp_bgr.shape[:2]:
        sharp_bgr = cv2.resize(sharp_bgr, (blur_bgr.shape[1], blur_bgr.shape[0]))
    gray_blur  = cv2.cvtColor(blur_bgr,  cv2.COLOR_BGR2GRAY)
    gray_sharp = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2GRAY)
    return compute_farneback_flow(gray_blur, gray_sharp)


def _flow_to_tensor(flow, target_h, target_w):
    """Convert (H, W, 2) flow to (2, H, W) tensor and resize to (target_h, target_w)."""
    if flow is None:
        return torch.zeros(2, target_h, target_w)
    flow_t = torch.from_numpy(flow.transpose(2, 0, 1))  # (2, H, W)
    if flow_t.shape[1] != target_h or flow_t.shape[2] != target_w:
        import torch.nn.functional as F
        flow_t = F.interpolate(flow_t.unsqueeze(0), size=(target_h, target_w),
                               mode='bilinear', align_corners=False).squeeze(0)
    return flow_t


def _motion_map_to_tensor(flow, target_h, target_w):
    """Convert flow to a (1, H, W) motion magnitude tensor."""
    if flow is None:
        return torch.zeros(1, target_h, target_w)
    from optical_flow_utils import flow_magnitude
    mag = flow_magnitude(flow).astype(np.float32)  # (H, W)
    mag_t = torch.from_numpy(mag).unsqueeze(0)      # (1, H, W)
    if mag_t.shape[1] != target_h or mag_t.shape[2] != target_w:
        import torch.nn.functional as F
        mag_t = F.interpolate(mag_t.unsqueeze(0), size=(target_h, target_w),
                              mode='bilinear', align_corners=False).squeeze(0)
    return mag_t


def _region_weights_to_tensor(flow, target_h, target_w):
    """Convert flow to a (1, H, W) region-weight tensor in [0, 1]."""
    if flow is None:
        return torch.ones(1, target_h, target_w)
    weights = flow_to_region_weights(flow).astype(np.float32)  # (H, W)
    w_t = torch.from_numpy(weights).unsqueeze(0)               # (1, H, W)
    if w_t.shape[1] != target_h or w_t.shape[2] != target_w:
        import torch.nn.functional as F
        w_t = F.interpolate(w_t.unsqueeze(0), size=(target_h, target_w),
                            mode='bilinear', align_corners=False).squeeze(0)
    return w_t


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class DataLoaderBladeFlowTrain(Dataset):
    """Training dataset for blade deblurring with optical flow labels.

    Each sample returns:
        tar_img   – (3, ps, ps) sharp (GT) patch tensor
        inp_img   – (3, ps, ps) blurred patch tensor
        flow      – (2, ps, ps) optical flow tensor (dx, dy) in pixels
        motion_map – (1, ps, ps) flow magnitude map
        region_w  – (1, ps, ps) region weight map (tip=high, root=low)
        filename  – sample identifier string

    Args:
        rgb_dir:    Root dataset directory.
        meta:       Meta list file path (``sharp_rel blur_rel`` per line).
        img_options: Dict with at least ``patch_size``.
        flow_dir:   Optional directory of pre-computed .npy flow files.
    """

    def __init__(self, rgb_dir, meta, img_options=None, flow_dir=None):
        super().__init__()
        with open(meta, 'r') as fin:
            self.tar_filenames = [line.strip('\n').split(' ')[0] for line in fin]
        with open(meta, 'r') as fin:
            self.inp_filenames = [line.strip('\n').split(' ')[1] for line in fin]

        self.img_options = img_options or {}
        self.rgb_dir     = rgb_dir
        self.flow_dir    = flow_dir
        self.sizex       = len(self.inp_filenames)
        self.ps          = self.img_options.get('patch_size', 256)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_rel = self.inp_filenames[index_]
        tar_rel = self.tar_filenames[index_]

        inp_path = os.path.join(self.rgb_dir, inp_rel)
        tar_path = os.path.join(self.rgb_dir, tar_rel) \
            if tar_rel != 'UNKNOWN' else None

        # ---- Load images ----
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB') \
            if tar_path and os.path.isfile(tar_path) else inp_img.copy()

        # ---- Padding ----
        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        if padw or padh:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        # ---- Colour augmentations ----
        if random.randint(0, 2) == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)
        if random.randint(0, 2) == 1:
            sat = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat)
            tar_img = TF.adjust_saturation(tar_img, sat)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr  = random.randint(0, hh - ps)
        cc  = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # ---- Geometric augmentations ----
        if aug == 1:
            inp_img = inp_img.flip(1); tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2); tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, k=2, dims=(1, 2))
            tar_img = torch.rot90(tar_img, k=2, dims=(1, 2))
        elif aug == 5:
            inp_img = torch.rot90(inp_img, k=3, dims=(1, 2))
            tar_img = torch.rot90(tar_img, k=3, dims=(1, 2))
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        # ---- Optical flow ----
        basename = os.path.splitext(os.path.basename(inp_rel))[0]
        flow = _load_flow(self.flow_dir, basename)
        if flow is None:
            flow = _compute_flow_from_images(inp_path, tar_path) \
                if tar_path and os.path.isfile(tar_path) else None

        flow_t      = _flow_to_tensor(flow, ps, ps)
        motion_map  = _motion_map_to_tensor(flow, ps, ps)
        region_w    = _region_weights_to_tensor(flow, ps, ps)

        filename = os.path.splitext(os.path.basename(tar_rel))[0] \
            if tar_rel != 'UNKNOWN' else basename

        return tar_img, inp_img, flow_t, motion_map, region_w, filename


# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class DataLoaderBladeFlowVal(Dataset):
    """Validation dataset for blade deblurring (centre-crop, no augmentation)."""

    def __init__(self, rgb_dir, meta, img_options=None, flow_dir=None):
        super().__init__()
        with open(meta, 'r') as fin:
            self.tar_filenames = [line.strip('\n').split(' ')[0] for line in fin]
        with open(meta, 'r') as fin:
            self.inp_filenames = [line.strip('\n').split(' ')[1] for line in fin]

        self.img_options = img_options or {}
        self.rgb_dir     = rgb_dir
        self.flow_dir    = flow_dir
        self.sizex       = len(self.inp_filenames)
        self.ps          = self.img_options.get('patch_size', 256)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_rel = self.inp_filenames[index_]
        tar_rel = self.tar_filenames[index_]

        inp_path = os.path.join(self.rgb_dir, inp_rel)
        tar_path = os.path.join(self.rgb_dir, tar_rel) \
            if tar_rel != 'UNKNOWN' else None

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB') \
            if tar_path and os.path.isfile(tar_path) else inp_img.copy()

        if ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        basename = os.path.splitext(os.path.basename(inp_rel))[0]
        flow = _load_flow(self.flow_dir, basename)
        if flow is None:
            flow = _compute_flow_from_images(inp_path, tar_path) \
                if tar_path and os.path.isfile(tar_path) else None

        flow_t     = _flow_to_tensor(flow, ps, ps)
        motion_map = _motion_map_to_tensor(flow, ps, ps)
        region_w   = _region_weights_to_tensor(flow, ps, ps)

        filename = os.path.splitext(os.path.basename(tar_rel))[0] \
            if tar_rel != 'UNKNOWN' else basename

        return tar_img, inp_img, flow_t, motion_map, region_w, filename
