"""
optical_flow_utils.py
---------------------
Optical flow extraction and processing utilities for wind turbine blade deblurring.

Supports:
  - Farneback dense optical flow (OpenCV, fast)
  - RAFT-based optical flow (high accuracy, requires raft package or torchvision)
  - Flow-to-PSF (point spread function) label generation
  - Flow visualization
  - Rotation center estimation from flow field
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Farneback optical flow
# ---------------------------------------------------------------------------

def compute_farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two grayscale frames using Farneback.

    Args:
        prev_gray: Previous frame (H x W, uint8).
        curr_gray: Current  frame (H x W, uint8).

    Returns:
        flow: np.ndarray of shape (H, W, 2) with (dx, dy) per pixel.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def extract_video_flow(video_path: str, blade_roi=None):
    """Extract per-frame Farneback optical flow from a video.

    Args:
        video_path: Path to the video file.
        blade_roi:  Optional (x1, y1, x2, y2) region-of-interest crop.

    Returns:
        flow_list:   List of np.ndarray (H, W, 2), one per consecutive frame pair.
        frame_shape: Shape of a single (possibly cropped) frame (H, W, C).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise IOError(f"Cannot read first frame from: {video_path}")

    def _crop(img, roi):
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]
        return img

    prev_gray = cv2.cvtColor(_crop(prev_frame, blade_roi), cv2.COLOR_BGR2GRAY)
    frame_shape = prev_gray.shape

    flow_list = []
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(_crop(curr_frame, blade_roi), cv2.COLOR_BGR2GRAY)
        flow = compute_farneback_flow(prev_gray, curr_gray)
        flow_list.append(flow)
        prev_gray = curr_gray.copy()

    cap.release()
    return flow_list, frame_shape


# ---------------------------------------------------------------------------
# RAFT optical flow wrapper (optional, requires torchvision >= 0.13 or raft pkg)
# ---------------------------------------------------------------------------

def compute_raft_flow(prev_rgb: np.ndarray, curr_rgb: np.ndarray,
                      device: str = 'cpu') -> np.ndarray:
    """Compute optical flow using RAFT via torchvision (if available).

    Falls back to Farneback if torchvision RAFT is not available.

    Args:
        prev_rgb: Previous frame (H x W x 3, uint8, BGR or RGB).
        curr_rgb: Current  frame (H x W x 3, uint8, BGR or RGB).
        device:   'cpu' or 'cuda'.

    Returns:
        flow: np.ndarray (H, W, 2).
    """
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        transforms = weights.transforms()
        model = raft_small(weights=weights).to(device).eval()

        def _to_tensor(img):
            t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0).to(device)

        img1 = _to_tensor(prev_rgb)
        img2 = _to_tensor(curr_rgb)
        img1, img2 = transforms(img1, img2)

        with torch.no_grad():
            flow_predictions = model(img1, img2)
        flow_tensor = flow_predictions[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow_tensor

    except Exception:
        # Fallback: Farneback
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_BGR2GRAY)
        return compute_farneback_flow(prev_gray, curr_gray)


# ---------------------------------------------------------------------------
# Flow field analysis utilities
# ---------------------------------------------------------------------------

def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return per-pixel magnitude (speed) of the flow field."""
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def flow_to_psf_label(flow: np.ndarray, max_kernel_size: int = 33) -> dict:
    """Convert optical flow statistics to PSF/blur-kernel labels.

    For each pixel the motion distance approximates the local blur kernel size.
    Three blade regions (tip / mid / root) are inferred from flow magnitude.

    Args:
        flow:            (H, W, 2) optical flow array.
        max_kernel_size: Upper bound on kernel size in pixels.

    Returns:
        dict with keys:
          'kernel_sizes'  – (H, W) estimated kernel size per pixel (float)
          'kernel_angles' – (H, W) estimated kernel angle in degrees
          'mean_speed'    – scalar mean speed (pixels/frame)
          'tip_speed'     – mean speed in the tip  region (top 33%)
          'mid_speed'     – mean speed in the mid  region (mid 33%)
          'root_speed'    – mean speed in the root region (bot 33%)
    """
    mag = flow_magnitude(flow)
    ang = np.degrees(np.arctan2(flow[..., 1], flow[..., 0]))

    kernel_sizes = np.clip(mag, 0, max_kernel_size)
    kernel_angles = ang % 180  # 0-180 for line PSF

    H = flow.shape[0]
    tip_region  = mag[:H // 3, :]
    mid_region  = mag[H // 3: 2 * H // 3, :]
    root_region = mag[2 * H // 3:, :]

    return {
        'kernel_sizes':  kernel_sizes,
        'kernel_angles': kernel_angles,
        'mean_speed':    float(mag.mean()),
        'tip_speed':     float(tip_region.mean()),
        'mid_speed':     float(mid_region.mean()),
        'root_speed':    float(root_region.mean()),
    }


def estimate_rotation_center(flow: np.ndarray) -> tuple:
    """Estimate the rotation center of the blade from the optical flow field.

    For pure rotation: dx = -omega*(y-cy), dy = omega*(x-cx).
    This gives:
      cx ≈ median(x + dy/omega)
      cy ≈ median(y - dx/omega)
    where omega is estimated from |v| / r using the image centre as the
    initial guess.

    Args:
        flow: (H, W, 2) optical flow.

    Returns:
        (cx, cy): estimated rotation center in pixel coordinates.
    """
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]

    dx = flow[..., 0].ravel().astype(np.float64)
    dy = flow[..., 1].ravel().astype(np.float64)
    x  = xs.ravel().astype(np.float64)
    y  = ys.ravel().astype(np.float64)

    mag = np.sqrt(dx ** 2 + dy ** 2)
    valid = mag > 1.0
    if valid.sum() < 10:
        return float(W / 2), float(H / 2)

    # Estimate omega using the image centre as the initial rotation centre guess
    x_c0, y_c0 = W / 2.0, H / 2.0
    r = np.sqrt((x[valid] - x_c0) ** 2 + (y[valid] - y_c0) ** 2)
    r = np.where(r < 1.0, 1.0, r)
    omega = float(np.median(mag[valid] / r))
    if abs(omega) < 1e-6:
        return float(x_c0), float(y_c0)

    cx = float(np.median(x[valid] + dy[valid] / omega))
    cy = float(np.median(y[valid] - dx[valid] / omega))
    return cx, cy


def flow_to_region_weights(flow: np.ndarray, n_regions: int = 3) -> np.ndarray:
    """Convert flow magnitude to per-pixel region weights (tip/mid/root).

    Args:
        flow:      (H, W, 2) optical flow.
        n_regions: number of blade regions to split into.

    Returns:
        weights: (H, W) array in [0, 1] representing relative blur strength.
    """
    mag = flow_magnitude(flow)
    max_mag = mag.max()
    if max_mag < 1e-6:
        return np.ones_like(mag)
    return mag / max_mag


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert flow field to an HSV color image (for visualization).

    Args:
        flow: (H, W, 2) flow array.

    Returns:
        BGR image (H, W, 3, uint8).
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2        # hue = direction
    hsv[..., 1] = 255                            # saturation = full
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
