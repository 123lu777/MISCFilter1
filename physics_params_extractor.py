"""
physics_params_extractor.py
---------------------------
Extract physical parameters of wind turbine blade rotation from optical flow
fields and video metadata.

Parameters extracted:
  - Rotation speed (RPM)
  - Exposure time (estimated or from metadata)
  - Rotation center (convergence point of the flow field)
  - Angular velocity (rad/frame, rad/s)
  - Per-region (tip/mid/root) linear velocity

These parameters are used as hard constraints in the physics-aware loss and
as conditioning signals for the MISCFilter model.
"""

import math
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List

from optical_flow_utils import (
    flow_magnitude,
    estimate_rotation_center,
)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def estimate_angular_velocity(flow: np.ndarray,
                               center: Tuple[float, float]
                               ) -> Tuple[float, float]:
    """Estimate angular velocity from a dense optical flow field.

    For pure rotation: v_tangential = omega * r
    → omega ≈ median(|v| / r)

    Args:
        flow:   (H, W, 2) optical flow (pixels/frame).
        center: (cx, cy) rotation center in pixel coordinates.

    Returns:
        omega_rad_per_frame: angular velocity in radians per frame.
        omega_sign:          +1 (CCW) or -1 (CW).
    """
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    dx = flow[..., 0]
    dy = flow[..., 1]
    cx, cy = center
    rx = xs - cx
    ry = ys - cy
    r = np.sqrt(rx ** 2 + ry ** 2)

    valid = r > 5.0
    if valid.sum() < 10:
        return 0.0, 1.0

    # Cross product (r x v) gives sign of rotation
    cross = rx[valid] * dy[valid] - ry[valid] * dx[valid]
    sign  = float(np.sign(np.median(cross)))
    if sign == 0.0:
        sign = 1.0

    mag = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)
    omega = float(np.median(mag / r[valid]))
    return omega, sign


def estimate_rotation_speed_rpm(flow_list: List[np.ndarray],
                                 frame_rate: float,
                                 center: Optional[Tuple[float, float]] = None
                                 ) -> float:
    """Estimate rotation speed in RPM from a sequence of optical flow fields.

    Accumulates per-frame angular velocities and converts to RPM.

    Args:
        flow_list:  List of (H, W, 2) flow arrays (consecutive frame pairs).
        frame_rate: Video frame rate (fps).
        center:     Optional known rotation center; estimated if None.

    Returns:
        Rotation speed in RPM.
    """
    if len(flow_list) == 0:
        return 0.0

    if center is None:
        center = estimate_rotation_center(flow_list[len(flow_list) // 2])

    omegas = []
    for flow in flow_list:
        omega, _ = estimate_angular_velocity(flow, center)
        omegas.append(omega)

    omega_rad_per_frame = float(np.median(omegas))
    omega_rad_per_sec   = omega_rad_per_frame * frame_rate
    rpm = omega_rad_per_sec * 60.0 / (2.0 * math.pi)
    return abs(rpm)


def estimate_exposure_time(flow: np.ndarray,
                            center: Tuple[float, float],
                            frame_rate: float,
                            known_omega: Optional[float] = None
                            ) -> float:
    """Estimate camera exposure time from blur magnitude and angular velocity.

    Blur extent (pixels) ≈ omega (rad/frame) * r (pixels) * T_exposure / T_frame

    Args:
        flow:        (H, W, 2) optical flow measured between two frames.
        center:      (cx, cy) rotation center.
        frame_rate:  Video FPS.
        known_omega: If provided, use this angular velocity (rad/frame);
                     otherwise estimate from flow.

    Returns:
        Estimated exposure time in seconds.
    """
    if known_omega is None:
        omega, _ = estimate_angular_velocity(flow, center)
    else:
        omega = abs(known_omega)

    if omega < 1e-6:
        return 1.0 / frame_rate

    # Mean blur magnitude
    mean_blur = float(flow_magnitude(flow).mean())

    # Mean radius from center
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    mean_r = float(r[r > 5.0].mean()) if (r > 5.0).any() else float(W / 4)

    if mean_r < 1.0 or omega * mean_r < 1.0e-6:
        return 1.0 / frame_rate

    # T_exposure/T_frame = blur / (omega * r)
    ratio = mean_blur / (omega * mean_r)
    ratio = min(ratio, 1.0)  # can't exceed one frame duration
    return ratio / frame_rate


def extract_region_velocities(flow: np.ndarray,
                               n_regions: int = 3
                               ) -> Dict[str, float]:
    """Compute mean linear velocity for blade tip / mid / root regions.

    Regions are split by image height into *n_regions* equal horizontal bands.
    The assumption is that the blade extends top-to-bottom in the cropped ROI,
    with the tip at the top (highest speed).

    Args:
        flow:      (H, W, 2) optical flow.
        n_regions: Number of blade span regions.

    Returns:
        Dict with region names and mean speeds (pixels/frame).
    """
    mag = flow_magnitude(flow)
    H = mag.shape[0]
    step = H // n_regions
    names = ['tip', 'mid', 'root'] if n_regions == 3 else \
            [f'region_{i}' for i in range(n_regions)]
    result = {}
    for i in range(n_regions):
        start = i * step
        end   = (i + 1) * step if i < n_regions - 1 else H
        result[names[i]] = float(mag[start:end, :].mean())
    return result


# ---------------------------------------------------------------------------
# Full extractor
# ---------------------------------------------------------------------------

def extract_physics_params(flow_list: List[np.ndarray],
                            frame_rate: float,
                            blade_roi: Optional[Tuple[int, int, int, int]] = None
                            ) -> Dict:
    """Extract all physics parameters from a sequence of optical flow fields.

    Args:
        flow_list:  List of (H, W, 2) optical flow arrays.
        frame_rate: Video FPS.
        blade_roi:  Optional (x1, y1, x2, y2) crop used (for metadata).

    Returns:
        Dictionary with keys:
          'rotation_center'  – (cx, cy)
          'omega_rad_frame'  – median angular velocity (rad/frame)
          'rotation_sign'    – +1 CCW, -1 CW
          'rpm'              – rotation speed in RPM
          'exposure_time_s'  – estimated exposure time (seconds)
          'mean_speed'       – mean pixel speed (pixels/frame)
          'region_velocities'– {tip, mid, root} speeds
          'frame_rate'       – input frame rate
    """
    if len(flow_list) == 0:
        return {}

    mid_flow = flow_list[len(flow_list) // 2]
    center = estimate_rotation_center(mid_flow)

    omegas, signs = [], []
    for flow in flow_list:
        omega, sign = estimate_angular_velocity(flow, center)
        omegas.append(omega)
        signs.append(sign)

    omega_med = float(np.median(omegas))
    sign_med  = float(np.sign(np.median(signs)))

    rpm = abs(omega_med) * frame_rate * 60.0 / (2.0 * math.pi)
    exposure = estimate_exposure_time(mid_flow, center, frame_rate,
                                      known_omega=omega_med)
    mean_speed = float(flow_magnitude(mid_flow).mean())
    region_vel = extract_region_velocities(mid_flow)

    return {
        'rotation_center':   center,
        'omega_rad_frame':   omega_med,
        'rotation_sign':     sign_med,
        'rpm':               rpm,
        'exposure_time_s':   exposure,
        'mean_speed':        mean_speed,
        'region_velocities': region_vel,
        'frame_rate':        frame_rate,
    }


# ---------------------------------------------------------------------------
# Utility: extract from a video file
# ---------------------------------------------------------------------------

def extract_physics_from_video(video_path: str,
                                blade_roi: Optional[Tuple[int, int, int, int]] = None,
                                max_frames: int = 100
                                ) -> Dict:
    """Convenience wrapper: open a video, compute flows, extract physics.

    Args:
        video_path: Path to video file.
        blade_roi:  Optional blade crop ROI.
        max_frames: Maximum number of frame pairs to process.

    Returns:
        Physics params dictionary (see extract_physics_params).
    """
    from optical_flow_utils import extract_video_flow

    flow_list, _ = extract_video_flow(video_path, blade_roi=blade_roi)
    flow_list = flow_list[:max_frames]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    return extract_physics_params(flow_list, frame_rate=fps,
                                   blade_roi=blade_roi)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(
        description='Extract wind turbine blade physics parameters from a video')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--roi', type=int, nargs=4, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--max_frames', type=int, default=100)
    args = parser.parse_args()

    params = extract_physics_from_video(
        args.video,
        blade_roi=tuple(args.roi) if args.roi else None,
        max_frames=args.max_frames,
    )
    print(json.dumps(params, indent=2, default=str))
