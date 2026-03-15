"""
video_processor.py
------------------
Video frame selection and optical-flow-guided blur synthesis for wind turbine
blade deblurring dataset generation.

Key functions:
  - extract_frames          : sample frames from a video at a given FPS
  - tenengrad_sharpness     : compute Tenengrad sharpness score for a frame
  - select_sharp_frames     : keep the top-k% sharpest frames as GT references
  - synthesize_blur_frame   : warp & accumulate a sharp frame along a flow to
                              produce a realistic motion-blur frame
  - build_paired_dataset    : end-to-end pipeline video → (blur, sharp) pairs
  - save_paired_dataset     : write pairs to disk in the format expected by the
                              existing DataLoaderFileTrain / DataLoaderFileVal
"""

import os
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple

from optical_flow_utils import (
    compute_farneback_flow,
    flow_magnitude,
    flow_to_psf_label,
)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str,
                   sample_fps: float = 3.0,
                   blade_roi: Optional[Tuple[int, int, int, int]] = None
                   ) -> List[np.ndarray]:
    """Extract frames from a video at *sample_fps* frames per second.

    Args:
        video_path: Path to the video file.
        sample_fps: Desired sampling rate (frames per second).
        blade_roi:  Optional (x1, y1, x2, y2) crop window.

    Returns:
        List of BGR frames (np.ndarray, H x W x 3, uint8).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0
    step = max(1, int(round(src_fps / sample_fps)))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if blade_roi is not None:
                x1, y1, x2, y2 = blade_roi
                frame = frame[y1:y2, x1:x2]
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Sharpness metrics
# ---------------------------------------------------------------------------

def tenengrad_sharpness(frame_bgr: np.ndarray) -> float:
    """Compute Tenengrad sharpness score (higher = sharper).

    Args:
        frame_bgr: BGR frame (H x W x 3, uint8).

    Returns:
        Tenengrad score (float).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_sharpness(frame_bgr: np.ndarray) -> float:
    """Compute Laplacian variance sharpness score."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Sharp frame selection
# ---------------------------------------------------------------------------

def select_sharp_frames(frames: List[np.ndarray],
                        top_percent: float = 0.30,
                        metric: str = 'tenengrad'
                        ) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Select the sharpest frames from a list.

    Args:
        frames:      List of BGR frames.
        top_percent: Fraction of frames to keep (0 < p ≤ 1).
        metric:      'tenengrad' or 'laplacian'.

    Returns:
        sharp_frames:  Selected frames.
        scores:        Sharpness scores of the selected frames.
        indices:       Original indices in *frames*.
    """
    fn = tenengrad_sharpness if metric == 'tenengrad' else laplacian_sharpness
    if top_percent > 1.0:
        raise ValueError(
            f"select_sharp_frames: top_percent must be a fraction between 0 and 1 "
            f"(got {top_percent}). Did you mean {top_percent / 100:.2f} instead of {top_percent}?"
        )
    scored = [(fn(f), i, f) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[0], reverse=True)
    n_keep = max(1, int(math.ceil(len(scored) * top_percent)))
    selected = scored[:n_keep]
    selected.sort(key=lambda x: x[1])  # restore temporal order
    sharp_frames = [s[2] for s in selected]
    scores       = [s[0] for s in selected]
    indices      = [s[1] for s in selected]
    return sharp_frames, scores, indices


# ---------------------------------------------------------------------------
# Blur synthesis via optical flow
# ---------------------------------------------------------------------------

def synthesize_blur_frame(sharp_frame: np.ndarray,
                          flow: np.ndarray,
                          n_steps: int = 8,
                          exposure_factor: float = 1.0
                          ) -> np.ndarray:
    """Synthesize a motion-blurred frame by warping *sharp_frame* along *flow*.

    The frame is repeatedly shifted by sub-steps of the flow vector and the
    results are averaged, mimicking camera integration over an exposure time.

    Args:
        sharp_frame:     BGR frame (H x W x 3, uint8), the GT sharp image.
        flow:            (H, W, 2) optical flow (pixels/frame).
        n_steps:         Number of intermediate warp steps for integration.
        exposure_factor: Scale factor applied to flow magnitude (simulates
                         different exposure times; 1.0 = one frame duration).

    Returns:
        Blurred frame (H x W x 3, uint8).
    """
    H, W = sharp_frame.shape[:2]
    sharp_f = sharp_frame.astype(np.float32)

    # Base grid
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    accumulated = np.zeros_like(sharp_f)
    for step in range(n_steps):
        alpha = (step / max(n_steps - 1, 1)) * exposure_factor
        map_x = (xs + flow[..., 0] * alpha).astype(np.float32)
        map_y = (ys + flow[..., 1] * alpha).astype(np.float32)
        warped = cv2.remap(sharp_f, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        accumulated += warped

    blurred = (accumulated / n_steps).clip(0, 255).astype(np.uint8)
    return blurred


# ---------------------------------------------------------------------------
# Paired dataset builder
# ---------------------------------------------------------------------------

def build_paired_dataset(video_path: str,
                         output_dir: str,
                         sample_fps: float = 3.0,
                         sharp_top_percent: float = 0.30,
                         blade_roi: Optional[Tuple[int, int, int, int]] = None,
                         n_blur_steps: int = 8,
                         exposure_factor: float = 1.0,
                         save_flow: bool = False
                         ) -> List[Tuple[str, str]]:
    """End-to-end pipeline: video → (blurred_path, sharp_path) pairs on disk.

    Flow:
      1. Extract frames at *sample_fps*.
      2. Select top-k% sharpest frames as GT.
      3. For each consecutive pair of GT frames, compute optical flow from the
         first to the second, then synthesize a blurred version of the first.
      4. Save pairs and return a list of (sharp_path, blur_path) tuples.

    Args:
        video_path:        Path to source video.
        output_dir:        Directory to write sharp/ and blur/ sub-dirs.
        sample_fps:        Frame sampling rate.
        sharp_top_percent: Top fraction of frames kept as GT.
        blade_roi:         Optional crop ROI.
        n_blur_steps:      Integration steps for blur synthesis.
        exposure_factor:   Exposure scale (flow magnitude multiplier).
        save_flow:         If True, also save flow visualizations.

    Returns:
        List of (sharp_path, blur_path) pairs (absolute paths).
    """
    sharp_dir = os.path.join(output_dir, 'sharp')
    blur_dir  = os.path.join(output_dir, 'blur')
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blur_dir,  exist_ok=True)
    if save_flow:
        flow_dir = os.path.join(output_dir, 'flow')
        os.makedirs(flow_dir, exist_ok=True)

    print(f"[VideoProcessor] Extracting frames from {video_path} ...")
    frames = extract_frames(video_path, sample_fps=sample_fps, blade_roi=blade_roi)
    print(f"[VideoProcessor]   Total frames sampled: {len(frames)}")

    sharp_frames, _, sharp_indices = select_sharp_frames(
        frames, top_percent=sharp_top_percent)
    print(f"[VideoProcessor]   Sharp frames selected: {len(sharp_frames)}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    pairs = []

    for i in range(len(sharp_frames) - 1):
        frame_a = sharp_frames[i]
        frame_b = sharp_frames[i + 1]

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        flow = compute_farneback_flow(gray_a, gray_b)

        blurred = synthesize_blur_frame(frame_a, flow,
                                        n_steps=n_blur_steps,
                                        exposure_factor=exposure_factor)

        idx_str = f"{video_name}_{i:06d}"
        sharp_path = os.path.join(sharp_dir, f"{idx_str}_sharp.png")
        blur_path  = os.path.join(blur_dir,  f"{idx_str}_blur.png")

        cv2.imwrite(sharp_path, frame_a)
        cv2.imwrite(blur_path,  blurred)

        if save_flow:
            from optical_flow_utils import flow_to_color
            flow_vis = flow_to_color(flow)
            cv2.imwrite(os.path.join(flow_dir, f"{idx_str}_flow.png"), flow_vis)

        pairs.append((sharp_path, blur_path))

    print(f"[VideoProcessor]   Generated {len(pairs)} pairs → {output_dir}")
    return pairs


def build_dataset_from_videos(video_dir: str,
                              output_dir: str,
                              sample_fps: float = 3.0,
                              sharp_top_percent: float = 0.30,
                              blade_roi: Optional[Tuple[int, int, int, int]] = None,
                              n_blur_steps: int = 8,
                              exposure_factor: float = 1.0
                              ) -> str:
    """Process videos and write a meta-list file.

    Args:
        video_dir:         Path to a single video file **or** a directory that
                           contains video files.  When a directory is given,
                           all video files found in it are processed.
        output_dir:        Root output directory.
        sample_fps:        Frame sampling rate (frames per second).
        sharp_top_percent: Top fraction of frames kept as GT.
        blade_roi:         Optional blade crop ROI (x1, y1, x2, y2).
        n_blur_steps:      Blur synthesis integration steps.
        exposure_factor:   Exposure scale factor applied to the flow magnitude.

    Returns:
        Path to the generated meta-list file (one line per pair:
        ``sharp_rel_path blur_rel_path``).
    """
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV'}
    if os.path.isfile(video_dir):
        # Single video file passed directly
        video_files = [video_dir]
    else:
        video_files = [
            os.path.join(video_dir, f)
            for f in sorted(os.listdir(video_dir))
            if os.path.splitext(f)[1] in video_exts
        ]

    all_pairs = []
    for vf in video_files:
        pairs = build_paired_dataset(
            vf, output_dir,
            sample_fps=sample_fps,
            sharp_top_percent=sharp_top_percent,
            blade_roi=blade_roi,
            n_blur_steps=n_blur_steps,
            exposure_factor=exposure_factor,
        )
        all_pairs.extend(pairs)

    meta_path = os.path.join(output_dir, 'blade_train_list.txt')
    with open(meta_path, 'w') as f:
        for sharp_path, blur_path in all_pairs:
            # Write relative paths from output_dir
            sharp_rel = os.path.relpath(sharp_path, output_dir)
            blur_rel  = os.path.relpath(blur_path,  output_dir)
            f.write(f"{sharp_rel} {blur_rel}\n")

    print(f"[VideoProcessor] Meta list written to {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build paired blade deblurring dataset from videos')
    parser.add_argument('--video_dir',   required=True, help='Single video file or directory of blade videos')
    parser.add_argument('--output_dir',  required=True, help='Output dataset directory')
    parser.add_argument('--sample_fps',  type=float, default=3.0)
    parser.add_argument('--sharp_pct',   type=float, default=0.30,
                        help='Top fraction of frames to keep as GT')
    parser.add_argument('--n_steps',     type=int,   default=8,
                        help='Blur integration steps')
    parser.add_argument('--exposure',    type=float, default=1.0,
                        help='Exposure factor for blur synthesis')
    parser.add_argument('--roi',         type=int,   nargs=4, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Blade region-of-interest crop')
    args = parser.parse_args()

    meta = build_dataset_from_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        sharp_top_percent=args.sharp_pct,
        blade_roi=tuple(args.roi) if args.roi else None,
        n_blur_steps=args.n_steps,
        exposure_factor=args.exposure,
    )
    print(f"Done. Meta list: {meta}")
