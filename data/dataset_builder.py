"""
data/dataset_builder.py
-----------------------
Automatically build a paired (blur, sharp) dataset from wind turbine blade
videos and/or a directory of blurry images.

Usage (CLI):
    python data/dataset_builder.py \\
        --video_dir  /path/to/videos \\
        --output_dir /path/to/dataset \\
        --blur_dir   /path/to/blur_images   # optional: add real blur images

The output directory will contain:
    <output_dir>/sharp/        – GT sharp frames (from video)
    <output_dir>/blur/         – Synthesised/real blurred frames
    <output_dir>/blade_train_list.txt
    <output_dir>/blade_val_list.txt
"""

import os
import random
import argparse
from typing import Optional, Tuple, List

from video_processor import build_dataset_from_videos


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_meta_list(meta_path: str,
                    val_ratio: float = 0.1,
                    seed: int = 42
                    ) -> Tuple[str, str]:
    """Split a meta list file into train/val splits.

    Args:
        meta_path: Path to the full meta list file (``sharp blur`` per line).
        val_ratio: Fraction of pairs to use for validation.
        seed:      Random seed for reproducible splits.

    Returns:
        (train_path, val_path) paths of the written split files.
    """
    with open(meta_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    random.seed(seed)
    random.shuffle(lines)
    n_val  = max(1, int(len(lines) * val_ratio))
    val_lines   = lines[:n_val]
    train_lines = lines[n_val:]

    base = os.path.dirname(meta_path)
    train_path = os.path.join(base, 'blade_train_list.txt')
    val_path   = os.path.join(base, 'blade_val_list.txt')

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_lines) + '\n')

    print(f"[DatasetBuilder] Train: {len(train_lines)} pairs → {train_path}")
    print(f"[DatasetBuilder] Val:   {len(val_lines)} pairs → {val_path}")
    return train_path, val_path


def append_blur_images(meta_path: str,
                       blur_dir: str,
                       sharp_placeholder: str = 'UNKNOWN'
                       ) -> str:
    """Append real blurry images (without GT) to the meta list.

    These entries are marked with *sharp_placeholder* as the GT path so that
    the dataset loader can identify them as unpaired samples for self-supervised
    or semi-supervised training.

    Args:
        meta_path:          Existing meta list file.
        blur_dir:           Directory of real blurry images.
        sharp_placeholder:  String written as the GT path for unpaired images.

    Returns:
        Updated meta_path (same file, appended in place).
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG'}
    blur_files = [
        f for f in sorted(os.listdir(blur_dir))
        if os.path.splitext(f)[1] in img_exts
    ]

    with open(meta_path, 'a') as f:
        for fname in blur_files:
            blur_rel = os.path.join(blur_dir, fname)
            f.write(f"{sharp_placeholder} {blur_rel}\n")

    print(f"[DatasetBuilder] Appended {len(blur_files)} blur images → {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_full_dataset(video_dir: str,
                       output_dir: str,
                       blur_dir: Optional[str] = None,
                       sample_fps: float = 3.0,
                       sharp_top_percent: float = 0.30,
                       blade_roi: Optional[Tuple[int, int, int, int]] = None,
                       n_blur_steps: int = 8,
                       exposure_factor: float = 1.0,
                       val_ratio: float = 0.1
                       ) -> Tuple[str, str]:
    """Build a complete paired dataset from videos (and optional blur images).

    Args:
        video_dir:          Directory containing video files.
        output_dir:         Root output directory.
        blur_dir:           Optional directory of real blurry images (no GT).
        sample_fps:         Frame sampling rate.
        sharp_top_percent:  Top fraction of frames to keep as GT.
        blade_roi:          Optional blade ROI crop.
        n_blur_steps:       Blur synthesis integration steps.
        exposure_factor:    Exposure scale factor.
        val_ratio:          Fraction for validation split.

    Returns:
        (train_meta_path, val_meta_path)
    """
    full_meta = build_dataset_from_videos(
        video_dir=video_dir,
        output_dir=output_dir,
        sample_fps=sample_fps,
        sharp_top_percent=sharp_top_percent,
        blade_roi=blade_roi,
        n_blur_steps=n_blur_steps,
        exposure_factor=exposure_factor,
    )

    # Split into train / val
    train_path, val_path = split_meta_list(full_meta, val_ratio=val_ratio)

    # Optionally append real blur images to the training set (unpaired)
    if blur_dir is not None and os.path.isdir(blur_dir):
        append_blur_images(train_path, blur_dir)

    return train_path, val_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build blade deblurring dataset from videos')
    parser.add_argument('--video_dir',   required=True,
                        help='Directory of blade video files')
    parser.add_argument('--output_dir',  required=True,
                        help='Output dataset root directory')
    parser.add_argument('--blur_dir',    default=None,
                        help='Optional directory of real blur images (no GT)')
    parser.add_argument('--sample_fps',  type=float, default=3.0)
    parser.add_argument('--sharp_pct',   type=float, default=0.30)
    parser.add_argument('--n_steps',     type=int,   default=8)
    parser.add_argument('--exposure',    type=float, default=1.0)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--roi',         type=int,   nargs=4, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'))
    args = parser.parse_args()

    train_meta, val_meta = build_full_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        blur_dir=args.blur_dir,
        sample_fps=args.sample_fps,
        sharp_top_percent=args.sharp_pct,
        blade_roi=tuple(args.roi) if args.roi else None,
        n_blur_steps=args.n_steps,
        exposure_factor=args.exposure,
        val_ratio=args.val_ratio,
    )
    print(f"Train meta: {train_meta}")
    print(f"Val   meta: {val_meta}")
