"""
temporal_eval.py
----------------
Temporal continuity evaluation for video deblurring results.

Metrics:
  - Inter-frame PSNR  : PSNR between consecutive deblurred frames
  - Optical-flow consistency  : photometric error after warping frame t → t+1
    using the ground-truth (or estimated) optical flow
  - Ghosting score   : ratio of high-frequency energy between consecutive frames
    (a proxy for temporal ringing/ghosting artefacts)
  - Temporal SSIM    : SSIM between consecutive deblurred frames

Usage:
    python temporal_eval.py \\
        --ckpt  checkpoints/blade/MISCFilter_blade_pretrain/model_best.pth \\
        --video dataset/videos/blade_video1.mp4 \\
        --output_dir results/temporal_eval
"""

import os
import math
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import utils
from models.MISCFilterNet import MISCKernelNet as myNet
from optical_flow_utils import compute_farneback_flow, flow_magnitude


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    """Simple luminance SSIM between two uint8 images."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (a.std() ** 2 + b.std() ** 2 + C2)
    return float(num / (den + 1e-10))


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp *frame* by *flow* using cv2.remap."""
    H, W = frame.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    return cv2.remap(frame, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def ghosting_score(a: np.ndarray, b: np.ndarray) -> float:
    """High-frequency energy ratio between consecutive frames (proxy for ghosting).

    A lower score means fewer temporal artefacts.
    """
    def hf_energy(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.mean(lap ** 2))

    e_a = hf_energy(a)
    e_b = hf_energy(b)
    if e_b < 1e-6:
        return float('inf')
    return e_a / e_b


def deblur_frame(model: torch.nn.Module,
                 frame_bgr: np.ndarray,
                 device) -> np.ndarray:
    """Run the deblurring model on a single BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = TF.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        restored, _ = model(inp)
    out = restored[0][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_temporal(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    model = myNet().to(args.device).eval()
    if args.ckpt and os.path.isfile(args.ckpt):
        utils.load_checkpoint(model, args.ckpt)
        print(f'Loaded checkpoint: {args.ckpt}')
    else:
        print('[WARNING] No checkpoint provided; using random weights.')

    # ---- Video ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    src_fps   = fps
    step      = max(1, int(round(src_fps / args.eval_fps)))
    max_frames = args.max_frames

    frames_raw = []
    idx = 0
    while len(frames_raw) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames_raw.append(frame)
        idx += 1
    cap.release()
    print(f'Loaded {len(frames_raw)} frames from {args.video}')

    if len(frames_raw) < 2:
        print('Not enough frames for temporal evaluation.')
        return

    # ---- Deblur all frames ----
    frames_deblurred = []
    for f in frames_raw:
        db = deblur_frame(model, f, device=args.device)
        frames_deblurred.append(db)

    # ---- Temporal metrics ----
    inter_psnr      = []
    inter_ssim      = []
    flow_consistency = []
    ghost_scores     = []

    for i in range(len(frames_deblurred) - 1):
        fa = frames_deblurred[i]
        fb = frames_deblurred[i + 1]

        # Inter-frame PSNR / SSIM
        inter_psnr.append(psnr_np(fa, fb))
        inter_ssim.append(ssim_np(fa, fb))

        # Ghosting score
        ghost_scores.append(ghosting_score(fa, fb))

        # Flow consistency (warp fa → fb, measure photometric error)
        gray_a = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        flow   = compute_farneback_flow(gray_a, gray_b)
        warped = warp_frame(fa, flow)
        err    = np.abs(warped.astype(np.float64) - fb.astype(np.float64)).mean()
        flow_consistency.append(float(err))

        # Save side-by-side pair
        if args.save_images and i < 20:
            pair = np.concatenate([fa, fb], axis=1)
            cv2.imwrite(os.path.join(args.output_dir, f'pair_{i:04d}.png'), pair)

    avg_psnr  = float(np.mean(inter_psnr))
    avg_ssim  = float(np.mean(inter_ssim))
    avg_flow  = float(np.mean(flow_consistency))
    avg_ghost = float(np.nanmean(ghost_scores))

    summary = (
        f"\n{'='*60}\n"
        f"Temporal Evaluation ({len(frames_deblurred)} frames)\n"
        f"{'='*60}\n"
        f"  Inter-frame PSNR        : {avg_psnr:.4f} dB\n"
        f"  Inter-frame SSIM        : {avg_ssim:.4f}\n"
        f"  Flow consistency error  : {avg_flow:.4f}  (lower=better)\n"
        f"  Ghosting score          : {avg_ghost:.4f}  (≈1.0=ideal)\n"
        f"{'='*60}\n"
    )
    print(summary)

    report_path = os.path.join(args.output_dir, 'temporal_report.txt')
    with open(report_path, 'w') as f:
        f.write(summary)
    print(f'Report saved to {report_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal continuity evaluation')
    parser.add_argument('--ckpt',       required=True, help='Model checkpoint path')
    parser.add_argument('--video',      required=True, help='Input video path')
    parser.add_argument('--output_dir', default='./results/temporal_eval')
    parser.add_argument('--eval_fps',   type=float, default=5.0,
                        help='Evaluation sampling rate (frames/sec)')
    parser.add_argument('--max_frames', type=int,   default=100)
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    evaluate_temporal(args)
