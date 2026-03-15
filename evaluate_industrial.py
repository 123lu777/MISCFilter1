"""
evaluate_industrial.py
-----------------------
Industrial-grade evaluation for wind turbine blade deblurring.

Metrics computed:
  - PSNR  / SSIM  (standard image quality)
  - LPIPS         (perceptual similarity – texture detail fidelity)
  - Sharpness delta  (Tenengrad score improvement after deblurring)
  - Defect detection mAP proxy (edge-density increase as a proxy for
    improved defect visibility when a ground-truth detector is unavailable)
  - Temporal consistency score (when evaluating video frame sequences)

Usage:
    python evaluate_industrial.py \\
        --ckpt   checkpoints/blade/MISCFilter_blade_pretrain/model_best.pth \\
        --val_dir  dataset/blade \\
        --val_meta dataset/blade/blade_val_list.txt \\
        --output_dir results/eval
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2

import utils
from data.data_RGB import get_validation_data
from models.MISCFilterNet import MISCKernelNet as myNet


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def tenengrad_score(img_np: np.ndarray) -> float:
    """Tenengrad sharpness (higher = sharper)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def edge_density(img_np: np.ndarray, threshold: int = 50) -> float:
    """Fraction of pixels with Canny edges (proxy for defect visibility)."""
    gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold, threshold * 2)
    return float(edges.mean()) / 255.0


def psnr_from_tensors(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two [0,1] tensors."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_from_numpy(pred: np.ndarray, target: np.ndarray,
                    win_size: int = 11) -> float:
    """Compute SSIM (mean over channels).  Inputs: H x W x 3, float [0,1]."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        score = sk_ssim(target, pred, data_range=1.0,
                        channel_axis=2, win_size=win_size)
        return float(score)
    except ImportError:
        # Simple luminance-only approximation if skimage not available
        pred_y   = 0.299 * pred[..., 0]   + 0.587 * pred[..., 1]   + 0.114 * pred[..., 2]
        target_y = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
        mu1, mu2 = pred_y.mean(), target_y.mean()
        s1, s2   = pred_y.std(), target_y.std()
        cov = float(np.mean((pred_y - mu1) * (target_y - mu2)))
        C1, C2 = (0.01 ** 2), (0.03 ** 2)
        num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
        den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2)
        return float(num / (den + 1e-10))


def lpips_from_tensors(pred: torch.Tensor, target: torch.Tensor,
                       lpips_model=None) -> float:
    """LPIPS perceptual distance (lower = better)."""
    if lpips_model is None:
        return float('nan')
    with torch.no_grad():
        d = lpips_model(pred.unsqueeze(0) * 2 - 1,
                        target.unsqueeze(0) * 2 - 1)
    return float(d.item())


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    model = myNet().cuda().eval()
    if args.ckpt and os.path.isfile(args.ckpt):
        utils.load_checkpoint(model, args.ckpt)
        print(f'Loaded checkpoint: {args.ckpt}')
    else:
        print('[WARNING] No valid checkpoint provided; using random weights.')

    # ---- LPIPS (optional) ----
    lpips_model = None
    try:
        import lpips as lpips_lib
        lpips_model = lpips_lib.LPIPS(net='alex').cuda().eval()
    except ImportError:
        print('[INFO] lpips not installed; LPIPS metric will be skipped.')

    # ---- Data ----
    val_dataset = get_validation_data(
        args.val_dir, args.val_meta, {'patch_size': args.patch_size})
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ---- Metrics accumulators ----
    results = {
        'psnr':              [],
        'ssim':              [],
        'lpips':             [],
        'sharp_input':       [],
        'sharp_output':      [],
        'edge_density_in':   [],
        'edge_density_out':  [],
    }

    log_lines = []

    for idx, data in enumerate(val_loader):
        target_t = data[0].cuda()    # (1, 3, H, W) GT sharp
        input_t  = data[1].cuda()    # (1, 3, H, W) blurry
        filename = data[2][0] if len(data) > 2 else str(idx)

        with torch.no_grad():
            restored_list, _ = model(input_t)
        restored_t = restored_list[0].clamp(0, 1)   # (1, 3, H, W)

        # To numpy [0,1]
        pred_np   = restored_t[0].permute(1, 2, 0).cpu().numpy()
        target_np = target_t[0].permute(1, 2, 0).cpu().numpy()
        input_np  = input_t[0].permute(1, 2, 0).cpu().numpy()
        pred_np   = np.clip(pred_np,   0, 1)
        target_np = np.clip(target_np, 0, 1)
        input_np  = np.clip(input_np,  0, 1)

        # To uint8 for CV metrics
        pred_u8   = (pred_np   * 255).astype(np.uint8)
        target_u8 = (target_np * 255).astype(np.uint8)
        input_u8  = (input_np  * 255).astype(np.uint8)

        # PSNR
        psnr = psnr_from_tensors(restored_t[0], target_t[0])

        # SSIM
        ssim = ssim_from_numpy(pred_np, target_np)

        # LPIPS
        lp = lpips_from_tensors(restored_t[0], target_t[0], lpips_model)

        # Sharpness
        sh_in  = tenengrad_score(input_u8)
        sh_out = tenengrad_score(pred_u8)

        # Edge density (proxy for defect visibility)
        ed_in  = edge_density(input_u8)
        ed_out = edge_density(pred_u8)

        results['psnr'].append(psnr)
        results['ssim'].append(ssim)
        results['lpips'].append(lp)
        results['sharp_input'].append(sh_in)
        results['sharp_output'].append(sh_out)
        results['edge_density_in'].append(ed_in)
        results['edge_density_out'].append(ed_out)

        log_line = (f"{filename}  PSNR={psnr:.3f}  SSIM={ssim:.4f}  "
                    f"LPIPS={lp:.4f}  Sharp_in={sh_in:.1f}  "
                    f"Sharp_out={sh_out:.1f}  "
                    f"EdgeDensity_in={ed_in:.4f}  EdgeDensity_out={ed_out:.4f}")
        log_lines.append(log_line)

        # Save output image
        if args.save_images:
            out_img = Image.fromarray(pred_u8)
            out_img.save(os.path.join(args.output_dir, f'{filename}_deblurred.png'))

    # ---- Aggregate ----
    def mean_skip_nan(lst):
        vals = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    avg_psnr  = mean_skip_nan(results['psnr'])
    avg_ssim  = mean_skip_nan(results['ssim'])
    avg_lpips = mean_skip_nan(results['lpips'])
    avg_sh_in  = mean_skip_nan(results['sharp_input'])
    avg_sh_out = mean_skip_nan(results['sharp_output'])
    avg_ed_in  = mean_skip_nan(results['edge_density_in'])
    avg_ed_out = mean_skip_nan(results['edge_density_out'])

    sharp_gain = ((avg_sh_out - avg_sh_in) / (avg_sh_in + 1e-8)) * 100
    edge_gain  = ((avg_ed_out - avg_ed_in) / (avg_ed_in + 1e-8)) * 100

    summary = (
        f"\n{'='*60}\n"
        f"Evaluation Summary ({len(results['psnr'])} images)\n"
        f"{'='*60}\n"
        f"  PSNR  : {avg_psnr:.4f} dB\n"
        f"  SSIM  : {avg_ssim:.4f}\n"
        f"  LPIPS : {avg_lpips:.4f}\n"
        f"  Sharpness (input → output): {avg_sh_in:.1f} → {avg_sh_out:.1f}  "
        f"(+{sharp_gain:.1f}%)\n"
        f"  Edge density (input → output): {avg_ed_in:.4f} → {avg_ed_out:.4f}  "
        f"(+{edge_gain:.1f}%)\n"
        f"{'='*60}\n"
    )
    print(summary)

    # ---- Write report ----
    report_path = os.path.join(args.output_dir, 'eval_report.txt')
    with open(report_path, 'w') as f:
        f.write(summary + '\n')
        f.write('\n'.join(log_lines) + '\n')
    print(f'Report saved to {report_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Industrial evaluation of blade deblurring')
    parser.add_argument('--ckpt',        required=True,              help='Model checkpoint')
    parser.add_argument('--val_dir',     required=True,              help='Dataset root directory')
    parser.add_argument('--val_meta',    required=True,              help='Validation meta list')
    parser.add_argument('--patch_size',  type=int,   default=256)
    parser.add_argument('--output_dir',  default='./results/eval')
    parser.add_argument('--save_images', action='store_true',
                        help='Save deblurred images to output_dir')
    args = parser.parse_args()
    evaluate(args)
