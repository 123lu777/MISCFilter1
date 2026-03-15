"""
train_GoPro_enhanced.py
-----------------------
Enhanced three-stage training script for the MISCFilter blade deblurring model.

Stages:
  Stage 1 – pretrain : Train on video-synthesised paired data with optical flow
                        prior losses and physics constraint losses.
  Stage 2 – finetune : Fine-tune on real blurry images with a low learning rate.
  Stage 3 – eval     : Full validation pass (no weight updates).

Usage examples:
  # Stage 1 – pre-training (video-synthesised pairs):
  python train_GoPro_enhanced.py \\
      --stage pretrain \\
      --train_dir ./dataset/blade \\
      --train_meta ./dataset/blade/blade_train_list.txt \\
      --val_dir   ./dataset/blade \\
      --val_meta  ./dataset/blade/blade_val_list.txt \\
      --num_epochs 70

  # Stage 2 – fine-tuning (real blur images):
  python train_GoPro_enhanced.py \\
      --stage finetune \\
      --train_dir ./dataset/blade \\
      --train_meta ./dataset/blade/blade_train_list.txt \\
      --val_dir   ./dataset/blade \\
      --val_meta  ./dataset/blade/blade_val_list.txt \\
      --pretrain_ckpt ./checkpoints/blade/pretrain/model_best.pth \\
      --num_epochs 15
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data.data_RGB import get_training_data, get_validation_data
from data.data_RGB import get_blade_training_data, get_blade_validation_data
from models.MISCFilterNet import MISCKernelNet as myNet
from loss import losses
from loss.motion_prior_loss import MotionPriorLoss, PhysicsConstraintLoss
from loss.temporal_consistency_loss import TemporalConsistencyLoss
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tools.get_parameter_number import get_parameter_number
import kornia
import argparse

######### Reproducibility ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Enhanced Blade Deblurring Training')

# Data
parser.add_argument('--train_dir',    default='./dataset/blade',  type=str)
parser.add_argument('--train_meta',   default='./dataset/blade/blade_train_list.txt', type=str)
parser.add_argument('--val_dir',      default='./dataset/blade',  type=str)
parser.add_argument('--val_meta',     default='./dataset/blade/blade_val_list.txt',   type=str)
parser.add_argument('--flow_dir',     default=None,               type=str,
                    help='Optional directory of pre-computed .npy flow files')

# Checkpoints
parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
parser.add_argument('--pretrain_ckpt',  default='',              type=str,
                    help='Path to pre-trained checkpoint (for finetune stage)')
parser.add_argument('--resume_ckpt',    default='',              type=str,
                    help='Resume from this checkpoint')

# Training
parser.add_argument('--stage',       default='pretrain', type=str,
                    choices=['pretrain', 'finetune', 'eval'],
                    help='Training stage')
parser.add_argument('--dataset',     default='blade',    type=str)
parser.add_argument('--session',     default='MISCFilter_blade', type=str)
parser.add_argument('--patch_size',  default=256,        type=int)
parser.add_argument('--num_epochs',  default=70,         type=int)
parser.add_argument('--batch_size',  default=8,          type=int)
parser.add_argument('--val_epochs',  default=5,          type=int)
parser.add_argument('--print_epochs', default=1,         type=int)

# Learning rates
parser.add_argument('--start_lr', default=2e-4, type=float)
parser.add_argument('--end_lr',   default=1e-6, type=float)
parser.add_argument('--finetune_lr', default=1e-5, type=float,
                    help='Learning rate for the finetune stage')

# Loss weights
parser.add_argument('--w_motion',   default=0.05, type=float,
                    help='Weight for motion-prior loss')
parser.add_argument('--w_temporal', default=0.05, type=float,
                    help='Weight for temporal consistency loss')
parser.add_argument('--w_physics',  default=0.01, type=float,
                    help='Weight for physics constraint loss (radial)')

# Dataset mode: 'blade' uses optical-flow-aware loaders; 'gopro' uses original
parser.add_argument('--data_mode', default='blade', type=str,
                    choices=['blade', 'gopro'],
                    help='"blade" loads optical flow labels; "gopro" is the original mode')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Directories and logging
# ---------------------------------------------------------------------------

stage   = args.stage
dataset = args.dataset
session = args.session + f'_{stage}'
ps      = args.patch_size

model_dir = os.path.join(args.model_save_dir, dataset, session)
utils.mkdir(model_dir)
log_dir   = os.path.join(model_dir, 'log.txt')

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model_restoration = myNet()

total_num, trainable_num = get_parameter_number(model_restoration)
print(f'Total params: {total_num}  |  Trainable: {trainable_num}')
with open(log_dir, 'a+') as f:
    f.write(f'Stage: {stage}\n')
    f.write(f'Total: {total_num}  Trainable: {trainable_num}\n')

model_restoration.cuda()

device_ids = list(range(torch.cuda.device_count()))
if len(device_ids) > 1:
    print(f"\nUsing {len(device_ids)} GPUs\n")

# ---------------------------------------------------------------------------
# Optimiser and scheduler
# ---------------------------------------------------------------------------

if stage == 'finetune':
    lr = args.finetune_lr
else:
    lr = args.start_lr

optimizer = optim.Adam(model_restoration.parameters(),
                       lr=lr, betas=(0.9, 0.999), eps=1e-8)

warmup_epochs = 3
num_epochs    = args.num_epochs
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs - warmup_epochs, eta_min=args.end_lr)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=warmup_epochs,
    after_scheduler=scheduler_cosine)

start_epoch = 1

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
    utils.load_checkpoint(model_restoration, args.pretrain_ckpt)
    print(f'Loaded pre-trained weights: {args.pretrain_ckpt}')

if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
    utils.load_checkpoint(model_restoration, args.resume_ckpt)
    start_epoch = utils.load_start_epoch(args.resume_ckpt) + 1
    utils.load_optim(optimizer, args.resume_ckpt)
    for _ in range(1, start_epoch):
        scheduler.step()
    print(f'Resumed from epoch {start_epoch}  lr={scheduler.get_lr()[0]:.2e}')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

criterion_char    = losses.CharbonnierLoss()
criterion_edge    = losses.EdgeLoss()
criterion_fft     = losses.fftLoss()
criterion_motion  = MotionPriorLoss(weight=args.w_motion)
criterion_physics = PhysicsConstraintLoss(w_radial=args.w_physics,
                                          w_rpm=0.0, w_exp=0.0)
criterion_temporal = TemporalConsistencyLoss(weight=args.w_temporal)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

use_blade_mode = (args.data_mode == 'blade')

if use_blade_mode:
    train_dataset = get_blade_training_data(
        args.train_dir, args.train_meta,
        {'patch_size': ps}, flow_dir=args.flow_dir)
    val_dataset   = get_blade_validation_data(
        args.val_dir, args.val_meta,
        {'patch_size': ps}, flow_dir=args.flow_dir)
else:
    train_dataset = get_training_data(
        args.train_dir, args.train_meta, {'patch_size': ps})
    val_dataset   = get_validation_data(
        args.val_dir, args.val_meta, {'patch_size': ps})

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=False,
                          pin_memory=True)
val_loader   = DataLoader(dataset=val_dataset,   batch_size=8,
                          shuffle=False, num_workers=4, drop_last=False,
                          pin_memory=True)

print(f'===> Start Epoch {start_epoch}  End Epoch {num_epochs + 1}')
print(f'===> Stage: {stage}  |  Data mode: {args.data_mode}')
with open(log_dir, 'a+') as f:
    f.write(f'===> Start Epoch {start_epoch}  End Epoch {num_epochs + 1}\n')

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

best_psnr  = 0.0
best_epoch = 0
global_iter = 0

if stage == 'eval':
    num_epochs = 0   # skip training, jump straight to evaluation

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start = time.time()
    epoch_loss  = 0.0
    model_restoration.train()

    for i, data in enumerate(train_loader):
        for param in model_restoration.parameters():
            param.grad = None

        if use_blade_mode:
            # data = (tar, inp, flow, motion_map, region_w, filename)
            target_      = data[0].cuda()
            input_       = data[1].cuda()
            # data[2]: (B, 2, H, W) optical flow tensor
            # data[3]: (B, 1, H, W) per-pixel motion magnitude map (from flow)
            motion_map_label = data[3].cuda()
            region_w         = data[4].cuda()   # (B, 1, H, W) region weights
        else:
            target_          = data[0].cuda()
            input_           = data[1].cuda()
            motion_map_label = None
            region_w         = None

        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored, restored_inter = model_restoration(input_)

        # --- Standard losses ---
        loss_fft  = sum(criterion_fft(restored[s], target[s])  for s in range(3))
        loss_char = sum(criterion_char(restored[s], target[s]) for s in range(3))
        loss_edge = sum(criterion_edge(restored[s], target[s]) for s in range(3))
        loss_char_inter = sum(
            criterion_char(restored_inter[s], target[s]) for s in range(3))

        loss = (loss_char + loss_char_inter
                + 0.01 * loss_fft
                + 0.05 * loss_edge)

        # --- Motion prior loss (blade mode only) ---
        if use_blade_mode and motion_map_label is not None:
            # Use the residual magnitude |restored - input| as a differentiable
            # proxy for the motion magnitude predicted by the model.
            # Clamp spatial dims to match restored output (which may differ from
            # input if the model uses sub-pixel convolutions or padding changes).
            h_r, w_r = restored[0].shape[2], restored[0].shape[3]
            input_crop = input_[:, :, :h_r, :w_r]
            pred_diff = (restored[0] - input_crop).abs().mean(dim=1, keepdim=True)
            loss_motion = criterion_motion(pred_diff, motion_map_label)
            loss = loss + loss_motion

        # --- Physics constraint (radial velocity gradient) ---
        if use_blade_mode and motion_map_label is not None:
            loss_phys = criterion_physics(motion_map=motion_map_label)
            loss = loss + loss_phys

        # --- Temporal consistency (pair consecutive samples in the batch) ---
        if use_blade_mode and restored[0].shape[0] >= 2 and data[2] is not None:
            flow_t = data[2].cuda()   # (B, 2, H, W) optical flow
            # Compare odd-even pairs within the batch
            n = restored[0].shape[0] // 2 * 2
            loss_temp = criterion_temporal(
                restored[0][:n:2], restored[0][1:n:2],
                flow=flow_t[:n:2])
            loss = loss + loss_temp

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        global_iter += 1

    # ---- Logging ----
    if epoch % args.print_epochs == 0:
        elapsed = time.time() - epoch_start
        lr_now  = scheduler.get_lr()[0]
        msg = (f"Epoch {epoch}/{num_epochs}  "
               f"Loss {epoch_loss:.4f}  "
               f"LR {lr_now:.2e}  "
               f"Time {elapsed:.1f}s")
        print(msg)
        with open(log_dir, 'a+') as f:
            f.write(msg + '\n')

    # ---- Validation ----
    if epoch % args.val_epochs == 0 or epoch == num_epochs:
        model_restoration.eval()
        psnr_vals = []
        with torch.no_grad():
            for data_val in val_loader:
                if use_blade_mode:
                    target_v = data_val[0].cuda()
                    input_v  = data_val[1].cuda()
                else:
                    target_v = data_val[0].cuda()
                    input_v  = data_val[1].cuda()

                restored_v, _ = model_restoration(input_v)
                for res, tar in zip(restored_v[0], target_v):
                    psnr_vals.append(utils.torchPSNR(res, tar))

        psnr_val = torch.stack(psnr_vals).mean().item()
        print(f'[Val] Epoch {epoch}  PSNR {psnr_val:.4f}  '
              f'(best: {best_psnr:.4f} @ epoch {best_epoch})')
        with open(log_dir, 'a+') as f:
            f.write(f'[Val] Epoch {epoch}  PSNR {psnr_val:.4f}\n')

        if psnr_val > best_psnr:
            best_psnr  = psnr_val
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer':  optimizer.state_dict()},
                       os.path.join(model_dir, 'model_best.pth'))
            print(f'  --> Saved best model (PSNR {best_psnr:.4f})')

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer':  optimizer.state_dict()},
                   os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

    scheduler.step()

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer':  optimizer.state_dict()},
               os.path.join(model_dir, 'model_latest.pth'))

# ---------------------------------------------------------------------------
# Final evaluation (also runs when stage == 'eval')
# ---------------------------------------------------------------------------

print('\n===> Final Evaluation')
model_restoration.eval()
psnr_vals = []
with torch.no_grad():
    for data_val in val_loader:
        if use_blade_mode:
            target_v = data_val[0].cuda()
            input_v  = data_val[1].cuda()
        else:
            target_v = data_val[0].cuda()
            input_v  = data_val[1].cuda()

        restored_v, _ = model_restoration(input_v)
        for res, tar in zip(restored_v[0], target_v):
            psnr_vals.append(utils.torchPSNR(res, tar))

final_psnr = torch.stack(psnr_vals).mean().item()
msg = f'Final PSNR: {final_psnr:.4f}  Best PSNR: {best_psnr:.4f} @ epoch {best_epoch}'
print(msg)
with open(log_dir, 'a+') as f:
    f.write(msg + '\n')
