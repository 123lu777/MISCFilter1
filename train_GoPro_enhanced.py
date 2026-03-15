"""
train_GoPro_enhanced.py
-----------------------
MISCFilter 叶片去模糊 – 一键三步训练脚本
=========================================

使用方法（无需终端）
--------------------
1. 在下方「用户配置」区域填写你的路径和参数。
2. 在 VSCode / PyCharm 中右键此文件 → "运行 Python 文件"即可。
   脚本会自动依次完成：
     Step 1 – 从视频生成配对数据集（若已有数据集可跳过，保持 video_dir = None）
     Step 2 – 预训练（~70 epoch，视频合成配对数据）
     Step 3 – 微调  （~15 epoch，真实模糊图片，低学习率）

只运行单个阶段
--------------
将 stage 改为 'pretrain'、'finetune' 或 'eval' 即可。
  - 'finetune' / 'eval' 时需填写 pretrain_ckpt。
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

import types
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
from tools.get_parameter_number import get_parameter_number
import kornia

######### Reproducibility ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# ===========================================================================
#  用户配置区  ★ 只需修改这里，然后右键运行即可 ★
# ===========================================================================

args = types.SimpleNamespace(

    # ── Step 1：数据集构建 ──────────────────────────────────────────────────
    # 有视频文件时填写 video_dir，脚本自动抽帧并生成配对数据集。
    # 数据集已存在时保持 video_dir = None，直接读取 output_dir 中的文件。
    video_dir       = None,              # 示例: r'D:\blade_videos'
    output_dir      = './dataset/blade', # 数据集根目录（生成或已有）
    blur_dir        = None,              # 可选：真实模糊图目录（无GT）
                                         # 示例: r'D:\real_blur'
    sample_fps      = 3.0,   # 视频抽帧帧率
    sharp_pct       = 0.30,  # Tenengrad筛选：保留最清晰的前30%帧
    n_blur_steps    = 8,     # 模糊合成积分步数
    exposure_factor = 1.0,   # 曝光缩放系数
    val_ratio       = 0.1,   # 验证集比例
    blade_roi       = None,  # 可选叶片ROI裁剪，示例: [x1, y1, x2, y2]

    # ── 数据路径（已有数据集时可覆盖；通常保持 None 自动推断） ──────────────
    train_dir  = None,  # 默认等于 output_dir
    train_meta = None,  # 默认: <output_dir>/blade_train_list.txt
    val_dir    = None,  # 默认等于 output_dir
    val_meta   = None,  # 默认: <output_dir>/blade_val_list.txt
    flow_dir   = None,  # 预计算光流 .npy 文件目录（可选）

    # ── 检查点 ──────────────────────────────────────────────────────────────
    model_save_dir = './checkpoints',
    # stage='finetune'/'eval' 时，填写预训练权重路径；stage='all' 时自动传递
    pretrain_ckpt  = '',  # 示例: './checkpoints/blade/MISCFilter_blade_pretrain/model_best.pth'
    resume_ckpt    = '',  # 断点续训（可选）

    # ── 训练阶段 ─────────────────────────────────────────────────────────────
    # 'all'      → Step1（若 video_dir 非空）+ 预训练 + 微调  ← 推荐
    # 'pretrain' → 仅预训练
    # 'finetune' → 仅微调（需填写 pretrain_ckpt）
    # 'eval'     → 仅评估（需填写 pretrain_ckpt）
    stage           = 'all',

    dataset         = 'blade',
    session         = 'MISCFilter_blade',
    patch_size      = 256,
    pretrain_epochs = 70,   # 预训练轮数
    finetune_epochs = 15,   # 微调轮数
    num_epochs      = None, # 单阶段运行时覆盖上方轮数（None=使用上方默认值）
    batch_size      = 8,
    val_epochs      = 5,    # 每隔多少 epoch 做一次验证
    print_epochs    = 1,    # 每隔多少 epoch 打印一次日志
    warmup_epochs   = 3,    # 学习率热身轮数

    # ── 学习率 ───────────────────────────────────────────────────────────────
    start_lr    = 2e-4,  # 预训练初始学习率
    end_lr      = 1e-6,  # CosineAnnealing 最小学习率
    finetune_lr = 1e-5,  # 微调学习率

    # ── 损失权重 ─────────────────────────────────────────────────────────────
    w_motion   = 0.05,
    w_temporal = 0.05,
    w_physics  = 0.01,

    # ── 数据加载模式 ─────────────────────────────────────────────────────────
    # 'blade' = 光流感知加载器（推荐）   'gopro' = 原始GoPro加载器
    data_mode = 'blade',
)

# ===========================================================================
#  配置结束  ★ 以下内容无需修改 ★
# ===========================================================================

# ---------------------------------------------------------------------------
# Resolve output/data paths
# ---------------------------------------------------------------------------

output_dir  = args.output_dir
train_dir   = args.train_dir  or output_dir
val_dir     = args.val_dir    or output_dir
train_meta  = args.train_meta or os.path.join(output_dir, 'blade_train_list.txt')
val_meta    = args.val_meta   or os.path.join(output_dir, 'blade_val_list.txt')

# ---------------------------------------------------------------------------
# Step 1 – Dataset building (only when --video_dir is provided)
# ---------------------------------------------------------------------------

if args.video_dir is not None and args.stage in ('all', 'pretrain'):
    print('\n' + '='*60)
    print('STEP 1 – Building paired dataset from videos')
    print('='*60)
    # Import here so the rest of the script works even without video_processor
    from data.dataset_builder import build_full_dataset
    train_meta, val_meta = build_full_dataset(
        video_dir=args.video_dir,
        output_dir=output_dir,
        blur_dir=args.blur_dir,
        sample_fps=args.sample_fps,
        sharp_top_percent=args.sharp_pct,
        blade_roi=tuple(args.blade_roi) if args.blade_roi else None,
        n_blur_steps=args.n_blur_steps,
        exposure_factor=args.exposure_factor,
        val_ratio=args.val_ratio,
    )
    train_dir = output_dir
    val_dir   = output_dir
    print(f'Dataset ready.  Train: {train_meta}  Val: {val_meta}')

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

use_blade_mode = (args.data_mode == 'blade')
ps             = args.patch_size


def _make_loaders(train_dir, train_meta, val_dir, val_meta):
    """Build DataLoader objects for the current data paths."""
    if use_blade_mode:
        tr_ds = get_blade_training_data(
            train_dir, train_meta, {'patch_size': ps}, flow_dir=args.flow_dir)
        va_ds = get_blade_validation_data(
            val_dir,   val_meta,   {'patch_size': ps}, flow_dir=args.flow_dir)
    else:
        tr_ds = get_training_data(train_dir, train_meta, {'patch_size': ps})
        va_ds = get_validation_data(val_dir, val_meta,   {'patch_size': ps})

    tr_loader = DataLoader(dataset=tr_ds, batch_size=args.batch_size,
                           shuffle=True,  num_workers=4, drop_last=False,
                           pin_memory=True)
    va_loader = DataLoader(dataset=va_ds, batch_size=8,
                           shuffle=False, num_workers=4, drop_last=False,
                           pin_memory=True)
    return tr_loader, va_loader


def _make_criteria():
    return (
        losses.CharbonnierLoss(),
        losses.EdgeLoss(),
        losses.fftLoss(),
        MotionPriorLoss(weight=args.w_motion),
        PhysicsConstraintLoss(w_radial=args.w_physics, w_rpm=0.0, w_exp=0.0),
        TemporalConsistencyLoss(weight=args.w_temporal),
    )


# ---------------------------------------------------------------------------
# Core training function  (used for both pretrain and finetune stages)
# ---------------------------------------------------------------------------

def train_one_stage(stage_name: str,
                    model,
                    num_epochs: int,
                    train_loader,
                    val_loader,
                    model_dir: str,
                    log_path: str,
                    lr: float,
                    resume_ckpt: str = '',
                    pretrain_ckpt: str = '') -> str:
    """Train the model for *num_epochs* and return the path to the best checkpoint.

    Args:
        stage_name:    'pretrain' or 'finetune'.
        model:         Unwrapped (single-GPU) model.
        num_epochs:    How many epochs to run (0 → evaluation only).
        train_loader:  DataLoader for training.
        val_loader:    DataLoader for validation.
        model_dir:     Directory for checkpoints and log.
        log_path:      Path to append log lines.
        lr:            Initial learning rate.
        resume_ckpt:   Optional checkpoint to resume training from.
        pretrain_ckpt: Optional checkpoint to initialise weights from.

    Returns:
        Path of the best saved checkpoint.
    """
    utils.mkdir(model_dir)

    (criterion_char, criterion_edge, criterion_fft,
     criterion_motion, criterion_physics, criterion_temporal) = _make_criteria()

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    warmup_epochs = min(args.warmup_epochs, num_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(num_epochs - warmup_epochs, 1), eta_min=args.end_lr)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs,
        after_scheduler=scheduler_cosine)

    start_epoch = 1

    # Load pre-trained weights (does not restore optimiser state)
    if pretrain_ckpt and os.path.isfile(pretrain_ckpt):
        utils.load_checkpoint(model, pretrain_ckpt)
        print(f'  Loaded weights from: {pretrain_ckpt}')

    # Resume (restores weights + optimiser + epoch)
    if resume_ckpt and os.path.isfile(resume_ckpt):
        utils.load_checkpoint(model, resume_ckpt)
        start_epoch = utils.load_start_epoch(resume_ckpt) + 1
        utils.load_optim(optimizer, resume_ckpt)
        for _ in range(1, start_epoch):
            scheduler.step()
        print(f'  Resumed from epoch {start_epoch}  '
              f'lr={scheduler.get_lr()[0]:.2e}')

    # Wrap in DataParallel if multi-GPU
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        model_dp = nn.DataParallel(model, device_ids=device_ids)
        print(f'  Using {len(device_ids)} GPUs')
    else:
        model_dp = model

    with open(log_path, 'a+') as f:
        f.write(f'\n=== Stage: {stage_name}  epochs: {num_epochs} ===\n')

    best_psnr   = 0.0
    best_epoch  = 0
    best_ckpt   = os.path.join(model_dir, 'model_best.pth')

    print(f'\n===> [{stage_name}] Start Epoch {start_epoch}  End Epoch {num_epochs}')

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss  = 0.0
        model_dp.train()

        for data in train_loader:
            for param in model_dp.parameters():
                param.grad = None

            if use_blade_mode:
                target_          = data[0].cuda()
                input_           = data[1].cuda()
                motion_map_label = data[3].cuda()
                region_w         = data[4].cuda()
            else:
                target_          = data[0].cuda()
                input_           = data[1].cuda()
                motion_map_label = None
                region_w         = None

            target   = kornia.geometry.transform.build_pyramid(target_, 3)
            restored, restored_inter = model_dp(input_)

            loss_fft        = sum(criterion_fft(restored[s], target[s])  for s in range(3))
            loss_char       = sum(criterion_char(restored[s], target[s]) for s in range(3))
            loss_edge       = sum(criterion_edge(restored[s], target[s]) for s in range(3))
            loss_char_inter = sum(criterion_char(restored_inter[s], target[s]) for s in range(3))

            loss = (loss_char + loss_char_inter
                    + 0.01 * loss_fft
                    + 0.05 * loss_edge)

            if use_blade_mode and motion_map_label is not None:
                h_r, w_r   = restored[0].shape[2], restored[0].shape[3]
                input_crop = input_[:, :, :h_r, :w_r]
                pred_diff  = (restored[0] - input_crop).abs().mean(dim=1, keepdim=True)
                loss += criterion_motion(pred_diff, motion_map_label)
                loss += criterion_physics(motion_map=motion_map_label)

            if use_blade_mode and restored[0].shape[0] >= 2 and data[2] is not None:
                flow_t = data[2].cuda()
                n = restored[0].shape[0] // 2 * 2
                loss += criterion_temporal(
                    restored[0][:n:2], restored[0][1:n:2],
                    flow=flow_t[:n:2])

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ---- Logging ----
        if epoch % args.print_epochs == 0:
            elapsed = time.time() - epoch_start
            lr_now  = scheduler.get_lr()[0]
            msg = (f"[{stage_name}] Epoch {epoch}/{num_epochs}  "
                   f"Loss {epoch_loss:.4f}  LR {lr_now:.2e}  "
                   f"Time {elapsed:.1f}s")
            print(msg)
            with open(log_path, 'a+') as f:
                f.write(msg + '\n')

        # ---- Validation ----
        if epoch % args.val_epochs == 0 or epoch == num_epochs:
            model_dp.eval()
            psnr_vals = []
            with torch.no_grad():
                for data_val in val_loader:
                    target_v = data_val[0].cuda()
                    input_v  = data_val[1].cuda()
                    restored_v, _ = model_dp(input_v)
                    for res, tar in zip(restored_v[0], target_v):
                        psnr_vals.append(utils.torchPSNR(res, tar))

            psnr_val = torch.stack(psnr_vals).mean().item()
            print(f'  [Val] Epoch {epoch}  PSNR {psnr_val:.4f}  '
                  f'(best {best_psnr:.4f} @ ep {best_epoch})')
            with open(log_path, 'a+') as f:
                f.write(f'  [Val] Epoch {epoch}  PSNR {psnr_val:.4f}\n')

            if psnr_val > best_psnr:
                best_psnr  = psnr_val
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer':  optimizer.state_dict()},
                           best_ckpt)
                print(f'    --> Saved best (PSNR {best_psnr:.4f})')

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer':  optimizer.state_dict()},
                       os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

        scheduler.step()
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict()},
                   os.path.join(model_dir, 'model_latest.pth'))

    # ---- Final evaluation ----
    print(f'\n===> [{stage_name}] Final Evaluation')
    model_dp.eval()
    psnr_vals = []
    with torch.no_grad():
        for data_val in val_loader:
            target_v = data_val[0].cuda()
            input_v  = data_val[1].cuda()
            restored_v, _ = model_dp(input_v)
            for res, tar in zip(restored_v[0], target_v):
                psnr_vals.append(utils.torchPSNR(res, tar))

    final_psnr = torch.stack(psnr_vals).mean().item()
    msg = (f'[{stage_name}] Final PSNR {final_psnr:.4f}  '
           f'Best PSNR {best_psnr:.4f} @ epoch {best_epoch}')
    print(msg)
    with open(log_path, 'a+') as f:
        f.write(msg + '\n')

    return best_ckpt


# ---------------------------------------------------------------------------
# Build model (shared across stages)
# ---------------------------------------------------------------------------

model_restoration = myNet()
total_num, trainable_num = get_parameter_number(model_restoration)
print(f'Total params: {total_num}  |  Trainable: {trainable_num}')
model_restoration.cuda()

# ---------------------------------------------------------------------------
# Determine which stages to run and execute them
# ---------------------------------------------------------------------------

stage = args.stage

# In 'all' mode we always run pretrain then finetune.
# In single-stage mode we respect --num_epochs if given, otherwise use defaults.
def _resolve_epochs(stage_name: str) -> int:
    if args.num_epochs is not None and args.stage == stage_name:
        return args.num_epochs
    return args.pretrain_epochs if stage_name == 'pretrain' else args.finetune_epochs

pretrain_epochs = _resolve_epochs('pretrain')
finetune_epochs = _resolve_epochs('finetune')

train_loader, val_loader = _make_loaders(train_dir, train_meta, val_dir, val_meta)

print(f'\n===> Data mode: {args.data_mode}  |  Stage(s): {stage}')

if stage in ('all', 'pretrain'):
    print('\n' + '='*60)
    print('STEP 2 – Pre-training')
    print('='*60)
    pretrain_session = args.session + '_pretrain'
    pretrain_dir     = os.path.join(args.model_save_dir, args.dataset, pretrain_session)
    pretrain_log     = os.path.join(pretrain_dir, 'log.txt')
    utils.mkdir(pretrain_dir)
    with open(pretrain_log, 'a+') as f:
        f.write(f'Total: {total_num}  Trainable: {trainable_num}\n')

    best_pretrain_ckpt = train_one_stage(
        stage_name='pretrain',
        model=model_restoration,
        num_epochs=pretrain_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=pretrain_dir,
        log_path=pretrain_log,
        lr=args.start_lr,
        resume_ckpt=args.resume_ckpt,
        pretrain_ckpt='',
    )

if stage in ('all', 'finetune'):
    print('\n' + '='*60)
    print('STEP 3 – Fine-tuning')
    print('='*60)
    # In 'all' mode use the best pretrain checkpoint automatically;
    # in 'finetune' mode use --pretrain_ckpt if given.
    if stage == 'all':
        ft_pretrain_ckpt = best_pretrain_ckpt
    else:
        ft_pretrain_ckpt = args.pretrain_ckpt

    finetune_session = args.session + '_finetune'
    finetune_dir     = os.path.join(args.model_save_dir, args.dataset, finetune_session)
    finetune_log     = os.path.join(finetune_dir, 'log.txt')
    utils.mkdir(finetune_dir)
    with open(finetune_log, 'a+') as f:
        f.write(f'Total: {total_num}  Trainable: {trainable_num}\n')

    best_finetune_ckpt = train_one_stage(
        stage_name='finetune',
        model=model_restoration,
        num_epochs=finetune_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=finetune_dir,
        log_path=finetune_log,
        lr=args.finetune_lr,
        resume_ckpt='' if stage == 'all' else args.resume_ckpt,
        pretrain_ckpt=ft_pretrain_ckpt,
    )

if stage == 'eval':
    eval_ckpt = args.pretrain_ckpt
    print('\n' + '='*60)
    print('EVAL – Evaluation only')
    print('='*60)
    eval_session = args.session + '_eval'
    eval_dir     = os.path.join(args.model_save_dir, args.dataset, eval_session)
    eval_log     = os.path.join(eval_dir, 'log.txt')
    utils.mkdir(eval_dir)
    train_one_stage(
        stage_name='eval',
        model=model_restoration,
        num_epochs=0,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=eval_dir,
        log_path=eval_log,
        lr=args.finetune_lr,
        pretrain_ckpt=eval_ckpt,
    )
