import os
import sys
import time
import json
import argparse
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from archs.xyether_compact import XyetherCompactNet
from archs.discriminator import UNetDiscriminatorSN
from losses.xyether_losses import CharbonnierLoss, MSSSIMLoss, ColorLuvLoss, FocalFrequencyLoss, GANLoss
from dataset.otf_dataset import AnimeOTFDataset
from dataset.download_dataset import download_and_extract_dataset
from benchmark.xyether_evaluator import XyetherEvaluator
from train_phase1 import ModelEMA

def train_phase2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[PHASE 2] Initializing Sharp Polish on device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    drive_backup_dir = None
    if os.path.exists("/content/drive/MyDrive"):
        drive_backup_dir = Path("/content/drive/MyDrive/Xyether_Checkpoints/Phase2")
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DRIVE] Google Drive backup active -> {drive_backup_dir}")

    # Dataset
    if not os.path.exists(args.data_dir):
        download_and_extract_dataset(args.data_dir)

    dataset = AnimeOTFDataset(data_dir=args.data_dir, patch_size=args.patch_size, repeat=20)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    # 1. Models: Generator & Discriminator
    net_g = XyetherCompactNet().to(device)
    net_d = UNetDiscriminatorSN(num_in_ch=3, num_feat=64).to(device)

    print(f"[MODEL] Loading Phase 1 checkpoint from {args.pretrained_g}...")
    ckpt_g = torch.load(args.pretrained_g, map_location='cpu')
    sd_g = ckpt_g.get('params_ema', ckpt_g.get('params', ckpt_g))
    net_g.load_state_dict(sd_g, strict=True)
    print("[MODEL] Phase 1 Base weights successfully loaded!")

    # 2. Losses
    crit_charb = CharbonnierLoss(loss_weight=0.50).to(device)
    crit_msssim = MSSSIMLoss(loss_weight=0.50).to(device)
    crit_color = ColorLuvLoss(loss_weight=0.70).to(device)
    crit_ffl = FocalFrequencyLoss(loss_weight=args.ffl_weight, alpha=1.0).to(device)
    crit_gan = GANLoss(loss_weight=args.gan_weight).to(device)

    # 3. Optimizers with Low Learning Rate (3e-5) to polish Reconstruction Head
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=args.lr_g, betas=(0.9, 0.999), weight_decay=0.0)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=args.lr_d, betas=(0.9, 0.999), weight_decay=0.0)

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=args.total_iters, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=args.total_iters, eta_min=1e-6)

    ema_g = ModelEMA(net_g, decay=args.ema_decay)
    if hasattr(torch.amp, 'GradScaler'):
        scaler_g = torch.amp.GradScaler('cuda', enabled=args.use_amp and torch.cuda.is_available())
        scaler_d = torch.amp.GradScaler('cuda', enabled=args.use_amp and torch.cuda.is_available())
    else:
        scaler_g = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
        scaler_d = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
    evaluator = XyetherEvaluator(device=device)

    print(f"[TRAIN] Starting Phase 2 Polish (Target: {args.total_iters} iterations)...")
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < args.total_iters:
        try:
            lr_batch, hr_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            lr_batch, hr_batch = next(data_iter)

        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ----------------- Train Generator -----------------
        optim_g.zero_grad()
        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            sr_batch = net_g(lr_batch)

            # Spatial & Frequency Losses
            l_pix = crit_charb(sr_batch, hr_batch)
            l_ssim = crit_msssim(sr_batch, hr_batch)
            l_col = crit_color(sr_batch, hr_batch)
            l_ffl = crit_ffl(sr_batch, hr_batch)

            # GAN Generator Loss (D-UNet)
            d_fake = net_d(sr_batch)
            l_gan_g = crit_gan(d_fake, is_real=True)

            total_loss_g = l_pix + l_ssim + l_col + l_ffl + l_gan_g

        scaler_g.scale(total_loss_g).backward()
        scaler_g.step(optim_g)
        scaler_g.update()
        sched_g.step()
        ema_g.update(net_g)

        # ----------------- Train Discriminator -----------------
        optim_d.zero_grad()
        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            d_real = net_d(hr_batch)
            d_fake_detached = net_d(sr_batch.detach())

            l_d_real = crit_gan(d_real, is_real=True)
            l_d_fake = crit_gan(d_fake_detached, is_real=False)
            total_loss_d = (l_d_real + l_d_fake) * 0.5

        scaler_d.scale(total_loss_d).backward()
        scaler_d.step(optim_d)
        scaler_d.update()
        sched_d.step()

        step += 1

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            ips = step / max(1, elapsed)
            print(
                f"[Step {step:5d}/{args.total_iters}] "
                f"G_Loss: {total_loss_g.item():.4f} (FFL: {l_ffl.item():.4f}, GAN: {l_gan_g.item():.4f}) | "
                f"D_Loss: {total_loss_d.item():.4f} | Speed: {ips:.2f} it/s"
            )

        if step % args.eval_every == 0 or step == args.total_iters:
            print(f"\n[BENCHMARK] Phase 2 Evaluation at Step {step}...")
            report = evaluator.evaluate_model(ema_g.model)
            print(f"  • Overshoot / Halo:     {report['overshoot_halo']} (Target: 0.00)")
            print(f"  • Noise Residual:       {report['noise_residual_std']} std ({report['noise_elimination_pct']}% killed)")
            print(f"  • Line Depth:           {report['line_depth_luma']} luma (Target: <= 8.0)")
            print(f"  • Sharpness (LapVar):   {report['sharpness_lapvar']} (Target: 18k - 25k)")
            print(f"  • Recommendation:       {report['recommendation']}\n")

            with torch.no_grad():
                sample_lr = lr_batch[0:1]
                sample_hr = hr_batch[0:1]
                sample_sr = ema_g.model(sample_lr)
                sample_nearest = F.interpolate(sample_lr, scale_factor=2, mode='nearest')
                grid = torch.cat([sample_nearest, sample_sr, sample_hr], dim=3)
                grid_np = (grid[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy())
                cv2.imwrite(str(sample_dir / f"phase2_step_{step:06d}.png"), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))

        if step % args.save_every == 0 or step == args.total_iters:
            ckpt_path = out_dir / f"xyether_phase2_step_{step}.pth"
            torch.save({
                'step': step,
                'params': net_g.state_dict(),
                'params_ema': ema_g.state_dict(),
                'params_d': net_d.state_dict()
            }, str(ckpt_path))
            print(f"[CHECKPOINT] Saved Phase 2 checkpoint -> {ckpt_path}")

            if drive_backup_dir:
                try:
                    import shutil
                    shutil.copy(str(ckpt_path), str(drive_backup_dir / ckpt_path.name))
                    print(f"[DRIVE] Checkpoint backed up to Google Drive!")
                except Exception as e:
                    print(f"[WARN] Drive backup error: {e}")

    # Export Final Production Model to ONNX & PyTorch
    final_pth = out_dir / "Xyether_Strong_v3_Final.pth"
    torch.save({"params_ema": ema_g.state_dict()}, str(final_pth))
    print(f"\n[FINAL EXPORT] Production PyTorch weights saved -> {final_pth}")

    try:
        final_onnx = out_dir / "Xyether_Strong_v3_Final.onnx"
        dummy_in = torch.randn(1, 3, 256, 256, device=device)
        torch.onnx.export(
            ema_g.model,
            dummy_in,
            str(final_onnx),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
            opset_version=14
        )
        print(f"[FINAL EXPORT] Production ONNX graph saved -> {final_onnx}")
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

    print("\n[PHASE 2 COMPLETE] Sharp Polish training successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Xyether Phase 2 (Sharp Polish)")
    parser.add_argument("--data_dir", type=str, default="/content/dataset", help="Dataset folder")
    parser.add_argument("--pretrained_g", type=str, required=True, help="Phase 1 EMA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./experiments/phase2", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size (12 for L4/A100)")
    parser.add_argument("--patch_size", type=int, default=128, help="HR Patch size")
    parser.add_argument("--total_iters", type=int, default=6000, help="Total iterations")
    parser.add_argument("--lr_g", type=float, default=3e-5, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=3e-5, help="Discriminator learning rate")
    parser.add_argument("--ffl_weight", type=float, default=0.05, help="FocalFrequencyLoss weight")
    parser.add_argument("--gan_weight", type=float, default=0.10, help="GAN loss weight")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="EMA decay")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use FP16 AMP")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log interval")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval interval")
    parser.add_argument("--save_every", type=int, default=2000, help="Save interval")
    args = parser.parse_args()
    train_phase2(args)
