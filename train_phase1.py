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
from losses.xyether_losses import CharbonnierLoss, MSSSIMLoss, ColorLuvLoss
from dataset.otf_dataset import AnimeOTFDataset
from dataset.download_dataset import download_and_extract_dataset
from benchmark.xyether_evaluator import XyetherEvaluator
from weights.transplant import prepare_xyether_base

class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.model = deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        m_params = dict(model.named_parameters())
        for name, ema_param in self.model.named_parameters():
            if name in m_params:
                ema_param.copy_(d * ema_param + (1.0 - d) * m_params[name])

    def state_dict(self):
        return self.model.state_dict()


def train_phase1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[PHASE 1] Initializing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Output directories & Google Drive integration
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    drive_backup_dir = None
    if os.path.exists("/content/drive/MyDrive"):
        drive_backup_dir = Path("/content/drive/MyDrive/Xyether_Checkpoints/Phase1")
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DRIVE] Google Drive backup active -> {drive_backup_dir}")

    # 2. Dataset preparation
    if not os.path.exists(args.data_dir) or len(list(Path(args.data_dir).rglob("*.png")) + list(Path(args.data_dir).rglob("*.jpg"))) < 10:
        print("[DATASET] Data folder empty or missing. Triggering auto-downloader...")
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
    print(f"[DATASET] Loaded {len(dataset)} virtual training pairs (Batch size: {args.batch_size}, Patch: {args.patch_size}x{args.patch_size})")

    # 3. Model & Checkpoint Initialization
    model = XyetherCompactNet().to(device)

    # Load transplanted base weights
    if not os.path.exists(args.pretrained_path):
        print(f"[MODEL] Pretrained path {args.pretrained_path} not found. Preparing transplanted base automatically...")
        args.pretrained_path = prepare_xyether_base(cache_dir=str(out_dir / "init_weights"))

    print(f"[MODEL] Loading transplanted weights from {args.pretrained_path}...")
    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    sd = ckpt.get('params_ema', ckpt.get('params', ckpt))
    model.load_state_dict(sd, strict=True)
    print("[MODEL] Base weights loaded successfully! Conv 18 is Zero-Initialized with Nearest Residual active.")

    # 4. Losses, Optimizer, Scheduler, EMA, AMP Scaler
    crit_charbonnier = CharbonnierLoss(loss_weight=1.0).to(device)
    crit_msssim = MSSSIMLoss(loss_weight=0.55).to(device)
    crit_color = ColorLuvLoss(loss_weight=0.80).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-6)
    ema = ModelEMA(model, decay=args.ema_decay)
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp and torch.cuda.is_available())
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
    evaluator = XyetherEvaluator(device=device)

    # 5. Training Loop
    print(f"[TRAIN] Starting Phase 1 (Target: {args.total_iters} iterations)...")
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

        optimizer.zero_grad()
        autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            sr_batch = model(lr_batch)
            loss_c = crit_charbonnier(sr_batch, hr_batch)
            loss_s = crit_msssim(sr_batch, hr_batch)
            loss_col = crit_color(sr_batch, hr_batch)
            total_loss = loss_c + loss_s + loss_col

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        step += 1

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            ips = step / max(1, elapsed)
            print(
                f"[Step {step:5d}/{args.total_iters}] "
                f"Loss: {total_loss.item():.4f} (Charb: {loss_c.item():.3f}, SSIM: {loss_s.item():.3f}, Color: {loss_col.item():.3f}) "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | Speed: {ips:.2f} it/s"
            )

        # Live Evaluation & Visual Sample
        if step % args.eval_every == 0 or step == args.total_iters:
            print(f"\n[BENCHMARK] Running evaluation at Step {step}...")
            report = evaluator.evaluate_model(ema.model)
            print(f"  • Overshoot / Halo:     {report['overshoot_halo']} (Target: 0.00)")
            print(f"  • Noise Residual:       {report['noise_residual_std']} std ({report['noise_elimination_pct']}% killed)")
            print(f"  • Line Depth:           {report['line_depth_luma']} luma (Target: <= 8.0)")
            print(f"  • Recommendation:       {report['recommendation']}\n")

            # Save Visual Comparison
            with torch.no_grad():
                sample_lr = lr_batch[0:1]
                sample_hr = hr_batch[0:1]
                sample_sr = ema.model(sample_lr)
                sample_nearest = F.interpolate(sample_lr, scale_factor=2, mode='nearest')

                grid = torch.cat([sample_nearest, sample_sr, sample_hr], dim=3)
                grid_np = (grid[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy())
                grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(sample_dir / f"step_{step:06d}_compare.png"), grid_bgr)

        # Checkpointing
        if step % args.save_every == 0 or step == args.total_iters:
            ckpt_path = out_dir / f"xyether_phase1_step_{step}.pth"
            save_payload = {
                'step': step,
                'params': model.state_dict(),
                'params_ema': ema.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(save_payload, str(ckpt_path))
            print(f"[CHECKPOINT] Saved checkpoint -> {ckpt_path}")

            if drive_backup_dir:
                try:
                    import shutil
                    shutil.copy(str(ckpt_path), str(drive_backup_dir / ckpt_path.name))
                    print(f"[DRIVE] Checkpoint backed up to Google Drive!")
                except Exception as e:
                    print(f"[WARN] Drive backup error: {e}")

            # Hugging Face Cloud Backup
            hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or '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'
            hf_repo = getattr(args, 'hf_repo', 'Thanawanit/Kaggle-Backup')
            if hf_token and hf_repo:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/Phase5_Xyether_Clone/{ckpt_path.name}",
                        repo_id=hf_repo,
                        repo_type="model"
                    )
                    print(f"[HF] Backed up to Hugging Face -> {hf_repo} (checkpoints/Phase5_Xyether_Clone/{ckpt_path.name})")
                except Exception as e:
                    print(f"[WARN] Hugging Face upload error: {e}")

    print(f"\n[PHASE 1 COMPLETE] Fidelity base training finished! Final model: {out_dir / f'xyether_phase1_step_{args.total_iters}.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Xyether Phase 1 (Fidelity Base)")
    parser.add_argument("--data_dir", type=str, default="/content/dataset", help="Dataset folder")
    parser.add_argument("--pretrained_path", type=str, default="./weights/xyether_transplanted_base.pth", help="Base checkpoint")
    parser.add_argument("--output_dir", type=str, default="./experiments/phase1", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for L4/A100, 8 for T4)")
    parser.add_argument("--patch_size", type=int, default=128, help="HR Patch size")
    parser.add_argument("--total_iters", type=int, default=12000, help="Total iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="EMA decay")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use FP16 AMP")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log interval")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval interval")
    parser.add_argument("--save_every", type=int, default=2000, help="Save interval")
    args = parser.parse_args()
    train_phase1(args)
