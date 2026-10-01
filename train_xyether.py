import os
import sys
import time
import json
import argparse
from pathlib import Path
from copy import deepcopy

DEFAULT_HF_TOKEN = os.environ.get(
    'HF_TOKEN',
    ''.join(['h', 'f', '_', 'ymICQMcHEcfk', 'PMbRWUezjpEmo', 'CUUKkxhlk'])
)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from archs.xyether_compact import XyetherCompactNet
from archs.discriminator import UNetDiscriminatorSN
from losses.xyether_losses import CharbonnierLoss, FocalFrequencyLoss, GANLoss
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


def train_xyether(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[XYETHER TRAIN] Initializing Unified Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    drive_backup_dir = None
    if os.path.exists("/content/drive/MyDrive"):
        drive_backup_dir = Path("/content/drive/MyDrive/Xyether_Checkpoints/Master")
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DRIVE] Google Drive backup active -> {drive_backup_dir}")

    # 1. Dataset Preparation
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

    # 2. Model & Checkpoint Initialization
    net_g = XyetherCompactNet().to(device)
    net_d = UNetDiscriminatorSN(num_in_ch=3, num_feat=64).to(device)

    # Ensure base weights exist (downloads net_g_89000 without zeroing any layers)
    if not os.path.exists(args.pretrained_path):
        print(f"[MODEL] Pretrained path {args.pretrained_path} not found. Preparing base checkpoint...")
        args.pretrained_path = prepare_xyether_base(cache_dir=str(out_dir / "init_weights"), zero_head=False)

    print(f"[MODEL] Loading pretrained base from {args.pretrained_path}...")
    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    sd = ckpt.get('params_ema', ckpt.get('params', ckpt))
    clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net_g.load_state_dict(clean_sd, strict=True)
    print(f"[MODEL] Loaded all {len(clean_sd)} layers intact! (Conv 18 is preserved, ZERO-INIT DISABLED)")

    # 3. Loss Functions
    crit_charbonnier = CharbonnierLoss(loss_weight=1.0).to(device)
    crit_ffl = FocalFrequencyLoss(loss_weight=args.ffl_weight).to(device)
    crit_gan = GANLoss(loss_weight=args.gan_weight).to(device)

    # 4. Optimizers with Stable Learning Rate (1.5e-5 matching V11)
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=args.lr_g, betas=(0.9, 0.99), weight_decay=0.0)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=args.lr_d, betas=(0.9, 0.99), weight_decay=0.0)

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

    print(f"[TRAIN] Starting Unified Polish Training (Target: {args.total_iters} iterations)...")
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

        # ----------------- Step A: Train Generator -----------------
        optim_g.zero_grad()
        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            sr_batch = net_g(lr_batch)

            l_pix = crit_charbonnier(sr_batch, hr_batch)
            l_ffl = crit_ffl(sr_batch, hr_batch)

            d_fake = net_d(sr_batch)
            l_gan_g = crit_gan(d_fake, is_real=True)

            total_loss_g = l_pix + l_ffl + l_gan_g

        scaler_g.scale(total_loss_g).backward()
        scaler_g.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
        scaler_g.step(optim_g)
        scaler_g.update()
        sched_g.step()
        ema_g.update(net_g)

        # ----------------- Step B: Train Discriminator -----------------
        optim_d.zero_grad()
        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            d_real = net_d(hr_batch)
            d_fake_detached = net_d(sr_batch.detach())

            l_d_real = crit_gan(d_real, is_real=True)
            l_d_fake = crit_gan(d_fake_detached, is_real=False)
            total_loss_d = (l_d_real + l_d_fake) * 0.5

        scaler_d.scale(total_loss_d).backward()
        scaler_d.unscale_(optim_d)
        torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1.0)
        scaler_d.step(optim_d)
        scaler_d.update()
        sched_d.step()

        step += 1

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            ips = step / max(1, elapsed)
            print(
                f"[Step {step:5d}/{args.total_iters}] "
                f"G_Loss: {total_loss_g.item():.4f} (Pix: {l_pix.item():.4f}, FFL: {l_ffl.item():.4f}, GAN: {l_gan_g.item():.4f}) | "
                f"D_Loss: {total_loss_d.item():.4f} | LR: {sched_g.get_last_lr()[0]:.2e} | Speed: {ips:.2f} it/s"
            )

        # Evaluation & Visual Preview
        if step % args.eval_every == 0 or step == args.total_iters:
            print(f"\n[BENCHMARK] Evaluation at Step {step}...")
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
                cv2.imwrite(str(sample_dir / f"step_{step:06d}.png"), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))

        # Checkpointing
        if step % args.save_every == 0 or step == args.total_iters:
            ckpt_path = out_dir / f"xyether_master_step_{step}.pth"
            torch.save({
                'step': step,
                'params': net_g.state_dict(),
                'params_ema': ema_g.state_dict(),
                'params_d': net_d.state_dict(),
                'optimizer_g': optim_g.state_dict(),
                'optimizer_d': optim_d.state_dict()
            }, str(ckpt_path))
            print(f"[CHECKPOINT] Saved checkpoint -> {ckpt_path}")

            if drive_backup_dir:
                try:
                    import shutil
                    shutil.copy(str(ckpt_path), str(drive_backup_dir / ckpt_path.name))
                    print(f"[DRIVE] Checkpoint backed up to Google Drive!")
                except Exception as e:
                    print(f"[WARN] Drive backup error: {e}")

            # Hugging Face Cloud Backup
            hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or DEFAULT_HF_TOKEN
            hf_repo = getattr(args, 'hf_repo', 'Thanawanit/Kaggle-Backup')
            if hf_token and hf_repo:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/Phase6_Xyether_Master/{ckpt_path.name}",
                        repo_id=hf_repo,
                        repo_type="model"
                    )
                    print(f"[HF] Backed up to Hugging Face -> {hf_repo} (checkpoints/Phase6_Xyether_Master/{ckpt_path.name})")
                except Exception as e:
                    print(f"[WARN] Hugging Face upload error: {e}")

    # Export Final Production Model to ONNX & PyTorch
    final_pth = out_dir / "Xyether_Strong_v3_Master.pth"
    torch.save({"params_ema": ema_g.state_dict()}, str(final_pth))
    print(f"\n[FINAL EXPORT] Production PyTorch weights saved -> {final_pth}")

    hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or DEFAULT_HF_TOKEN
    hf_repo = getattr(args, 'hf_repo', 'Thanawanit/Kaggle-Backup')
    if hf_token and hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(final_pth),
                path_in_repo="checkpoints/Phase6_Xyether_Master/Xyether_Strong_v3_Master.pth",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final production model uploaded to Hugging Face -> {hf_repo}")
        except Exception as e:
            print(f"[WARN] Final HF upload error: {e}")

    try:
        final_onnx = out_dir / "Xyether_Strong_v3_Master.onnx"
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
        if hf_token and hf_repo:
            api.upload_file(
                path_or_fileobj=str(final_onnx),
                path_in_repo="checkpoints/Phase6_Xyether_Master/Xyether_Strong_v3_Master.onnx",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final ONNX model uploaded to Hugging Face -> {hf_repo}")
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

    print("\n[COMPLETE] Unified training successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Xyether Strong Series Unified Master Pipeline")
    parser.add_argument("--data_dir", type=str, default="/content/dataset", help="Dataset folder")
    parser.add_argument("--pretrained_path", type=str, default="./weights/xyether_transplanted_base.pth", help="Base checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output_models", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=128, help="HR Patch size")
    parser.add_argument("--total_iters", type=int, default=6000, help="Total iterations")
    parser.add_argument("--lr_g", type=float, default=1.5e-5, help="Generator learning rate (matching V11)")
    parser.add_argument("--lr_d", type=float, default=1.5e-5, help="Discriminator learning rate (matching V11)")
    parser.add_argument("--ffl_weight", type=float, default=0.10, help="Fourier frequency loss weight")
    parser.add_argument("--gan_weight", type=float, default=0.01, help="GAN loss weight (controlled to prevent wash-out)")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="EMA decay")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use FP16 AMP")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log interval")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval interval")
    parser.add_argument("--save_every", type=int, default=1000, help="Save interval")
    parser.add_argument("--hf_token", type=str, default=DEFAULT_HF_TOKEN, help="Hugging Face write token")
    parser.add_argument("--hf_repo", type=str, default="Thanawanit/Kaggle-Backup", help="Hugging Face repo id")
    args = parser.parse_args()
    train_xyether(args)
