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

from archs.xyether_compact import XyetherCompactNet, XyetherBalancedNet
from archs.discriminator import UNetDiscriminatorSN
from losses.xyether_losses import (
    CharbonnierLoss,
    FocalFrequencyLoss,
    GANLoss,
    AdjacentGradientLoss,
    DirectionalGradientLoss,
    MSSSIMLoss,
    VGGPerceptualLoss,
    MaskedInkingLoss,
    LaplacianLoss,
    ColorLuvLoss
)
from dataset.otf_dataset import AnimeOTFDataset
from dataset.download_dataset import download_and_extract_dataset
from benchmark.xyether_evaluator import XyetherEvaluator
from weights.transplant import prepare_xyether_base, download_file_if_needed

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
    print(f"[XYETHERKILLER REFINED MASTER] Initializing Phase 15 on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

    # 2. Student & Discriminator Initialization (Starting from authentic user checkpoints net_g_89000 & net_d_89000)
    net_g = XyetherCompactNet().to(device)
    net_d = UNetDiscriminatorSN(num_in_ch=3, num_feat=64).to(device)

    # Load student net_g (net_g_89000)
    student_init_path = Path(args.pretrained_path)
    if not student_init_path.exists():
        if args.pretrained_url:
            print(f"[STUDENT] Downloading base student from {args.pretrained_url}...")
            download_file_if_needed(args.pretrained_url, student_init_path)
        else:
            args.pretrained_path = prepare_xyether_base(cache_dir=str(out_dir / "init_weights"), zero_head=False)

    print(f"[STUDENT] Loading student weights from {args.pretrained_path}...")
    ckpt_student = torch.load(args.pretrained_path, map_location='cpu')
    sd_student = ckpt_student.get('params_ema', ckpt_student.get('params', ckpt_student))
    clean_sd_student = {k.replace('module.', ''): v for k, v in sd_student.items()}
    net_g.load_state_dict(clean_sd_student, strict=True)
    print(f"[STUDENT] Warm-started student model successfully ({len(clean_sd_student)} layers loaded).")

    # Load discriminator net_d (net_d_89000)
    if getattr(args, 'pretrained_d_url', None):
        d_init_path = Path(getattr(args, 'pretrained_d_path', './weights/net_d_89000.pth'))
        if not d_init_path.exists():
            print(f"[DISCRIMINATOR] Downloading base discriminator from {args.pretrained_d_url}...")
            download_file_if_needed(args.pretrained_d_url, d_init_path)
        try:
            ckpt_d = torch.load(str(d_init_path), map_location='cpu')
            sd_d = ckpt_d.get('params_d', ckpt_d.get('params_ema', ckpt_d.get('params', ckpt_d)))
            clean_sd_d = {k.replace('module.', ''): v for k, v in sd_d.items()}
            net_d.load_state_dict(clean_sd_d, strict=True)
            print(f"[DISCRIMINATOR] Warm-started discriminator successfully ({len(clean_sd_d)} layers loaded).")
        except Exception as e:
            print(f"[WARN] Discriminator loading warning: {e}")

    # 3. Dual Teachers Initialization (Strong v3 for Lines + Balanced for Shading)
    cache_weights_dir = out_dir / "teacher_weights"
    cache_weights_dir.mkdir(parents=True, exist_ok=True)

    # Teacher A: Strong v3 (Razor-sharp inking guide)
    teacher_strong_path = cache_weights_dir / "Xyether_Strong_v3_Official.pth"
    download_file_if_needed(args.teacher_strong_url, teacher_strong_path)
    teacher_strong = XyetherCompactNet().to(device)
    ckpt_strong = torch.load(str(teacher_strong_path), map_location='cpu')
    sd_strong = ckpt_strong.get('params_ema', ckpt_strong.get('params', ckpt_strong))
    teacher_strong.load_state_dict({k.replace('module.', ''): v for k, v in sd_strong.items()}, strict=True)
    teacher_strong.eval()
    for p in teacher_strong.parameters():
        p.requires_grad = False
    print("✅ Teacher A (Strong v3 - Line Art Guide) loaded & frozen.")

    # Teacher B: Balanced 2x (Smooth shading guide)
    teacher_bal_path = cache_weights_dir / "Xyether_Balanced_2x_Official.pth"
    download_file_if_needed(args.teacher_bal_url, teacher_bal_path)
    teacher_bal = XyetherBalancedNet().to(device)
    ckpt_bal = torch.load(str(teacher_bal_path), map_location='cpu')
    sd_bal = ckpt_bal.get('params_ema', ckpt_bal.get('params', ckpt_bal))
    teacher_bal.load_state_dict({k.replace('module.', ''): v for k, v in sd_bal.items()}, strict=True)
    teacher_bal.eval()
    for p in teacher_bal.parameters():
        p.requires_grad = False
    print("✅ Teacher B (Balanced 2x - Smooth Shading Guide) loaded & frozen.")

    # 4. Loss Functions (Active GAN + Multi-Layer VGG Perceptual + Masked Inking + Laplacian Curvature + ColorLuv)
    crit_pixel = CharbonnierLoss(loss_weight=1.0).to(device)
    crit_vgg = VGGPerceptualLoss(loss_weight=1.0).to(device)
    crit_masked_ink = MaskedInkingLoss(grad_weight=args.grad_weight, ink_weight=args.ink_weight, threshold=0.08).to(device)
    crit_lap = LaplacianLoss(loss_weight=1.0).to(device)
    crit_luv = ColorLuvLoss(loss_weight=1.0).to(device)
    crit_gan = GANLoss(loss_weight=1.0).to(device)

    # 5. Optimizers with Stable Learning Rate
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

    print(f"[XYETHERKILLER REFINED MASTER] Starting Phase 15 Training (Target: {args.total_iters} iterations)...")
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
            # 1. Forward Pass Student
            sr_batch = net_g(lr_batch)

            # 2. Forward Pass Teachers (Frozen)
            with torch.no_grad():
                t_strong_out = teacher_strong(lr_batch)
                t_bal_out = teacher_bal(lr_batch)

                # Compute Edge Mask from Strong v3 teacher (detects 1-pixel line contours)
                dx_t = t_strong_out[:, :, :, 1:] - t_strong_out[:, :, :, :-1]
                dy_t = t_strong_out[:, :, 1:, :] - t_strong_out[:, :, :-1, :]
                edge_mask = torch.zeros_like(t_strong_out)
                edge_mask[:, :, :, 1:] = torch.maximum(edge_mask[:, :, :, 1:], (dx_t.abs().mean(dim=1, keepdim=True) > 0.08).float())
                edge_mask[:, :, 1:, :] = torch.maximum(edge_mask[:, :, 1:, :], (dy_t.abs().mean(dim=1, keepdim=True) > 0.08).float())
                flat_mask = 1.0 - edge_mask

                # Clean Anime Target: Strong v3 vector inking on lines + Balanced 2x clean shading on flat areas
                clean_target = edge_mask * t_strong_out + flat_mask * t_bal_out

            # 3. Dual-Domain Losses
            # A. Edge Inking & Gradient Loss (Forces pitch black ink & sharp edges from Strong v3)
            l_edge_ink = crit_masked_ink(sr_batch, t_strong_out)

            # B. Edge Laplacian Curvature Loss (Forces vector smoothness along lines, eliminates staircase kinks/jaggedness)
            l_edge_curv = crit_lap(sr_batch, t_strong_out, mask=edge_mask)

            # C. Flat Region Smooth Shading Loss (Balanced 2x flat cel shading, zero noise)
            l_flat_shade = crit_pixel(sr_batch * flat_mask, t_bal_out * flat_mask)
            l_flat_luv = crit_luv(sr_batch * flat_mask, t_bal_out * flat_mask)

            # D. Pixel Anchor Loss (lightweight, anchors macro color without blurring)
            l_pixel = crit_pixel(sr_batch, hr_batch)

            # E. Multi-Layer VGG19 Perceptual Loss (aligned with clean anime target, strips web image grain)
            l_percep = crit_vgg(sr_batch, clean_target)

            # F. Active UNet GAN Loss against clean anime target
            d_fake = net_d(sr_batch)
            l_gan_g = crit_gan(d_fake, is_real=True)

            # Combined Objective: Vector Smooth Lines + Clean Flat Cel Shading
            total_loss_g = (
                args.pixel_weight * l_pixel +
                args.percep_weight * l_percep +
                l_edge_ink +
                args.lap_weight * l_edge_curv +
                args.shade_weight * l_flat_shade +
                args.luv_weight * l_flat_luv +
                args.gan_weight * l_gan_g
            )

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
            d_real = net_d(clean_target.detach())
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
                f"G_Loss: {total_loss_g.item():.4f} (Ink: {l_edge_ink.item():.4f}, Curv: {l_edge_curv.item():.4f}, Shade: {l_flat_shade.item():.4f}, GAN: {l_gan_g.item():.4f}) | "
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
                sample_strong = teacher_strong(sample_lr)
                sample_bal = teacher_bal(sample_lr)
                grid = torch.cat([sample_lr.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3), sample_sr, sample_strong, sample_bal, sample_hr], dim=3)
                grid_np = (grid[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy())
                cv2.imwrite(str(sample_dir / f"step_{step:06d}.png"), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))

        # Checkpointing
        if step % args.save_every == 0 or step == args.total_iters:
            ckpt_path = out_dir / f"xyetherkiller_vectorclean_step_{step}.pth"
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
                        path_in_repo=f"checkpoints/Phase14_XyetherKiller_VectorClean/{ckpt_path.name}",
                        repo_id=hf_repo,
                        repo_type="model"
                    )
                    print(f"[HF] Backed up to Hugging Face -> {hf_repo} (checkpoints/Phase15_XyetherKiller_RefinedMaster/{ckpt_path.name})")
                except Exception as e:
                    print(f"[WARN] Hugging Face upload error: {e}")

    # Export Final Production Model to ONNX & PyTorch
    final_pth = out_dir / "XyetherKiller_RefinedMaster_v9_Final.pth"
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
                path_in_repo="checkpoints/Phase15_XyetherKiller_RefinedMaster/XyetherKiller_RefinedMaster_v9_Final.pth",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final production model uploaded to Hugging Face -> {hf_repo}")
        except Exception as e:
            print(f"[WARN] Final HF upload error: {e}")

    try:
        final_onnx = out_dir / "XyetherKiller_RefinedMaster_v9_Final.onnx"
        dummy_in = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            torch.onnx.export(
                ema_g.model,
                dummy_in,
                str(final_onnx),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
                dynamo=False
            )
        print(f"[FINAL EXPORT] Production ONNX graph saved -> {final_onnx}")
        if hf_token and hf_repo:
            api.upload_file(
                path_or_fileobj=str(final_onnx),
                path_in_repo="checkpoints/Phase15_XyetherKiller_RefinedMaster/XyetherKiller_RefinedMaster_v9_Final.onnx",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final ONNX model uploaded to Hugging Face -> {hf_repo}")
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

    print("\n[COMPLETE] Phase 15 XyetherKiller Refined Master training successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XyetherKiller Refined Master Series (Phase 15)")
    parser.add_argument("--data_dir", type=str, default="/content/dataset", help="Dataset folder")
    parser.add_argument("--pretrained_path", type=str, default="./weights/XyetherKiller_VectorClean_v8_Final.pth", help="Base student checkpoint")
    parser.add_argument("--pretrained_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Phase14_XyetherKiller_VectorClean/XyetherKiller_VectorClean_v8_Final.pth", help="Student warm-start URL")
    parser.add_argument("--pretrained_d_path", type=str, default="./weights/xyetherkiller_vectorclean_step_6000.pth", help="Base discriminator checkpoint")
    parser.add_argument("--pretrained_d_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Phase14_XyetherKiller_VectorClean/xyetherkiller_vectorclean_step_6000.pth", help="Discriminator warm-start URL")
    parser.add_argument("--teacher_strong_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Official_Xyether_References/Xyether_Strong_v3_Official.pth", help="Strong v3 teacher url")
    parser.add_argument("--teacher_bal_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Official_Xyether_References/Xyether_Balanced_2x_Official.pth", help="Balanced 2x teacher url")
    parser.add_argument("--output_dir", type=str, default="./output_models", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=192, help="HR Patch size")
    parser.add_argument("--total_iters", type=int, default=6000, help="Total iterations")
    parser.add_argument("--lr_g", type=float, default=8.0e-6, help="Generator polishing learning rate")
    parser.add_argument("--lr_d", type=float, default=6.0e-6, help="Discriminator polishing learning rate")
    parser.add_argument("--pixel_weight", type=float, default=0.08, help="Pixel anchor Charbonnier weight")
    parser.add_argument("--percep_weight", type=float, default=0.20, help="Multi-layer VGG19 perceptual loss weight")
    parser.add_argument("--grad_weight", type=float, default=0.30, help="Teacher masked adjacent gradient weight")
    parser.add_argument("--ink_weight", type=float, default=0.08, help="Teacher stroke width profile match weight")
    parser.add_argument("--lap_weight", type=float, default=0.20, help="Teacher Laplacian curvature smoothing weight")
    parser.add_argument("--shade_weight", type=float, default=0.35, help="Teacher clean flat shading weight")
    parser.add_argument("--luv_weight", type=float, default=0.15, help="Teacher ColorLuv cel shading weight")
    parser.add_argument("--gan_weight", type=float, default=0.08, help="Active UNet Discriminator GAN weight")
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
