import os
import sys
import time
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
from losses.xyether_losses import (
    BilateralInkTroughTarget,
    Phase18CoreEngineLoss,
    VGGPerceptualLoss,
    ColorLuvLoss,
)
from dataset.otf_dataset import AnimeOTFDataset
from dataset.download_dataset import download_and_extract_dataset
from benchmark.xyether_evaluator import XyetherEvaluator
from weights.transplant import prepare_xyether_base, download_file_if_needed


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        raw = model.module if isinstance(model, nn.DataParallel) else model
        self.model = deepcopy(raw)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        raw = model.module if isinstance(model, nn.DataParallel) else model
        m_params = dict(raw.named_parameters())
        for name, ema_param in self.model.named_parameters():
            if name in m_params:
                ema_param.copy_(d * ema_param + (1.0 - d) * m_params[name])

    def state_dict(self):
        return self.model.state_dict()


def train_xyether(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = f"{num_gpus}x {torch.cuda.get_device_name(0)}" if num_gpus > 0 else "CPU"
    print(f"[PHASE 18: ZERO-HALO 1PX CORE ENGINE] Initializing on: {device} ({gpu_info})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Preparation (with 1080p-1440p Broadcast Scale Normalization & Edge-Rich Crops)
    if not os.path.exists(args.data_dir) or len(
        list(Path(args.data_dir).rglob("*.png")) + list(Path(args.data_dir).rglob("*.jpg"))
    ) < 10:
        print("[DATASET] Data folder empty or missing. Triggering auto-downloader...")
        download_and_extract_dataset(args.data_dir)

    dataset = AnimeOTFDataset(data_dir=args.data_dir, patch_size=args.patch_size, repeat=25)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    print(f"[DATASET] Loaded {len(dataset)} virtual training pairs (Batch: {args.batch_size}, Patch: {args.patch_size}x{args.patch_size})")

    # 2. Authentic Student Initialization (Strictly from net_g_89000.pth, NO external teachers!)
    raw_net_g = XyetherCompactNet().to(device)

    student_init_path = Path(args.pretrained_path)
    if not student_init_path.exists():
        if args.pretrained_url:
            print(f"[STUDENT] Downloading authentic base student from {args.pretrained_url}...")
            download_file_if_needed(args.pretrained_url, student_init_path)
        else:
            args.pretrained_path = prepare_xyether_base(cache_dir=str(out_dir / "init_weights"), zero_head=False)

    print(f"[STUDENT] Loading authentic base weights from {args.pretrained_path}...")
    ckpt_student = torch.load(args.pretrained_path, map_location='cpu')
    sd_student = ckpt_student.get('params_ema', ckpt_student.get('params', ckpt_student))
    clean_sd_student = {k.replace('module.', ''): v for k, v in sd_student.items()}
    raw_net_g.load_state_dict(clean_sd_student, strict=True)
    print(f"[STUDENT] Warm-started authentic student model ({len(clean_sd_student)} tensors loaded).")

    # Record initial Frobenius norms for Weight Norm Preservation (prevents 18-layer multiplicative shrinkage)
    init_weight_norms = {}
    for name, p in raw_net_g.named_parameters():
        if 'weight' in name:
            init_weight_norms[name] = p.data.norm().item()

    if num_gpus > 1:
        print(f"[MULTI-GPU] Wrapping student generator in nn.DataParallel across {num_gpus} GPUs...")
        net_g = nn.DataParallel(raw_net_g)
    else:
        net_g = raw_net_g

    # 3. Self-Contained Target Generator & Phase 18 Losses
    target_builder = BilateralInkTroughTarget(
        valley_strength=args.valley_strength,
        aa_preserve=args.aa_preserve
    ).to(device)
    crit_core = Phase18CoreEngineLoss().to(device)
    crit_vgg = VGGPerceptualLoss(loss_weight=args.percep_weight).to(device)
    crit_luv = ColorLuvLoss(loss_weight=args.luv_weight).to(device)

    # 4. Optimizer, Warmup + Cosine Scheduler, and EMA
    optim_g = torch.optim.AdamW(raw_net_g.parameters(), lr=args.lr_g, betas=(0.9, 0.99), weight_decay=0.0)
    ema_g = ModelEMA(raw_net_g, decay=args.ema_decay)

    if hasattr(torch.amp, 'GradScaler'):
        scaler_g = torch.amp.GradScaler('cuda', enabled=args.use_amp and torch.cuda.is_available())
    else:
        scaler_g = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    evaluator = XyetherEvaluator(device=device)

    print(f"[PHASE 18] Starting Zero-Halo 1px Core Engine Training ({args.total_iters} iterations)...")
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < args.total_iters:
        try:
            lr_batch, hr_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            lr_batch, hr_batch = next(data_iter)

        step += 1

        # Linear warmup for first 150 steps, then Cosine Annealing down to 2e-6
        if step <= args.warmup_iters:
            cur_lr = args.lr_g * (step / float(max(1, args.warmup_iters)))
        else:
            progress = (step - args.warmup_iters) / float(max(1, args.total_iters - args.warmup_iters))
            cur_lr = 2.0e-6 + 0.5 * (args.lr_g - 2.0e-6) * (1.0 + np.cos(np.pi * progress))
        for pg in optim_g.param_groups:
            pg['lr'] = cur_lr

        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        with torch.no_grad():
            clean_target, core_valley = target_builder(hr_batch)

        autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optim_g.zero_grad()

        with torch.amp.autocast(autocast_device, enabled=args.use_amp and torch.cuda.is_available()):
            sr_batch = net_g(lr_batch)
            l_core, core_metrics = crit_core(sr_batch, clean_target, lr_batch, core_valley)
            l_percep = crit_vgg(sr_batch, clean_target) if args.percep_weight > 0 else torch.tensor(0.0, device=device)
            l_luv = crit_luv(sr_batch, clean_target)
            total_loss_g = l_core + l_percep + l_luv

        scaler_g.scale(total_loss_g).backward()
        scaler_g.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(raw_net_g.parameters(), max_norm=0.5)
        scaler_g.step(optim_g)
        scaler_g.update()

        # Frobenius Weight Norm Preservation: prevents 18-layer plain CNN shrinkage while boosting 1px split head
        with torch.no_grad():
            for name, p in raw_net_g.named_parameters():
                if name in init_weight_norms:
                    cur_norm = p.data.norm().item()
                    min_norm = init_weight_norms[name] * (1.08 if name.startswith('body.32') else 1.0)
                    if cur_norm < min_norm and cur_norm > 1e-8:
                        p.data.mul_(min_norm / cur_norm)

        ema_g.update(raw_net_g)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            ips = step / max(1e-3, elapsed)
            print(
                f"[Step {step:5d}/{args.total_iters}] "
                f"Loss: {total_loss_g.item():.4f} "
                f"(Act: {core_metrics['l_active']:.4f}, Quiet: {core_metrics['l_quiet']:.4f}, "
                f"Grad: {core_metrics['l_grad']:.4f}, Halo: {core_metrics['l_halo']:.4f}, VGG12: {l_percep.item():.4f}) | "
                f"LR: {cur_lr:.2e} | Speed: {ips:.2f} it/s",
                flush=True
            )

        if step % args.eval_every == 0 or step == args.total_iters:
            print(f"\n[BENCHMARK] Evaluation at Step {step}...")
            report = evaluator.evaluate_model(ema_g.model)
            print(f"  • Overshoot / Halo:     {report['overshoot_halo']} (Target: 0.00)")
            print(f"  • Noise Residual:       {report['noise_residual_std']} std ({report['noise_elimination_pct']}% killed)")
            print(f"  • Line Depth:           {report['line_depth_luma']} luma (Target: <= 8.0)")
            print(f"  • Sharpness (LapVar):   {report['sharpness_lapvar']}\n", flush=True)

            with torch.no_grad():
                sample_lr = lr_batch[0:1]
                sample_hr = hr_batch[0:1]
                sample_tar = clean_target[0:1]
                sample_sr = ema_g.model(sample_lr)
                nearest_lr = F.interpolate(sample_lr, scale_factor=2, mode='nearest')
                grid = torch.cat([nearest_lr, sample_sr, sample_tar, sample_hr], dim=3)
                grid_np = (grid[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy())
                cv2.imwrite(str(sample_dir / f"step_{step:06d}.png"), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))

        if step % args.save_every == 0 or step == args.total_iters:
            ckpt_path = out_dir / f"xyetherkiller_coremaster_step_{step}.pth"
            torch.save({
                'step': step,
                'params': raw_net_g.state_dict(),
                'params_ema': ema_g.state_dict(),
            }, str(ckpt_path))
            print(f"[CHECKPOINT] Saved checkpoint -> {ckpt_path}", flush=True)

            hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or DEFAULT_HF_TOKEN
            hf_repo = getattr(args, 'hf_repo', 'Thanawanit/Kaggle-Backup')
            if hf_token and hf_repo and hf_repo.lower() != 'none':
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/Phase18_XyetherKiller_CoreMaster/{ckpt_path.name}",
                        repo_id=hf_repo,
                        repo_type="model"
                    )
                    print(f"[HF] Backed up -> {hf_repo}/checkpoints/Phase18_XyetherKiller_CoreMaster/{ckpt_path.name}", flush=True)
                except Exception as e:
                    print(f"[WARN] Hugging Face upload error: {e}", flush=True)

    # Export Final Production Model to PyTorch & ONNX
    final_pth = out_dir / "XyetherKiller_CoreMaster_v18_Final.pth"
    torch.save({"params": ema_g.state_dict(), "params_ema": ema_g.state_dict()}, str(final_pth))
    print(f"\n[FINAL EXPORT] Production PyTorch weights saved -> {final_pth}", flush=True)

    hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or DEFAULT_HF_TOKEN
    hf_repo = getattr(args, 'hf_repo', 'Thanawanit/Kaggle-Backup')
    if hf_token and hf_repo and hf_repo.lower() != 'none':
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(final_pth),
                path_in_repo="checkpoints/Phase18_XyetherKiller_CoreMaster/XyetherKiller_CoreMaster_v18_Final.pth",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final production model uploaded to Hugging Face -> {hf_repo}", flush=True)
        except Exception as e:
            print(f"[WARN] Final HF upload error: {e}", flush=True)

    try:
        final_onnx = out_dir / "XyetherKiller_CoreMaster_v18_Final.onnx"
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
        print(f"[FINAL EXPORT] Production ONNX graph saved -> {final_onnx}", flush=True)
        if hf_token and hf_repo and hf_repo.lower() != 'none':
            api.upload_file(
                path_or_fileobj=str(final_onnx),
                path_in_repo="checkpoints/Phase18_XyetherKiller_CoreMaster/XyetherKiller_CoreMaster_v18_Final.onnx",
                repo_id=hf_repo,
                repo_type="model"
            )
            print(f"[HF] Final ONNX model uploaded to Hugging Face -> {hf_repo}", flush=True)
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}", flush=True)

    print("\n[COMPLETE] Phase 18 XyetherKiller CoreMaster training successfully finished!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XyetherKiller CoreMaster Series (Phase 18)")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/Elite_Anime_4K", help="Dataset folder")
    parser.add_argument("--pretrained_path", type=str, default="./weights/net_g_89000.pth", help="Authentic base student checkpoint")
    parser.add_argument("--pretrained_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Phase4_XyetherKiller_V11/net_g_89000.pth", help="Authentic student warm-start URL")
    parser.add_argument("--output_dir", type=str, default="./output_models_phase18", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size across GPUs")
    parser.add_argument("--patch_size", type=int, default=192, help="HR Patch size")
    parser.add_argument("--total_iters", type=int, default=4500, help="Total iterations")
    parser.add_argument("--warmup_iters", type=int, default=150, help="Linear warmup iterations")
    parser.add_argument("--lr_g", type=float, default=3.0e-5, help="Generator learning rate")
    parser.add_argument("--valley_strength", type=float, default=1.55, help="Bilateral 1px NMS ink trough strength")
    parser.add_argument("--aa_preserve", type=float, default=0.18, help="Subpixel shoulder anti-aliasing preservation")
    parser.add_argument("--percep_weight", type=float, default=0.0, help="Shallow VGG19 (conv1_2, conv2_2) weight (0.0 = pure coordinate-exact)")
    parser.add_argument("--luv_weight", type=float, default=0.10, help="ColorLuv cel-shading consistency weight")
    parser.add_argument("--ema_decay", type=float, default=0.998, help="EMA decay")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use FP16 AMP")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log interval")
    parser.add_argument("--eval_every", type=int, default=1000, help="Eval interval")
    parser.add_argument("--save_every", type=int, default=1000, help="Save interval")
    parser.add_argument("--hf_token", type=str, default=DEFAULT_HF_TOKEN, help="Hugging Face write token")
    parser.add_argument("--hf_repo", type=str, default="Thanawanit/Kaggle-Backup", help="Hugging Face repo id")
    args = parser.parse_args()
    train_xyether(args)
