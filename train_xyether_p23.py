import argparse
import copy
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure workspace root is importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from archs.xyether_lineflow import XyetherLineFlowNet
from benchmark.xyether_evaluator import XyetherEvaluator
from dataset.otf_dataset import AnimeOTFDataset
from losses.xyether_losses import (
    ColorLuvLoss,
    Phase23LineFlowLoss,
    Phase23ZeroHaloAPISRTeacher,
)


class ModelEMA:
    """Exponential Moving Average with step warmup for clean from-scratch convergence."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def ensure_file(path: str, url: str, label: str) -> str:
    if os.path.exists(path):
        return path
    if not url:
        raise FileNotFoundError(f"Missing {label} at {path} and no download URL supplied.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"Downloading {label} -> {path} ...")
    urllib.request.urlretrieve(url, path)
    return path


def build_lr_lambda(warmup_iters: int, total_iters: int, lr_g: float, min_lr: float = 1e-5):
    min_ratio = min_lr / max(lr_g, 1e-8)

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return max(0.05, float(step + 1) / float(max(1, warmup_iters)))
        progress = float(step - warmup_iters) / float(max(1, total_iters - warmup_iters))
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def export_onnx(ema_model: nn.Module, onnx_path: str) -> str:
    try:
        import onnx  # noqa: F401
    except ImportError:
        print(f"[WARN] Skipping ONNX export ({onnx_path}): 'onnx' package not installed.")
        return ""
    model_cpu = copy.deepcopy(ema_model).eval().cpu()
    dummy = torch.randn(1, 3, 64, 64)
    kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    }
    try:
        torch.onnx.export(model_cpu, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, onnx_path, **kwargs)
    print(f"Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB)")
    return onnx_path


def upload_hf_artifacts(paths, hf_token: str, hf_repo: str, subdir: str) -> None:
    if not hf_token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        for p in paths:
            if p and os.path.exists(p):
                rel = f"{subdir}/{os.path.basename(p)}"
                api.upload_file(
                    path_or_fileobj=p,
                    path_in_repo=rel,
                    repo_id=hf_repo,
                    repo_type="model",
                )
                print(f"Uploaded to HF: {hf_repo}/{rel}")
    except Exception as exc:
        print(f"[WARN] HF upload skipped/failed: {exc}")


def save_checkpoint(model, ema, args, step: int, tag: str) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    pth_path = os.path.join(args.output_dir, f"XyetherLineFlow_v23_{tag}.pth")
    torch.save(
        {
            "params": model.state_dict(),
            "params_ema": ema.ema.state_dict(),
            "arch": {
                "name": "XyetherLineFlowNet",
                "num_feat": args.num_feat,
                "num_blocks": args.num_blocks,
                "upscale": 2,
            },
            "iter": step,
        },
        pth_path,
    )
    print(f"Saved checkpoint: {pth_path}")
    return pth_path


def train_p23(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 23 XyetherLineFlowNet Training | device={device} | scratch_init=True")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_file(args.teacher_onnx_path, args.teacher_onnx_url, "APISR.onnx teacher")

    dataset = AnimeOTFDataset(data_dir=args.data_dir, patch_size=args.patch_size, repeat=40)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_images = [p for p in Path(args.data_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}][:5]

    model = XyetherLineFlowNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=args.num_feat,
        num_blocks=args.num_blocks,
        upscale=2,
    ).to(device)

    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        sd = ckpt.get("params", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
        print(f"Resumed Phase 23 checkpoint: {args.resume_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Student XyetherLineFlowNet(feat={args.num_feat}, blocks={args.num_blocks}): {total_params:,} params")

    ema = ModelEMA(model, decay=args.ema_decay)
    teacher = Phase23ZeroHaloAPISRTeacher(
        onnx_path=args.teacher_onnx_path,
    ).to(device)
    criterion = Phase23LineFlowLoss().to(device)
    luv_loss = ColorLuvLoss(loss_weight=args.luv_weight).to(device) if args.luv_weight > 0.0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_g, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_iters, args.total_iters, args.lr_g, min_lr=1e-5)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.use_amp))
    evaluator = XyetherEvaluator(device=device)

    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(1, args.total_iters + 1):
        try:
            lr_b, gt_b = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            lr_b, gt_b = next(data_iter)

        lr_b = lr_b.to(device, non_blocking=True)
        gt_b = gt_b.to(device, non_blocking=True)

        with torch.no_grad():
            tar_b, edge_zone = teacher(gt_b)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
            pred = model(lr_b)
            loss_main, metrics = criterion(pred, tar_b, gt_b, lr_b, edge_zone=edge_zone)
            loss = loss_main
            if luv_loss is not None:
                l_luv = luv_loss(pred.float(), tar_b.float())
                loss = loss + l_luv
                metrics["l_luv"] = l_luv.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model, step)

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]["lr"]
            m_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[{step:>5d}/{args.total_iters}] {elapsed:6.1f}s | lr={lr_now:.2e} | loss={loss.item():.4f} | {m_str}")

        if args.eval_every > 0 and step % args.eval_every == 0 and step < args.total_iters:
            rep = evaluator.evaluate_model(ema.ema, test_images=test_images)
            print(f"  [Eval@{step}] {rep}")
            model.train()

        if args.save_every > 0 and step % args.save_every == 0 and step < args.total_iters:
            pth_path = save_checkpoint(model, ema, args, step, f"iter{step}")
            upload_hf_artifacts([pth_path], args.hf_token, args.hf_repo, args.hf_subdir)

    final_pth = save_checkpoint(model, ema, args, args.total_iters, "Final")
    final_onnx = export_onnx(ema.ema, os.path.join(args.output_dir, "XyetherLineFlow_v23_Final.onnx"))
    final_rep = evaluator.evaluate_model(ema.ema, test_images=test_images)
    print(f"  [Final Eval] {final_rep}")
    upload_hf_artifacts([final_pth, final_onnx], args.hf_token, args.hf_repo, args.hf_subdir)
    print("Phase 23 Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 23 XyetherLineFlowNet from Scratch")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/Elite_Anime_4K")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--teacher_onnx_path", type=str, default="./reference_models/Model/APISR.onnx")
    parser.add_argument("--teacher_onnx_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/teachers/APISR.onnx")
    parser.add_argument("--output_dir", type=str, default="./output_models_phase23")
    parser.add_argument("--num_feat", type=int, default=56)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=160)
    parser.add_argument("--total_iters", type=int, default=6000)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_g", type=float, default=3.5e-4)
    parser.add_argument("--luv_weight", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1500)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf_repo", type=str, default="Thanawanit/Kaggle-Backup")
    parser.add_argument("--hf_subdir", type=str, default="checkpoints/Phase23_XyetherLineFlow_v1")
    args = parser.parse_args()
    train_p23(args)
