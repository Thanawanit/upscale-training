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
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure workspace root is importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from archs.xyether_lineflow import (
    XyetherLineFlowChrDeblockNet,
    XyetherLineFlowChrSnapNet,
    XyetherLineFlowNet,
    XyetherLineFlowP35Net,
)
from archs.xyether_rephybrid import XyetherRepHybridNet
from benchmark.xyether_evaluator import XyetherEvaluator
from dataset.otf_dataset import AnimeOTFDataset
from losses.xyether_losses import (
    ColorLuvLoss,
    LPIPSPerceptualLoss,
    PatchVectorDiscriminator,
    Phase23LineFlowLoss,
    Phase23ZeroHaloAPISRTeacher,
    Phase24LineFlowLoss,
    Phase25LineFlowLoss,
    Phase26LineFlowLoss,
    Phase27LineFlowLoss,
    Phase28DegradGatedSWTLoss,
    Phase29VectorCleanLoss,
    Phase30APISRVectorLoss,
    Phase31SilkyVectorLoss,
    Phase32EdgeBandLoss,
    Phase33MacroblockDebleedLoss,
    Phase34ChrSnapLoss,
    Phase35SharpColoredVectorLoss,
    Phase36NaturalRGBVectorLoss,
    VGGPerceptualLoss,
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
    if hasattr(model_cpu, "switch_to_deploy"):
        model_cpu.switch_to_deploy()
    dummy = torch.randn(1, 3, 64, 64)
    kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch", 2: "height_in", 3: "width_in"},
            "output": {0: "batch", 2: "height_out", 3: "width_out"},
        },
    }
    try:
        torch.onnx.export(model_cpu, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, onnx_path, **kwargs)
    print(f"Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB)")
    return onnx_path


def export_fp16_onnx(fp32_onnx_path: str, fp16_onnx_path: str) -> str:
    if not fp32_onnx_path or not os.path.exists(fp32_onnx_path):
        return ""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print(f"[WARN] Skipping FP16 ONNX export ({fp16_onnx_path}): 'onnxconverter_common' not installed.")
        return ""
    model_fp32 = onnx.load(fp32_onnx_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
    onnx.save(model_fp16, fp16_onnx_path)
    print(f"Exported FP16 ONNX: {fp16_onnx_path} ({os.path.getsize(fp16_onnx_path) / (1024 * 1024):.2f} MB)")
    return fp16_onnx_path


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
    pth_path = os.path.join(args.output_dir, f"{args.model_prefix}_{tag}.pth")
    if args.arch == "rephybrid":
        arch_name = "XyetherRepHybridNet"
    elif args.arch == "lineflow_p35":
        arch_name = "XyetherLineFlowP35Net"
    elif args.arch == "lineflow_chrsnap":
        arch_name = "XyetherLineFlowChrSnapNet"
    elif args.arch == "lineflow_chrdeblock":
        arch_name = "XyetherLineFlowChrDeblockNet"
    else:
        arch_name = "XyetherLineFlowNet"
    torch.save(
        {
            "params": model.state_dict(),
            "params_ema": ema.ema.state_dict(),
            "arch": {
                "name": arch_name,
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


def save_deploy_checkpoint(ema_model: nn.Module, args, tag: str = "Final_Deploy") -> str:
    if not hasattr(ema_model, "switch_to_deploy"):
        return ""
    os.makedirs(args.output_dir, exist_ok=True)
    dep_model = copy.deepcopy(ema_model).eval().cpu()
    dep_model.switch_to_deploy()
    dep_path = os.path.join(args.output_dir, f"{args.model_prefix}_{tag}.pth")
    torch.save(
        {
            "params_ema": dep_model.state_dict(),
            "arch": {
                "name": "XyetherRepHybridNet",
                "num_feat": args.num_feat,
                "num_blocks": args.num_blocks,
                "upscale": 2,
                "deployed": True,
            },
            "iter": args.total_iters,
        },
        dep_path,
    )
    print(f"Saved deployed checkpoint: {dep_path}")
    return dep_path


def train_p23(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resuming = bool(args.resume_path or args.resume_url)
    is_p28 = args.loss_version >= 28
    is_p29 = args.loss_version >= 29
    is_p30 = args.loss_version >= 30
    is_p31 = args.loss_version >= 31
    is_p32 = args.loss_version >= 32
    is_p33 = args.loss_version >= 33
    print(f"Phase {args.loss_version} Training | arch={args.arch} | device={device} | scratch_init={not resuming}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_file(args.teacher_onnx_path, args.teacher_onnx_url, "APISR.onnx teacher")
    if args.resume_path and args.resume_url and not os.path.exists(args.resume_path):
        ensure_file(args.resume_path, args.resume_url, "own student checkpoint")
    if args.warmstart_ckpt and args.warmstart_url and not os.path.exists(args.warmstart_ckpt):
        ensure_file(args.warmstart_ckpt, args.warmstart_url, "own LineFlow warm-start teacher")

    dataset = AnimeOTFDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        repeat=40,
        return_regime=is_p28,
        clean_prob=(0.22 if is_p33 else (0.25 if is_p31 else (0.30 if is_p30 else 0.20))),
        use_h264=is_p32,
        h264_direct_prob=(0.50 if is_p33 else 0.0),
    )
    if is_p32:
        print(f"Dataset Real H.264 (PyAV libx264) Active: {dataset.use_h264} | direct_prob={dataset.h264_direct_prob:.2f}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    test_images = [p for p in Path(args.data_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}][:5]

    if args.arch == "rephybrid":
        model = XyetherRepHybridNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=2,
        ).to(device)
    elif args.arch == "lineflow_p35":
        model = XyetherLineFlowP35Net(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=2,
        ).to(device)
    elif args.arch == "lineflow_chrsnap":
        model = XyetherLineFlowChrSnapNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=2,
        ).to(device)
    elif args.arch == "lineflow_chrdeblock":
        model = XyetherLineFlowChrDeblockNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=2,
        ).to(device)
    else:
        model = XyetherLineFlowNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=2,
        ).to(device)

    new_head_keys = set()
    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        sd = ckpt.get("params_ema", ckpt.get("params", ckpt))
        clean_sd = {k.replace("module.", ""): v.float() for k, v in sd.items()}
        if args.arch == "lineflow_p35":
            # Load all matching 8B 56ch backbone weights from own v34.pth; train fresh PUChr16 head with 5x LR
            msd = model.state_dict()
            compat_sd = {k: v for k, v in clean_sd.items() if k in msd and msd[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(compat_sd, strict=False)
            new_head_keys = set(missing)
            print(
                f"Resumed own v34 backbone into P35Net: {args.resume_path} "
                f"(loaded={len(compat_sd)}/{len(msd)}, new_puchr_tensors={len(missing)}, unexp={len(unexpected)})"
            )
        elif args.arch == "lineflow_chrsnap":
            # Drop old 1x1 c_ref keys from v33 if shapes mismatch the new DW3x3 c_ref
            msd = model.state_dict()
            compat_sd = {k: v for k, v in clean_sd.items() if k in msd and msd[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(compat_sd, strict=False)
            new_head_keys = set(missing) | {k for k in msd if k.startswith("cd1.")}
            print(
                f"Resumed own v33 weights into ChrSnapNet: {args.resume_path} "
                f"(loaded={len(compat_sd)}/{len(msd)}, new_dw3_tensors={len(missing)}, unexp={len(unexpected)})"
            )
        elif args.arch == "lineflow_chrdeblock":
            missing, unexpected = model.load_state_dict(clean_sd, strict=False)
            new_head_keys = set(missing)
            print(
                f"Resumed own LineFlow backbone into ChrDeblockNet: {args.resume_path} "
                f"(new head tensors={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            model.load_state_dict(clean_sd, strict=True)
            print(f"Resumed own student checkpoint (EMA weights): {args.resume_path}")

    warmstart_model = None
    if args.warmstart_ckpt and os.path.exists(args.warmstart_ckpt):
        warmstart_model = XyetherLineFlowNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=args.warmstart_feat,
            num_blocks=args.warmstart_blocks,
            upscale=2,
        ).to(device).eval()
        ws_ckpt = torch.load(args.warmstart_ckpt, map_location=device)
        ws_sd = ws_ckpt.get("params_ema", ws_ckpt.get("params", ws_ckpt))
        warmstart_model.load_state_dict({k.replace("module.", ""): v.float() for k, v in ws_sd.items()}, strict=True)
        for p in warmstart_model.parameters():
            p.requires_grad_(False)
        print(f"Loaded own LineFlow warm-start teacher: {args.warmstart_ckpt} ({args.warmstart_iters} decay iters)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Student {args.arch}(feat={args.num_feat}, blocks={args.num_blocks}): {total_params:,} params")

    ema = ModelEMA(model, decay=args.ema_decay)
    teacher = Phase23ZeroHaloAPISRTeacher(
        onnx_path=args.teacher_onnx_path,
        full_color_consolidation=(args.loss_version >= 26),
        freq_micro_rescue=(args.loss_version == 27 or args.loss_version == 28),
        unclipped=is_p29,
    ).to(device)
    if args.loss_version >= 36:
        criterion = Phase36NaturalRGBVectorLoss().to(device)
    elif args.loss_version >= 35:
        criterion = Phase35SharpColoredVectorLoss().to(device)
    elif args.loss_version >= 34:
        criterion = Phase34ChrSnapLoss().to(device)
    elif args.loss_version >= 33:
        criterion = Phase33MacroblockDebleedLoss().to(device)
    elif args.loss_version >= 32:
        criterion = Phase32EdgeBandLoss(edgeband_w=2.8).to(device)
    elif args.loss_version >= 31:
        criterion = Phase31SilkyVectorLoss().to(device)
    elif args.loss_version >= 30:
        criterion = Phase30APISRVectorLoss().to(device)
    elif args.loss_version >= 29:
        criterion = Phase29VectorCleanLoss().to(device)
    elif args.loss_version == 28:
        criterion = Phase28DegradGatedSWTLoss().to(device)
    elif args.loss_version == 27:
        criterion = Phase27LineFlowLoss().to(device)
    elif args.loss_version == 26:
        criterion = Phase26LineFlowLoss().to(device)
    elif args.loss_version == 25:
        criterion = Phase25LineFlowLoss().to(device)
    elif args.loss_version == 24:
        criterion = Phase24LineFlowLoss().to(device)
    else:
        criterion = Phase23LineFlowLoss().to(device)
    luv_loss = ColorLuvLoss(loss_weight=args.luv_weight).to(device) if args.luv_weight > 0.0 else None
    vgg_loss = VGGPerceptualLoss(loss_weight=args.vgg_weight).to(device) if args.vgg_weight > 0.0 else None
    lpips_loss = (
        LPIPSPerceptualLoss(loss_weight=args.lpips_weight).to(device)
        if (is_p29 and args.lpips_weight > 0.0)
        else None
    )
    disc = PatchVectorDiscriminator(nf=48).to(device) if (is_p29 and args.gan_weight > 0.0) else None

    if new_head_keys:
        base_params = [p for n, p in model.named_parameters() if n not in new_head_keys]
        head_params = [p for n, p in model.named_parameters() if n in new_head_keys]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": args.lr_g},
                {"params": head_params, "lr": args.lr_g * 5.0},
            ],
            weight_decay=1e-4,
        )
        print(f"Differential LR active: backbone={args.lr_g:.2e} | new_chr_head={args.lr_g * 5.0:.2e}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_g, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_iters, args.total_iters, args.lr_g, min_lr=args.min_lr)
    )
    opt_d = (
        torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=1e-4)
        if disc is not None
        else None
    )
    sch_d = (
        torch.optim.lr_scheduler.LambdaLR(
            opt_d, build_lr_lambda(args.warmup_iters, args.total_iters, args.lr_d, min_lr=args.min_lr)
        )
        if opt_d is not None
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.use_amp))
    evaluator = XyetherEvaluator(device=device)

    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(1, args.total_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if is_p28:
            lr_b, gt_b, regime_b = batch
            regime_b = regime_b.to(device, non_blocking=True)
        else:
            lr_b, gt_b = batch
            regime_b = None

        lr_b = lr_b.to(device, non_blocking=True)
        gt_b = gt_b.to(device, non_blocking=True)

        with torch.no_grad():
            tar_b, edge_zone = teacher(gt_b)
            warmstart_out = None
            ws_weight = 0.0
            if warmstart_model is not None and args.warmstart_iters > 0 and step <= args.warmstart_iters:
                ws_weight = args.warmstart_weight * 0.5 * (1.0 + math.cos(math.pi * float(step) / float(args.warmstart_iters)))
                warmstart_out = warmstart_model(lr_b)

        loss_d_val = None
        if disc is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
                    pred_det = model(lr_b)
            d_real = disc(tar_b.float())
            d_fake = disc(pred_det.float())
            loss_d = 0.5 * (
                F.relu(1.0 - (d_real - d_fake.mean())).mean()
                + F.relu(1.0 + (d_fake - d_real.mean())).mean()
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
            sch_d.step()
            loss_d_val = loss_d.item()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
            pred = model(lr_b)
            if is_p28:
                loss_main, metrics = criterion(
                    pred,
                    tar_b,
                    gt_b,
                    lr_b,
                    edge_zone=edge_zone,
                    regime=regime_b,
                    warmstart_out=warmstart_out,
                    warmstart_weight=ws_weight,
                )
            else:
                loss_main, metrics = criterion(pred, tar_b, gt_b, lr_b, edge_zone=edge_zone)
            loss = loss_main
            pred_f = pred.float()
            if luv_loss is not None:
                if is_p33:
                    luv_ref = tar_b.float()
                elif is_p31:
                    luv_ref = (0.90 * tar_b + 0.10 * gt_b).float()
                elif is_p30:
                    luv_ref = gt_b.float()
                elif is_p29:
                    luv_ref = (0.70 * tar_b + 0.30 * gt_b).float()
                else:
                    luv_ref = (gt_b if is_p28 else tar_b).float()
                l_luv = luv_loss(pred_f, luv_ref)
                loss = loss + l_luv
                metrics["l_luv"] = l_luv.item()
            if lpips_loss is not None:
                l_lpips = lpips_loss(pred_f, tar_b.float())
                loss = loss + l_lpips
                metrics["l_lpips"] = l_lpips.item()
            if vgg_loss is not None:
                perc_ref = gt_b if (is_p28 and not is_p29) else tar_b
                l_vgg = vgg_loss(pred_f, perc_ref.float())
                loss = loss + l_vgg
                metrics["l_vgg"] = l_vgg.item()
            if disc is not None:
                d_r_g = disc(tar_b.float()).detach()
                d_f_g = disc(pred_f)
                l_gan = args.gan_weight * 0.5 * (
                    F.relu(1.0 + (d_r_g - d_f_g.mean())).mean()
                    + F.relu(1.0 - (d_f_g - d_r_g.mean())).mean()
                )
                loss = loss + l_gan
                metrics["l_gan"] = l_gan.item()
                if loss_d_val is not None:
                    metrics["l_d"] = loss_d_val

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
    final_dep_pth = save_deploy_checkpoint(ema.ema, args, "Final_Deploy")
    final_onnx = export_onnx(ema.ema, os.path.join(args.output_dir, f"{args.model_prefix}_Final.onnx"))
    final_fp16_onnx = (
        export_fp16_onnx(final_onnx, os.path.join(args.output_dir, f"{args.model_prefix}_Final_FP16.onnx"))
        if (args.export_fp16 or is_p29)
        else ""
    )
    final_rep = evaluator.evaluate_model(ema.ema, test_images=test_images)
    print(f"  [Final Eval] {final_rep}")
    upload_hf_artifacts([final_pth, final_dep_pth, final_onnx, final_fp16_onnx], args.hf_token, args.hf_repo, args.hf_subdir)
    print(f"Phase {args.loss_version} Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 23-29 Xyether AnimeSR")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/Elite_Anime_4K")
    parser.add_argument(
        "--arch",
        type=str,
        default="lineflow",
        choices=["lineflow", "lineflow_chrdeblock", "lineflow_chrsnap", "lineflow_p35", "rephybrid"],
    )
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--resume_url", type=str, default="")
    parser.add_argument("--warmstart_ckpt", type=str, default="")
    parser.add_argument("--warmstart_url", type=str, default="")
    parser.add_argument("--warmstart_iters", type=int, default=1500)
    parser.add_argument("--warmstart_weight", type=float, default=0.80)
    parser.add_argument("--warmstart_feat", type=int, default=56)
    parser.add_argument("--warmstart_blocks", type=int, default=8)
    parser.add_argument("--teacher_onnx_path", type=str, default="./reference_models/Model/APISR.onnx")
    parser.add_argument("--teacher_onnx_url", type=str, default="https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/teachers/APISR.onnx")
    parser.add_argument("--output_dir", type=str, default="./output_models_phase29")
    parser.add_argument("--model_prefix", type=str, default="XyetherLineFlow_v29")
    parser.add_argument("--loss_version", type=int, default=29)
    parser.add_argument("--num_feat", type=int, default=56)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=160)
    parser.add_argument("--total_iters", type=int, default=6500)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_g", type=float, default=1.5e-4)
    parser.add_argument("--lr_d", type=float, default=1.2e-4)
    parser.add_argument("--min_lr", type=float, default=8e-6)
    parser.add_argument("--luv_weight", type=float, default=0.18)
    parser.add_argument("--lpips_weight", type=float, default=0.45)
    parser.add_argument("--vgg_weight", type=float, default=0.0)
    parser.add_argument("--gan_weight", type=float, default=0.08)
    parser.add_argument("--ema_decay", type=float, default=0.9992)
    parser.add_argument("--export_fp16", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf_repo", type=str, default="Thanawanit/Kaggle-Backup")
    parser.add_argument("--hf_subdir", type=str, default="checkpoints/Phase29_XyetherLineFlow_v1")
    args = parser.parse_args()
    train_p23(args)
