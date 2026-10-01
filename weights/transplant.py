import os
import sys
import shutil
import subprocess
from pathlib import Path
from collections import OrderedDict
import torch

DEFAULT_CKPT_URL = "https://huggingface.co/Thanawanit/Kaggle-Backup/resolve/main/checkpoints/Phase4_XyetherKiller_V11/net_g_89000.pth"

def download_file_if_needed(url, dest_path):
    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 1000000:
        print(f"[TRANSPLANT] Found existing checkpoint: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TRANSPLANT] Downloading {url} -> {dest}...")

    if shutil.which("aria2c"):
        cmd = ["aria2c", "-q", "-c", "-x", "8", "-s", "8", "-o", dest.name, url]
        subprocess.run(cmd, cwd=str(dest.parent), check=False)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))

    if not dest.exists() or dest.stat().st_size < 1000000:
        raise RuntimeError(f"Failed to download checkpoint from {url}")

    print(f"[TRANSPLANT] Download complete: {dest.stat().st_size / (1024*1024):.2f} MB")
    return dest


def perform_weight_surgery(source_ckpt_path, output_ckpt_path, zero_head=False):
    """
    Transplants weights from base checkpoint (net_g_89000):
    1. Retains all 89,000-step anime feature representations across all 53 layers.
    2. Preserves Conv 18 weights by default to prevent PixelShuffle asymmetric checkerboard artifacts.
    """
    source_path = Path(source_ckpt_path)
    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TRANSPLANT] Loading source checkpoint: {source_path}")
    data = torch.load(str(source_path), map_location='cpu')

    if isinstance(data, dict):
        if 'params_ema' in data:
            source_dict = data['params_ema']
            print("[TRANSPLANT] Extracted 'params_ema' weights.")
        elif 'params' in data:
            source_dict = data['params']
            print("[TRANSPLANT] Extracted 'params' weights.")
        else:
            source_dict = data
    else:
        source_dict = data

    new_dict = OrderedDict()
    head_keys = []

    for k, v in source_dict.items():
        # Clean prefix if needed
        clean_k = k
        if clean_k.startswith("module."):
            clean_k = clean_k[7:]

        if zero_head and "34" in clean_k: # Conv 18 Reconstruction Head
            head_keys.append(clean_k)
            new_dict[clean_k] = torch.zeros_like(v)
        else:
            new_dict[clean_k] = v.clone()

    print(f"[TRANSPLANT] Transplanted {len(new_dict)} layers.")
    if head_keys:
        print(f"[TRANSPLANT] Head layers zero-initialized: {head_keys}")
    else:
        print(f"[TRANSPLANT] All 53 layers preserved intact (Zero-Init disabled to prevent dot grid).")

    # Save format compatible with BasicSR, traiNNer-redux, and native PyTorch
    torch.save({
        'params': new_dict,
        'params_ema': new_dict
    }, str(output_path))

    print(f"[TRANSPLANT] Successfully saved ready-to-train model -> {output_path}")
    return str(output_path)


def prepare_xyether_base(ckpt_url=DEFAULT_CKPT_URL, cache_dir="./weights", zero_head=False):
    cache_path = Path(cache_dir)
    source_file = cache_path / "net_g_89000.pth"
    target_file = cache_path / "xyether_transplanted_base.pth"

    download_file_if_needed(ckpt_url, source_file)
    return perform_weight_surgery(source_file, target_file, zero_head=zero_head)


if __name__ == "__main__":
    prepare_xyether_base()
