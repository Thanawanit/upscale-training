import os
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class AnimeOTFDataset(Dataset):
    """
    On-The-Fly (OTF) Degradation Dataset for Anime Super-Resolution.
    Implements Xyether degradation envelope:
    - Random Gaussian Blur (sigma 0.4 to 3.2, p=0.8)
    - Downsample 2x (Random Area, Bilinear, Bicubic)
    - High-range Gaussian Noise (sigma 2 to 20, p=0.6)
    - JPEG Compression (Quality 40 to 90, p=0.7)
    - Geometric Augmentation (Flip, Rot90)
    """
    def __init__(self, data_dir, patch_size=128, repeat=10):
        self.patch_size = patch_size
        self.repeat = repeat

        data_path = Path(data_dir)
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        self.files = [
            str(p) for p in data_path.rglob('*')
            if p.suffix.lower() in valid_extensions and p.is_file()
        ]

        if len(self.files) == 0:
            print(f"[WARN] No images found in {data_dir}. Creating placeholder for verification.")
            placeholder_dir = data_path / "placeholders"
            placeholder_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(placeholder_dir / f"dummy_{i}.png"), dummy)
            self.files = [str(p) for p in placeholder_dir.glob("*.png")]

    def __len__(self):
        return len(self.files) * self.repeat

    def _apply_gaussian_blur(self, img_np):
        # Mild blur envelope [0.1, 1.0] to focus on fine line reconstruction (matching V11)
        sigma = random.uniform(0.1, 1.0)
        ks = int(2 * round(3 * sigma) + 1)
        if ks % 2 == 0:
            ks += 1
        return cv2.GaussianBlur(img_np, (ks, ks), sigmaX=sigma, sigmaY=sigma)

    def _apply_jpeg_compression(self, img_np):
        # High quality JPEG [70, 95] (matching V11)
        quality = random.randint(70, 95)
        _, enc = cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _apply_noise(self, img_float):
        # Subtle noise in [1, 8] out of 255 -> [0.0039, 0.0314] (matching V11)
        sigma = random.uniform(1.0, 8.0) / 255.0
        noise = np.random.normal(0, sigma, img_float.shape).astype(np.float32)
        return np.clip(img_float + noise, 0.0, 1.0)

    def __getitem__(self, idx):
        file_path = self.files[idx % len(self.files)]
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if img is None:
            img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            target_h = max(h, self.patch_size)
            target_w = max(w, self.patch_size)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        # 1. Random Crop Ground Truth (HR Patch)
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        hr_patch = img[top:top + self.patch_size, left:left + self.patch_size]

        # 2. Geometric Augmentation
        if random.random() < 0.5:
            hr_patch = np.fliplr(hr_patch)
        if random.random() < 0.5:
            hr_patch = np.flipud(hr_patch)
        if random.random() < 0.5:
            rot_k = random.choice([1, 2, 3])
            hr_patch = np.rot90(hr_patch, rot_k)

        # 3. Simulate Mild Degradation on LR
        lr_patch = hr_patch.copy()

        # A. Mild Blur (50% chance)
        if random.random() < 0.5:
            lr_patch = self._apply_gaussian_blur(lr_patch)

        # B. Downsample 2x with random kernel (including 10% Nearest for de-aliasing training)
        interp = random.choices(
            [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST],
            weights=[0.35, 0.30, 0.25, 0.10]
        )[0]
        lr_h = self.patch_size // 2
        lr_w = self.patch_size // 2
        lr_patch = cv2.resize(lr_patch, (lr_w, lr_h), interpolation=interp)

        # Convert to RGB float [0, 1]
        hr_rgb = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lr_rgb = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # C. Add Subtle Noise (5% chance, matching V11)
        if random.random() < 0.05:
            lr_rgb = self._apply_noise(lr_rgb)

        # D. JPEG Compression (20% chance, matching V11)
        if random.random() < 0.20:
            lr_bgr_uint8 = cv2.cvtColor((lr_rgb * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            lr_bgr_jpeg = self._apply_jpeg_compression(lr_bgr_uint8)
            lr_rgb = cv2.cvtColor(lr_bgr_jpeg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Convert to PyTorch [C, H, W]
        hr_tensor = torch.from_numpy(hr_rgb.transpose(2, 0, 1).copy())
        lr_tensor = torch.from_numpy(lr_rgb.transpose(2, 0, 1).copy())

        return lr_tensor, hr_tensor
