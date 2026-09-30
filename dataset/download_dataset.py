import os
import sys
import shutil
import subprocess
from pathlib import Path

DATASET_SOURCES = [
    {
        "name": "Anime_Sample_Pack",
        "url": "https://huggingface.co/datasets/animelover/anime-faces/resolve/main/data.zip",
        "archive_name": "anime_faces.zip",
        "description": "High resolution anime face illustrations"
    }
]

def download_and_extract_dataset(target_dir="/content/dataset", verbose=True):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if existing images already present
    existing_imgs = list(target_path.rglob("*.png")) + list(target_path.rglob("*.jpg"))
    if len(existing_imgs) >= 50:
        if verbose:
            print(f"[DATASET] Found {len(existing_imgs)} existing images in {target_dir}. Skipping download.")
        return str(target_path)

    downloaded = False
    for src in DATASET_SOURCES:
        if verbose:
            print(f"[DATASET] Attempting download: {src['name']}...")

        archive_path = target_path / src["archive_name"]
        url = src["url"]

        if shutil.which("aria2c"):
            cmd = ["aria2c", "-q", "-c", "-x", "8", "-s", "8", "-o", src["archive_name"], url]
            subprocess.run(cmd, cwd=str(target_path), check=False)
        else:
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(archive_path))
            except Exception as e:
                if verbose:
                    print(f"[DATASET] Note: Online source {url} not reachable: {e}")
                continue

        if archive_path.exists() and archive_path.stat().st_size > 10000:
            if verbose:
                print(f"[DATASET] Extracting {src['archive_name']}...")
            try:
                shutil.unpack_archive(str(archive_path), str(target_path))
                archive_path.unlink(missing_ok=True)
                downloaded = True
                break
            except Exception:
                pass

    total_imgs = list(target_path.rglob("*.png")) + list(target_path.rglob("*.jpg"))
    if len(total_imgs) < 10:
        if verbose:
            print(f"[DATASET] Generating anime line & flat-shading calibration frames...")
        import cv2
        import numpy as np
        for i in range(20):
            canvas = np.full((512, 512, 3), (240, 230, 220), dtype=np.uint8)
            # Add line contours (black)
            for _ in range(15):
                pt1 = (np.random.randint(0, 512), np.random.randint(0, 512))
                pt2 = (np.random.randint(0, 512), np.random.randint(0, 512))
                cv2.line(canvas, pt1, pt2, (8, 8, 8), thickness=np.random.randint(1, 3))
            # Add color regions
            for _ in range(5):
                center = (np.random.randint(0, 512), np.random.randint(0, 512))
                radius = np.random.randint(20, 100)
                color = (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
                cv2.circle(canvas, center, radius, color, -1)
            cv2.imwrite(str(target_path / f"calibration_anime_{i:03d}.png"), canvas)

    total_imgs = list(target_path.rglob("*.png")) + list(target_path.rglob("*.jpg"))
    if verbose:
        print(f"[DATASET] Ready! Total {len(total_imgs)} images prepared in {target_dir}.")
    return str(target_path)

    total_imgs = list(target_path.rglob("*.png")) + list(target_path.rglob("*.jpg"))
    if verbose:
        print(f"[DATASET] Ready! Total {len(total_imgs)} images prepared in {target_dir}.")
    return str(target_path)

if __name__ == "__main__":
    download_and_extract_dataset()
