import os
import sys
import shutil
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2

def download_safebooru_anime(target_path: Path, max_pages: int = 6, verbose: bool = True) -> int:
    """
    Downloads high-resolution anime art from Safebooru (Absurdres 4K).
    Fetches up to max_pages * 100 images with multi-threaded downloads.
    """
    if verbose:
        print("📡 Querying Safebooru Absurdres 4K API...")

    image_urls = []
    for page in range(1, max_pages + 1):
        try:
            api_url = (
                f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1"
                f"&limit=100&pid={page}&tags=absurdres+rating:safe+-comic+-text"
            )
            resp = requests.get(api_url, timeout=15)
            if resp.status_code == 200:
                posts = resp.json()
                for post in posts:
                    dir_name = post.get("directory", "")
                    img_name = post.get("image", "")
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        img_url = f"https://safebooru.org/images/{dir_name}/{img_name}"
                        image_urls.append(img_url)
        except Exception as e:
            if verbose:
                print(f"⚠️ Safebooru Page {page} warning: {e}")

    if verbose:
        print(f"📡 Found {len(image_urls)} candidate anime images from Safebooru. Downloading...")

    def download_single(url):
        try:
            fname = url.split("/")[-1]
            dest = target_path / fname
            if not dest.exists() or dest.stat().st_size == 0:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(dest, "wb") as f:
                        f.write(r.content)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=8) as exe:
        list(exe.map(download_single, image_urls))

    # Validate image integrity
    valid_count = 0
    for f in list(target_path.glob("*.*")):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            try:
                img = cv2.imread(str(f))
                if img is not None and img.shape[0] >= 200 and img.shape[1] >= 200:
                    valid_count += 1
                else:
                    f.unlink(missing_ok=True)
            except Exception:
                f.unlink(missing_ok=True)

    if verbose:
        print(f"✅ Safebooru download complete: {valid_count} valid high-res anime frames ready!")
    return valid_count


def download_and_extract_dataset(target_dir="/content/dataset", verbose=True):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 1. Check if existing images already present
    existing_imgs = [
        f for f in target_path.rglob("*.*")
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]
    if len(existing_imgs) >= 100:
        if verbose:
            print(f"[DATASET] Found {len(existing_imgs)} existing images in {target_dir}. Skipping download.")
        return str(target_path)

    # 2. Check if Kaggle /input has mounted anime dataset
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate_dir in kaggle_input.rglob("*"):
            if candidate_dir.is_dir():
                found_imgs = [
                    f for f in candidate_dir.glob("*.*")
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
                ]
                if len(found_imgs) >= 50:
                    if verbose:
                        print(f"[DATASET] Detected mounted Kaggle dataset in {candidate_dir} ({len(found_imgs)} images).")
                    for img in found_imgs[:1000]:
                        try:
                            shutil.copy(str(img), str(target_path / img.name))
                        except Exception:
                            pass
                    break

    existing_imgs = [
        f for f in target_path.rglob("*.*")
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]
    if len(existing_imgs) >= 100:
        if verbose:
            print(f"[DATASET] Ready with {len(existing_imgs)} images.")
        return str(target_path)

    # 3. Download high-res anime art from Safebooru
    valid_count = download_safebooru_anime(target_path, max_pages=6, verbose=verbose)

    # 4. Strict Guard: NEVER fallback to synthetic random shapes
    if valid_count < 50:
        raise RuntimeError(
            f"❌ [DATASET ERROR] Failed to download sufficient anime training images (only {valid_count} available). "
            "Need at least 50 valid anime frames to train Xyether."
        )

    return str(target_path)


if __name__ == "__main__":
    download_and_extract_dataset()
