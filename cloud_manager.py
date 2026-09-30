import os
import sys
import subprocess
from pathlib import Path

def check_kaggle_auth():
    kaggle_dir = Path.home() / ".kaggle"
    token_file = kaggle_dir / "kaggle.json"
    if token_file.exists():
        print(f"[KAGGLE] Found credentials: {token_file}")
        res = subprocess.run(["kaggle", "competitions", "list"], capture_output=True, text=True)
        if res.returncode == 0:
            print("[KAGGLE] Authentication: SUCCESS [OK]")
            return True
        else:
            print(f"[KAGGLE] Auth error: {res.stderr[:200]}")
            return False
    else:
        print(f"[KAGGLE] Missing {token_file}. Place your kaggle.json in ~/.kaggle/ to enable headless push.")
        return False

def check_lightning_auth():
    lightning_cli = Path(r"C:\Users\IDAEPAD\AppData\Local\Programs\Python\Python311\Scripts\lightning.exe")
    res = subprocess.run([str(lightning_cli), "--version"], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[LIGHTNING] CLI ready: {res.stdout.strip()} [OK]")
        return True
    return False

if __name__ == "__main__":
    print("=== CLOUD RUNTIME READINESS CHECK ===")
    check_kaggle_auth()
    check_lightning_auth()
