"""
Model Weights Downloader & Setup Helper for Member 03
Downloads baseline TranSalNet_Res and ResNet-50 weights.
"""

import os
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def download_file(url, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"[OK] {target_path.name} already exists.")
        return
    print(f"Downloading {target_path.name} from {url} ...")
    urllib.request.urlretrieve(url, str(target_path))
    print(f"[Done] Saved to {target_path}")

def main():
    models_dir = ROOT_DIR / "models" / "transalnet_pretrained"
    checkpoints_dir = ROOT_DIR / "models" / "checkpoints"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ResNet-50 PyTorch official weights
    resnet_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    download_file(resnet_url, models_dir / "resnet50-0676ba61.pth")
    
    # 2. TranSalNet_Res SALICON baseline weights
    print("\nNote: TranSalNet_Res.pth and fine-tuned best.pt are large binary files (>250MB).")
    print("If not already present, please download them from the project Google Drive / Zenodo storage")
    print(f"and place them into:")
    print(f"  - Baseline:    {models_dir / 'TranSalNet_Res.pth'}")
    print(f"  - Fine-Tuned:  {checkpoints_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
