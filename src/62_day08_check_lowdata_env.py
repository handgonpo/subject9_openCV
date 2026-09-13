from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import PIL
from PIL import Image


ROOT = Path("samples/day07/original/bottle")

required_dirs = [
    ROOT / "train/good",
    ROOT / "test/good",
    ROOT / "test/broken_small",
    ROOT / "test/contamination",
    ROOT / "ground_truth/broken_small",
    ROOT / "ground_truth/contamination",
]


print("=" * 70)
print("DAY08 LOW-DATA ENV CHECK")
print("=" * 70)

print("Torch      :", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV     :", cv2.__version__)
print("NumPy      :", np.__version__)
print("Pandas     :", pd.__version__)
print("Pillow     :", PIL.__version__)
print("CUDA       :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))

print()
print("Dataset    :", ROOT)

for directory in required_dirs:
    if not directory.exists():
        raise FileNotFoundError(directory)
    print("[OK]", directory)


counts = {
    "train_good": len(list((ROOT / "train/good").glob("*.png"))),
    "test_good": len(list((ROOT / "test/good").glob("*.png"))),
    "broken_small": len(list((ROOT / "test/broken_small").glob("*.png"))),
    "contamination": len(list((ROOT / "test/contamination").glob("*.png"))),
}

print()
for name, count in counts.items():
    print(f"{name:15s}: {count}")


minimum_required = {
    "train_good": 41,
    "test_good": 10,
    "broken_small": 17,
    "contamination": 2,
}

for name, minimum in minimum_required.items():
    if counts[name] < minimum:
        raise RuntimeError(
            f"{name} 이미지가 부족합니다. "
            f"필요: {minimum}, 현재: {counts[name]}"
        )

print()
print("Day08 Environment OK")