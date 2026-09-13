from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


ROOT = Path(
    "samples/day07/original/bottle"
)

required_dirs = [
    ROOT / "train/good",

    ROOT / "test/good",
    ROOT / "test/broken_large",
    ROOT / "test/broken_small",
    ROOT / "test/contamination",

    ROOT / "ground_truth/broken_large",
    ROOT / "ground_truth/broken_small",
    ROOT / "ground_truth/contamination",
]


print("=" * 70)
print("DAY07 ANOMALY ENV CHECK")
print("=" * 70)

# --------------------------------------------------
# 1. 실행 환경 확인
# --------------------------------------------------
print("Torch      :", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV     :", cv2.__version__)
print("NumPy      :", np.__version__)
print("CUDA       :", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU        :",
        torch.cuda.get_device_name(0),
    )


# --------------------------------------------------
# 2. Dataset 폴더 확인
# --------------------------------------------------
print()
print("Dataset:", ROOT)

all_ok = True

for directory in required_dirs:
    exists = directory.exists()

    if not exists:
        all_ok = False

    print(
        f"[{'OK' if exists else 'MISSING'}]",
        directory,
    )

if not all_ok:
    raise FileNotFoundError(
        "MVTec bottle 폴더 구조를 다시 확인하세요."
    )


# --------------------------------------------------
# 3. 이미지 수 확인
# --------------------------------------------------
print()

print(
    "Train Good        :",
    len(
        list(
            (ROOT / "train/good").glob("*.png")
        )
    ),
)

print(
    "Test Good         :",
    len(
        list(
            (ROOT / "test/good").glob("*.png")
        )
    ),
)

print(
    "Broken Large      :",
    len(
        list(
            (ROOT / "test/broken_large").glob("*.png")
        )
    ),
)

print(
    "Broken Small      :",
    len(
        list(
            (ROOT / "test/broken_small").glob("*.png")
        )
    ),
)

print(
    "Contamination     :",
    len(
        list(
            (ROOT / "test/contamination").glob("*.png")
        )
    ),
)


print()
print("Day07 Environment OK")