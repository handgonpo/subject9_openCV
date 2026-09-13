from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

blur_path = Path(
    "samples/day02/blur.jpg"
)

output_dir = Path(
    "outputs/day02/sharpen"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 기준 ROI 이미지 읽기
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: "
        f"{reference_path}"
    )


# --------------------------------------------------
# 실습용 Blur 이미지 만들기
# --------------------------------------------------

blurred = cv2.GaussianBlur(
    reference,
    (7, 7),
    0,
)

cv2.imwrite(
    str(blur_path),
    blurred,
)

print(
    "문제 이미지 생성:",
    blur_path,
)


# --------------------------------------------------
# Sharpening Kernel
# --------------------------------------------------

kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


# --------------------------------------------------
# Sharpening 적용
# --------------------------------------------------

sharpened = cv2.filter2D(
    blurred,
    -1,
    kernel,
)


# --------------------------------------------------
# 비교 결과 저장
# --------------------------------------------------

cv2.imwrite(
    str(
        output_dir / "reference.jpg"
    ),
    reference,
)

cv2.imwrite(
    str(
        output_dir / "blurred.jpg"
    ),
    blurred,
)

cv2.imwrite(
    str(
        output_dir / "sharpened.jpg"
    ),
    sharpened,
)

print(
    "Sharpening 결과 저장 완료"
)