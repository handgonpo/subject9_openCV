from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

noise_path = Path(
    "samples/day02/noise.jpg"
)

output_dir = Path(
    "outputs/day02/filter"
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
# 실습용 Noise 이미지 만들기
# --------------------------------------------------

# 같은 결과가 반복되도록 Seed 고정
rng = np.random.default_rng(42)


# 1. 작은 Gaussian Noise 추가
gaussian_noise = rng.normal(
    loc=0,
    scale=18,
    size=reference.shape,
)

noisy = reference.astype(
    np.float32
) + gaussian_noise

noisy = np.clip(
    noisy,
    0,
    255,
).astype(np.uint8)


# 2. 일부 픽셀에 점 형태 Noise 추가
height, width = noisy.shape[:2]

noise_count = int(
    height * width * 0.01
)

ys = rng.integers(
    0,
    height,
    noise_count,
)

xs = rng.integers(
    0,
    width,
    noise_count,
)

half = noise_count // 2

noisy[
    ys[:half],
    xs[:half],
] = 0

noisy[
    ys[half:],
    xs[half:],
] = 255


# 문제 이미지 저장
cv2.imwrite(
    str(noise_path),
    noisy,
)

print(
    "문제 이미지 생성:",
    noise_path,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

gray = cv2.cvtColor(
    noisy,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 여러 Filter 적용
# --------------------------------------------------

results = {
    "original_gray.jpg": gray,

    "gaussian_3.jpg": cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    ),

    "gaussian_7.jpg": cv2.GaussianBlur(
        gray,
        (7, 7),
        0,
    ),

    "median_3.jpg": cv2.medianBlur(
        gray,
        3,
    ),

    "median_7.jpg": cv2.medianBlur(
        gray,
        7,
    ),

    "bilateral.jpg": cv2.bilateralFilter(
        gray,
        9,
        75,
        75,
    ),
}


# --------------------------------------------------
# 결과 저장
# --------------------------------------------------

for filename, result in results.items():

    output_path = (
        output_dir / filename
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "저장:",
        output_path,
    )