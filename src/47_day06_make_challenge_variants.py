from pathlib import Path

import cv2
import numpy as np

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
)


# ========================================
# 대표 SN ROI 찾기
# ========================================

rules = load_rules()

reference_file = rules[
    "reference_file"
]

reference_stem = (
    Path(reference_file)
    .stem
)

SOURCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "roi"
    / f"{reference_stem}_sn_roi.jpg"
)

ROOT = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "challenge"
)


# ========================================
# 정상 SN ROI 읽기
# ========================================

image = cv2.imread(
    str(SOURCE_PATH)
)

if image is None:
    raise FileNotFoundError(
        SOURCE_PATH
    )


height, width = (
    image.shape[:2]
)


# ========================================
# 1. Low Contrast
# 문자와 배경 차이를 약하게 만듦
# ========================================

neutral = np.full_like(
    image,
    128,
)

low_contrast = cv2.addWeighted(
    image,
    0.45,
    neutral,
    0.55,
    0,
)


# ========================================
# 2. Blur
# 문자를 흐리게 만듦
# ========================================

blur = cv2.GaussianBlur(
    image,
    (9, 9),
    0,
)


# ========================================
# 3. Tilt
# 문자를 +8도 기울임
# ========================================

center = (
    width / 2,
    height / 2,
)

matrix = (
    cv2.getRotationMatrix2D(
        center,
        8.0,
        1.0,
    )
)

tilt = cv2.warpAffine(
    image,
    matrix,
    (width, height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(255, 255, 255),
)


# ========================================
# 4. Glare
# 빛 반사 영역을 인위적으로 추가
# ========================================

overlay = image.copy()

cv2.ellipse(
    overlay,
    (
        int(width * 0.65),
        int(height * 0.50),
    ),
    (
        max(
            10,
            int(width * 0.24),
        ),
        max(
            8,
            int(height * 0.20),
        ),
    ),
    -20,
    0,
    360,
    (255, 255, 255),
    -1,
)

glare = cv2.addWeighted(
    overlay,
    0.60,
    image,
    0.40,
    0,
)


# ========================================
# 테스트 이미지 저장
# ========================================

variants = {
    "low_contrast":
        low_contrast,

    "blur":
        blur,

    "tilt":
        tilt,

    "glare":
        glare,
}


for condition, result in (
    variants.items()
):

    output_dir = (
        ROOT
        / condition
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{condition}_01.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "Saved:",
        output_path
    )


print()
print(
    "Environment Test Images Created"
)