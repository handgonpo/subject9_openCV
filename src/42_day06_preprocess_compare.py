from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    crop_roi,
    get_roi_for_file,
    load_rules,
    make_preprocess_variants,
)


# ========================================
# Rule 읽기
# ========================================

rules = load_rules()


# ========================================
# 대표 이미지
# ========================================

file_name = rules[
    "reference_file"
]

IMAGE_PATH = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
    / file_name
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "preprocess"
)


# ========================================
# 이미지 읽기
# ========================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


# ========================================
# 대표 이미지의 ROI 좌표 가져오기
# ========================================

roi_coordinates = (
    get_roi_for_file(
        rules,
        file_name,
    )
)


# ========================================
# SN 영역 Crop
# ========================================

roi = crop_roi(
    image,
    roi_coordinates,
)


# ========================================
# 전처리 후보 만들기
# ========================================

variants = (
    make_preprocess_variants(
        roi,
        rules,
    )
)


# ========================================
# 결과 폴더
# ========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ========================================
# 전처리 결과 저장
# ========================================

for index, (
    name,
    result,
) in enumerate(
    variants.items(),
    start=1,
):

    output_path = (
        OUTPUT_DIR
        / f"{index:02d}_{name}.jpg"
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
    "Reference:",
    file_name
)

print(
    "ROI:",
    roi_coordinates
)

print(
    "Preprocess Compare Complete"
)