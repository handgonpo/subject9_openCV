import csv
from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    crop_roi,
    get_roi_for_file,
    load_rules,
    make_preprocess_variants,
    run_ocr,
)


# ========================================
# Rule 읽기
# ========================================

rules = load_rules()


# ========================================
# 대표 이미지 확인
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

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_preprocess_compare.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "preprocess"
)


# ========================================
# 파일명에서 실제 SN 가져오기
# ========================================

expected_sn = (
    Path(file_name)
    .stem
    .split("_side")[0]
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
# 해당 이미지의 ROI 가져오기
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


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


rows = []


# ========================================
# 전처리별 OCR 비교
# ========================================

for index, (
    name,
    processed,
) in enumerate(
    variants.items(),
    start=1,
):

    # ------------------------------------
    # 작은 문자 ROI를 OCR용으로 4배 확대
    # ------------------------------------

    enlarged = cv2.resize(
        processed,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )


    # ------------------------------------
    # Tesseract OCR
    # ------------------------------------

    raw_text = run_ocr(
        enlarged,
        rules,
    )


    # 줄바꿈과 앞뒤 공백만 제거
    ocr_text = raw_text.strip()


    # ------------------------------------
    # 실제 SN과 정확히 같은지 확인
    # ------------------------------------

    exact_match = (
        ocr_text
        == expected_sn
    )


    # ------------------------------------
    # OCR 입력 이미지도 저장
    # ------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"ocr_{index:02d}_{name}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        enlarged,
    )


    # ------------------------------------
    # 결과 기록
    # ------------------------------------

    rows.append(
        {
            "pipeline":
                name,

            "expected_sn":
                expected_sn,

            "raw_text":
                repr(raw_text),

            "ocr_text":
                ocr_text,

            "exact_match":
                exact_match,
        }
    )


    print(
        "=" * 60
    )

    print(
        "Pipeline:",
        name
    )

    print(
        "Expected:",
        expected_sn
    )

    print(
        "Raw OCR:",
        repr(raw_text)
    )

    print(
        "OCR Text:",
        ocr_text
    )

    print(
        "Exact Match:",
        exact_match
    )


# ========================================
# CSV 저장
# ========================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "pipeline",
            "expected_sn",
            "raw_text",
            "ocr_text",
            "exact_match",
        ],
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()

print(
    "Saved:",
    REPORT_PATH
)

print(
    "OCR Compare Complete"
)