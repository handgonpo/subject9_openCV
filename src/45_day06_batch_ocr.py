import csv

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    crop_roi,
    get_roi_for_file,
    load_rules,
    preprocess_for_ocr,
    run_ocr,
)

from day06_ocr_validation import (
    inspect_text,
)


# ========================================
# 경로 설정
# ========================================

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "batch"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_results.csv"
)


# ========================================
# OCR Rule 읽기
# ========================================

rules = load_rules()


# ========================================
# Freeze된 Rule인지 확인
# ========================================

if rules[
    "baseline_version"
] != "day06-ocr-v1.0":

    raise RuntimeError(
        "OCR Rule이 Freeze되지 않았습니다."
    )


# ========================================
# Label CSV 존재 확인
# ========================================

if not LABELS_PATH.exists():

    raise FileNotFoundError(
        "Batch 검사에 필요한 "
        "day06_labels.csv가 없습니다.\n"
        f"확인 경로: {LABELS_PATH}"
    )


# ========================================
# 출력 폴더 준비
# ========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ========================================
# Ground Truth / Expected 값 읽기
# ========================================

with LABELS_PATH.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:

    labels = list(
        csv.DictReader(file)
    )


if not labels:

    raise RuntimeError(
        "day06_labels.csv에 "
        "검사 데이터가 없습니다."
    )


rows = []


# ========================================
# 제품 이미지 하나씩 검사
# ========================================

for item in labels:

    file_name = item[
        "file"
    ]

    image_path = (
        PRODUCT_DIR
        / file_name
    )


    # ------------------------------------
    # 이미지 읽기
    # ------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"제품 이미지를 찾을 수 없습니다: "
            f"{image_path}"
        )


    # ------------------------------------
    # 이미지별 ROI 좌표 가져오기
    # ------------------------------------

    roi_coordinates = (
        get_roi_for_file(
            rules,
            file_name,
        )
    )


    # ------------------------------------
    # SN 영역 Crop
    # ------------------------------------

    roi = crop_roi(
        image,
        roi_coordinates,
    )


    # ------------------------------------
    # Freeze된 전처리 적용
    # 현재: original
    # ------------------------------------

    processed = (
        preprocess_for_ocr(
            roi,
            rules,
        )
    )


    # ------------------------------------
    # 43번과 동일하게 4배 확대
    # ------------------------------------

    ocr_input = cv2.resize(
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
        ocr_input,
        rules,
    )


    # ------------------------------------
    # 문자열 검사 + 제품 판정
    # ------------------------------------

    result = inspect_text(
        raw_text=raw_text,

        expected_text=item[
            "expected_text"
        ],

        pattern=rules[
            "expected_pattern"
        ],
    )


    # ------------------------------------
    # OCR 자체의 정확도 확인
    # ------------------------------------

    ocr_exact = (
        result["final_text"]
        == item["actual_text"]
    )


    # ------------------------------------
    # 최종 제품 판정 정확도 확인
    # ------------------------------------

    inspection_correct = (
        result[
            "predicted_status"
        ]
        == item[
            "actual_status"
        ]
    )


    # ------------------------------------
    # 실제 OCR 입력 이미지 저장
    # ------------------------------------

    output_image_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_ocr_input.jpg"
    )

    cv2.imwrite(
        str(output_image_path),
        ocr_input,
    )


    # ------------------------------------
    # 결과 저장
    # ------------------------------------

    row = {
        "file":
            file_name,

        "actual_text":
            item["actual_text"],

        "expected_text":
            item["expected_text"],

        "actual_status":
            item["actual_status"],

        "raw_text":
            repr(
                result["raw_text"]
            ),

        "cleaned_text":
            result["cleaned_text"],

        "final_text":
            result["final_text"],

        "format_ok":
            result["format_ok"],

        "expected_match":
            result["expected_match"],

        "ocr_exact":
            ocr_exact,

        "predicted_status":
            result["predicted_status"],

        "inspection_correct":
            inspection_correct,

        "reasons":
            "|".join(
                result["reasons"]
            ),
    }

    rows.append(
        row
    )


    # ------------------------------------
    # 터미널 출력
    # ------------------------------------

    print(
        "=" * 70
    )

    print(
        "File:",
        file_name
    )

    print(
        "Actual:",
        item["actual_text"]
    )

    print(
        "Expected:",
        item["expected_text"]
    )

    print(
        "OCR:",
        result["final_text"]
    )

    print(
        "Format OK:",
        result["format_ok"]
    )

    print(
        "Expected Match:",
        result["expected_match"]
    )

    print(
        "OCR Exact:",
        ocr_exact
    )

    print(
        "Predicted:",
        result["predicted_status"]
    )

    print(
        "Actual Status:",
        item["actual_status"]
    )

    print(
        "Inspection Correct:",
        inspection_correct
    )

    print(
        "Reasons:",
        result["reasons"]
    )


# ========================================
# Batch 결과 CSV 저장
# ========================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fieldnames = [
    "file",
    "actual_text",
    "expected_text",
    "actual_status",
    "raw_text",
    "cleaned_text",
    "final_text",
    "format_ok",
    "expected_match",
    "ocr_exact",
    "predicted_status",
    "inspection_correct",
    "reasons",
]


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
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
    "Day06 Batch OCR Complete"
)