import csv

from day06_ocr_utils import (
    PROJECT_ROOT,
)


# ========================================
# 경로 설정
# ========================================

RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_results.csv"
)

METRIC_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_metrics.md"
)


# ========================================
# Batch 결과 파일 확인
# ========================================

if not RESULT_PATH.exists():

    raise FileNotFoundError(
        "먼저 45_day06_batch_ocr.py를 "
        "실행해야 합니다.\n"
        f"결과 파일이 없습니다: "
        f"{RESULT_PATH}"
    )


# ========================================
# CSV 읽기
# ========================================

with RESULT_PATH.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:

    rows = list(
        csv.DictReader(file)
    )


if not rows:

    raise RuntimeError(
        "day06_ocr_results.csv에 "
        "결과가 없습니다."
    )


# ========================================
# 문자열 True / False 변환
# ========================================

def as_bool(
    value,
):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


# ========================================
# 기본 개수 계산
# ========================================

count = len(
    rows
)


ocr_correct = sum(
    as_bool(
        row["ocr_exact"]
    )
    for row in rows
)


inspection_correct_count = sum(
    as_bool(
        row["inspection_correct"]
    )
    for row in rows
)


format_ok_count = sum(
    as_bool(
        row["format_ok"]
    )
    for row in rows
)


expected_match_count = sum(
    as_bool(
        row["expected_match"]
    )
    for row in rows
)


# ========================================
# 정확도 계산
# ========================================

ocr_accuracy = (
    ocr_correct
    / count
)


inspection_accuracy = (
    inspection_correct_count
    / count
)


# ========================================
# 실패 이미지 찾기
# ========================================

ocr_failed_files = [
    row["file"]
    for row in rows
    if not as_bool(
        row["ocr_exact"]
    )
]


inspection_failed_files = [
    row["file"]
    for row in rows
    if not as_bool(
        row["inspection_correct"]
    )
]


# ========================================
# Markdown Report 작성
# ========================================

ocr_failure_text = (
    "\n".join(
        "- " + name
        for name in ocr_failed_files
    )
    if ocr_failed_files
    else "- 없음"
)


inspection_failure_text = (
    "\n".join(
        "- " + name
        for name in inspection_failed_files
    )
    if inspection_failed_files
    else "- 없음"
)


text = f"""# Day 06 OCR Metrics

## Dataset

Images: {count}

## OCR Exact Accuracy

{ocr_accuracy:.4f}

OCR Correct: {ocr_correct} / {count}

## Inspection Accuracy

{inspection_accuracy:.4f}

Inspection Correct: {inspection_correct_count} / {count}

## Format

Format OK: {format_ok_count} / {count}

## Expected Match

Expected Match: {expected_match_count} / {count}

## OCR Failures

{ocr_failure_text}

## Inspection Failures

{inspection_failure_text}
"""


# ========================================
# Report 저장
# ========================================

METRIC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


METRIC_PATH.write_text(
    text,
    encoding="utf-8",
)


print(
    text
)

print(
    "Saved:",
    METRIC_PATH
)