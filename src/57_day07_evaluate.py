import shutil
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
INPUT_CSV = Path(
    "reports/day07_anomaly_scores.csv"
)

REPORT_PATH = Path(
    "reports/day07_test_metrics.md"
)

FP_DIR = Path(
    "outputs/day07/fp"
)

FN_DIR = Path(
    "outputs/day07/fn"
)


# --------------------------------------------------
# 2. 0으로 나누는 오류 방지
# --------------------------------------------------
def safe_divide(a, b):
    return a / b if b else 0.0


# --------------------------------------------------
# 3. Test 결과 CSV 확인
# --------------------------------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Test 결과 CSV를 확인하세요: {INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)

if df.empty:
    raise RuntimeError(
        "Test 결과가 없습니다."
    )


# --------------------------------------------------
# 4. TP / TN / FP / FN 개수 계산
# --------------------------------------------------
counts = (
    df["error_type"]
    .value_counts()
    .to_dict()
)

tp = int(
    counts.get("TP", 0)
)

tn = int(
    counts.get("TN", 0)
)

fp = int(
    counts.get("FP", 0)
)

fn = int(
    counts.get("FN", 0)
)

total = (
    tp
    + tn
    + fp
    + fn
)


# --------------------------------------------------
# 5. 성능 지표 계산
# --------------------------------------------------
accuracy = safe_divide(
    tp + tn,
    total,
)

precision = safe_divide(
    tp,
    tp + fp,
)

recall = safe_divide(
    tp,
    tp + fn,
)


# --------------------------------------------------
# 6. FP / FN 폴더 준비
# --------------------------------------------------
for directory in [
    FP_DIR,
    FN_DIR,
]:
    if directory.exists():
        shutil.rmtree(
            directory
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# --------------------------------------------------
# 7. FP / FN 이미지 따로 복사
# --------------------------------------------------
for _, row in df.iterrows():
    source = Path(
        row["file"]
    )

    if not source.exists():
        continue

    if row["error_type"] == "FP":
        shutil.copy2(
            source,
            FP_DIR / source.name,
        )

    elif row["error_type"] == "FN":
        defect_type = row[
            "defect_type"
        ]

        destination = (
            FN_DIR
            / f"{defect_type}_{source.name}"
        )

        shutil.copy2(
            source,
            destination,
        )


# --------------------------------------------------
# 8. 이미지 종류별 정확도
# --------------------------------------------------
per_type = (
    df.groupby(
        "defect_type"
    )["correct"]
    .mean()
    .sort_index()
)


# --------------------------------------------------
# 9. 평가 Report 작성
# --------------------------------------------------
report_lines = [
    "# Day07 Test Metrics",
    "",
    "## Confusion Matrix",
    "",
    "| Actual / Predicted | Anomaly | Normal |",
    "|---|---:|---:|",
    f"| Anomaly | TP = {tp} | FN = {fn} |",
    f"| Normal | FP = {fp} | TN = {tn} |",
    "",
    "## Metrics",
    "",
    f"- Accuracy: {accuracy:.4f}",
    f"- Anomaly Precision: {precision:.4f}",
    f"- Anomaly Recall: {recall:.4f}",
    f"- FP: {fp}",
    f"- FN: {fn}",
    "",
    "## Type Accuracy",
    "",
]


for defect_type, value in per_type.items():
    report_lines.append(
        f"- {defect_type}: {value:.4f}"
    )


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# --------------------------------------------------
# 10. 결과 출력
# --------------------------------------------------
print("=" * 70)
print("DAY07 TEST EVALUATION")
print("=" * 70)

print(
    "Total    :",
    total,
)

print(
    "TP       :",
    tp,
)

print(
    "TN       :",
    tn,
)

print(
    "FP       :",
    fp,
)

print(
    "FN       :",
    fn,
)

print(
    "Accuracy :",
    round(
        accuracy,
        4,
    ),
)

print(
    "Precision:",
    round(
        precision,
        4,
    ),
)

print(
    "Recall   :",
    round(
        recall,
        4,
    ),
)

print()

print(
    per_type
)

print()

print(
    "Report:",
    REPORT_PATH,
)