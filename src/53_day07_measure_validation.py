import csv
from pathlib import Path

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
    load_reference,
)


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
VAL_DIR = Path(
    "samples/day07/validation/normal"
)

REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

OUTPUT_CSV = Path(
    "reports/day07_validation_scores.csv"
)


# --------------------------------------------------
# 2. 필요한 데이터가 있는지 확인
# --------------------------------------------------
if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation 폴더를 확인하세요: {VAL_DIR}"
    )

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Normal Reference를 확인하세요: {REFERENCE_PATH}"
    )


image_files = sorted(
    VAL_DIR.glob("*.png")
)

if not image_files:
    raise RuntimeError(
        "Validation 이미지가 없습니다."
    )


# --------------------------------------------------
# 3. Normal Reference 불러오기
# --------------------------------------------------
reference_vector, _ = load_reference(
    REFERENCE_PATH
)


# --------------------------------------------------
# 4. Feature 추출용 ResNet18 준비
# --------------------------------------------------
model = create_backbone()


# --------------------------------------------------
# 5. Validation 이미지의 Score 계산
# --------------------------------------------------
rows = []

print(
    "Normal Validation:",
    len(image_files),
)

for index, image_path in enumerate(
    image_files,
    start=1,
):
    vector, _ = extract_features(
        model,
        image_path,
    )

    score = anomaly_score(
        vector,
        reference_vector,
    )

    rows.append(
        {
            "file": image_path.name,
            "actual": "Normal",
            "anomaly_score": round(
                score,
                6,
            ),
        }
    )

    if (
        index == 1
        or index % 10 == 0
        or index == len(image_files)
    ):
        print(
            f"[{index}/{len(image_files)}]",
            image_path.name,
        )


# --------------------------------------------------
# 6. CSV 저장
# --------------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "actual",
            "anomaly_score",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# 7. 결과 확인
# --------------------------------------------------
print("=" * 70)
print("Validation Score 측정 완료")
print("=" * 70)

print(
    "Images:",
    len(rows),
)

print(
    "Saved :",
    OUTPUT_CSV,
)