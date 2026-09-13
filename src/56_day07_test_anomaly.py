import csv
import json
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
TEST_NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

TEST_ANOMALY_DIR = Path(
    "samples/day07/test/anomaly"
)

REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)

OUTPUT_CSV = Path(
    "reports/day07_anomaly_scores.csv"
)


# --------------------------------------------------
# 2. 필요한 파일 / 폴더 확인
# --------------------------------------------------
required_paths = [
    TEST_NORMAL_DIR,
    TEST_ANOMALY_DIR,
    REFERENCE_PATH,
    RULE_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더를 확인하세요: {path}"
        )


# --------------------------------------------------
# 3. Freeze한 Rule 불러오기
# --------------------------------------------------
rules = json.loads(
    RULE_PATH.read_text(
        encoding="utf-8"
    )
)

threshold = float(
    rules["anomaly_threshold"]
)


# --------------------------------------------------
# 4. Normal Reference와 모델 준비
# --------------------------------------------------
reference_vector, _ = load_reference(
    REFERENCE_PATH
)

model = create_backbone()

rows = []


# --------------------------------------------------
# 5. 이미지 한 장을 검사하는 함수
# --------------------------------------------------
def inspect_image(
    image_path,
    actual,
    defect_type,
):
    vector, _ = extract_features(
        model,
        image_path,
    )

    score = anomaly_score(
        vector,
        reference_vector,
    )

    predicted = (
        "Anomaly"
        if score > threshold
        else "Normal"
    )


    # TP / TN / FP / FN 구분
    if (
        actual == "Anomaly"
        and predicted == "Anomaly"
    ):
        error_type = "TP"

    elif (
        actual == "Normal"
        and predicted == "Normal"
    ):
        error_type = "TN"

    elif (
        actual == "Normal"
        and predicted == "Anomaly"
    ):
        error_type = "FP"

    else:
        error_type = "FN"


    rows.append(
        {
            "file": str(image_path),
            "actual": actual,
            "defect_type": defect_type,
            "anomaly_score": round(
                score,
                6,
            ),
            "threshold": round(
                threshold,
                6,
            ),
            "predicted": predicted,
            "error_type": error_type,
            "correct": actual == predicted,
        }
    )


# --------------------------------------------------
# 6. Normal Test 검사
# --------------------------------------------------
for image_path in sorted(
    TEST_NORMAL_DIR.glob("*.png")
):
    inspect_image(
        image_path,
        actual="Normal",
        defect_type="good",
    )


# --------------------------------------------------
# 7. Anomaly Test 검사
# --------------------------------------------------
for defect_dir in sorted(
    TEST_ANOMALY_DIR.iterdir()
):
    if not defect_dir.is_dir():
        continue

    for image_path in sorted(
        defect_dir.glob("*.png")
    ):
        inspect_image(
            image_path,
            actual="Anomaly",
            defect_type=defect_dir.name,
        )


# --------------------------------------------------
# 8. 결과 확인
# --------------------------------------------------
if not rows:
    raise RuntimeError(
        "검사할 Test 이미지가 없습니다."
    )


# --------------------------------------------------
# 9. CSV 저장
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
            "defect_type",
            "anomaly_score",
            "threshold",
            "predicted",
            "error_type",
            "correct",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# 10. 결과 출력
# --------------------------------------------------
print("=" * 70)
print("DAY07 TEST COMPLETE")
print("=" * 70)

print(
    "Threshold:",
    threshold,
)

print(
    "Images   :",
    len(rows),
)

print(
    "Saved    :",
    OUTPUT_CSV,
)