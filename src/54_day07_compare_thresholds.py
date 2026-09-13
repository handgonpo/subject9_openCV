from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. 입력 / 출력 파일
# --------------------------------------------------
INPUT_CSV = Path(
    "reports/day07_validation_scores.csv"
)

OUTPUT_CSV = Path(
    "reports/day07_threshold_compare.csv"
)


# --------------------------------------------------
# 2. Validation Score 불러오기
# --------------------------------------------------
df = pd.read_csv(
    INPUT_CSV
)

scores = df[
    "anomaly_score"
].to_numpy()


# --------------------------------------------------
# 3. Threshold 후보 계산
# --------------------------------------------------
candidates = []

for percentile in [
    90,
    95,
    99,
]:
    threshold = float(
        np.percentile(
            scores,
            percentile,
        )
    )

    # 해당 Threshold보다 높은 정상 이미지 수
    normal_over = int(
        (
            scores
            > threshold
        ).sum()
    )

    candidates.append(
        {
            "percentile": percentile,
            "threshold": round(
                threshold,
                6,
            ),
            "validation_normal_over_threshold": normal_over,
        }
    )


# --------------------------------------------------
# 4. 결과를 표 형태로 만들기
# --------------------------------------------------
result = pd.DataFrame(
    candidates
)


# --------------------------------------------------
# 5. CSV 저장
# --------------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# --------------------------------------------------
# 6. 결과 출력
# --------------------------------------------------
print("=" * 70)
print("Validation Score Summary")
print("=" * 70)

print(
    "Count:",
    len(scores),
)

print(
    "Min  :",
    round(
        float(scores.min()),
        6,
    ),
)

print(
    "Mean :",
    round(
        float(scores.mean()),
        6,
    ),
)

print(
    "Max  :",
    round(
        float(scores.max()),
        6,
    ),
)

print()

print(
    result.to_string(
        index=False
    )
)

print()

print(
    "Saved:",
    OUTPUT_CSV,
)