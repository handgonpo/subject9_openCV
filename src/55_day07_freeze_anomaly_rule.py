import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
VALIDATION_CSV = Path(
    "reports/day07_validation_scores.csv"
)

RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)


# --------------------------------------------------
# 2. 오늘 선택한 Threshold 기준
# --------------------------------------------------
SELECTED_PERCENTILE = 95


# --------------------------------------------------
# 3. Validation Score 확인
# --------------------------------------------------
if not VALIDATION_CSV.exists():
    raise FileNotFoundError(
        f"Validation Score 파일을 확인하세요: {VALIDATION_CSV}"
    )


df = pd.read_csv(
    VALIDATION_CSV
)

if "anomaly_score" not in df.columns:
    raise RuntimeError(
        "anomaly_score 컬럼이 없습니다."
    )

scores = df[
    "anomaly_score"
].to_numpy()

if len(scores) == 0:
    raise RuntimeError(
        "Validation Score가 없습니다."
    )


# --------------------------------------------------
# 4. 선택한 95 Percentile Threshold 계산
# --------------------------------------------------
threshold = round(
    float(
        np.percentile(
            scores,
            SELECTED_PERCENTILE,
        )
    ),
    6,
)


# --------------------------------------------------
# 5. Baseline Rule 구성
# --------------------------------------------------
rules = {
    "baseline_version": "day07-anomaly-v1.0",
    "dataset": "MVTec AD bottle",
    "backbone": "resnet18_imagenet_pretrained",
    "feature_layer": "layer3",
    "reference": "mean_normal_feature",
    "reference_path": "artifacts/day07/normal_reference.npz",
    "image_size": 224,
    "selected_percentile": SELECTED_PERCENTILE,
    "threshold_method": "normal_validation_95_percentile",
    "anomaly_threshold": threshold,
    "decision_rule": "score > threshold => Anomaly",
}


# --------------------------------------------------
# 6. Rule JSON 저장
# --------------------------------------------------
RULE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RULE_PATH.write_text(
    json.dumps(
        rules,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# --------------------------------------------------
# 7. 결과 확인
# --------------------------------------------------
print("=" * 70)
print("DAY07 RULE FREEZE")
print("=" * 70)

print(
    "Version   :",
    rules["baseline_version"],
)

print(
    "Percentile:",
    rules["selected_percentile"],
)

print(
    "Threshold :",
    rules["anomaly_threshold"],
)

print(
    "Rule      :",
    rules["decision_rule"],
)

print(
    "Saved     :",
    RULE_PATH,
)