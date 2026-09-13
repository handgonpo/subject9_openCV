import csv
import json
from pathlib import Path

import numpy as np

from day08_feature_core import (
    create_feature_model,
    cosine_similarity,
    extract_feature,
)


REFERENCE_DIR = Path("samples/day08/one_shot/reference")
CALIBRATION_DIR = Path("samples/day08/one_shot/calibration/normal")

ARTIFACT_PATH = Path("artifacts/day08/one_shot_reference.npy")
RULE_PATH = Path("configs/day08_one_shot_rules.json")
CSV_PATH = Path("reports/day08_one_shot_calibration.csv")

PERCENTILE = 5


reference_files = sorted(REFERENCE_DIR.glob("*.png"))

if len(reference_files) != 1:
    raise RuntimeError(
        f"Reference는 1장이어야 합니다. 현재: {len(reference_files)}"
    )

reference_path = reference_files[0]
calibration_files = sorted(CALIBRATION_DIR.glob("*.png"))

if not calibration_files:
    raise RuntimeError("Calibration 이미지가 없습니다.")


model = create_feature_model()
reference_feature = extract_feature(model, reference_path)

ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(ARTIFACT_PATH, reference_feature)


rows = []
similarities = []

for image_path in calibration_files:
    feature = extract_feature(model, image_path)
    similarity = cosine_similarity(
        reference_feature,
        feature,
    )

    similarities.append(similarity)

    rows.append(
        {
            "file": image_path.name,
            "similarity": round(similarity, 6),
        }
    )


threshold = float(
    np.percentile(
        similarities,
        PERCENTILE,
    )
)


CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "similarity"],
    )
    writer.writeheader()
    writer.writerows(rows)


rules = {
    "baseline_version": "day08-one-shot-v1.0",
    "dataset": "MVTec AD bottle",
    "method": "one_reference_cosine_similarity",
    "backbone": "resnet18_imagenet_pretrained",
    "reference_count": 1,
    "reference_file": reference_path.name,
    "reference_path": str(ARTIFACT_PATH),
    "calibration_count": len(calibration_files),
    "threshold_method": "normal_calibration_5_percentile",
    "similarity_threshold": threshold,
    "decision_rule": "similarity >= threshold => Normal",
}

RULE_PATH.parent.mkdir(parents=True, exist_ok=True)
RULE_PATH.write_text(
    json.dumps(
        rules,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 70)
print("DAY08 ONE-SHOT CALIBRATION")
print("=" * 70)
print("Reference          :", reference_path.name)
print("Calibration Images :", len(calibration_files))
print("Similarity Min     :", round(min(similarities), 6))
print("Similarity Mean    :", round(float(np.mean(similarities)), 6))
print("Similarity Max     :", round(max(similarities), 6))
print("5 Percentile       :", round(threshold, 6))
print("Rule               : similarity >= threshold => Normal")
print("Saved Rule         :", RULE_PATH)