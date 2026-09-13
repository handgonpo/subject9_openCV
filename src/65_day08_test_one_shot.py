import csv
import json
from pathlib import Path

import numpy as np

from day08_feature_core import (
    create_feature_model,
    cosine_similarity,
    extract_feature,
)


QUERY_ROOT = Path("samples/day08/common_query")
REFERENCE_PATH = Path("artifacts/day08/one_shot_reference.npy")
RULE_PATH = Path("configs/day08_one_shot_rules.json")
CSV_PATH = Path("reports/day08_one_shot_results.csv")


reference_feature = np.load(REFERENCE_PATH)

rules = json.loads(
    RULE_PATH.read_text(encoding="utf-8")
)

threshold = float(rules["similarity_threshold"])
model = create_feature_model()

rows = []


def inspect(image_path, actual, sample_id):
    feature = extract_feature(model, image_path)

    similarity = cosine_similarity(
        reference_feature,
        feature,
    )

    predicted = (
        "Normal"
        if similarity >= threshold
        else "Anomaly"
    )

    rows.append(
        {
            "sample_id": sample_id,
            "file": image_path.name,
            "actual": actual,
            "similarity": round(similarity, 6),
            "threshold": round(threshold, 6),
            "predicted": predicted,
            "correct": predicted == actual,
        }
    )


for image_path in sorted(
    (QUERY_ROOT / "normal").glob("*.png")
):
    inspect(
        image_path,
        "Normal",
        f"normal/{image_path.name}",
    )


anomaly_dir = (
    QUERY_ROOT
    / "anomaly"
    / "broken_small"
)

for image_path in sorted(anomaly_dir.glob("*.png")):
    inspect(
        image_path,
        "Anomaly",
        f"anomaly/broken_small/{image_path.name}",
    )


CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)


print("=" * 70)
print("DAY08 ONE-SHOT TEST")
print("=" * 70)
print("Threshold:", round(threshold, 6))
print()

for row in rows:
    print(
        row["sample_id"],
        "actual=",
        row["actual"],
        "predicted=",
        row["predicted"],
        "similarity=",
        row["similarity"],
        "correct=",
        row["correct"],
    )

print()
print("Saved:", CSV_PATH)