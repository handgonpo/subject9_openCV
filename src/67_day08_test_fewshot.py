import csv
from pathlib import Path

import numpy as np

from day08_feature_core import (
    create_feature_model,
    cosine_similarity,
    extract_feature,
)


QUERY_ROOT = Path("samples/day08/common_query")
ARTIFACT_PATH = Path("artifacts/day08/fewshot_prototypes.npz")
CSV_PATH = Path("reports/day08_fewshot_results.csv")


prototype_data = np.load(ARTIFACT_PATH)
normal_prototype = prototype_data["normal_prototype"]
anomaly_prototype = prototype_data["anomaly_prototype"]

model = create_feature_model()
rows = []


def inspect(image_path, actual, sample_id):
    feature = extract_feature(model, image_path)

    normal_similarity = cosine_similarity(
        feature,
        normal_prototype,
    )

    anomaly_similarity = cosine_similarity(
        feature,
        anomaly_prototype,
    )

    predicted = (
        "Normal"
        if normal_similarity >= anomaly_similarity
        else "Anomaly"
    )

    rows.append(
        {
            "sample_id": sample_id,
            "file": image_path.name,
            "actual": actual,
            "normal_similarity": round(normal_similarity, 6),
            "anomaly_similarity": round(anomaly_similarity, 6),
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
print("DAY08 FEW-SHOT TEST")
print("=" * 70)

for row in rows:
    print(
        row["sample_id"],
        "actual=",
        row["actual"],
        "predicted=",
        row["predicted"],
        "normal_sim=",
        row["normal_similarity"],
        "anomaly_sim=",
        row["anomaly_similarity"],
        "correct=",
        row["correct"],
    )

print()
print("Saved:", CSV_PATH)