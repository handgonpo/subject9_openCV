import csv
from pathlib import Path

import pandas as pd


ONE_PATH = Path("reports/day08_one_shot_results.csv")
FEW_PATH = Path("reports/day08_fewshot_results.csv")
OUTPUT_PATH = Path("reports/day08_method_comparison.csv")


one = pd.read_csv(ONE_PATH)
few = pd.read_csv(FEW_PATH)


if set(one["sample_id"]) != set(few["sample_id"]):
    raise RuntimeError(
        "One-shot과 Few-shot의 Common Query가 서로 다릅니다."
    )


def metrics(df):
    actual = df["actual"]
    predicted = df["predicted"]

    tp = int(((actual == "Anomaly") & (predicted == "Anomaly")).sum())
    tn = int(((actual == "Normal") & (predicted == "Normal")).sum())
    fp = int(((actual == "Normal") & (predicted == "Anomaly")).sum())
    fn = int(((actual == "Anomaly") & (predicted == "Normal")).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "samples": total,
        "accuracy": accuracy,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


rows = []

for method, df in [
    ("one_shot", one),
    ("few_shot", few),
]:
    result = metrics(df)
    result["method"] = method
    rows.append(result)


fieldnames = [
    "method",
    "samples",
    "accuracy",
    "anomaly_precision",
    "anomaly_recall",
    "tp",
    "tn",
    "fp",
    "fn",
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )
    writer.writeheader()

    for row in rows:
        output = dict(row)
        output["accuracy"] = round(output["accuracy"], 4)
        output["anomaly_precision"] = round(output["anomaly_precision"], 4)
        output["anomaly_recall"] = round(output["anomaly_recall"], 4)
        writer.writerow(output)


print("=" * 70)
print("DAY08 METHOD COMPARISON")
print("=" * 70)

for row in rows:
    print()
    print("Method            :", row["method"])
    print("Samples           :", row["samples"])
    print("Accuracy          :", f"{row['accuracy']:.4f}")
    print("Anomaly Precision :", f"{row['anomaly_precision']:.4f}")
    print("Anomaly Recall    :", f"{row['anomaly_recall']:.4f}")
    print("TP / TN / FP / FN :", row["tp"], row["tn"], row["fp"], row["fn"])

print()
print("Saved:", OUTPUT_PATH)