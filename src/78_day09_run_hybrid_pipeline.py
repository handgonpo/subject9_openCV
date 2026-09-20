from pathlib import Path
import csv
import json

from day09_hybrid_core import hybrid_judge


POLICY_PATH = Path("configs/day09/hybrid_policy.json")
INPUT_PATH = Path("reports/day09/hybrid_demo_inputs.csv")
OUTPUT_PATH = Path("reports/day09/hybrid_results.csv")


policy = json.loads(
    POLICY_PATH.read_text(encoding="utf-8")
)

with INPUT_PATH.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:
    input_rows = list(csv.DictReader(file))

if not input_rows:
    raise RuntimeError("Hybrid Demo Input이 없습니다.")

rows = []

for item in input_rows:
    anomaly_score = float(item["anomaly_score"])
    anomaly_threshold = float(item["anomaly_threshold"])

    status_from_score = (
        "Anomaly"
        if anomaly_score > anomaly_threshold
        else "Normal"
    )

    score_status_ok = (
        status_from_score == item["anomaly_status"]
    )

    result = hybrid_judge(
        policy=policy,
        rule_status=item["rule_status"],
        rule_reason=item["rule_reason"],
        ocr_status=item["ocr_status"],
        ocr_reason=item["ocr_reason"],
        anomaly_status=item["anomaly_status"],
        anomaly_score=anomaly_score,
    )

    expected_final = item["expected_final"]
    final_decision = result["final_decision"]

    final_ok = final_decision == expected_final

    rows.append(
        {
            "product_id": item["product_id"],
            "rule_status": item["rule_status"],
            "ocr_status": item["ocr_status"],
            "anomaly_status": item["anomaly_status"],
            "anomaly_score": anomaly_score,
            "anomaly_threshold": anomaly_threshold,
            "score_status_check": (
                "PASS" if score_status_ok else "FAIL"
            ),
            "final_decision": final_decision,
            "expected_final": expected_final,
            "final_check": (
                "PASS" if final_ok else "FAIL"
            ),
            "reasons": "|".join(result["reasons"]),
        }
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 70)
print("DAY09 HYBRID RESULT")
print("=" * 70)

for row in rows:
    print(
        row["product_id"],
        "→ Score/Status",
        row["score_status_check"],
        "→ Final",
        row["final_check"],
        "→",
        row["reasons"],
    )

all_pass = all(
    row["score_status_check"] == "PASS"
    and row["final_check"] == "PASS"
    for row in rows
)

if not all_pass:
    raise RuntimeError("Hybrid Self Check에 실패했습니다.")

print()
print("Hybrid Self Check: PASS")
print("Saved:", OUTPUT_PATH)