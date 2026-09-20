from pathlib import Path
import csv
import json


DAY07_RULE_PATH = Path("configs/day07_anomaly_rules.json")
POLICY_PATH = Path("configs/day09/hybrid_policy.json")
INPUT_PATH = Path("reports/day09/hybrid_demo_inputs.csv")


if not DAY07_RULE_PATH.exists():
    raise FileNotFoundError(DAY07_RULE_PATH)

day07_rule = json.loads(
    DAY07_RULE_PATH.read_text(encoding="utf-8")
)

threshold = float(day07_rule["anomaly_threshold"])

if threshold <= 0:
    raise RuntimeError(
        f"Day07 Anomaly Threshold가 올바르지 않습니다: {threshold}"
    )

margin = max(threshold * 0.25, 0.0001)

normal_score_1 = max(0.0, threshold - margin)
normal_score_2 = max(0.0, threshold - margin * 0.5)
anomaly_score_1 = threshold + margin
anomaly_score_2 = threshold + margin * 2.0


policy = {
    "policy_version": "day09-hybrid-v1.0",
    "rule_required": True,
    "ocr_required": True,
    "anomaly_required": True,
    "final_rule": "RULE_OK AND OCR_OK AND ANOMALY_NORMAL",
}

cases = [
    {
        "product_id": "CASE-001",
        "rule_status": "OK",
        "rule_reason": "",
        "ocr_status": "OK",
        "ocr_reason": "",
        "anomaly_status": "Normal",
        "anomaly_score": round(normal_score_1, 6),
        "anomaly_threshold": round(threshold, 6),
        "expected_final": "OK",
    },
    {
        "product_id": "CASE-002",
        "rule_status": "NG",
        "rule_reason": "LOW_SIFT_MATCH",
        "ocr_status": "OK",
        "ocr_reason": "",
        "anomaly_status": "Normal",
        "anomaly_score": round(normal_score_2, 6),
        "anomaly_threshold": round(threshold, 6),
        "expected_final": "NG",
    },
    {
        "product_id": "CASE-003",
        "rule_status": "OK",
        "rule_reason": "",
        "ocr_status": "NG",
        "ocr_reason": "NG_EXPECTED_MISMATCH",
        "anomaly_status": "Normal",
        "anomaly_score": round(normal_score_1, 6),
        "anomaly_threshold": round(threshold, 6),
        "expected_final": "NG",
    },
    {
        "product_id": "CASE-004",
        "rule_status": "OK",
        "rule_reason": "",
        "ocr_status": "OK",
        "ocr_reason": "",
        "anomaly_status": "Anomaly",
        "anomaly_score": round(anomaly_score_1, 6),
        "anomaly_threshold": round(threshold, 6),
        "expected_final": "NG",
    },
    {
        "product_id": "CASE-005",
        "rule_status": "NG",
        "rule_reason": "LOW_INLIER_RATIO",
        "ocr_status": "NG",
        "ocr_reason": "NG_FORMAT",
        "anomaly_status": "Anomaly",
        "anomaly_score": round(anomaly_score_2, 6),
        "anomaly_threshold": round(threshold, 6),
        "expected_final": "NG",
    },
]


POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
POLICY_PATH.write_text(
    json.dumps(policy, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with INPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(cases[0].keys()),
    )
    writer.writeheader()
    writer.writerows(cases)

print("Policy            :", POLICY_PATH)
print("Cases             :", INPUT_PATH)
print("Count             :", len(cases))
print("Day07 Threshold   :", round(threshold, 6))