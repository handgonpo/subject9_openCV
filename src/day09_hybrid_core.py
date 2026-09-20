def hybrid_judge(
    policy,
    rule_status,
    rule_reason,
    ocr_status,
    ocr_reason,
    anomaly_status,
    anomaly_score,
):
    reasons = []

    rule_ok = rule_status == "OK"
    ocr_ok = ocr_status == "OK"
    anomaly_ok = anomaly_status == "Normal"

    if policy.get("rule_required", True) and not rule_ok:
        reasons.append(
            "RULE:" + (rule_reason or "NG_RULE")
        )

    if policy.get("ocr_required", True) and not ocr_ok:
        reasons.append(
            "OCR:" + (ocr_reason or "NG_OCR")
        )

    if policy.get("anomaly_required", True) and not anomaly_ok:
        reasons.append("ANOMALY:ANOMALY_SCORE_HIGH")

    required_checks = []

    if policy.get("rule_required", True):
        required_checks.append(rule_ok)

    if policy.get("ocr_required", True):
        required_checks.append(ocr_ok)

    if policy.get("anomaly_required", True):
        required_checks.append(anomaly_ok)

    final_ok = all(required_checks) if required_checks else True

    return {
        "final_decision": "OK" if final_ok else "NG",
        "reasons": reasons,
        "rule_ok": rule_ok,
        "ocr_ok": ocr_ok,
        "anomaly_ok": anomaly_ok,
        "anomaly_score": float(anomaly_score),
    }