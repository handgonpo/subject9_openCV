import json
from pathlib import Path


# ========================================
# 프로젝트 경로
# ========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)


# ========================================
# 기존 Rule 읽기
# ========================================

rules = json.loads(
    CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


# ========================================
# 실험 결과를 반영하여 Rule Freeze
# ========================================

rules[
    "baseline_version"
] = "day06-ocr-v1.0"

rules[
    "preprocess_mode"
] = "original"

rules[
    "ocr_psm"
] = 7

rules[
    "whitelist"
] = "0123456789"

rules[
    "expected_pattern"
] = r"^[0-9]{13}$"

rules[
    "allow_context_correction"
] = False


# ========================================
# Config 저장
# ========================================

CONFIG_PATH.write_text(
    json.dumps(
        rules,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ========================================
# 결과 출력
# ========================================

print(
    "Day06 OCR Rule Freeze"
)

print(
    "Version:",
    rules["baseline_version"]
)

print(
    "Preprocess:",
    rules["preprocess_mode"]
)

print(
    "PSM:",
    rules["ocr_psm"]
)

print(
    "Whitelist:",
    rules["whitelist"]
)

print(
    "Pattern:",
    rules["expected_pattern"]
)

print(
    "Context Correction:",
    rules["allow_context_correction"]
)

print(
    "Saved:",
    CONFIG_PATH
)