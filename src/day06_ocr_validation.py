import re


# ========================================
# OCR 문자열 정리
# ========================================

def clean_ocr_text(
    raw_text,
):
    # 앞뒤 공백과 줄바꿈 제거
    text = raw_text.strip()

    # 중간에 들어간 공백 제거
    text = re.sub(
        r"\s+",
        "",
        text,
    )

    return text


# ========================================
# SN 형식 검사
# ========================================

def validate_pattern(
    text,
    pattern,
):
    return bool(
        re.fullmatch(
            pattern,
            text,
        )
    )


# ========================================
# 최종 제품 판정
# ========================================

def inspect_text(
    raw_text,
    expected_text,
    pattern,
):
    # 1. OCR 문자열 정리
    cleaned_text = (
        clean_ocr_text(
            raw_text
        )
    )

    reasons = []


    # 2. OCR 결과가 비어 있는지 확인
    if not cleaned_text:
        reasons.append(
            "NG_OCR_EMPTY"
        )


    # 3. 13자리 SN 형식인지 확인
    format_ok = (
        validate_pattern(
            cleaned_text,
            pattern,
        )
    )

    if not format_ok:
        reasons.append(
            "NG_FORMAT"
        )


    # 4. Expected SN과 비교
    expected_match = (
        cleaned_text
        == expected_text
    )

    if not expected_match:
        reasons.append(
            "NG_EXPECTED_MISMATCH"
        )


    # 5. 최종 OK / NG
    predicted_status = (
        "OK"
        if not reasons
        else "NG"
    )


    return {
        "raw_text":
            raw_text,

        "cleaned_text":
            cleaned_text,

        "final_text":
            cleaned_text,

        "format_ok":
            format_ok,

        "expected_match":
            expected_match,

        "predicted_status":
            predicted_status,

        "reasons":
            reasons,
    }