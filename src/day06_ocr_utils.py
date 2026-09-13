import json
from pathlib import Path

import cv2
import pytesseract


# ========================================
# 프로젝트 경로
# ========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DEFAULT_RULES_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)


# ========================================
# OCR Rule 읽기
# ========================================

def load_rules(
    path=DEFAULT_RULES_PATH,
):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


# ========================================
# 이미지별 ROI 좌표 가져오기
# ========================================

def get_roi_for_file(
    rules,
    file_name,
):
    roi_by_file = rules[
        "roi_by_file"
    ]

    if file_name not in roi_by_file:
        raise KeyError(
            f"ROI 정보가 없습니다: {file_name}"
        )

    return roi_by_file[
        file_name
    ]


# ========================================
# ROI Crop
# ========================================

def crop_roi(
    image,
    roi,
):
    x1, y1, x2, y2 = roi

    return image[
        y1:y2,
        x1:x2,
    ]


# ========================================
# CLAHE
# ========================================

def apply_clahe(
    gray,
    rules,
):
    clahe = cv2.createCLAHE(
        clipLimit=float(
            rules["clahe_clip_limit"]
        ),
        tileGridSize=tuple(
            rules["clahe_grid"]
        ),
    )

    return clahe.apply(
        gray
    )


# ========================================
# 전처리 후보 만들기
# ========================================

def make_preprocess_variants(
    roi,
    rules,
):
    # Gray
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    # CLAHE
    clahe = apply_clahe(
        gray,
        rules,
    )

    # Gray + Otsu
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # CLAHE + Otsu
    _, clahe_otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    return {
        "original": roi,
        "gray": gray,
        "clahe": clahe,
        "otsu": otsu,
        "clahe_otsu": clahe_otsu,
    }


# ========================================
# 선택된 전처리 적용
# ========================================

def preprocess_for_ocr(
    roi,
    rules,
):
    variants = (
        make_preprocess_variants(
            roi,
            rules,
        )
    )

    mode = rules[
        "preprocess_mode"
    ]

    if mode not in variants:
        raise ValueError(
            "아직 사용할 전처리가 "
            f"선택되지 않았습니다: {mode}"
        )

    return variants[
        mode
    ]


# ========================================
# Tesseract 설정 만들기
# ========================================

def build_ocr_config(
    rules,
):
    return (
        f"--oem {rules['ocr_oem']} "
        f"--psm {rules['ocr_psm']} "
        "-c "
        f"tessedit_char_whitelist="
        f"{rules['whitelist']}"
    )


# ========================================
# OCR 실행
# ========================================

def run_ocr(
    image,
    rules,
):
    config = (
        build_ocr_config(
            rules
        )
    )

    return pytesseract.image_to_string(
        image,
        lang=rules["ocr_lang"],
        config=config,
    )


def rotate_image(
    image,
    angle,
):
    height, width = (
        image.shape[:2]
    )

    center = (
        width / 2,
        height / 2,
    )

    matrix = (
        cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )
    )

    if image.ndim == 2:
        border_value = 255
    else:
        border_value = (
            255,
            255,
            255,
        )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )