import json
from pathlib import Path

import cv2


# ========================================
# 프로젝트 경로
# ========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "roi"
)


# ========================================
# 제품 이미지 찾기
# ========================================

IMAGE_PATHS = sorted(
    [
        path
        for path in PRODUCT_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]
)


if not IMAGE_PATHS:
    raise FileNotFoundError(
        f"이미지가 없습니다: {PRODUCT_DIR}"
    )


print(
    "찾은 제품 이미지:",
    len(IMAGE_PATHS),
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ========================================
# 이미지별 ROI 저장
# ========================================

roi_by_file = {}


for image_path in IMAGE_PATHS:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )


    print()
    print(
        "Image:",
        image_path.name
    )

    print(
        "SN의 13자리 숫자 영역만 선택한 후 "
        "ENTER 또는 SPACE를 누르세요."
    )


    x, y, w, h = cv2.selectROI(
        f"Select SN ROI - {image_path.name}",
        image,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyAllWindows()


    if w == 0 or h == 0:
        raise RuntimeError(
            f"ROI가 선택되지 않았습니다: "
            f"{image_path.name}"
        )


    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)


    # ====================================
    # ROI Crop
    # ====================================

    roi = image[
        y1:y2,
        x1:x2,
    ]


    # ====================================
    # ROI 위치 Preview
    # ====================================

    preview = image.copy()

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )


    # ====================================
    # 결과 저장
    # ====================================

    preview_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_roi_preview.jpg"
    )

    roi_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_sn_roi.jpg"
    )


    cv2.imwrite(
        str(preview_path),
        preview,
    )

    cv2.imwrite(
        str(roi_path),
        roi,
    )


    roi_by_file[
        image_path.name
    ] = [
        x1,
        y1,
        x2,
        y2,
    ]


    print(
        "ROI:",
        roi_by_file[
            image_path.name
        ],
    )


# ========================================
# Day06 OCR Rule 초안
# ========================================

rules = {
    "baseline_version":
        "day06-ocr-draft",

    "reference_file":
        IMAGE_PATHS[0].name,

    "roi_by_file":
        roi_by_file,

    "preprocess_mode":
        "NOT_SELECTED",

    "clahe_clip_limit":
        2.0,

    "clahe_grid":
        [8, 8],

    "ocr_lang":
        "eng",

    "ocr_oem":
        3,

    "ocr_psm":
        7,

    "whitelist":
        "0123456789",

    "expected_pattern":
        r"^[0-9]{13}$",

    "allow_context_correction":
        False,
}


# ========================================
# Config 저장
# ========================================

CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CONFIG_PATH.write_text(
    json.dumps(
        rules,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "Saved:",
    CONFIG_PATH
)

print(
    "Day06 SN ROI Setup Complete"
)