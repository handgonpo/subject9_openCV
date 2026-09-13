from pathlib import Path

import cv2


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

low_contrast_path = Path(
    "samples/day02/low_contrast.jpg"
)

output_dir = Path(
    "outputs/day02/threshold"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# low_contrast.jpg 확인
# --------------------------------------------------

image = cv2.imread(
    str(low_contrast_path)
)


# low_contrast.jpg가 없다면
# 기준 ROI에서 다시 생성
if image is None:

    print(
        "low_contrast.jpg가 없어 "
        "기준 이미지에서 생성합니다."
    )

    reference = cv2.imread(
        str(reference_path)
    )

    if reference is None:
        raise FileNotFoundError(
            f"기준 이미지를 읽지 못했습니다: "
            f"{reference_path}"
        )

    image = cv2.convertScaleAbs(
        reference,
        alpha=0.35,
        beta=80,
    )

    cv2.imwrite(
        str(low_contrast_path),
        image,
    )

    print(
        "문제 이미지 생성:",
        low_contrast_path,
    )


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)

cv2.imwrite(
    str(output_dir / "gray.jpg"),
    gray,
)


# --------------------------------------------------
# 고정 Threshold 비교
# --------------------------------------------------

threshold_values = [
    80,
    120,
    160,
]


for value in threshold_values:

    _, binary = cv2.threshold(
        gray,
        value,
        255,
        cv2.THRESH_BINARY_INV,
    )

    output_path = (
        output_dir
        / f"threshold_{value}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        binary,
    )

    print(
        "저장:",
        output_path,
    )


# --------------------------------------------------
# Otsu Threshold
# --------------------------------------------------

otsu_value, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)

otsu_path = (
    output_dir / "otsu.jpg"
)

cv2.imwrite(
    str(otsu_path),
    otsu,
)

print(
    "Otsu Threshold:",
    otsu_value,
)

print(
    "저장:",
    otsu_path,
)