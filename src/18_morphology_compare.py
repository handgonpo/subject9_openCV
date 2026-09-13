from pathlib import Path

import cv2
import numpy as np


image_path = Path(
    "samples/day03/reference/cap_on.jpg"
)

output_dir = Path(
    "outputs/day03/morphology"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 1. 이미 ROI 이미지이므로 다시 Crop하지 않습니다.
# --------------------------------------------------

roi = image


# --------------------------------------------------
# 2. Gray
# --------------------------------------------------

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 3. Day02 기준과 같은 Otsu Binary
# 어두운 병뚜껑을 흰색 객체로 봅니다.
# --------------------------------------------------

otsu_value, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


cv2.imwrite(
    str(output_dir / "01_gray.jpg"),
    gray,
)

cv2.imwrite(
    str(output_dir / "02_binary.jpg"),
    binary,
)


# --------------------------------------------------
# 4. Kernel 크기를 바꾸어 비교합니다.
# --------------------------------------------------

kernel_sizes = [
    3,
    5,
    9,
]


for size in kernel_sizes:

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"kernel_{size}_opening.jpg"
        ),
        opened,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"kernel_{size}_closing.jpg"
        ),
        cleaned,
    )


print("입력:", image_path)
print("Otsu Threshold:", otsu_value)
print("Morphology 비교 결과 저장 완료")