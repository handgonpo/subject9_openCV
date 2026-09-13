from pathlib import Path

import cv2
import numpy as np


image_path = Path(
    "samples/day03/reference/cap_on.jpg"
)

output_dir = Path(
    "outputs/day03/contour"
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


roi = image

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY,
)


_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3),
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


contours, _ = cv2.findContours(
    cleaned.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)


print("전체 Contour 수:", len(contours))


if not contours:

    cv2.imwrite(
        str(output_dir / "binary.jpg"),
        binary,
    )

    cv2.imwrite(
        str(output_dir / "cleaned.jpg"),
        cleaned,
    )

    print("Contour를 찾지 못했습니다.")

    raise SystemExit


largest = max(
    contours,
    key=cv2.contourArea,
)


area = float(
    cv2.contourArea(largest)
)


x, y, w, h = cv2.boundingRect(
    largest
)


moments = cv2.moments(
    largest
)


center = None

if moments["m00"] != 0:

    cx = int(
        moments["m10"]
        / moments["m00"]
    )

    cy = int(
        moments["m01"]
        / moments["m00"]
    )

    center = (
        cx,
        cy,
    )


height, width = roi.shape[:2]

roi_area = float(
    width * height
)

area_ratio = (
    area / roi_area
    if roi_area > 0
    else 0.0
)


result = roi.copy()


cv2.drawContours(
    result,
    [largest],
    -1,
    (0, 255, 0),
    2,
)


cv2.rectangle(
    result,
    (x, y),
    (x + w, y + h),
    (255, 0, 0),
    2,
)


if center is not None:

    cv2.circle(
        result,
        center,
        5,
        (0, 0, 255),
        -1,
    )


cv2.putText(
    result,
    f"AreaRatio: {area_ratio:.3f}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2,
)


cv2.imwrite(
    str(output_dir / "binary.jpg"),
    binary,
)

cv2.imwrite(
    str(output_dir / "cleaned.jpg"),
    cleaned,
)

cv2.imwrite(
    str(output_dir / "contour_result.jpg"),
    result,
)


print("Largest Area:", area)
print("Area Ratio:", area_ratio)
print("BBox:", (x, y, w, h))
print("Center:", center)
print("Contour 측정 완료")