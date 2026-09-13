from pathlib import Path
import csv

import cv2
import numpy as np


input_files = {
    "cap_on": Path(
        "samples/day03/reference/cap_on.jpg"
    ),
    "cap_off": Path(
        "samples/day03/reference/cap_off.jpg"
    ),
}


output_dir = Path(
    "outputs/day03/feature"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


report_path = Path(
    "reports/day03_reference_features.csv"
)


# 작은 Noise Contour를 제외하기 위한
# 교육용 시작 비율입니다.
MIN_CONTOUR_AREA_RATIO = 0.001


def measure_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    otsu_value, binary = cv2.threshold(
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

    height, width = image.shape[:2]

    image_area = float(
        width * height
    )

    min_area = (
        image_area
        * MIN_CONTOUR_AREA_RATIO
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= min_area
    ]

    features = {
        "otsu_value": float(otsu_value),
        "valid_count": len(valid_contours),
        "largest_area": 0.0,
        "largest_area_ratio": 0.0,
        "center_x_norm": "",
        "center_y_norm": "",
        "touches_border": "",
    }

    annotated = image.copy()

    if not valid_contours:

        return (
            features,
            binary,
            cleaned,
            annotated,
        )

    largest = max(
        valid_contours,
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

    cx = None
    cy = None

    if moments["m00"] != 0:

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

    touches_border = (
        x <= 0
        or y <= 0
        or x + w >= width
        or y + h >= height
    )

    features[
        "largest_area"
    ] = area

    features[
        "largest_area_ratio"
    ] = (
        area / image_area
        if image_area > 0
        else 0.0
    )

    if cx is not None:
        features[
            "center_x_norm"
        ] = cx / width

    if cy is not None:
        features[
            "center_y_norm"
        ] = cy / height

    features[
        "touches_border"
    ] = touches_border

    cv2.drawContours(
        annotated,
        [largest],
        -1,
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2,
    )

    if (
        cx is not None
        and cy is not None
    ):

        cv2.circle(
            annotated,
            (
                int(cx),
                int(cy),
            ),
            5,
            (0, 0, 255),
            -1,
        )

    return (
        features,
        binary,
        cleaned,
        annotated,
    )


rows = []


for label, image_path in input_files.items():

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: "
            f"{image_path}"
        )

    (
        features,
        binary,
        cleaned,
        annotated,
    ) = measure_features(
        image
    )

    row = {
        "label": label,
        "file": image_path.name,
        **features,
    }

    rows.append(
        row
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_binary.jpg"
        ),
        binary,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_cleaned.jpg"
        ),
        cleaned,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_contour.jpg"
        ),
        annotated,
    )

    print()
    print("Label:", label)

    for key, value in features.items():
        print(
            f"{key}:",
            value,
        )


report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with report_path.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [
        "label",
        "file",
        "otsu_value",
        "valid_count",
        "largest_area",
        "largest_area_ratio",
        "center_x_norm",
        "center_y_norm",
        "touches_border",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print()
print(
    "측정 결과 CSV:",
    report_path,
)