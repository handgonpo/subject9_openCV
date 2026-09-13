from pathlib import Path

import cv2
import numpy as np


CAP_ON_PATH = Path(
    "samples/day03/reference/cap_on.jpg"
)

CAP_OFF_PATH = Path(
    "samples/day03/reference/cap_off.jpg"
)

OUTPUT_DIR = Path(
    "outputs/day03/rule"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MIN_CONTOUR_AREA_RATIO = 0.001


def measure_area_ratio(
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: "
            f"{image_path}"
        )

    gray = cv2.cvtColor(
        image,
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

    height, width = image.shape[:2]

    image_area = float(
        width * height
    )

    min_area = (
        image_area
        * MIN_CONTOUR_AREA_RATIO
    )

    valid = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= min_area
    ]

    if not valid:

        return (
            0.0,
            image,
            None,
        )

    largest = max(
        valid,
        key=cv2.contourArea,
    )

    largest_area = float(
        cv2.contourArea(largest)
    )

    area_ratio = (
        largest_area / image_area
        if image_area > 0
        else 0.0
    )

    return (
        area_ratio,
        image,
        largest,
    )


cap_on_ratio, _, _ = (
    measure_area_ratio(
        CAP_ON_PATH
    )
)

cap_off_ratio, _, _ = (
    measure_area_ratio(
        CAP_OFF_PATH
    )
)


candidate_threshold = (
    cap_on_ratio
    + cap_off_ratio
) / 2.0


cap_present_when_greater = (
    cap_on_ratio
    > cap_off_ratio
)


def judge_cap(
    area_ratio,
):

    if cap_present_when_greater:

        cap_present = (
            area_ratio
            >= candidate_threshold
        )

    else:

        cap_present = (
            area_ratio
            <= candidate_threshold
        )

    if cap_present:

        return (
            "OK",
            "",
        )

    return (
        "NG",
        "NG_CAP_AREA",
    )


print(
    "cap_on ratio :",
    cap_on_ratio,
)

print(
    "cap_off ratio:",
    cap_off_ratio,
)

print(
    "reference gap:",
    abs(
        cap_on_ratio
        - cap_off_ratio
    ),
)

print(
    "candidate threshold:",
    candidate_threshold,
)

print(
    "cap present when:",
    "GREATER"
    if cap_present_when_greater
    else "LESS",
)


for label, image_path in [
    (
        "cap_on",
        CAP_ON_PATH,
    ),
    (
        "cap_off",
        CAP_OFF_PATH,
    ),
]:

    area_ratio, image, contour = (
        measure_area_ratio(
            image_path
        )
    )

    decision, reason = judge_cap(
        area_ratio
    )

    result = image.copy()

    if contour is not None:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

    text = (
        f"{decision} "
        f"ratio={area_ratio:.3f}"
    )

    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            200,
            0,
        )
        if decision == "OK"
        else (
            0,
            0,
            255,
        ),
        2,
    )

    output_path = (
        OUTPUT_DIR
        / f"{label}_rule.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print()
    print("file:", image_path.name)
    print("area_ratio:", area_ratio)
    print("decision:", decision)
    print("reason:", reason)