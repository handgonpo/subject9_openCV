from pathlib import Path

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


REFERENCE_PATH = Path(
    "samples/day04/reference/reference_label.jpg"
)

TARGET_PATH = Path(
    "samples/day05/resized/ok_01.jpg"
)


# --------------------------------------------------
# 1. Reference 준비
# --------------------------------------------------

reference_data = prepare_reference(
    REFERENCE_PATH
)


print(
    "Reference Keypoints:",
    len(
        reference_data["keypoints"]
    )
)


# --------------------------------------------------
# 2. 실제 Target 한 장 측정
# --------------------------------------------------

measurement = measure_target(
    TARGET_PATH,
    reference_data
)


# --------------------------------------------------
# 3. 결과 확인
# --------------------------------------------------

print()
print("=" * 60)

print(
    "Target:",
    TARGET_PATH.name
)

print(
    "Template Score:",
    round(
        measurement["template_score"],
        4
    )
)

print(
    "SIFT Good Match:",
    measurement[
        "sift_good_matches"
    ]
)

print(
    "Homography Found:",
    measurement[
        "homography_found"
    ]
)

print(
    "Homography Inlier:",
    measurement[
        "homography_inliers"
    ]
)

print(
    "Inlier Ratio:",
    round(
        measurement["inlier_ratio"],
        4
    )
)

print("=" * 60)