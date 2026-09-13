from pathlib import Path
import csv

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# --------------------------------------------------
# 경로 설정
# --------------------------------------------------

REFERENCE_PATH = Path(
    "samples/day04/reference/reference_label.jpg"
)

DATA_DIR = Path(
    "samples/day05/resized"
)

OUTPUT_CSV = Path(
    "reports/day05_calibration_measurements.csv"
)


# --------------------------------------------------
# Calibration 이미지 목록을 만듭니다.
# --------------------------------------------------

def find_calibration_images():

    image_infos = []


    # OK 6장
    for image_path in sorted(
        DATA_DIR.glob("ok_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "OK"
            )
        )


    # NG 6장
    for image_path in sorted(
        DATA_DIR.glob("ng_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "NG"
            )
        )


    return image_infos


# --------------------------------------------------
# 실행
# --------------------------------------------------

def main():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # ==============================================
    # 1. 기준 로고 준비
    # ==============================================

    reference_data = prepare_reference(
        REFERENCE_PATH
    )


    print(
        "Reference Keypoints:",
        len(
            reference_data["keypoints"]
        )
    )


    # ==============================================
    # 2. Calibration 이미지 확인
    # ==============================================

    image_infos = (
        find_calibration_images()
    )


    print(
        "Calibration Images:",
        len(image_infos)
    )


    if len(image_infos) != 12:

        print(
            "주의: Calibration 이미지는 "
            "OK 6장 + NG 6장 = 12장을 예상합니다."
        )


    rows = []


    # ==============================================
    # 3. 12장에 같은 측정을 반복
    # ==============================================

    for image_path, actual in image_infos:

        measurement = measure_target(
            image_path,
            reference_data
        )


        row = {

            "file":
                image_path.name,

            "actual":
                actual,

            "template_score":
                round(
                    measurement[
                        "template_score"
                    ],
                    6
                ),

            "sift_good_matches":
                measurement[
                    "sift_good_matches"
                ],

            "homography_found":
                measurement[
                    "homography_found"
                ],

            "homography_inliers":
                measurement[
                    "homography_inliers"
                ],

            "inlier_ratio":
                round(
                    measurement[
                        "inlier_ratio"
                    ],
                    6
                ),
        }


        rows.append(
            row
        )


        print(
            row
        )


    # ==============================================
    # 4. CSV 저장
    # ==============================================

    fieldnames = [

        "file",

        "actual",

        "template_score",

        "sift_good_matches",

        "homography_found",

        "homography_inliers",

        "inlier_ratio",
    ]


    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ==============================================
    # 5. 완료 확인
    # ==============================================

    print()
    print("=" * 70)

    print(
        "Calibration 측정 완료"
    )

    print(
        "전체 이미지:",
        len(rows)
    )

    print(
        "저장:",
        OUTPUT_CSV
    )

    print("=" * 70)


if __name__ == "__main__":
    main()