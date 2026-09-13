from pathlib import Path
import csv
import json

import cv2
import numpy as np

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# --------------------------------------------------
# 경로
# --------------------------------------------------

RULE_PATH = Path(
    "configs/day05_logo_rules.json"
)

DATA_DIR = Path(
    "samples/day05/resized"
)

OUTPUT_DIR = Path(
    "outputs/day05/test"
)

OUTPUT_CSV = Path(
    "reports/day05_test_results.csv"
)


# --------------------------------------------------
# 1. Test 이미지 목록 만들기
# --------------------------------------------------

def find_test_images():

    image_infos = []


    # 실제 정상 Test 이미지
    for image_path in sorted(
        DATA_DIR.glob("test_ok_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "OK"
            )
        )


    # 실제 불량 Test 이미지
    for image_path in sorted(
        DATA_DIR.glob("test_ng_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "NG"
            )
        )


    return image_infos


# --------------------------------------------------
# 2. 측정값에 Rule 적용
# --------------------------------------------------

def judge_measurement(
    measurement,
    rules,
):

    reasons = []

    required = rules[
        "required_rules"
    ]


    # Homography 성공 여부
    if (
        required["homography_found"]
        and measurement["homography_found"] != 1
    ):

        reasons.append(
            "HOMOGRAPHY_NOT_FOUND"
        )


    # SIFT Good Match 검사
    if (
        measurement["sift_good_matches"]
        < required["sift_good_matches_min"]
    ):

        reasons.append(
            "LOW_SIFT_MATCH"
        )


    # Inlier Ratio 검사
    if (
        measurement["inlier_ratio"]
        < required["inlier_ratio_min"]
    ):

        reasons.append(
            "LOW_INLIER_RATIO"
        )


    # 실패 이유가 하나도 없으면 OK
    if not reasons:

        predicted = "OK"

    else:

        predicted = "NG"


    return predicted, reasons


# --------------------------------------------------
# 3. 실제 정답과 프로그램 판정 비교
#
# 이번 수업에서는 NG를 Positive로 둡니다.
# --------------------------------------------------

def classify_error(
    actual,
    predicted,
):

    if (
        actual == "NG"
        and predicted == "NG"
    ):
        return "TP"


    if (
        actual == "OK"
        and predicted == "OK"
    ):
        return "TN"


    if (
        actual == "OK"
        and predicted == "NG"
    ):
        return "FP"


    if (
        actual == "NG"
        and predicted == "OK"
    ):
        return "FN"


    return "UNKNOWN"


# --------------------------------------------------
# 4. 결과 이미지 만들기
# --------------------------------------------------

def annotate_result(
    measurement,
    actual,
    predicted,
    reasons,
):

    image = (
        measurement["target_image"]
        .copy()
    )


    polygon = measurement.get(
        "polygon"
    )


    # Homography로 찾은 로고 영역 표시
    if polygon is not None:

        points = np.int32(
            polygon
        ).reshape(
            -1,
            1,
            2
        )


        # OK와 NG의 선 색상을 구분
        if predicted == "OK":

            line_color = (
                0,
                255,
                0
            )

        else:

            line_color = (
                0,
                0,
                255
            )


        cv2.polylines(
            image,
            [points],
            True,
            line_color,
            3
        )


    reason_text = (
        "+".join(reasons)
        if reasons
        else "PASS"
    )


    lines = [

        f"Actual: {actual}",

        f"Predicted: {predicted}",

        (
            "SIFT Good: "
            f"{measurement['sift_good_matches']}"
        ),

        (
            "Inlier Ratio: "
            f"{measurement['inlier_ratio']:.4f}"
        ),

        (
            "Template: "
            f"{measurement['template_score']:.4f}"
        ),

        f"Reason: {reason_text}",
    ]


    y = 40


    for line in lines:

        cv2.putText(
            image,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 35


    return image


# --------------------------------------------------
# 실행
# --------------------------------------------------

def main():

    # ==============================================
    # 1. 동결한 Rule 읽기
    # ==============================================

    if not RULE_PATH.exists():

        raise FileNotFoundError(
            "Rule JSON이 없습니다. "
            "먼저 36_day05_freeze_rule.py를 실행하세요."
        )


    with RULE_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        rules = json.load(
            file
        )


    print("=" * 70)

    print(
        "Baseline Rule:",
        rules["baseline_version"]
    )

    print(
        "SIFT >=",
        rules[
            "required_rules"
        ][
            "sift_good_matches_min"
        ]
    )

    print(
        "Inlier Ratio >=",
        rules[
            "required_rules"
        ][
            "inlier_ratio_min"
        ]
    )


    # ==============================================
    # 2. Reference 준비
    # ==============================================

    reference_path = Path(
        rules[
            "reference"
        ][
            "reference_label"
        ]
    )


    reference_data = prepare_reference(
        reference_path
    )


    # ==============================================
    # 3. Test 이미지 확인
    # ==============================================

    image_infos = (
        find_test_images()
    )


    print(
        "Test Images:",
        len(image_infos)
    )


    if len(image_infos) != 6:

        print(
            "주의: Test 이미지는 "
            "OK 3장 + NG 3장 = 6장을 예상합니다."
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    rows = []


    # ==============================================
    # 4. Test 6장 검사
    # ==============================================

    for image_path, actual in image_infos:

        # ------------------------------------------
        # 측정
        # ------------------------------------------

        measurement = measure_target(
            image_path,
            reference_data
        )


        # ------------------------------------------
        # 동결 Rule로 판정
        # ------------------------------------------

        predicted, reasons = (
            judge_measurement(
                measurement,
                rules
            )
        )


        # ------------------------------------------
        # 실제 정답과 비교
        # ------------------------------------------

        error_type = classify_error(
            actual,
            predicted
        )


        # ------------------------------------------
        # 결과 이미지 생성
        # ------------------------------------------

        result_image = annotate_result(
            measurement,
            actual,
            predicted,
            reasons
        )


        output_image = (
            OUTPUT_DIR
            / (
                f"{image_path.stem}"
                "_result.jpg"
            )
        )


        cv2.imwrite(
            str(output_image),
            result_image
        )


        # ------------------------------------------
        # CSV 결과
        # ------------------------------------------

        row = {

            "file":
                image_path.name,

            "actual":
                actual,

            "predicted":
                predicted,

            "error_type":
                error_type,

            "ng_reason":
                (
                    ";".join(reasons)
                    if reasons
                    else "PASS"
                ),

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

            "result_image":
                str(output_image),
        }


        rows.append(
            row
        )


        print(
            row
        )


    # ==============================================
    # 5. CSV 저장
    # ==============================================

    fieldnames = [
        "file",
        "actual",
        "predicted",
        "error_type",
        "ng_reason",
        "template_score",
        "sift_good_matches",
        "homography_found",
        "homography_inliers",
        "inlier_ratio",
        "result_image",
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
    # 6. Test 결과 요약
    # ==============================================

    tp = sum(
        row["error_type"] == "TP"
        for row in rows
    )

    tn = sum(
        row["error_type"] == "TN"
        for row in rows
    )

    fp = sum(
        row["error_type"] == "FP"
        for row in rows
    )

    fn = sum(
        row["error_type"] == "FN"
        for row in rows
    )


    total = len(rows)


    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )


    print()

    print("=" * 70)

    print(
        "Test 완료"
    )

    print("=" * 70)

    print(
        "전체:",
        total
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "TP:",
        tp
    )

    print(
        "TN:",
        tn
    )

    print(
        "FP:",
        fp
    )

    print(
        "FN:",
        fn
    )

    print(
        "CSV:",
        OUTPUT_CSV
    )

    print(
        "결과 이미지:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()