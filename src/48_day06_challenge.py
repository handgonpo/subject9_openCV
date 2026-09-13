import csv
from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
    preprocess_for_ocr,
    rotate_image,
    run_ocr,
)

from day06_ocr_validation import (
    inspect_text,
)


# ========================================
# 경로 설정
# ========================================

ROOT = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "challenge"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "challenge"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_challenge_results.csv"
)


# ========================================
# Freeze된 OCR Rule 읽기
# ========================================

rules = load_rules()


if rules[
    "baseline_version"
] != "day06-ocr-v1.0":

    raise RuntimeError(
        "OCR Rule이 Freeze되지 않았습니다."
    )


# ========================================
# 대표 SN 정답 가져오기
# ========================================

reference_file = rules[
    "reference_file"
]

EXPECTED_SN = (
    Path(reference_file)
    .stem
    .split("_side")[0]
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


rows = []


# ========================================
# 4가지 환경 변화 검사
# ========================================

for condition in [
    "low_contrast",
    "blur",
    "tilt",
    "glare",
]:

    input_dir = (
        ROOT
        / condition
    )

    for image_path in sorted(
        input_dir.glob("*.jpg")
    ):

        roi = cv2.imread(
            str(image_path)
        )

        if roi is None:
            continue


        # ==================================
        # Freeze된 전처리 적용
        # 현재: original
        # ==================================

        processed = (
            preprocess_for_ocr(
                roi,
                rules,
            )
        )


        # ==================================
        # Batch와 동일하게 4배 확대
        # ==================================

        ocr_input = cv2.resize(
            processed,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        )


        # ==================================
        # OCR
        # ==================================

        raw_text = run_ocr(
            ocr_input,
            rules,
        )


        # ==================================
        # 문자열 검사
        # ==================================

        result = inspect_text(
            raw_text=raw_text,
            expected_text=EXPECTED_SN,
            pattern=rules[
                "expected_pattern"
            ],
        )


        ocr_exact = (
            result["final_text"]
            == EXPECTED_SN
        )


        # ==================================
        # 실제 OCR 입력 이미지 저장
        # ==================================

        output_path = (
            OUTPUT_DIR
            / f"{condition}_ocr_input.jpg"
        )

        cv2.imwrite(
            str(output_path),
            ocr_input,
        )


        rows.append(
            {
                "condition":
                    condition,

                "deskew":
                    "no",

                "expected_sn":
                    EXPECTED_SN,

                "raw_text":
                    repr(
                        result["raw_text"]
                    ),

                "final_text":
                    result["final_text"],

                "format_ok":
                    result["format_ok"],

                "expected_match":
                    result["expected_match"],

                "ocr_exact":
                    ocr_exact,

                "status":
                    result[
                        "predicted_status"
                    ],

                "reasons":
                    "|".join(
                        result["reasons"]
                    ),
            }
        )


        print(
            "=" * 60
        )

        print(
            "Condition:",
            condition
        )

        print(
            "Deskew: no"
        )

        print(
            "OCR:",
            result["final_text"]
        )

        print(
            "Exact:",
            ocr_exact
        )


        # ==================================
        # Tilt만 Deskew 후 다시 검사
        # ==================================

        if condition == "tilt":

            deskewed = rotate_image(
                roi,
                -8.0,
            )

            deskew_processed = (
                preprocess_for_ocr(
                    deskewed,
                    rules,
                )
            )

            deskew_input = cv2.resize(
                deskew_processed,
                None,
                fx=4.0,
                fy=4.0,
                interpolation=cv2.INTER_CUBIC,
            )

            deskew_raw = run_ocr(
                deskew_input,
                rules,
            )

            deskew_result = (
                inspect_text(
                    raw_text=deskew_raw,
                    expected_text=EXPECTED_SN,
                    pattern=rules[
                        "expected_pattern"
                    ],
                )
            )

            deskew_exact = (
                deskew_result[
                    "final_text"
                ]
                == EXPECTED_SN
            )


            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / "tilt_deskew_ocr_input.jpg"
                ),
                deskew_input,
            )


            rows.append(
                {
                    "condition":
                        "tilt",

                    "deskew":
                        "yes",

                    "expected_sn":
                        EXPECTED_SN,

                    "raw_text":
                        repr(
                            deskew_result[
                                "raw_text"
                            ]
                        ),

                    "final_text":
                        deskew_result[
                            "final_text"
                        ],

                    "format_ok":
                        deskew_result[
                            "format_ok"
                        ],

                    "expected_match":
                        deskew_result[
                            "expected_match"
                        ],

                    "ocr_exact":
                        deskew_exact,

                    "status":
                        deskew_result[
                            "predicted_status"
                        ],

                    "reasons":
                        "|".join(
                            deskew_result[
                                "reasons"
                            ]
                        ),
                }
            )


            print(
                "Deskew: yes"
            )

            print(
                "OCR:",
                deskew_result[
                    "final_text"
                ]
            )

            print(
                "Exact:",
                deskew_exact
            )


# ========================================
# 결과 확인
# ========================================

if not rows:
    raise RuntimeError(
        "환경 변화 테스트 이미지가 없습니다."
    )


# ========================================
# CSV 저장
# ========================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()

print(
    "Saved:",
    REPORT_PATH
)

print(
    "Environment Stress Test Complete"
)