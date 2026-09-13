from pathlib import Path
import json


# --------------------------------------------------
# 저장할 JSON 파일
# --------------------------------------------------

OUTPUT_PATH = Path(
    "configs/day05_logo_rules.json"
)


# --------------------------------------------------
# Calibration 분석에서 선택한 Threshold
# --------------------------------------------------

SIFT_GOOD_MATCHES_MIN = 129

INLIER_RATIO_MIN = 0.8664


# --------------------------------------------------
# Rule 저장
# --------------------------------------------------

def main():

    rules = {

        # Rule 버전
        "baseline_version": "day05-logo-v1.0",

        # 무엇을 검사하는가?
        "inspection_target":
            "광천김 로고 존재 및 정합 검사",

        # 이번 수업에서는 NG를 Positive로 정의
        "positive_class": "NG",

        # 비교에 사용하는 기준 로고
        "reference": {
            "reference_label":
                "samples/day04/reference/reference_label.jpg"
        },

        # 실제 OK / NG 판정에 사용할 조건
        "required_rules": {

            "homography_found":
                True,

            "sift_good_matches_min":
                SIFT_GOOD_MATCHES_MIN,

            "inlier_ratio_min":
                INLIER_RATIO_MIN,
        },

        # Calibration에서 확인한 결과
        "calibration_result": {

            "accuracy":
                0.9167,

            "fp":
                1,

            "fn":
                0,
        },

        "note":
            "Calibration 결과로 결정한 Day05 Baseline Rule"
    }


    # --------------------------------------------------
    # configs 폴더가 없으면 생성
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------
    # JSON 저장
    # --------------------------------------------------

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            rules,
            file,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------
    # 결과 확인
    # --------------------------------------------------

    print("=" * 70)

    print(
        "Rule 동결 완료"
    )

    print("=" * 70)

    print(
        "저장:",
        OUTPUT_PATH
    )

    print()

    print(
        json.dumps(
            rules,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()