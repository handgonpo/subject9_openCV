from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 파일 경로
# --------------------------------------------------

CSV_PATH = Path(
    "reports/day05_calibration_measurements.csv"
)

CANDIDATE_CSV = Path(
    "reports/day05_rule_candidates.csv"
)

PLOT_PATH = Path(
    "outputs/day05/calibration/sift_inlier_distribution.png"
)


# --------------------------------------------------
# 1. Threshold 후보 만들기
# --------------------------------------------------

def make_candidates(
    series,
    count=15,
    integer=False,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if values.empty:
        return []


    low = float(
        values.min()
    )

    high = float(
        values.max()
    )


    # 모든 값이 같으면 후보도 하나만 사용
    if low == high:

        if integer:
            return [int(low)]

        return [low]


    # 최소값부터 최대값까지
    # 15개의 후보를 균등하게 생성
    candidates = np.linspace(
        low,
        high,
        count,
    )


    if integer:

        return sorted(
            set(
                int(round(value))
                for value in candidates
            )
        )


    return sorted(
        set(
            round(
                float(value),
                4,
            )
            for value in candidates
        )
    )


# --------------------------------------------------
# 2. 실제 정답과 임시 판정 비교
#
# 제조검사에서는 NG를 Positive로 정의합니다.
# --------------------------------------------------

def classify_error(
    actual,
    predicted,
):

    # 실제 NG → NG로 검출
    if (
        actual == "NG"
        and predicted == "NG"
    ):
        return "TP"


    # 실제 OK → OK로 통과
    if (
        actual == "OK"
        and predicted == "OK"
    ):
        return "TN"


    # 실제 OK → NG로 잘못 판정
    if (
        actual == "OK"
        and predicted == "NG"
    ):
        return "FP"


    # 실제 NG → OK로 잘못 통과
    if (
        actual == "NG"
        and predicted == "OK"
    ):
        return "FN"


    return "UNKNOWN"


# --------------------------------------------------
# 실행
# --------------------------------------------------

def main():

    # ==============================================
    # 1. Calibration CSV 읽기
    # ==============================================

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            CSV_PATH
        )


    df = pd.read_csv(
        CSV_PATH
    )


    print("=" * 70)
    print("Calibration 데이터")
    print("=" * 70)

    print(
        "전체 이미지:",
        len(df)
    )


    print(
        df["actual"].value_counts()
    )


    # ==============================================
    # 2. OK / NG 기본 통계 확인
    # ==============================================

    numeric_columns = [
        "template_score",
        "sift_good_matches",
        "homography_inliers",
        "inlier_ratio",
    ]


    print()

    print("=" * 70)
    print("OK / NG 측정값 통계")
    print("=" * 70)


    statistics = (
        df.groupby("actual")[
            numeric_columns
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )


    print(
        statistics
    )


    # ==============================================
    # 3. Threshold 후보 만들기
    # ==============================================

    sift_candidates = make_candidates(
        df["sift_good_matches"],
        count=15,
        integer=True,
    )


    inlier_candidates = make_candidates(
        df["inlier_ratio"],
        count=15,
        integer=False,
    )


    print()

    print(
        "SIFT Threshold 후보:"
    )

    print(
        sift_candidates
    )


    print()

    print(
        "Inlier Ratio Threshold 후보:"
    )

    print(
        inlier_candidates
    )


    # ==============================================
    # 4. 모든 후보 조합 시험
    # ==============================================

    rows = []


    for sift_min in sift_candidates:

        for inlier_min in inlier_candidates:

            tp = 0
            tn = 0
            fp = 0
            fn = 0


            # Calibration 12장에
            # 현재 후보 Rule을 적용
            for _, row in df.iterrows():

                predicted = (
                    "OK"

                    if (
                        int(
                            row["homography_found"]
                        )
                        == 1

                        and int(
                            row["sift_good_matches"]
                        )
                        >= sift_min

                        and float(
                            row["inlier_ratio"]
                        )
                        >= inlier_min
                    )

                    else "NG"
                )


                error = classify_error(
                    row["actual"],
                    predicted,
                )


                if error == "TP":
                    tp += 1

                elif error == "TN":
                    tn += 1

                elif error == "FP":
                    fp += 1

                elif error == "FN":
                    fn += 1


            # --------------------------------------
            # 정확도 계산
            # --------------------------------------

            total = (
                tp
                + tn
                + fp
                + fn
            )


            accuracy = (
                (tp + tn) / total
                if total > 0
                else 0.0
            )


            # 현재 후보 결과 저장
            rows.append(
                {
                    "sift_good_matches_min":
                        sift_min,

                    "inlier_ratio_min":
                        inlier_min,

                    "accuracy":
                        round(
                            accuracy,
                            4,
                        ),

                    "tp":
                        tp,

                    "tn":
                        tn,

                    "fp":
                        fp,

                    "fn":
                        fn,

                    "total_error":
                        fp + fn,
                }
            )


    # ==============================================
    # 5. 후보 Rule을 좋은 순서로 정렬
    # ==============================================

    result_df = pd.DataFrame(
        rows
    )


    result_df = (
        result_df.sort_values(
            [
                "fn",
                "total_error",
                "fp",
                "sift_good_matches_min",
                "inlier_ratio_min",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


    # ==============================================
    # 6. 후보 Rule CSV 저장
    # ==============================================

    CANDIDATE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    result_df.to_csv(
        CANDIDATE_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    print()

    print("=" * 70)
    print("후보 Rule 상위 15개")
    print("=" * 70)


    print(
        result_df
        .head(15)
        .to_string(
            index=False
        )
    )


    print()

    print(
        "저장:",
        CANDIDATE_CSV
    )


    # ==============================================
    # 7. 가장 위의 후보 가져오기
    # ==============================================

    best = result_df.iloc[0]


    best_sift = int(
        best["sift_good_matches_min"]
    )

    best_inlier = float(
        best["inlier_ratio_min"]
    )


    print()

    print("=" * 70)
    print("현재 가장 좋은 후보")
    print("=" * 70)

    print(
        "SIFT Good Match >=",
        best_sift
    )

    print(
        "Inlier Ratio >=",
        best_inlier
    )

    print(
        "Accuracy:",
        best["accuracy"]
    )

    print(
        "FP:",
        int(best["fp"])
    )

    print(
        "FN:",
        int(best["fn"])
    )


    # ==============================================
    # 8. 분포 그래프 만들기
    # ==============================================

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    plt.figure(
        figsize=(9, 6)
    )


    # OK와 NG를 다른 모양으로 표시
    for actual, marker in [
        ("OK", "o"),
        ("NG", "x"),
    ]:

        subset = df[
            df["actual"] == actual
        ]


        plt.scatter(
            subset["sift_good_matches"],
            subset["inlier_ratio"],
            label=actual,
            marker=marker,
        )


        # 각 점에 파일명 표시
        for _, row in subset.iterrows():

            plt.annotate(
                row["file"],
                (
                    row["sift_good_matches"],
                    row["inlier_ratio"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )


    # 선택된 후보 Threshold 표시
    plt.axvline(
        best_sift,
        linestyle="--",
        label=(
            f"SIFT >= {best_sift}"
        ),
    )


    plt.axhline(
        best_inlier,
        linestyle="--",
        label=(
            f"Inlier >= {best_inlier:.4f}"
        ),
    )


    plt.xlabel(
        "SIFT Good Matches"
    )

    plt.ylabel(
        "Homography Inlier Ratio"
    )

    plt.title(
        "Day05 Calibration Distribution"
    )


    plt.legend()

    plt.grid(
        alpha=0.2
    )


    plt.savefig(
        PLOT_PATH,
        dpi=150,
        bbox_inches="tight",
    )


    plt.close()


    print()

    print(
        "분포 이미지:",
        PLOT_PATH
    )


if __name__ == "__main__":
    main()