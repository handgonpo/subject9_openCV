from pathlib import Path
import shutil
import json

import pandas as pd


# --------------------------------------------------
# 입력 파일
# --------------------------------------------------

RESULT_CSV = Path(
    "reports/day05_test_results.csv"
)

RULE_PATH = Path(
    "configs/day05_logo_rules.json"
)


# --------------------------------------------------
# 출력 파일
# --------------------------------------------------

REPORT_PATH = Path(
    "reports/day05_test_metrics.md"
)

FP_DIR = Path(
    "outputs/day05/fp"
)

FN_DIR = Path(
    "outputs/day05/fn"
)


# --------------------------------------------------
# 0으로 나누는 오류를 막는 함수
# --------------------------------------------------

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


# --------------------------------------------------
# 실행
# --------------------------------------------------

def main():

    # ==============================================
    # 1. Test 결과 CSV 확인
    # ==============================================

    if not RESULT_CSV.exists():

        raise FileNotFoundError(
            "Test 결과 CSV가 없습니다. "
            "먼저 37_day05_test.py를 실행하세요."
        )


    df = pd.read_csv(
        RESULT_CSV
    )


    required_columns = {
        "file",
        "actual",
        "predicted",
        "error_type",
        "result_image",
    }


    missing = (
        required_columns
        - set(df.columns)
    )


    if missing:

        raise RuntimeError(
            f"필요한 컬럼이 없습니다: {missing}"
        )


    # ==============================================
    # 2. 사용한 Rule 버전 확인
    # ==============================================

    baseline_version = "UNKNOWN"


    if RULE_PATH.exists():

        with RULE_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            rules = json.load(
                file
            )


        baseline_version = rules.get(
            "baseline_version",
            "UNKNOWN"
        )


    # ==============================================
    # 3. TP / TN / FP / FN 집계
    # ==============================================

    counts = (
        df["error_type"]
        .value_counts()
        .to_dict()
    )


    tp = int(
        counts.get("TP", 0)
    )

    tn = int(
        counts.get("TN", 0)
    )

    fp = int(
        counts.get("FP", 0)
    )

    fn = int(
        counts.get("FN", 0)
    )


    total = (
        tp
        + tn
        + fp
        + fn
    )


    # ==============================================
    # 4. 성능 지표 계산
    # ==============================================

    accuracy = safe_divide(
        tp + tn,
        total
    )


    # NG를 Positive로 정의합니다.
    precision_ng = safe_divide(
        tp,
        tp + fp
    )


    recall_ng = safe_divide(
        tp,
        tp + fn
    )


    # ==============================================
    # 5. 기존 FP / FN 폴더 초기화
    #
    # 이전 실행 결과가 섞이지 않도록 합니다.
    # ==============================================

    for folder in [
        FP_DIR,
        FN_DIR,
    ]:

        if folder.exists():

            shutil.rmtree(
                folder
            )


        folder.mkdir(
            parents=True,
            exist_ok=True
        )


    # ==============================================
    # 6. FP / FN 결과 이미지를 따로 복사
    # ==============================================

    for _, row in df.iterrows():

        error_type = row[
            "error_type"
        ]

        source = Path(
            row["result_image"]
        )


        if not source.exists():
            continue


        if error_type == "FP":

            shutil.copy2(
                source,
                FP_DIR / source.name
            )


        elif error_type == "FN":

            shutil.copy2(
                source,
                FN_DIR / source.name
            )


    # ==============================================
    # 7. 평가 보고서 만들기
    # ==============================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    report = f"""# Day05 Test Metrics

## Baseline

- Rule Version: {baseline_version}
- Positive Class: NG
- Test Images: {total}

## Confusion Matrix

| Actual / Predicted | NG | OK |
|---|---:|---:|
| NG | TP = {tp} | FN = {fn} |
| OK | FP = {fp} | TN = {tn} |

## Metrics

- Accuracy: {accuracy:.4f}
- NG Precision: {precision_ng:.4f}
- NG Recall: {recall_ng:.4f}
- FP: {fp}
- FN: {fn}

## Interpretation

- FP: 실제 OK인데 NG로 잘못 판정
- FN: 실제 NG인데 OK로 잘못 통과
"""


    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )


    # ==============================================
    # 8. 터미널 출력
    # ==============================================

    print("=" * 70)

    print(
        "Day05 Test Evaluation"
    )

    print("=" * 70)


    print(
        "Rule:",
        baseline_version
    )

    print(
        "Test Images:",
        total
    )

    print()

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

    print()

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "NG Precision:",
        round(
            precision_ng,
            4
        )
    )

    print(
        "NG Recall:",
        round(
            recall_ng,
            4
        )
    )

    print()

    print(
        "평가 보고서:",
        REPORT_PATH
    )

    print(
        "FP 이미지:",
        FP_DIR
    )

    print(
        "FN 이미지:",
        FN_DIR
    )


if __name__ == "__main__":
    main()