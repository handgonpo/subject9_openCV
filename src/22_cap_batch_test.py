from pathlib import Path
import csv

import cv2


# --------------------------------------------------
# 데이터 경로
# --------------------------------------------------

CAP_ON_DIR = Path(
    "samples/day03/cap_on"
)

CAP_OFF_DIR = Path(
    "samples/day03/cap_off"
)


# 기준값을 정할 Reference 이미지
REFERENCE_ON = (
    CAP_ON_DIR / "cap_on_001.jpg"
)

REFERENCE_OFF = (
    CAP_OFF_DIR / "cap_off_001.jpg"
)


# 결과 저장 위치
OUTPUT_DIR = Path(
    "outputs/day03/batch"
)

REPORT_PATH = Path(
    "reports/day03_inspection_results.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# 너무 작은 Contour는 제외
MIN_CONTOUR_AREA_RATIO = 0.001


# --------------------------------------------------
# 이미지 측정 함수
# --------------------------------------------------

def measure(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None


    # Gray 변환
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    # Otsu Threshold
    otsu_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )


    # Morphology
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


    # Contour 찾기
    contours, _ = cv2.findContours(
        cleaned.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )


    height, width = image.shape[:2]

    image_area = float(
        width * height
    )


    # 이미지 전체 크기에 비례하여
    # 너무 작은 Contour 제거
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


    largest_contour = None
    largest_area_ratio = 0.0

    center_x_norm = ""
    center_y_norm = ""


    # 가장 큰 Contour 선택
    if valid_contours:

        largest_contour = max(
            valid_contours,
            key=cv2.contourArea,
        )

        largest_area = float(
            cv2.contourArea(
                largest_contour
            )
        )


        # 단순 픽셀 면적이 아니라
        # 전체 ROI에서 차지하는 비율 사용
        largest_area_ratio = (
            largest_area / image_area
            if image_area > 0
            else 0.0
        )


        # Contour 중심점 계산
        moments = cv2.moments(
            largest_contour
        )

        if moments["m00"] != 0:

            cx = (
                moments["m10"]
                / moments["m00"]
            )

            cy = (
                moments["m01"]
                / moments["m00"]
            )


            center_x_norm = (
                cx / width
            )

            center_y_norm = (
                cy / height
            )


    return {
        "image": image,
        "otsu_value": float(
            otsu_value
        ),
        "valid_count": len(
            valid_contours
        ),
        "largest_contour": (
            largest_contour
        ),
        "largest_area_ratio": (
            largest_area_ratio
        ),
        "center_x_norm": (
            center_x_norm
        ),
        "center_y_norm": (
            center_y_norm
        ),
    }


# --------------------------------------------------
# Reference 이미지 측정
# --------------------------------------------------

reference_on = measure(
    REFERENCE_ON
)

reference_off = measure(
    REFERENCE_OFF
)


if (
    reference_on is None
    or reference_off is None
):

    raise RuntimeError(
        "Reference 이미지를 확인하세요."
    )


on_ratio = (
    reference_on[
        "largest_area_ratio"
    ]
)

off_ratio = (
    reference_off[
        "largest_area_ratio"
    ]
)


# 두 Reference 값의 중간을
# 임시 기준값으로 사용
candidate_threshold = (
    on_ratio + off_ratio
) / 2.0


# cap_on일 때 값이 더 큰지 확인
cap_present_when_greater = (
    on_ratio > off_ratio
)


print(
    "Reference ON :",
    round(on_ratio, 4),
)

print(
    "Reference OFF:",
    round(off_ratio, 4),
)

print(
    "Threshold    :",
    round(candidate_threshold, 4),
)


# --------------------------------------------------
# 판정 함수
# --------------------------------------------------

def judge(area_ratio):

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
            "cap_on",
            "OK",
            "",
        )


    return (
        "cap_off",
        "NG",
        "NG_CAP_AREA",
    )


# --------------------------------------------------
# Test 이미지
# --------------------------------------------------

TEST_IMAGES = [
    (
        "cap_on",
        CAP_ON_DIR
        / "cap_on_002.jpg",
    ),
    (
        "cap_on",
        CAP_ON_DIR
        / "cap_on_003.jpg",
    ),
    (
        "cap_off",
        CAP_OFF_DIR
        / "cap_off_002.jpg",
    ),
    (
        "cap_off",
        CAP_OFF_DIR
        / "cap_off_003.jpg",
    ),
]


rows = []

total = 0
correct = 0


# --------------------------------------------------
# 같은 Rule을 Test 이미지에 적용
# --------------------------------------------------

for expected, image_path in TEST_IMAGES:

    measured = measure(
        image_path
    )

    if measured is None:

        print(
            "읽기 실패:",
            image_path,
        )

        continue


    predicted, decision, reason = (
        judge(
            measured[
                "largest_area_ratio"
            ]
        )
    )


    is_correct = (
        predicted == expected
    )


    total += 1

    if is_correct:
        correct += 1


    # ----------------------------------------------
    # 결과 이미지 생성
    # ----------------------------------------------

    result = measured[
        "image"
    ].copy()


    contour = measured[
        "largest_contour"
    ]


    if contour is not None:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )


    text = (
        f"{predicted} "
        f"{measured['largest_area_ratio']:.3f}"
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
        if is_correct
        else (
            0,
            0,
            255,
        ),
        2,
    )


    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_result.jpg"
    )


    cv2.imwrite(
        str(output_path),
        result,
    )


    # ----------------------------------------------
    # CSV 기록
    # ----------------------------------------------

    rows.append(
        {
            "file": image_path.name,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "largest_area_ratio": measured[
                "largest_area_ratio"
            ],
            "valid_count": measured[
                "valid_count"
            ],
            "center_x_norm": measured[
                "center_x_norm"
            ],
            "center_y_norm": measured[
                "center_y_norm"
            ],
            "decision": decision,
            "reason": reason,
        }
    )


# --------------------------------------------------
# 결과 확인
# --------------------------------------------------

if total == 0:

    print(
        "검사할 이미지가 없습니다."
    )

    raise SystemExit


# --------------------------------------------------
# CSV 저장
# --------------------------------------------------

with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [
        "file",
        "expected",
        "predicted",
        "correct",
        "largest_area_ratio",
        "valid_count",
        "center_x_norm",
        "center_y_norm",
        "decision",
        "reason",
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()

    writer.writerows(
        rows
    )


# --------------------------------------------------
# 최종 결과 출력
# --------------------------------------------------

print()
print("==============================")
print("Batch Test 결과")
print("==============================")

print(
    "확인 이미지 수:",
    total,
)

print(
    "기준과 일치:",
    correct,
)

print(
    "불일치:",
    total - correct,
)

print(
    "Accuracy:",
    round(
        correct / total,
        3,
    ),
)

print(
    "CSV:",
    REPORT_PATH,
)

print(
    "결과 이미지:",
    OUTPUT_DIR,
)