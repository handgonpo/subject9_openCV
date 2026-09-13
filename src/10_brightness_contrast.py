from pathlib import Path

import cv2


# 1. 1일차에서 만든 기준 ROI
reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

# 2. 오늘 사용할 문제 이미지
dark_path = Path(
    "samples/day02/dark.jpg"
)

# 3. 결과 저장 위치
output_dir = Path(
    "outputs/day02/brightness"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# 기준 이미지 읽기
reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: {reference_path}"
    )


# --------------------------------------------------
# 실습용 어두운 이미지 만들기
# --------------------------------------------------

dark = cv2.convertScaleAbs(
    reference,
    alpha=0.55,
    beta=0,
)

cv2.imwrite(
    str(dark_path),
    dark,
)

print("문제 이미지 생성:", dark_path)


# --------------------------------------------------
# Brightness / Contrast 비교
# --------------------------------------------------

settings = [
    (1.0, 0),
    (1.0, 20),
    (1.0, 50),
    (1.0, 80),
]


for alpha, beta in settings:

    result = cv2.convertScaleAbs(
        dark,
        alpha=alpha,
        beta=beta,
    )

    filename = (
        f"alpha_{alpha}_beta_{beta}.jpg"
    )

    output_path = (
        output_dir / filename
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print("저장:", output_path)