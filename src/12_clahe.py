from pathlib import Path

import cv2


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

low_contrast_path = Path(
    "samples/day02/low_contrast.jpg"
)

output_dir = Path(
    "outputs/day02/clahe"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 기준 ROI 이미지 읽기
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: "
        f"{reference_path}"
    )


# --------------------------------------------------
# 실습용 저대비 이미지 만들기
# --------------------------------------------------

low_contrast = cv2.convertScaleAbs(
    reference,
    alpha=0.35,
    beta=80,
)

cv2.imwrite(
    str(low_contrast_path),
    low_contrast,
)

print(
    "문제 이미지 생성:",
    low_contrast_path,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
)

low_contrast_gray = cv2.cvtColor(
    low_contrast,
    cv2.COLOR_BGR2GRAY,
)


# 기준 이미지와 저대비 이미지 저장
cv2.imwrite(
    str(
        output_dir
        / "reference_gray.jpg"
    ),
    reference_gray,
)

cv2.imwrite(
    str(
        output_dir
        / "low_contrast_gray.jpg"
    ),
    low_contrast_gray,
)


# --------------------------------------------------
# CLAHE 설정
# --------------------------------------------------

settings = [
    (1.0, (8, 8)),
    (2.0, (8, 8)),
    (3.0, (8, 8)),
    (4.0, (8, 8)),
]


# --------------------------------------------------
# 여러 clipLimit 비교
# --------------------------------------------------

for clip_limit, tile_size in settings:

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size,
    )

    result = clahe.apply(
        low_contrast_gray
    )

    output_path = (
        output_dir
        / f"clahe_clip_{clip_limit}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "저장:",
        output_path,
    )