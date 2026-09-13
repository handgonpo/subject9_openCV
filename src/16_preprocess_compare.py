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
    "outputs/day02/compare"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# low_contrast.jpg 확인
# --------------------------------------------------

image = cv2.imread(
    str(low_contrast_path)
)


# low_contrast.jpg가 없다면
# 기준 ROI에서 다시 생성
if image is None:

    print(
        "low_contrast.jpg가 없어 "
        "기준 이미지에서 생성합니다."
    )

    reference = cv2.imread(
        str(reference_path)
    )

    if reference is None:
        raise FileNotFoundError(
            f"기준 이미지를 읽지 못했습니다: "
            f"{reference_path}"
        )

    image = cv2.convertScaleAbs(
        reference,
        alpha=0.35,
        beta=80,
    )

    cv2.imwrite(
        str(low_contrast_path),
        image,
    )

    print(
        "문제 이미지 생성:",
        low_contrast_path,
    )


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 1. 전처리 없이 Otsu
# --------------------------------------------------

original_value, original_otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 2. Gaussian 후 Otsu
# --------------------------------------------------

gaussian = cv2.GaussianBlur(
    gray,
    (5, 5),
    0,
)

gaussian_value, gaussian_otsu = cv2.threshold(
    gaussian,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 3. CLAHE 후 Otsu
# --------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

clahe_image = clahe.apply(
    gray
)

clahe_value, clahe_otsu = cv2.threshold(
    clahe_image,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# Otsu가 선택한 값 확인
# --------------------------------------------------

print(
    "Original Otsu:",
    original_value,
)

print(
    "Gaussian Otsu:",
    gaussian_value,
)

print(
    "CLAHE Otsu:",
    clahe_value,
)


# --------------------------------------------------
# 비교용 이미지 만들기
# --------------------------------------------------

view_size = (
    400,
    300,
)


def make_view(
    source,
    title,
    is_binary=False,
):

    if is_binary:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA

    resized = cv2.resize(
        source,
        view_size,
        interpolation=interpolation,
    )

    if len(resized.shape) == 2:
        canvas = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        canvas = resized.copy()

    cv2.rectangle(
        canvas,
        (0, 0),
        (view_size[0], 40),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return canvas


# --------------------------------------------------
# 2 × 2 비교 화면
# --------------------------------------------------

top = cv2.hconcat(
    [
        make_view(
            gray,
            "Low Contrast Gray",
        ),
        make_view(
            original_otsu,
            "Original + Otsu",
            is_binary=True,
        ),
    ]
)

bottom = cv2.hconcat(
    [
        make_view(
            gaussian_otsu,
            "Gaussian + Otsu",
            is_binary=True,
        ),
        make_view(
            clahe_otsu,
            "CLAHE + Otsu",
            is_binary=True,
        ),
    ]
)

compare = cv2.vconcat(
    [
        top,
        bottom,
    ]
)


# --------------------------------------------------
# 결과 저장
# --------------------------------------------------

output_path = (
    output_dir
    / "preprocess_compare.png"
)

cv2.imwrite(
    str(output_path),
    compare,
)

print(
    "저장:",
    output_path,
)