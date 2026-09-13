import shutil
from pathlib import Path

import cv2

from day07_anomaly_core import (
    anomaly_map,
    create_backbone,
    extract_features,
    load_reference,
    make_heatmap_overlay,
)


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

TEST_NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

TEST_ANOMALY_DIR = Path(
    "samples/day07/test/anomaly"
)

GROUND_TRUTH_DIR = Path(
    "samples/day07/ground_truth"
)

OUTPUT_DIR = Path(
    "outputs/day07/heatmap"
)


# --------------------------------------------------
# 2. 필요한 파일 / 폴더 확인
# --------------------------------------------------
required_paths = [
    REFERENCE_PATH,
    TEST_NORMAL_DIR,
    TEST_ANOMALY_DIR,
    GROUND_TRUTH_DIR,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더를 확인하세요: {path}"
        )


# --------------------------------------------------
# 3. 이전 Heatmap 결과 초기화
# --------------------------------------------------
if OUTPUT_DIR.exists():
    shutil.rmtree(
        OUTPUT_DIR
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 4. Normal Reference Map 불러오기
# --------------------------------------------------
_, reference_map = load_reference(
    REFERENCE_PATH
)


# --------------------------------------------------
# 5. Feature 추출용 ResNet18 준비
# --------------------------------------------------
model = create_backbone()


# --------------------------------------------------
# 6. 시각화할 대표 이미지 선택
# --------------------------------------------------
samples = []

normal_files = sorted(
    TEST_NORMAL_DIR.glob(
        "*.png"
    )
)

for path in normal_files[:2]:
    samples.append(
        (
            "good",
            path,
        )
    )


for defect_name in [
    "broken_large",
    "broken_small",
    "contamination",
]:
    defect_files = sorted(
        (
            TEST_ANOMALY_DIR
            / defect_name
        ).glob("*.png")
    )

    for path in defect_files[:2]:
        samples.append(
            (
                defect_name,
                path,
            )
        )


if not samples:
    raise RuntimeError(
        "Heatmap을 만들 Test 이미지가 없습니다."
    )


# --------------------------------------------------
# 7. 이미지별 Heatmap 생성
# --------------------------------------------------
for defect_type, image_path in samples:

    # Test 이미지의 Feature Map 추출
    _, spatial_map = extract_features(
        model,
        image_path,
    )

    # Normal Reference Map과 위치별 차이 계산
    score_map = anomaly_map(
        spatial_map,
        reference_map,
    )

    # Heatmap + Overlay 생성
    (
        _,
        heatmap_color,
        overlay,
    ) = make_heatmap_overlay(
        image_path,
        score_map,
    )


    stem = (
        f"{defect_type}_"
        f"{image_path.stem}"
    )


    # Heatmap 저장
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{stem}_heatmap.jpg"
        ),
        heatmap_color,
    )


    # 원본 + Heatmap Overlay 저장
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{stem}_overlay.jpg"
        ),
        overlay,
    )


    # --------------------------------------------------
    # 8. Anomaly라면 Ground Truth Mask도 저장
    # --------------------------------------------------
    if defect_type != "good":

        mask_path = (
            GROUND_TRUTH_DIR
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        if mask_path.exists():

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / f"{stem}_gt_mask.png"
                ),
                mask,
            )


    print(
        "Saved:",
        stem,
    )


print("=" * 70)
print("DAY07 HEATMAP COMPLETE")
print("=" * 70)

print(
    "Samples:",
    len(samples),
)

print(
    "Saved  :",
    OUTPUT_DIR,
)