from pathlib import Path

import numpy as np

from day07_anomaly_core import (
    create_backbone,
    extract_features,
)


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
TRAIN_DIR = Path(
    "samples/day07/train/normal"
)

OUTPUT_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Feature 추출용 ResNet18 준비
# --------------------------------------------------
model = create_backbone()


# --------------------------------------------------
# 3. 정상 이미지의 Feature를 저장할 리스트
# --------------------------------------------------
vectors = []
maps = []


# --------------------------------------------------
# 4. Normal Train 이미지 읽기
# --------------------------------------------------
image_files = sorted(
    TRAIN_DIR.glob("*.png")
)

if not image_files:
    raise RuntimeError(
        "Normal Train 이미지가 없습니다."
    )

print(
    "Normal Train:",
    len(image_files),
)


# --------------------------------------------------
# 5. 정상 이미지마다 Feature 추출
# --------------------------------------------------
for index, image_path in enumerate(
    image_files,
    start=1,
):
    vector, spatial_map = extract_features(
        model,
        image_path,
    )

    vectors.append(vector)
    maps.append(spatial_map)

    if (
        index == 1
        or index % 20 == 0
        or index == len(image_files)
    ):
        print(
            f"[{index}/{len(image_files)}]",
            image_path.name,
        )


# --------------------------------------------------
# 6. 여러 이미지의 Feature를 하나의 배열로 묶기
# --------------------------------------------------
vectors = np.stack(
    vectors
)

maps = np.stack(
    maps
)


# --------------------------------------------------
# 7. Feature Vector 평균
#    → 이미지 전체 비교용 정상 기준
# --------------------------------------------------
reference_vector = np.mean(
    vectors,
    axis=0,
)

vector_norm = np.linalg.norm(
    reference_vector
)

if vector_norm == 0:
    raise RuntimeError(
        "Normal Reference Vector를 만들 수 없습니다."
    )

reference_vector = (
    reference_vector
    / vector_norm
)


# --------------------------------------------------
# 8. Feature Map 평균
#    → Heatmap 비교용 정상 기준
# --------------------------------------------------
reference_map = np.mean(
    maps,
    axis=0,
)


# --------------------------------------------------
# 9. Normal Reference 저장
# --------------------------------------------------
np.savez_compressed(
    OUTPUT_PATH,
    reference_vector=reference_vector,
    reference_map=reference_map,
)


# --------------------------------------------------
# 10. 결과 확인
# --------------------------------------------------
print("=" * 70)
print("Normal Reference 생성 완료")
print("=" * 70)

print(
    "Vector shape:",
    reference_vector.shape,
)

print(
    "Map shape   :",
    reference_map.shape,
)

print(
    "Saved       :",
    OUTPUT_PATH,
)