import random
import shutil
from pathlib import Path


# --------------------------------------------------
# 1. 원본과 실습용 데이터 경로
# --------------------------------------------------
RAW_ROOT = Path(
    "samples/day07/original/bottle"
)

OUTPUT_ROOT = Path(
    "samples/day07"
)

SEED = 42
TRAIN_RATIO = 0.8

random.seed(SEED)


# --------------------------------------------------
# 2. 원본 데이터 확인
# --------------------------------------------------
if not RAW_ROOT.exists():
    raise FileNotFoundError(
        f"MVTec bottle 원본을 확인하세요: {RAW_ROOT}"
    )


# --------------------------------------------------
# 3. 이전에 생성한 실습용 데이터만 초기화
#    original 폴더는 절대 삭제하지 않습니다.
# --------------------------------------------------
generated_dirs = [
    OUTPUT_ROOT / "train",
    OUTPUT_ROOT / "validation",
    OUTPUT_ROOT / "test",
    OUTPUT_ROOT / "ground_truth",
]

for directory in generated_dirs:
    if directory.exists():
        shutil.rmtree(directory)


# --------------------------------------------------
# 4. 실습용 폴더 생성
# --------------------------------------------------
train_dir = (
    OUTPUT_ROOT
    / "train/normal"
)

val_dir = (
    OUTPUT_ROOT
    / "validation/normal"
)

test_normal_dir = (
    OUTPUT_ROOT
    / "test/normal"
)

test_anomaly_dir = (
    OUTPUT_ROOT
    / "test/anomaly"
)

ground_truth_dir = (
    OUTPUT_ROOT
    / "ground_truth"
)


for directory in [
    train_dir,
    val_dir,
    test_normal_dir,
    test_anomaly_dir,
    ground_truth_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# --------------------------------------------------
# 5. Normal Train / Validation 분리
# --------------------------------------------------
normal_files = sorted(
    (RAW_ROOT / "train/good").glob("*.png")
)

if not normal_files:
    raise FileNotFoundError(
        RAW_ROOT / "train/good"
    )

random.shuffle(normal_files)

split_index = int(
    len(normal_files)
    * TRAIN_RATIO
)

train_files = (
    normal_files[:split_index]
)

val_files = (
    normal_files[split_index:]
)


for path in train_files:
    shutil.copy2(
        path,
        train_dir / path.name,
    )

for path in val_files:
    shutil.copy2(
        path,
        val_dir / path.name,
    )


# --------------------------------------------------
# 6. Normal Test 복사
# --------------------------------------------------
for path in sorted(
    (RAW_ROOT / "test/good").glob("*.png")
):
    shutil.copy2(
        path,
        test_normal_dir / path.name,
    )


# --------------------------------------------------
# 7. Anomaly Test 복사
# --------------------------------------------------
defect_names = [
    "broken_large",
    "broken_small",
    "contamination",
]

for defect_name in defect_names:

    source_dir = (
        RAW_ROOT
        / "test"
        / defect_name
    )

    destination = (
        test_anomaly_dir
        / defect_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in sorted(
        source_dir.glob("*.png")
    ):
        shutil.copy2(
            path,
            destination / path.name,
        )


# --------------------------------------------------
# 8. Ground Truth Mask 복사
# --------------------------------------------------
for defect_name in defect_names:

    source_dir = (
        RAW_ROOT
        / "ground_truth"
        / defect_name
    )

    destination = (
        ground_truth_dir
        / defect_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source_dir.exists():
        continue

    for path in sorted(
        source_dir.glob("*.png")
    ):
        shutil.copy2(
            path,
            destination / path.name,
        )


# --------------------------------------------------
# 9. 결과 확인
# --------------------------------------------------
print("=" * 70)
print("DAY07 DATASET PREPARED")
print("=" * 70)

print(
    "Normal Train      :",
    len(train_files),
)

print(
    "Normal Validation :",
    len(val_files),
)

print(
    "Normal Test       :",
    len(
        list(
            test_normal_dir.glob("*.png")
        )
    ),
)


for defect_name in defect_names:

    count = len(
        list(
            (
                test_anomaly_dir
                / defect_name
            ).glob("*.png")
        )
    )

    print(
        f"{defect_name:18}:",
        count,
    )