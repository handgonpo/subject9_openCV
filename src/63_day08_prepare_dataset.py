import random
import shutil
from pathlib import Path


RAW_ROOT = Path("samples/day07/original/bottle")
DAY08_ROOT = Path("samples/day08")
REPORT_PATH = Path("reports/day08_dataset_summary.md")

SEED = 42


# --------------------------------------------------
# 1. Day08 폴더를 새로 준비
# --------------------------------------------------
if DAY08_ROOT.exists():
    shutil.rmtree(DAY08_ROOT)


def copy_files(files, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for path in files:
        shutil.copy2(path, destination / path.name)


def copy_defect_sources(files, defect_type, destination_root):
    image_dir = destination_root / defect_type / "images"
    mask_dir = destination_root / defect_type / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for image_path in files:
        mask_path = (
            RAW_ROOT
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(mask_path, mask_dir / mask_path.name)


# --------------------------------------------------
# 2. 원본 목록
# --------------------------------------------------
normal_train = sorted((RAW_ROOT / "train/good").glob("*.png"))
normal_test = sorted((RAW_ROOT / "test/good").glob("*.png"))
broken_small = sorted((RAW_ROOT / "test/broken_small").glob("*.png"))
contamination = sorted((RAW_ROOT / "test/contamination").glob("*.png"))


rng_normal = random.Random(SEED)
rng_broken = random.Random(SEED + 1)
rng_contamination = random.Random(SEED + 2)

rng_normal.shuffle(normal_train)
rng_broken.shuffle(broken_small)
rng_contamination.shuffle(contamination)


# --------------------------------------------------
# 3. Normal 데이터 역할 분리
# --------------------------------------------------
one_reference = normal_train[0:1]
one_calibration = normal_train[1:21]
few_normal_support = normal_train[21:26]
cutpaste_normal = normal_train[26:36]
mini_normal = normal_train[36:41]

common_normal_query = normal_test[:10]


# --------------------------------------------------
# 4. broken_small 역할 분리
# --------------------------------------------------
few_anomaly_support = broken_small[0:5]
common_anomaly_query = broken_small[5:15]
cutpaste_defect_source = broken_small[15:17]


# --------------------------------------------------
# 5. contamination은 Mini Challenge용
# --------------------------------------------------
mini_defect_source = contamination[:2]


# --------------------------------------------------
# 6. 중복 검사
# --------------------------------------------------
normal_groups = [
    set(one_reference),
    set(one_calibration),
    set(few_normal_support),
    set(cutpaste_normal),
    set(mini_normal),
]

for i in range(len(normal_groups)):
    for j in range(i + 1, len(normal_groups)):
        if normal_groups[i] & normal_groups[j]:
            raise RuntimeError("Normal Train 역할 간 데이터가 겹칩니다.")

broken_groups = [
    set(few_anomaly_support),
    set(common_anomaly_query),
    set(cutpaste_defect_source),
]

for i in range(len(broken_groups)):
    for j in range(i + 1, len(broken_groups)):
        if broken_groups[i] & broken_groups[j]:
            raise RuntimeError("broken_small 역할 간 데이터가 겹칩니다.")


# --------------------------------------------------
# 7. 복사
# --------------------------------------------------
copy_files(
    one_reference,
    DAY08_ROOT / "one_shot/reference",
)

copy_files(
    one_calibration,
    DAY08_ROOT / "one_shot/calibration/normal",
)

copy_files(
    few_normal_support,
    DAY08_ROOT / "few_shot/support/normal",
)

copy_files(
    few_anomaly_support,
    DAY08_ROOT / "few_shot/support/anomaly",
)

copy_files(
    common_normal_query,
    DAY08_ROOT / "common_query/normal",
)

copy_files(
    common_anomaly_query,
    DAY08_ROOT / "common_query/anomaly/broken_small",
)

copy_files(
    cutpaste_normal,
    DAY08_ROOT / "cutpaste/normal",
)

copy_defect_sources(
    cutpaste_defect_source,
    "broken_small",
    DAY08_ROOT / "cutpaste/defect_source",
)

copy_files(
    mini_normal,
    DAY08_ROOT / "mini_challenge/normal",
)

copy_defect_sources(
    mini_defect_source,
    "contamination",
    DAY08_ROOT / "mini_challenge/defect_source",
)


# --------------------------------------------------
# 8. Summary 저장
# --------------------------------------------------
summary = f"""# Day08 Dataset Summary

Seed: {SEED}

## One-shot

- Reference: {len(one_reference)}
- Normal Calibration: {len(one_calibration)}

## Few-shot Support

- Normal: {len(few_normal_support)}
- Anomaly(broken_small): {len(few_anomaly_support)}

## Common Query

- Normal: {len(common_normal_query)}
- Anomaly(broken_small): {len(common_anomaly_query)}

## Cut-Paste

- Normal Target: {len(cutpaste_normal)}
- Defect Source(broken_small): {len(cutpaste_defect_source)}

## Mini Challenge

- Normal Target: {len(mini_normal)}
- Defect Source(contamination): {len(mini_defect_source)}

## Data Leakage Check

PASS
"""

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(summary, encoding="utf-8")


print("=" * 70)
print("DAY08 DATASET PREPARE")
print("=" * 70)
print("One-shot Reference        :", len(one_reference))
print("One-shot Calibration      :", len(one_calibration))
print("Few-shot Normal Support   :", len(few_normal_support))
print("Few-shot Anomaly Support  :", len(few_anomaly_support))
print("Common Query Normal       :", len(common_normal_query))
print("Common Query Anomaly      :", len(common_anomaly_query))
print("Cut-Paste Normal          :", len(cutpaste_normal))
print("Cut-Paste Defect Source   :", len(cutpaste_defect_source))
print("Mini Defect Source        :", len(mini_defect_source))
print("Data Leakage Check        : PASS")
print("Saved                     :", REPORT_PATH)