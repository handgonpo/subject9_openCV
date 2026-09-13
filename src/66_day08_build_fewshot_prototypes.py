import json
from pathlib import Path

import numpy as np

from day08_feature_core import (
    build_prototype,
    create_feature_model,
    extract_feature,
)


SUPPORT_ROOT = Path("samples/day08/few_shot/support")
ARTIFACT_PATH = Path("artifacts/day08/fewshot_prototypes.npz")
CONFIG_PATH = Path("configs/day08_fewshot_config.json")


model = create_feature_model()


def extract_folder(folder):
    files = sorted(folder.glob("*.png"))

    if not files:
        raise RuntimeError(f"Support 이미지가 없습니다: {folder}")

    features = [
        extract_feature(model, image_path)
        for image_path in files
    ]

    return files, features


normal_files, normal_features = extract_folder(
    SUPPORT_ROOT / "normal"
)

anomaly_files, anomaly_features = extract_folder(
    SUPPORT_ROOT / "anomaly"
)


normal_prototype = build_prototype(normal_features)
anomaly_prototype = build_prototype(anomaly_features)


ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    ARTIFACT_PATH,
    normal_prototype=normal_prototype,
    anomaly_prototype=anomaly_prototype,
)


config = {
    "baseline_version": "day08-few-shot-v1.0",
    "dataset": "MVTec AD bottle",
    "method": "prototype_cosine_similarity",
    "backbone": "resnet18_imagenet_pretrained",
    "normal_support_count": len(normal_files),
    "anomaly_support_count": len(anomaly_files),
    "anomaly_support_type": "broken_small",
    "decision_rule": "choose prototype with higher similarity",
}

CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 70)
print("DAY08 FEW-SHOT PROTOTYPE")
print("=" * 70)
print("Normal Support  :", len(normal_files))
print("Anomaly Support :", len(anomaly_files))
print("Anomaly Type    : broken_small")
print("Saved Artifact  :", ARTIFACT_PATH)
print("Saved Config    :", CONFIG_PATH)