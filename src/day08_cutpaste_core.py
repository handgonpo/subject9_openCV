import csv
import random
from pathlib import Path

import cv2
import numpy as np


def extract_patch(image_path, mask_path):
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)

    if mask is None:
        raise FileNotFoundError(mask_path)

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(binary)

    if points is None:
        raise RuntimeError(f"Mask가 비어 있습니다: {mask_path}")

    x, y, w, h = cv2.boundingRect(points)

    patch = image[y:y + h, x:x + w].copy()
    patch_mask = binary[y:y + h, x:x + w].copy()

    return patch, patch_mask


def rotate_pair(patch, mask, angle):
    height, width = patch.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated_patch = cv2.warpAffine(
        patch,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

    return rotated_patch, rotated_mask


def paste_patch(
    background,
    patch,
    mask,
    rng,
    scale_range=(0.7, 1.2),
    angle_range=(-15.0, 15.0),
):
    result = background.copy()

    scale = rng.uniform(*scale_range)
    angle = rng.uniform(*angle_range)

    new_width = max(4, int(patch.shape[1] * scale))
    new_height = max(4, int(patch.shape[0] * scale))

    patch = cv2.resize(
        patch,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = cv2.resize(
        mask,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    patch, mask = rotate_pair(
        patch,
        mask,
        angle,
    )

    image_h, image_w = result.shape[:2]
    patch_h, patch_w = patch.shape[:2]

    if patch_w >= image_w or patch_h >= image_h:
        raise RuntimeError("Defect Patch가 정상 이미지보다 큽니다.")

    # MVTec bottle은 제품이 중앙에 있으므로
    # 교육용 Baseline에서는 중앙 영역을 우선 사용합니다.
    x_min = int(image_w * 0.25)
    x_max = int(image_w * 0.75) - patch_w

    y_min = int(image_h * 0.10)
    y_max = int(image_h * 0.90) - patch_h

    if x_max <= x_min:
        x_min = 0
        x_max = image_w - patch_w

    if y_max <= y_min:
        y_min = 0
        y_max = image_h - patch_h

    x = rng.randint(x_min, x_max)
    y = rng.randint(y_min, y_max)

    roi = result[
        y:y + patch_h,
        x:x + patch_w,
    ]

    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = alpha[..., None]

    blended = (
        patch.astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha)
    )

    result[
        y:y + patch_h,
        x:x + patch_w,
    ] = np.clip(blended, 0, 255).astype(np.uint8)

    info = {
        "x": x,
        "y": y,
        "width": patch_w,
        "height": patch_h,
        "rotation": round(angle, 2),
        "scale": round(scale, 3),
    }

    return result, info


def generate_synthetic_set(
    normal_dir,
    defect_image_dir,
    defect_mask_dir,
    output_dir,
    metadata_path,
    defect_type,
    count,
    seed,
    scale_range=(0.7, 1.2),
    angle_range=(-15.0, 15.0),
):
    normal_files = sorted(Path(normal_dir).glob("*.png"))
    defect_files = sorted(Path(defect_image_dir).glob("*.png"))

    if not normal_files:
        raise RuntimeError("Normal Target 이미지가 없습니다.")

    if not defect_files:
        raise RuntimeError("Defect Source 이미지가 없습니다.")

    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # 재실행할 때 이전 synthetic 결과를 섞지 않습니다.
    for old_file in output_dir.glob("synthetic_*.png"):
        old_file.unlink()

    rng = random.Random(seed)
    rows = []

    for index in range(count):
        normal_path = rng.choice(normal_files)
        defect_path = rng.choice(defect_files)

        mask_path = (
            Path(defect_mask_dir)
            / f"{defect_path.stem}_mask.png"
        )

        background = cv2.imread(str(normal_path))

        if background is None:
            raise FileNotFoundError(normal_path)

        patch, mask = extract_patch(
            defect_path,
            mask_path,
        )

        synthetic, info = paste_patch(
            background,
            patch,
            mask,
            rng,
            scale_range=scale_range,
            angle_range=angle_range,
        )

        output_name = f"synthetic_{index:03d}.png"
        output_path = output_dir / output_name

        cv2.imwrite(str(output_path), synthetic)

        rows.append(
            {
                "synthetic_file": output_name,
                "source_normal": normal_path.name,
                "source_defect": defect_path.name,
                "defect_type": defect_type,
                "seed": seed,
                **info,
            }
        )

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows