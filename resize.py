from pathlib import Path

import cv2


input_dir = Path("samples/day05/raw")
output_dir = Path("samples/day05/resized")

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


image_paths = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(
        input_dir.glob(extension)
    )


for image_path in sorted(image_paths):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            "읽기 실패:",
            image_path
        )
        continue

    resized = cv2.resize(
        image,
        (1280, 960),
        interpolation=cv2.INTER_AREA,
    )

    save_path = (
        output_dir
        / image_path.name
    )

    cv2.imwrite(
        str(save_path),
        resized
    )

    print(
        image_path.name,
        "→",
        resized.shape[1],
        "x",
        resized.shape[0]
    )


print("전체 변환 완료")