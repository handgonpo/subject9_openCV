import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# --------------------------------------------------
# 1. 실행 장치 설정
# --------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Pretrained ResNet18 설정
# --------------------------------------------------
WEIGHTS = ResNet18_Weights.DEFAULT

# ResNet18에 맞는 이미지 전처리 방법
PREPROCESS = WEIGHTS.transforms()


# --------------------------------------------------
# 3. Feature 추출용 ResNet18 생성
# --------------------------------------------------
def create_backbone():
    model = resnet18(
        weights=WEIGHTS
    )

    # ResNet18의 layer3 출력까지 사용합니다.
    # 이미지의 위치 정보가 어느 정도 남아 있기 때문에
    # Feature Vector와 Heatmap에 모두 사용할 수 있습니다.
    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-3]
    )

    # 학습하지 않고 Feature 추출용으로 사용합니다.
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    return feature_extractor


# --------------------------------------------------
# 4. 이미지를 ResNet18 입력 형식으로 변환
# --------------------------------------------------
def load_image_tensor(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = PREPROCESS(
        image
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# --------------------------------------------------
# 5. Feature Vector + Feature Map 추출
# --------------------------------------------------
@torch.no_grad()
def extract_features(
    model,
    image_path,
):
    tensor = load_image_tensor(
        image_path
    )

    feature_map = model(
        tensor
    )

    # ----------------------------------------------
    # 이미지 전체 특징
    # Feature Map → Pooling → Feature Vector
    # ----------------------------------------------
    pooled = F.adaptive_avg_pool2d(
        feature_map,
        (1, 1),
    ).flatten(1)

    pooled = F.normalize(
        pooled,
        dim=1,
    )

    # ----------------------------------------------
    # 위치별 특징
    # Heatmap에서 사용
    # ----------------------------------------------
    normalized_map = F.normalize(
        feature_map,
        dim=1,
    )

    vector = (
        pooled[0]
        .cpu()
        .numpy()
    )

    spatial_map = (
        normalized_map[0]
        .cpu()
        .numpy()
    )

    return vector, spatial_map


# --------------------------------------------------
# 6. 이미지 전체 Anomaly Score 계산
# --------------------------------------------------
def anomaly_score(
    feature_vector,
    reference_vector,
):
    difference = (
        feature_vector
        - reference_vector
    )

    score = np.linalg.norm(
        difference
    )

    return float(score)


# --------------------------------------------------
# 7. 위치별 Anomaly 차이 계산
# --------------------------------------------------
def anomaly_map(
    spatial_feature,
    reference_map,
):
    difference = (
        spatial_feature
        - reference_map
    )

    score_map = np.sqrt(
        np.sum(
            difference ** 2,
            axis=0,
        )
    )

    return score_map


# --------------------------------------------------
# 8. Heatmap 이미지 생성
# --------------------------------------------------
def make_heatmap_overlay(
    image_path,
    score_map,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    height, width = image.shape[:2]

    resized_map = cv2.resize(
        score_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    normalized = cv2.normalize(
        resized_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    heatmap_uint8 = (
        normalized.astype(
            np.uint8
        )
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap_color,
        0.35,
        0,
    )

    return (
        heatmap_uint8,
        heatmap_color,
        overlay,
    )


# --------------------------------------------------
# 9. 저장된 Normal Reference 불러오기
# --------------------------------------------------
def load_reference(
    reference_path,
):
    data = np.load(
        reference_path
    )

    return (
        data["reference_vector"],
        data["reference_map"],
    )