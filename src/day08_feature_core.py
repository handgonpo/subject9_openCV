import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def create_feature_model():
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    feature_model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_model.eval()
    feature_model.to(DEVICE)

    return feature_model


@torch.no_grad()
def extract_feature(model, image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = (
        PREPROCESS(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    feature = model(tensor).flatten(1)

    feature = F.normalize(
        feature,
        p=2,
        dim=1,
    )

    return feature[0].cpu().numpy()


def cosine_similarity(feature_a, feature_b):
    return float(
        np.dot(feature_a, feature_b)
    )


def build_prototype(features):
    stacked = np.stack(features)
    prototype = stacked.mean(axis=0)

    norm = np.linalg.norm(prototype)

    if norm == 0:
        raise RuntimeError("Prototype norm이 0입니다.")

    return prototype / norm