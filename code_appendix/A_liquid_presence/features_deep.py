"""Optional pretrained CNN embedding extraction."""

from typing import List, Tuple

import numpy as np
from PIL import Image


def is_torch_available() -> bool:
    """Return True when torch and torchvision can be imported."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


class DeepFeatureExtractor:
    """Extract ResNet embeddings with torchvision when available."""

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, device: str = "auto"):
        if not is_torch_available():
            raise ImportError("torch/torchvision is not available.")

        import torch
        from torchvision import models, transforms

        self.torch = torch
        self.transforms = transforms
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name

        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            self.embedding_dim = 2048
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            self.embedding_dim = 512
        else:
            raise ValueError("model_name must be 'resnet18' or 'resnet50'.")

        # 去掉最后分类层，保留全局平均池化后的 embedding。
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a 512-dim or 2048-dim embedding for one RGB crop."""
        if image_rgb is None or image_rgb.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        pil_image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            embedding = self.model(tensor).flatten().detach().cpu().numpy()
        return embedding.astype(np.float32)


def extract_deep_feature_matrix(
    images: List[np.ndarray],
    model_name: str = "resnet18",
    pretrained: bool = True,
) -> Tuple[np.ndarray, List[str], str]:
    """Extract deep embeddings for a list of images.

    Returns (matrix, feature_names, status_message). Import/download failures
    are reported as RuntimeError by the caller.
    """
    extractor = DeepFeatureExtractor(model_name=model_name, pretrained=pretrained)
    features = [extractor.extract(image) for image in images]
    matrix = np.vstack(features).astype(np.float32)
    names = [f"{model_name}_embedding_{i}" for i in range(matrix.shape[1])]
    return matrix, names, f"Deep features extracted with {model_name}."
