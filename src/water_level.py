"""PyTorch water-level classifier used by predict_all.py."""

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


LEVEL_NAMES = {
    "1": "low",
    "2": "medium",
    "3": "high",
}


def build_level_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a classifier architecture matching the saved checkpoint."""
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported level model: {model_name}")


def load_level_classifier(weights_path: str | Path, device: torch.device) -> Dict:
    """Load model, class mapping, and image transform from a checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"1": 0, "2": 1, "3": 2})
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    model_name = checkpoint.get("model_name", "efficientnet_b0")

    model = build_level_model(model_name, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    checkpoint_args = checkpoint.get("args", {})
    imgsz = int(checkpoint_args.get("imgsz", 320))
    transform = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return {
        "model": model,
        "idx_to_class": idx_to_class,
        "model_name": model_name,
        "imgsz": imgsz,
        "transform": transform,
    }


def predict_water_level(crop: Image.Image, weights_path: str | Path, device: torch.device) -> dict:
    """Predict liquid level class 1/2/3 from a cropped bottle image."""
    loaded = load_level_classifier(weights_path, device)
    tensor = loaded["transform"](crop.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = loaded["model"](tensor)
        probs_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()

    top_index = int(probs_tensor.argmax().item())
    level = str(loaded["idx_to_class"][top_index])
    probabilities = {
        str(loaded["idx_to_class"][index]): float(prob)
        for index, prob in enumerate(probs_tensor.tolist())
    }
    return {
        "model_name": loaded["model_name"],
        "imgsz": loaded["imgsz"],
        "level": level,
        "level_name": LEVEL_NAMES.get(level, level),
        "confidence": float(probs_tensor[top_index].item()),
        "probabilities": probabilities,
    }
