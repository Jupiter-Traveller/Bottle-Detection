"""Bottle detection and crop utilities for the final prediction pipeline."""

from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


def select_device(device_arg: str | None = None) -> torch.device:
    """Select cuda, mps, or cpu for inference."""
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _yolo_device(device: torch.device) -> int | str:
    if device.type == "cuda":
        return 0 if device.index is None else device.index
    return str(device)


def detect_bottle_crop(
    image_path: str | Path,
    weights_path: str | Path,
    conf: float = 0.25,
    padding: float = 0.05,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Detect the highest-confidence bottle and return a padded PIL crop."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = device or select_device()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image_width, image_height = image.size

    detector = YOLO(str(weights_path))
    result = detector.predict(
        source=str(image_path),
        conf=conf,
        device=_yolo_device(device),
        verbose=False,
    )[0]
    if len(result.boxes) == 0:
        raise RuntimeError(f"No bottle detected in image: {image_path}")

    best_box = max(result.boxes, key=lambda box: float(box.conf[0].item()))
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding
    pad_y = box_height * padding

    x1 = int(round(_clamp(x1 - pad_x, 0, image_width)))
    y1 = int(round(_clamp(y1 - pad_y, 0, image_height)))
    x2 = int(round(_clamp(x2 + pad_x, 0, image_width)))
    y2 = int(round(_clamp(y2 + pad_y, 0, image_height)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bottle crop box: {[x1, y1, x2, y2]}")

    return {
        "box_xyxy": [x1, y1, x2, y2],
        "confidence": float(best_box.conf[0].item()),
        "crop": image.crop((x1, y1, x2, y2)),
    }
