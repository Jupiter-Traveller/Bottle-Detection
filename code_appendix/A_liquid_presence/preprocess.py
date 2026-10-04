"""Image loading and bbox preprocessing utilities."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def read_image(image_path: Path) -> Optional[np.ndarray]:
    """Read an image as an RGB ndarray."""
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        return None


def sanitize_bbox(
    bbox: Sequence[float], image_shape: Tuple[int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """Clip a bbox to image bounds and return integer coordinates."""
    h, w = image_shape[:2]
    try:
        xmin, ymin, xmax, ymax = [int(round(float(v))) for v in bbox]
    except Exception:
        return None

    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h))

    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def crop_bbox(image: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    """Crop image by bbox after clipping to valid image coordinates."""
    clean_bbox = sanitize_bbox(bbox, image.shape)
    if clean_bbox is None:
        return None
    xmin, ymin, xmax, ymax = clean_bbox
    crop = image[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return None
    return crop


def resize_for_features(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to a fixed size for stable feature dimensions."""
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized)


def save_image(image_path: Path, image: np.ndarray) -> bool:
    """Save an RGB image."""
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(image_path)
        return True
    except Exception:
        return False
