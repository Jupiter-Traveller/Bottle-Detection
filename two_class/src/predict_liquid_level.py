"""Predict qualitative liquid level for one image using a trained level model."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from PIL import Image

from features_traditional import extract_traditional_features
from liquid_level_train_eval import yolo_to_xyxy
from preprocess import crop_bbox, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict liquid level for a single bottle image.")
    parser.add_argument("--model", required=True, help="Path to best_level_model.joblib.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--label", default=None, help="Optional YOLO label txt with one bbox line.")
    parser.add_argument(
        "--bbox",
        default=None,
        help="Optional xyxy bbox as xmin,ymin,xmax,ymax. Used when --label is not supplied.",
    )
    return parser.parse_args()


def parse_bbox_arg(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not value:
        return None
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must contain exactly four comma-separated values.")
    return tuple(parts)


def bbox_from_yolo_label(label_path: Path, image_path: Path) -> Tuple[float, float, float, float]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label_path} should contain exactly one label line.")
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"{label_path} should have 5 YOLO fields.")
    _, x_center, y_center, width, height = parts
    with Image.open(image_path) as image:
        image_w, image_h = image.size
    return yolo_to_xyxy(float(x_center), float(y_center), float(width), float(height), image_w, image_h)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    payload = joblib.load(model_path)
    model = payload["model"]
    class_names = payload.get("class_names", {})

    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    bbox = None
    if args.label:
        bbox = bbox_from_yolo_label(Path(args.label), image_path)
    else:
        bbox = parse_bbox_arg(args.bbox)
    if bbox is None:
        h, w = image.shape[:2]
        bbox = (0.0, 0.0, float(w), float(h))

    crop = crop_bbox(image, bbox)
    if crop is None:
        raise ValueError(f"Invalid bbox for image: {bbox}")

    vector, _ = extract_traditional_features(crop)
    X = vector.reshape(1, -1).astype(np.float32)
    pred_id = int(model.predict(X)[0])
    pred_name = class_names.get(pred_id, str(pred_id))

    print(f"image: {image_path}")
    print(f"bbox_xyxy: {','.join(f'{value:.2f}' for value in bbox)}")
    print(f"pred_class_id: {pred_id}")
    print(f"pred_class_name: {pred_name}")
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", payload.get("class_ids", range(len(prob))))
        pairs = [
            f"{int(cid)}:{class_names.get(int(cid), str(int(cid)))}={float(score):.4f}"
            for cid, score in zip(classes, prob)
        ]
        print("probabilities: " + ", ".join(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
