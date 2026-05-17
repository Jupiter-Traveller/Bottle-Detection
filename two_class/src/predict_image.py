"""Predict whether a bottle image/crop contains liquid residue."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from config import MODEL_DIR
from features_traditional import extract_traditional_features
from preprocess import crop_bbox, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict has_liquid for one image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Optional bottle bbox in pixel coordinates. If omitted, the whole image is used.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional experiment run directory containing models/. Overrides default model/scaler paths.",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_DIR / "best_model.joblib"),
        help="Path to best_model.joblib.",
    )
    parser.add_argument(
        "--scaler",
        default=str(MODEL_DIR / "scaler.joblib"),
        help="Path to scaler.joblib.",
    )
    return parser.parse_args()


def predict(image_path: Path, bbox, model_path: Path, scaler_path: Path) -> dict:
    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    used_bbox = bbox
    if bbox is not None:
        crop = crop_bbox(image, bbox)
        if crop is None:
            raise ValueError(f"Invalid bbox for image {image_path}: {bbox}")
    else:
        crop = image

    model_bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model = model_bundle["model"]
    feature_mode = model_bundle.get("feature_mode", "traditional")
    if feature_mode != "traditional":
        raise ValueError(f"This predictor currently supports traditional features, got {feature_mode}")

    features, feature_names = extract_traditional_features(crop)
    expected_names = model_bundle.get("feature_names")
    if expected_names and list(expected_names) != feature_names:
        raise ValueError("Feature names do not match the saved model.")

    X = scaler.transform(features.reshape(1, -1))
    label = int(model.predict(X)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))

    return {
        "image": str(image_path),
        "bbox": used_bbox,
        "prediction": label,
        "prediction_text": "has_liquid" if label == 1 else "empty",
        "probability_has_liquid": probability,
        "model_name": model_bundle.get("model_name", "unknown"),
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        model_path = run_dir / "models" / "best_model.joblib"
        scaler_path = run_dir / "models" / "scaler.joblib"

    result = predict(Path(args.image), args.bbox, model_path, scaler_path)
    print(f"image: {result['image']}")
    if result["bbox"] is None:
        print("bbox: full image used")
        print("note: no bottle detector is used yet; full-image prediction is only a rough check.")
    else:
        print(f"bbox: {result['bbox']}")
    print(f"model: {result['model_name']}")
    print(f"prediction: {result['prediction']} ({result['prediction_text']})")
    if result["probability_has_liquid"] is not None:
        print(f"probability_has_liquid: {result['probability_has_liquid']:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Prediction stopped] {exc}", file=sys.stderr)
        raise SystemExit(1)
