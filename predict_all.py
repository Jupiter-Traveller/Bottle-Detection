"""Run the final bottle liquid-residue prediction pipeline.

Pipeline:
1. Detect and crop the bottle with the YOLOv8 position model.
2. Predict whether liquid exists with the two-class traditional ML model.
3. If liquid exists, predict water level 1/2/3 with the PyTorch classifier.

Example:
python3 predict_all.py --image examples/input.jpg
python3 predict_all.py --image examples/input.jpg --json
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.bottle_detector import detect_bottle_crop, select_device
from src.liquid_presence import predict_liquid_presence
from src.water_level import LEVEL_NAMES, predict_water_level


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bottle_detection_matplotlib"))


DEFAULT_DET_WEIGHTS = WEIGHTS_DIR / "yolov8n_img512_light_best.pt"
DEFAULT_BINARY_MODEL = WEIGHTS_DIR / "two_class_best_model.joblib"
DEFAULT_BINARY_SCALER = WEIGHTS_DIR / "two_class_scaler.joblib"
DEFAULT_LEVEL_WEIGHTS = WEIGHTS_DIR / "efficientnet_b0_img320_color_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bottle position, liquid presence, and water level.")
    parser.add_argument("image_arg", nargs="?", help="Input image path. Same as --image.")
    parser.add_argument("--image", dest="image_opt", help="Input image path.")
    parser.add_argument("--det-weights", default=str(DEFAULT_DET_WEIGHTS), help="YOLO bottle detector weights.")
    parser.add_argument("--binary-model", default=str(DEFAULT_BINARY_MODEL), help="Two-class joblib model.")
    parser.add_argument("--binary-scaler", default=str(DEFAULT_BINARY_SCALER), help="Two-class joblib scaler.")
    parser.add_argument("--level-weights", default=str(DEFAULT_LEVEL_WEIGHTS), help="Water-level classifier checkpoint.")
    parser.add_argument("--det-conf", type=float, default=0.25, help="YOLO detector confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.05, help="Extra crop padding as a fraction of box size.")
    parser.add_argument("--device", help="cuda, cuda:0, cpu, or mps. Defaults to auto.")
    parser.add_argument("--save-crop", help="Optional path to save the detected bottle crop.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def require_existing_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_output(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image_opt or args.image_arg or "")
    if not image_path:
        raise ValueError("Pass an input image with --image path/to/image.jpg")
    require_existing_file(image_path, "Input image")

    det_weights = require_existing_file(args.det_weights, "Detector weights")
    binary_model = require_existing_file(args.binary_model, "Binary model")
    binary_scaler = require_existing_file(args.binary_scaler, "Binary scaler")
    level_weights = require_existing_file(args.level_weights, "Water-level weights")

    device = select_device(args.device)
    det_result = detect_bottle_crop(
        image_path=image_path,
        weights_path=det_weights,
        conf=args.det_conf,
        padding=args.padding,
        device=device,
    )
    crop = det_result["crop"]
    if args.save_crop:
        crop_path = Path(args.save_crop)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)

    binary_result = predict_liquid_presence(
        crop=crop,
        model_path=binary_model,
        scaler_path=binary_scaler,
    )

    output: dict[str, Any] = {
        "image": str(image_path),
        "device": str(device),
        "detector": {
            "weights": str(det_weights),
            "box_xyxy": det_result["box_xyxy"],
            "confidence": det_result["confidence"],
        },
        "has_liquid": {
            "weights": str(binary_model),
            **binary_result,
        },
        "level": None,
    }
    if args.save_crop:
        output["detector"]["saved_crop"] = str(Path(args.save_crop))

    if int(binary_result["prediction"]) == 1:
        output["level"] = {
            "weights": str(level_weights),
            **predict_water_level(crop=crop, weights_path=level_weights, device=device),
        }

    return output


def print_human(output: dict[str, Any]) -> None:
    print(f"image: {output['image']}")
    print(f"device: {output['device']}")
    print("detector:")
    print(f"  box_xyxy: {output['detector']['box_xyxy']}")
    print(f"  confidence: {output['detector']['confidence']:.4f}")
    if "saved_crop" in output["detector"]:
        print(f"  saved_crop: {output['detector']['saved_crop']}")

    print("has_liquid:")
    print(f"  model: {output['has_liquid']['model_name']}")
    print(f"  prediction: {output['has_liquid']['prediction']} ({output['has_liquid']['prediction_text']})")
    probability = output["has_liquid"]["probability_has_liquid"]
    if probability is not None:
        print(f"  probability_has_liquid: {probability:.4f}")

    if output["level"] is None:
        print("level: skipped because has_liquid prediction is empty")
        return

    print("level:")
    print(f"  model: {output['level']['model_name']}")
    print(f"  imgsz: {output['level']['imgsz']}")
    print(f"  prediction: {output['level']['level']} ({output['level']['level_name']})")
    print(f"  confidence: {output['level']['confidence']:.4f}")
    print("  probabilities:")
    for class_name in sorted(output["level"]["probabilities"]):
        readable_name = LEVEL_NAMES.get(class_name, class_name)
        probability = output["level"]["probabilities"][class_name]
        print(f"    {class_name} ({readable_name}): {probability:.4f}")


def main() -> None:
    args = parse_args()
    output = build_output(args)
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human(output)


if __name__ == "__main__":
    main()
