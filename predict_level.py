import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# python3 predict_level.py .jpg \
#  --cls-weights best.pt

#  python3 predict_level.py .jpg \
#   --crop \
#   --det-weights best.pt \
#   --cls-weights best.pt \
#   --padding 0.05


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_mixed_img640_base" / "weights" / "best.pt"
DEFAULT_CLS_WEIGHTS = PROJECT_ROOT / "weights_level" / "yolov8n_cls_level_img224_base" / "weights" / "best.pt"
DEFAULT_CROP_DIR = PROJECT_ROOT / "cropped_level_infer"
LEVEL_NAMES = {
    "1": "low",
    "2": "medium",
    "3": "high",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict water level 1/2/3 from an image.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument(
        "--cls-weights",
        default=str(DEFAULT_CLS_WEIGHTS),
        help="Water-level classification weights.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Run bottle detector first and classify the detected crop.",
    )
    parser.add_argument(
        "--det-weights",
        default=str(DEFAULT_DET_WEIGHTS),
        help="Bottle detector weights used when --crop is enabled.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detector confidence threshold used when --crop is enabled.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Extra crop padding as a fraction of detected box size.",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save the detector crop when --crop is enabled.",
    )
    parser.add_argument(
        "--crop-output-dir",
        default=str(DEFAULT_CROP_DIR),
        help="Directory for saved crops when --save-crop is enabled.",
    )
    parser.add_argument(
        "--device",
        help="Inference device, for example 0, cuda:0, cpu, or mps. Defaults to Ultralytics auto selection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser.parse_args()


def clamp(value, low, high):
    return max(low, min(high, value))


def detect_and_crop(image_path, det_weights, conf, padding, save_crop, crop_output_dir, device=None):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    model = YOLO(str(det_weights))
    predict_kwargs = {"source": str(image_path), "conf": conf, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    result = results[0]
    if len(result.boxes) == 0:
        raise RuntimeError(f"No bottle detected in image: {image_path}")

    best_box = max(result.boxes, key=lambda box: float(box.conf[0].item()))
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding
    pad_y = box_height * padding

    x1 = int(round(clamp(x1 - pad_x, 0, image_width)))
    y1 = int(round(clamp(y1 - pad_y, 0, image_height)))
    x2 = int(round(clamp(x2 + pad_x, 0, image_width)))
    y2 = int(round(clamp(y2 + pad_y, 0, image_height)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid detected crop box for image: {image_path}")

    cropped = image.crop((x1, y1, x2, y2))
    if save_crop:
        crop_output_dir = Path(crop_output_dir)
        crop_output_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_output_dir / f"{image_path.stem}_crop{image_path.suffix.lower()}"
    else:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=image_path.suffix.lower() or ".jpg",
            prefix=f"{image_path.stem}_crop_",
            delete=False,
        )
        crop_path = Path(temp_file.name)
        temp_file.close()

    cropped.save(crop_path)
    return {
        "image_path": crop_path,
        "box_xyxy": [x1, y1, x2, y2],
        "det_conf": float(best_box.conf[0].item()),
    }


def classify_level(image_path, cls_weights, device=None):
    model = YOLO(str(cls_weights))
    predict_kwargs = {"source": str(image_path), "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    result = model.predict(**predict_kwargs)[0]
    if result.probs is None:
        raise RuntimeError(f"Classifier did not return probabilities for image: {image_path}")

    top1_index = int(result.probs.top1)
    top1_conf = float(result.probs.top1conf.item())
    class_name = str(result.names[top1_index])

    probs = {}
    for index, prob in enumerate(result.probs.data.tolist()):
        probs[str(result.names[index])] = float(prob)

    return {
        "level": class_name,
        "level_name": LEVEL_NAMES.get(class_name, class_name),
        "confidence": top1_conf,
        "probabilities": probs,
    }


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    cls_input = image_path
    crop_info = None
    if args.crop:
        crop_info = detect_and_crop(
            image_path=image_path,
            det_weights=args.det_weights,
            conf=args.conf,
            padding=args.padding,
            save_crop=args.save_crop,
            crop_output_dir=args.crop_output_dir,
            device=args.device,
        )
        cls_input = crop_info["image_path"]

    cls_result = classify_level(
        image_path=cls_input,
        cls_weights=args.cls_weights,
        device=args.device,
    )

    output = {
        "input": str(image_path),
        "classifier_input": str(cls_input),
        "used_detector_crop": args.crop,
        **cls_result,
    }
    if crop_info is not None:
        output["detector"] = {
            "box_xyxy": crop_info["box_xyxy"],
            "confidence": crop_info["det_conf"],
        }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print(f"input: {output['input']}")
    if args.crop:
        print(f"crop: {output['classifier_input']}")
        print(f"detector_confidence: {output['detector']['confidence']:.4f}")
        print(f"detector_box_xyxy: {output['detector']['box_xyxy']}")
    print(f"level: {output['level']} ({output['level_name']})")
    print(f"confidence: {output['confidence']:.4f}")
    print("probabilities:")
    for class_name in sorted(output["probabilities"]):
        readable_name = LEVEL_NAMES.get(class_name, class_name)
        print(f"  {class_name} ({readable_name}): {output['probabilities'][class_name]:.4f}")


if __name__ == "__main__":
    main()
