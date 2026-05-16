"""
Predict water level with MobileNetV3-Small or EfficientNet-B0 checkpoints.

Classify an already-cropped bottle image:
python3 predict_level_torch.py path/to/bottle_crop.jpg \
  --weights weights_level/mobilenetv3_small_level_img224_base/best.pt

Run bottle detection first, then classify the crop:
python3 predict_level_torch.py path/to/original.jpg \
  --crop \
  --det-weights weights/yolov8s_mixed_img640_light_b16/weights/best.pt \
  --weights weights_level/efficientnet_b0_level_img224_base/best.pt
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8s_mixed_img640_light_b16" / "weights" / "best.pt"
DEFAULT_CROP_DIR = PROJECT_ROOT / "cropped_level_infer"
LEVEL_NAMES = {
    "1": "low",
    "2": "medium",
    "3": "high",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict water level using a PyTorch classifier.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("--weights", required=True, help="MobileNet/EfficientNet checkpoint path.")
    parser.add_argument(
        "--model-type",
        choices=("auto", "mobilenet_v3_small", "efficientnet_b0"),
        default="auto",
        help="Model architecture. Defaults to reading model_name from checkpoint.",
    )
    parser.add_argument("--imgsz", type=int, help="Input size. Defaults to checkpoint args.imgsz or 224.")
    parser.add_argument("--device", help="cuda, cuda:0, cpu, or mps. Defaults to auto.")
    parser.add_argument("--crop", action="store_true", help="Run bottle detector first and classify the crop.")
    parser.add_argument(
        "--det-weights",
        default=str(DEFAULT_DET_WEIGHTS),
        help="Bottle detector weights used when --crop is enabled.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.05, help="Detector crop padding fraction.")
    parser.add_argument("--save-crop", action="store_true", help="Save detector crop when --crop is enabled.")
    parser.add_argument(
        "--crop-output-dir",
        default=str(DEFAULT_CROP_DIR),
        help="Directory for saved crops when --save-crop is enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_type, num_classes):
    if model_type == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model type: {model_type}")


def load_classifier(weights_path, model_type_arg, device):
    checkpoint = torch.load(weights_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"1": 0, "2": 1, "3": 2})
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    model_type = model_type_arg
    if model_type == "auto":
        model_type = checkpoint.get("model_name")
    if model_type not in {"mobilenet_v3_small", "efficientnet_b0"}:
        raise ValueError(
            "Could not infer model type from checkpoint. Pass "
            "--model-type mobilenet_v3_small or --model-type efficientnet_b0."
        )

    model = build_model(model_type, num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    checkpoint_args = checkpoint.get("args", {})
    imgsz = checkpoint_args.get("imgsz", 224)
    return model, idx_to_class, model_type, int(imgsz)


def build_transform(imgsz):
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def clamp(value, low, high):
    return max(low, min(high, value))


def detect_and_crop(image_path, det_weights, conf, padding, save_crop, crop_output_dir, device):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    model = YOLO(str(det_weights))
    yolo_device = 0 if device.type == "cuda" else str(device)
    result = model.predict(source=str(image_path), conf=conf, device=yolo_device, verbose=False)[0]
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


def classify_image(image_path, model, idx_to_class, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()

    top_index = int(probs_tensor.argmax().item())
    class_name = str(idx_to_class[top_index])
    probabilities = {
        str(idx_to_class[index]): float(prob)
        for index, prob in enumerate(probs_tensor.tolist())
    }
    return {
        "level": class_name,
        "level_name": LEVEL_NAMES.get(class_name, class_name),
        "confidence": float(probs_tensor[top_index].item()),
        "probabilities": probabilities,
    }


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = select_device(args.device)
    model, idx_to_class, model_type, checkpoint_imgsz = load_classifier(args.weights, args.model_type, device)
    imgsz = args.imgsz or checkpoint_imgsz
    transform = build_transform(imgsz)

    classifier_input = image_path
    crop_info = None
    if args.crop:
        crop_info = detect_and_crop(
            image_path=image_path,
            det_weights=args.det_weights,
            conf=args.conf,
            padding=args.padding,
            save_crop=args.save_crop,
            crop_output_dir=args.crop_output_dir,
            device=device,
        )
        classifier_input = crop_info["image_path"]

    cls_result = classify_image(classifier_input, model, idx_to_class, transform, device)
    output = {
        "input": str(image_path),
        "classifier_input": str(classifier_input),
        "used_detector_crop": args.crop,
        "model_type": model_type,
        "imgsz": imgsz,
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
    print(f"model_type: {output['model_type']}")
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
