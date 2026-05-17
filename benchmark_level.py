"""
SCRIPT_GROUP: _evaluation
PURPOSE: Benchmark water-level classifier inference time.

This script measures classification latency on dataset_level/val or any image
directory. It supports YOLOv8-cls weights and the PyTorch checkpoints produced
by train_level_mobilenet.py / train_level_efficientnet.py.

YOLOv8-cls example:
python3 benchmark_level.py \
  --weights weights_level/yolov8n_cls_level_img224_base/weights/best.pt \
  --model-kind yolo \
  --images dataset_level/val \
  --imgsz 224 \
  --device 0 \
  --output weights_level/yolov8n_cls_level_img224_base/benchmark.csv

EfficientNet/MobileNet example:
python3 benchmark_level.py \
  --weights weights_level/efficientnet_b0_level_img320_color/best.pt \
  --model-kind torch \
  --images dataset_level/val \
  --device 0 \
  --output weights_level/efficientnet_b0_level_img320_color/benchmark.csv
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark water-level classifier inference time.")
    parser.add_argument("--weights", required=True, help="Classifier weights path.")
    parser.add_argument(
        "--model-kind",
        choices=("auto", "yolo", "torch"),
        default="auto",
        help="Classifier type. auto infers YOLO if checkpoint has parent weights/.",
    )
    parser.add_argument("--images", default="dataset_level/val", help="Image file or directory.")
    parser.add_argument("--imgsz", type=int, help="Input size. Defaults to checkpoint setting or 224.")
    parser.add_argument("--device", help="Device, e.g. 0, cuda, mps, or cpu. Defaults to cuda -> mps -> cpu.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images.")
    parser.add_argument("--repeat", type=int, default=5, help="Repeat rounds over all images.")
    parser.add_argument("--limit", type=int, help="Limit number of images.")
    parser.add_argument("--output", help="Optional CSV output path.")
    return parser.parse_args()


def select_torch_device(device_arg):
    if device_arg:
        if str(device_arg).isdigit():
            return torch.device(f"cuda:{device_arg}")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(device_arg):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found in: {path}")
    return images


def infer_model_kind(weights_path, model_kind):
    if model_kind != "auto":
        return model_kind
    weights_path = Path(weights_path)
    return "yolo" if weights_path.parent.name == "weights" else "torch"


def build_torch_model(model_name, num_classes):
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torch model: {model_name}")


def load_torch_classifier(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"1": 0, "2": 1, "3": 2})
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint missing model_name; cannot build torch classifier.")
    model = build_torch_model(model_name, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    imgsz = int(checkpoint.get("args", {}).get("imgsz", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform, imgsz


def benchmark_yolo(weights, images, imgsz, device, warmup, repeat):
    model = YOLO(str(weights))
    for image in images[: max(1, min(warmup, len(images)))]:
        model.predict(source=str(image), imgsz=imgsz, device=device, verbose=False)

    timings = []
    for _ in range(repeat):
        for image in images:
            started = time.perf_counter()
            model.predict(source=str(image), imgsz=imgsz, device=device, verbose=False)
            timings.append((time.perf_counter() - started) * 1000)
    return timings


def benchmark_torch(weights, images, device, warmup, repeat):
    model, transform, checkpoint_imgsz = load_torch_classifier(weights, device)
    tensors = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        tensors.append(transform(image).unsqueeze(0).to(device))

    with torch.no_grad():
        for tensor in tensors[: max(1, min(warmup, len(tensors)))]:
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(repeat):
            for tensor in tensors:
                started = time.perf_counter()
                model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
    return timings, checkpoint_imgsz


def summarize(timings):
    timings = sorted(timings)
    avg_ms = sum(timings) / len(timings)
    p50_ms = timings[len(timings) // 2]
    p95_ms = timings[int(len(timings) * 0.95) - 1]
    fps = 1000 / avg_ms if avg_ms > 0 else 0
    return avg_ms, p50_ms, p95_ms, fps


def main():
    args = parse_args()
    images = list_images(args.images)
    if args.limit:
        images = images[: args.limit]

    model_kind = infer_model_kind(args.weights, args.model_kind)
    if model_kind == "yolo":
        device = yolo_device(args.device)
        imgsz = args.imgsz or 224
        timings = benchmark_yolo(args.weights, images, imgsz, device, args.warmup, args.repeat)
    else:
        device = select_torch_device(args.device)
        timings, checkpoint_imgsz = benchmark_torch(args.weights, images, device, args.warmup, args.repeat)
        imgsz = args.imgsz or checkpoint_imgsz

    avg_ms, p50_ms, p95_ms, fps = summarize(timings)
    summary = {
        "weights": args.weights,
        "model_kind": model_kind,
        "images": args.images,
        "imgsz": imgsz,
        "device": str(device),
        "image_count": len(images),
        "repeat": args.repeat,
        "total_images": len(images) * args.repeat,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }

    print("Level classifier benchmark")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)


if __name__ == "__main__":
    main()
