"""
SCRIPT_GROUP: _evaluation
PURPOSE: Benchmark bottle detector inference time on Mac or Ubuntu.

This script measures only YOLO bottle-position inference latency. It does not
train models and does not change datasets.

Mac example:
python3 benchmark_detector.py \
  --weights weights/yolov8n_mixed_img512_lightaug/weights/best.pt \
  --images dataset_position/images/val \
  --imgsz 512

Ubuntu GPU example:
python3 benchmark_detector.py \
  --weights weights/yolov8n_mixed_img512_lightaug/weights/best.pt \
  --images dataset_position/images/val \
  --imgsz 512 \
  --device 0
"""

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO detector inference time.")
    parser.add_argument("--weights", required=True, help="Detector weights path.")
    parser.add_argument(
        "--images",
        default="dataset_position/images/val",
        help="Image file or directory for benchmarking.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        help="Inference device, e.g. 0, cpu, or mps. Defaults to cuda -> mps -> cpu.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat rounds over all images.")
    parser.add_argument("--limit", type=int, help="Limit number of images.")
    parser.add_argument("--output", help="Optional CSV output path.")
    return parser.parse_args()


def select_device(device_arg):
    if device_arg:
        return device_arg

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

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
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found in: {path}")
    return images


def main():
    args = parse_args()
    device = select_device(args.device)
    image_paths = list_images(args.images)
    if args.limit:
        image_paths = image_paths[: args.limit]

    model = YOLO(args.weights)

    warmup_images = image_paths[: max(1, min(args.warmup, len(image_paths)))]
    for image_path in warmup_images:
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )

    timings = []
    detections = 0
    for _ in range(args.repeat):
        for image_path in image_paths:
            started = time.perf_counter()
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings.append(elapsed_ms)
            detections += len(results[0].boxes)

    total_images = len(image_paths) * args.repeat
    avg_ms = sum(timings) / len(timings)
    sorted_timings = sorted(timings)
    p50_ms = sorted_timings[len(sorted_timings) // 2]
    p95_ms = sorted_timings[int(len(sorted_timings) * 0.95) - 1]
    fps = 1000 / avg_ms if avg_ms > 0 else 0

    summary = {
        "weights": args.weights,
        "images": str(args.images),
        "imgsz": args.imgsz,
        "device": device,
        "image_count": len(image_paths),
        "repeat": args.repeat,
        "total_images": total_images,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
        "avg_detections": detections / total_images,
    }

    print("Detector benchmark")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)


if __name__ == "__main__":
    main()
