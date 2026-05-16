"""
SCRIPT_GROUP: _data
PURPOSE: Auto-label bottle boxes with an existing YOLO detector.

Input:
  An image directory and trained bottle-detector weights.

Output:
  A YOLO-format image/label dataset with predicted bottle boxes.

Use this only as an auxiliary labeling tool; predicted labels still need manual
inspection before being treated as training data.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_img640_base" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label full bottle boxes with a trained YOLO model.")
    parser.add_argument("--image-dir", default="train/images", help="Source image directory.")
    parser.add_argument("--output-root", default="train_bottle_auto", help="Output YOLO dataset root.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Bottle detector weights.")
    parser.add_argument("--conf", type=float, default=0.15, help="Prediction confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.02, help="Box padding as a fraction of box size.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output-root/images/train.")
    return parser.parse_args()


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def padded_xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height, padding):
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding
    pad_y = box_height * padding

    x1 = clamp((x1 - pad_x) / image_width)
    y1 = clamp((y1 - pad_y) / image_height)
    x2 = clamp((x2 + pad_x) / image_width)
    y2 = clamp((y2 + pad_y) / image_height)

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return x_center, y_center, width, height


def write_yaml(output_root):
    yaml_text = """path: ./train_bottle_auto

train: images/train
val: images/train

names:
  0: bottle

task: detect
"""
    (output_root / "bottle_data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_root = Path(args.output_root)
    label_dir = output_root / "labels" / "train"
    output_image_dir = output_root / "images" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        output_image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    model = YOLO(args.weights)
    missing = []
    written = 0

    for image_path in image_paths:
        image = Image.open(image_path)
        image_width, image_height = image.size
        results = model.predict(source=str(image_path), conf=args.conf, verbose=False)
        result = results[0]
        if len(result.boxes) == 0:
            missing.append(image_path.name)
            continue

        best_box = max(result.boxes, key=lambda box: float(box.conf[0].item()))
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x_center, y_center, width, height = padded_xyxy_to_yolo(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
            args.padding,
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text(
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
            encoding="utf-8",
        )
        if args.copy_images:
            shutil.copy2(image_path, output_image_dir / image_path.name)
        written += 1

    write_yaml(output_root)

    if missing:
        missing_path = output_root / "missing_predictions.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing predictions: {len(missing)}. See {missing_path}")
    print(f"Wrote labels: {written}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
