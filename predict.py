"""
SCRIPT_GROUP: _predict
PURPOSE: Predict bottle-position boxes with a YOLO detector.

Input:
  A raw bottle image or image directory.

Output:
  One YOLO-normalized bbox line per image:
  class_id x_center y_center width height

For water-level classification inference, use predict_level.py or
predict_level_torch.py instead.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Predict bottle bounding boxes.")
    parser.add_argument("source", help="Path to a .jpg image or image directory.")
    parser.add_argument(
        "--weights",
        default="weights/bottle_detector/weights/best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf)

    for result in results:
        if len(result.boxes) == 0:
            continue

        best_box = max(result.boxes, key=lambda box: float(box.conf[0].item()))
        class_id = int(best_box.cls[0].item())
        x_center, y_center, width, height = best_box.xywhn[0].tolist()
        print(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")


if __name__ == "__main__":
    main()
