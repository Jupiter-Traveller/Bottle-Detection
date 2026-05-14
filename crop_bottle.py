import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_mixed_img640_base" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cropped"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONF = 0.001


def load_model(weights=DEFAULT_WEIGHTS):
    return YOLO(str(weights))


def crop_bottle(image_path, output_path=None, model=None, weights=DEFAULT_WEIGHTS, conf=0.25, padding=0):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_model(weights)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    result = results[0]
    if len(result.boxes) == 0:
        raise RuntimeError(f"No bottle detected in image: {image_path}")

    best_box = max(result.boxes, key=lambda box: float(box.conf[0].item()))
    x1, y1, x2, y2 = [int(round(value)) for value in best_box.xyxy[0].tolist()]

    image_height, image_width = image.shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_width, x2 + pad_x)
    y2 = min(image_height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop box for image: {image_path}")

    cropped = image[y1:y2, x1:x2]

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{image_path.stem}_crop{image_path.suffix}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), cropped):
        raise RuntimeError(f"Failed to write cropped image: {output_path}")

    return output_path


def list_images(input_path, limit=None):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_paths = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def crop_images(input_path, output_dir=DEFAULT_OUTPUT_DIR, weights=DEFAULT_WEIGHTS, limit=None, conf=0.25, padding=0):
    image_paths = list_images(input_path, limit=limit)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(weights)

    cropped_paths = []
    failed = []
    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}_crop{image_path.suffix}"
        try:
            cropped_paths.append(
                crop_bottle(
                    image_path=image_path,
                    output_path=output_path,
                    model=model,
                    conf=conf,
                    padding=padding,
                )
            )
        except Exception as exc:
            failed.append((image_path, exc))

    return cropped_paths, failed


def parse_args():
    parser = argparse.ArgumentParser(description="Crop detected bottles from an image or image directory.")
    parser.add_argument("input", help="Input image path or image directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Defaults to cropped/.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images to crop. Defaults to all images.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--no-conf",
        action="store_true",
        help="Disable practical confidence filtering by using a very low threshold.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0,
        help="Extra crop padding as a fraction of detected box size.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conf = MIN_CONF if args.no_conf else args.conf
    cropped_paths, failed = crop_images(
        input_path=args.input,
        output_dir=args.output_dir,
        weights=args.weights,
        limit=args.limit,
        conf=0,
        padding=args.padding,
    )
    print(f"Cropped images: {len(cropped_paths)}")
    print(f"Output directory: {Path(args.output_dir)}")
    if failed:
        print(f"Failed images: {len(failed)}")
        for image_path, exc in failed[:20]:
            print(f"  {image_path}: {exc}")


if __name__ == "__main__":
    main()
