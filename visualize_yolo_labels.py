"""
SCRIPT_GROUP: _data
PURPOSE: Render YOLO labels on images for annotation inspection.

Input:
  Image directory and matching YOLO label directory.

Output:
  Preview images with bounding boxes and class labels.

This script is for checking label quality before training.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO labels on sample images.")
    parser.add_argument("--image-dir", default="train/images", help="Image directory.")
    parser.add_argument("--label-dir", default="train/labels", help="YOLO label directory.")
    parser.add_argument("--output-dir", default="label_preview", help="Output directory.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to render. Use 0 for all.")
    parser.add_argument(
        "--class-names",
        default="",
        help="Comma-separated class names. Example: empty,full,half,three_quarters",
    )
    parser.add_argument(
        "--show-height",
        action="store_true",
        help="Write normalized YOLO box height on the image.",
    )
    parser.add_argument(
        "--show-id-only",
        action="store_true",
        help="Only write the class id instead of class id plus class name.",
    )
    parser.add_argument("--font-size", type=int, default=28, help="Label font size.")
    return parser.parse_args()


def yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    box_width = width * image_width
    box_height = height * image_height
    x1 = (x_center * image_width) - box_width / 2
    y1 = (y_center * image_height) - box_height / 2
    x2 = x1 + box_width
    y2 = y1 + box_height
    return x1, y1, x2, y2


def label_text(class_id, class_names, height, show_height, show_id_only):
    if show_id_only:
        text = class_id
    else:
        class_name = class_names.get(class_id, f"class {class_id}")
        text = f"{class_id} {class_name}"
    class_name = class_names.get(class_id, f"class {class_id}")
    if show_height:
        return f"{text} h={height:.3f}"
    return text


def load_font(font_size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_labels(image_path, label_path, output_path, class_names, show_height, show_id_only, font_size):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(font_size)
    image_width, image_height = image.size

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} must have 5 YOLO fields.")

        class_id = parts[0]
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height,
        )

        color = (255, 40, 40)
        line_width = max(2, round(min(image_width, image_height) / 180))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = label_text(class_id, class_names, height, show_height, show_id_only)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = max(0, x1)
        text_y = max(0, y1 - text_height - 6)
        pad_x = max(6, font_size // 3)
        pad_y = max(4, font_size // 5)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + pad_x * 2, text_y + text_height + pad_y * 2),
            fill=color,
        )
        draw.text((text_x + pad_x, text_y + pad_y), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    class_names = {
        str(index): name.strip()
        for index, name in enumerate(args.class_names.split(","))
        if name.strip()
    }

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    selected_paths = image_paths if args.count == 0 else image_paths[: args.count]
    for image_path in selected_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_dir / f"{image_path.stem}_boxed{image_path.suffix.lower()}"
        draw_labels(
            image_path,
            label_path,
            output_path,
            class_names,
            args.show_height,
            args.show_id_only,
            args.font_size,
        )

    print(f"Rendered {len(selected_paths)} preview images to {output_dir}")


if __name__ == "__main__":
    main()
