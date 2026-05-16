"""
SCRIPT_GROUP: _data
PURPOSE: Legacy alias for building the cropped 1/2/3 classification dataset.

Prefer prepare_level_dataset.py for new runs. This file is kept only for
backward compatibility with earlier local commands.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEVEL_CLASSES = {"1", "2", "3"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 1/2/3 water-level classification dataset from YOLO labels."
    )
    parser.add_argument(
        "--source-root",
        default="./dataset",
        help="Source dataset root. Supports dataset/dataset*/images+labels or images/train+labels/train.",
    )
    parser.add_argument(
        "--target-root",
        default="./dataset_level",
        help="Output classification dataset root.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio when source data is not already split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Extra crop padding as a fraction of bbox size.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear target directory before writing.",
    )
    return parser.parse_args()


def yolo_to_xyxy(label_line, image_width, image_height, padding):
    parts = label_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO label must have 5 fields, got: {label_line!r}")

    class_id = parts[0]
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    if any(value < 0 or value > 1 for value in (x_center, y_center, width, height)):
        raise ValueError(f"YOLO coordinates outside 0-1: {label_line!r}")

    box_width = width * image_width
    box_height = height * image_height
    x1 = (x_center * image_width) - box_width / 2
    y1 = (y_center * image_height) - box_height / 2
    x2 = (x_center * image_width) + box_width / 2
    y2 = (y_center * image_height) + box_height / 2

    pad_x = box_width * padding
    pad_y = box_height * padding
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_width, int(round(x2 + pad_x)))
    y2 = min(image_height, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box from label: {label_line!r}")

    return class_id, x1, y1, x2, y2


def list_images(image_dir):
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_from_image_label_dirs(image_dir, label_dir, source_name, subset=None):
    samples = []
    for image_path in list_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Empty label file: {label_path}")
        if len(lines) > 1:
            raise ValueError(f"Expected one bottle per image, got {len(lines)} labels in {label_path}")

        class_id = lines[0].split()[0]
        if class_id not in {"0", "1", "2", "3"}:
            raise ValueError(f"Invalid class id in {label_path}: {class_id}")
        if class_id == "0":
            continue

        samples.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "label_line": lines[0],
                "class_id": class_id,
                "source_name": source_name,
                "subset": subset,
            }
        )
    return samples


def collect_samples(source_root):
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    samples = []

    split_image_root = source_root / "images"
    split_label_root = source_root / "labels"
    if split_image_root.exists() and split_label_root.exists():
        for subset in ("train", "val"):
            image_dir = split_image_root / subset
            label_dir = split_label_root / subset
            if image_dir.exists() and label_dir.exists():
                samples.extend(
                    collect_from_image_label_dirs(
                        image_dir=image_dir,
                        label_dir=label_dir,
                        source_name=source_root.name,
                        subset=subset,
                    )
                )
        if samples:
            return samples

    for dataset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        image_dir = dataset_dir / "images"
        label_dir = dataset_dir / "labels"
        if image_dir.exists() and label_dir.exists():
            samples.extend(
                collect_from_image_label_dirs(
                    image_dir=image_dir,
                    label_dir=label_dir,
                    source_name=dataset_dir.name,
                    subset=None,
                )
            )

    if not samples:
        raise RuntimeError(
            "No 1/2/3 samples found. Expected either images/train+labels/train "
            "or dataset*/images+labels under source root."
        )
    return samples


def split_samples(samples, train_ratio, seed):
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    if all(sample["subset"] in {"train", "val"} for sample in samples):
        return samples

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["class_id"]].append(sample)

    rng = random.Random(seed)
    split = []
    for class_id in sorted(grouped):
        class_samples = grouped[class_id][:]
        rng.shuffle(class_samples)

        split_idx = int(len(class_samples) * train_ratio)
        split_idx = max(1, min(split_idx, len(class_samples) - 1))
        for sample in class_samples[:split_idx]:
            sample["subset"] = "train"
            split.append(sample)
        for sample in class_samples[split_idx:]:
            sample["subset"] = "val"
            split.append(sample)

    rng.shuffle(split)
    return split


def clear_target(target_root):
    target_root = Path(target_root)
    if target_root.exists():
        shutil.rmtree(target_root)


def prepare_target_dirs(target_root):
    target_root = Path(target_root)
    for subset in ("train", "val"):
        for class_id in sorted(LEVEL_CLASSES):
            (target_root / subset / class_id).mkdir(parents=True, exist_ok=True)


def crop_sample(sample, target_root, padding):
    if cv2 is not None:
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            raise ValueError(f"Failed to read image: {sample['image_path']}")
        image_height, image_width = image.shape[:2]
    else:
        image = Image.open(sample["image_path"]).convert("RGB")
        image_width, image_height = image.size

    class_id, x1, y1, x2, y2 = yolo_to_xyxy(
        sample["label_line"],
        image_width=image_width,
        image_height=image_height,
        padding=padding,
    )
    if class_id not in LEVEL_CLASSES:
        raise ValueError(f"Expected class 1/2/3, got {class_id}: {sample['label_path']}")

    target_dir = Path(target_root) / sample["subset"] / class_id
    target_name = f"{sample['source_name']}_{sample['image_path'].stem}{sample['image_path'].suffix.lower()}"
    target_path = target_dir / target_name

    if cv2 is not None:
        cropped = image[y1:y2, x1:x2]
        if not cv2.imwrite(str(target_path), cropped):
            raise RuntimeError(f"Failed to write crop: {target_path}")
    else:
        cropped = image.crop((x1, y1, x2, y2))
        cropped.save(target_path)
    return target_path


def print_counts(samples):
    counts = Counter((sample["subset"], sample["class_id"]) for sample in samples)
    print("Water-level classification dataset ready.")
    for subset in ("train", "val"):
        total = sum(counts[(subset, class_id)] for class_id in sorted(LEVEL_CLASSES))
        print(f"{subset}: {total}")
        for class_id in sorted(LEVEL_CLASSES):
            print(f"  class {class_id}: {counts[(subset, class_id)]}")


def main():
    args = parse_args()
    target_root = Path(args.target_root)

    if args.clear:
        clear_target(target_root)
    prepare_target_dirs(target_root)

    samples = collect_samples(args.source_root)
    samples = split_samples(samples, train_ratio=args.train_ratio, seed=args.seed)

    for sample in samples:
        crop_sample(sample, target_root=target_root, padding=args.padding)

    print_counts(samples)
    print(f"Target root: {target_root}")


if __name__ == "__main__":
    main()
