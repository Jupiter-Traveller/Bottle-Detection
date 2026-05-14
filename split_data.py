import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VALID_CLASSES = {"0", "1", "2", "3"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split bottle dataset for YOLO training.")
    parser.add_argument(
        "--raw-img-dir",
        default="./bottle_dataset/images/train",
        help="Source image directory.",
    )
    parser.add_argument(
        "--raw-label-dir",
        default="./bottle_dataset/labels/train",
        help="Source label directory.",
    )
    parser.add_argument(
        "--source-name",
        default="bottle",
        help="Name prefix for the main source dataset.",
    )
    parser.add_argument(
        "--target-root",
        default="./dataset",
        help="Target YOLO dataset root.",
    )
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        metavar="NAME:IMAGE_DIR:LABEL_DIR",
        help=(
            "Additional source dataset. Can be used multiple times, for example "
            "train1:./train1/images:./train1/labels."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def list_images(raw_img_dir):
    return sorted(
        path for path in Path(raw_img_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_class(label_path):
    classes = set()
    with label_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_no} must have 5 YOLO fields.")

            class_id = parts[0]
            if class_id not in VALID_CLASSES:
                raise ValueError(f"{label_path}:{line_no} has invalid class id: {class_id}")

            coords = [float(value) for value in parts[1:]]
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError(f"{label_path}:{line_no} has coordinates outside 0-1.")

            classes.add(class_id)

    if not classes:
        raise ValueError(f"{label_path} is empty.")

    if len(classes) > 1:
        raise ValueError(f"{label_path} contains multiple classes: {sorted(classes)}")

    return next(iter(classes))


def validate_source(raw_img_dir, raw_label_dir, source_name):
    raw_img_dir = Path(raw_img_dir)
    raw_label_dir = Path(raw_label_dir)

    if not raw_img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {raw_img_dir}")
    if not raw_label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {raw_label_dir}")

    images = list_images(raw_img_dir)
    if not images:
        raise RuntimeError(f"No images found in {raw_img_dir}")

    image_stems = {path.stem for path in images}
    label_paths = sorted(path for path in raw_label_dir.glob("*.txt") if path.name != "classes.txt")
    label_stems = {path.stem for path in label_paths}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels:
        raise RuntimeError(f"Images missing labels: {missing_labels[:10]}")
    if missing_images:
        raise RuntimeError(f"Labels missing images: {missing_images[:10]}")

    samples_by_group = defaultdict(list)
    for image_path in images:
        label_path = raw_label_dir / f"{image_path.stem}.txt"
        class_id = label_class(label_path)
        samples_by_group[(source_name, class_id)].append(
            {
                "source": source_name,
                "class_id": class_id,
                "image_path": image_path,
                "label_path": label_path,
                "target_name": f"{source_name}_{image_path.name}",
            }
        )

    return samples_by_group


def parse_extra_source(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(
            "--extra-source must use NAME:IMAGE_DIR:LABEL_DIR, "
            f"got: {value}"
        )
    return tuple(parts)


def collect_sources(raw_img_dir, raw_label_dir, source_name, extra_sources):
    sources = [(source_name, raw_img_dir, raw_label_dir)]
    sources.extend(parse_extra_source(source) for source in extra_sources)

    samples_by_group = defaultdict(list)
    for source_name, image_dir, label_dir in sources:
        source_groups = validate_source(image_dir, label_dir, source_name)
        for key, samples in source_groups.items():
            samples_by_group[key].extend(samples)
    return samples_by_group


def clear_subset_dirs(target_root):
    labels_root = Path(target_root) / "labels"
    for cache_file in ("train.cache", "val.cache"):
        cache_path = labels_root / cache_file
        if cache_path.exists():
            cache_path.unlink()

    for subset in ("train", "val"):
        for kind in ("images", "labels"):
            directory = Path(target_root) / kind / subset
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.iterdir():
                if path.name == ".gitkeep":
                    continue
                if path.is_file():
                    path.unlink()


def split_by_group(samples_by_group, train_ratio, seed):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    train_files = []
    val_files = []

    for group in sorted(samples_by_group):
        samples = samples_by_group[group][:]
        rng.shuffle(samples)

        if len(samples) == 1:
            train_files.extend(samples)
            continue

        split_idx = int(len(samples) * train_ratio)
        split_idx = max(1, min(split_idx, len(samples) - 1))

        train_files.extend(samples[:split_idx])
        val_files.extend(samples[split_idx:])

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    return train_files, val_files


def copy_files(samples, subset, target_root):
    target_img_dir = Path(target_root) / "images" / subset
    target_label_dir = Path(target_root) / "labels" / subset

    for sample in samples:
        image_src = sample["image_path"]
        label_src = sample["label_path"]
        target_name = sample["target_name"]
        target_label_name = f"{Path(target_name).stem}.txt"

        shutil.copy2(image_src, target_img_dir / target_name)
        shutil.copy2(label_src, target_label_dir / target_label_name)


def split_dataset(
    raw_img_dir,
    raw_label_dir,
    target_root,
    source_name="bottle",
    train_ratio=0.8,
    seed=42,
    extra_sources=None,
):
    samples_by_group = collect_sources(raw_img_dir, raw_label_dir, source_name, extra_sources or [])
    train_files, val_files = split_by_group(samples_by_group, train_ratio, seed)

    clear_subset_dirs(target_root)
    copy_files(train_files, "train", target_root)
    copy_files(val_files, "val", target_root)

    print("Dataset split complete.")
    print(f"Source images: {len(train_files) + len(val_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print("Group counts:")
    for source_name, class_id in sorted(samples_by_group):
        print(f"  {source_name} class {class_id}: {len(samples_by_group[(source_name, class_id)])}")


def main():
    args = parse_args()
    split_dataset(
        raw_img_dir=args.raw_img_dir,
        raw_label_dir=args.raw_label_dir,
        target_root=args.target_root,
        source_name=args.source_name,
        train_ratio=args.train_ratio,
        seed=args.seed,
        extra_sources=args.extra_source,
    )


if __name__ == "__main__":
    main()
