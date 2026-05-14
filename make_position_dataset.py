import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a single-class bottle dataset.")
    parser.add_argument("--source-root", default="./dataset", help="Source YOLO dataset root.")
    parser.add_argument(
        "--target-root",
        default="./dataset_position",
        help="Target single-class YOLO dataset root.",
    )
    return parser.parse_args()


def clear_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file():
            path.unlink()


def convert_label(source_label, target_label):
    converted_lines = []
    with source_label.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{source_label}:{line_no} must have 5 YOLO fields.")

            coords = [float(value) for value in parts[1:]]
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError(f"{source_label}:{line_no} has coordinates outside 0-1.")

            converted_lines.append(f"0 {' '.join(parts[1:])}")

    if not converted_lines:
        raise ValueError(f"{source_label} is empty.")

    target_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")


def convert_subset(source_root, target_root, subset):
    source_image_dir = source_root / "images" / subset
    source_label_dir = source_root / "labels" / subset
    target_image_dir = target_root / "images" / subset
    target_label_dir = target_root / "labels" / subset

    if not source_image_dir.exists() or not source_label_dir.exists():
        raise FileNotFoundError(f"Missing source subset directories for {subset}.")

    clear_directory(target_image_dir)
    clear_directory(target_label_dir)

    image_paths = sorted(
        path for path in source_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No source images found for {subset}.")

    for image_path in image_paths:
        source_label = source_label_dir / f"{image_path.stem}.txt"
        if not source_label.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {source_label}")

        shutil.copy2(image_path, target_image_dir / image_path.name)
        convert_label(source_label, target_label_dir / source_label.name)

    return len(image_paths)


def write_yaml(target_root):
    yaml_text = """path: ./dataset_position

train: images/train
val: images/val

names:
  0: bottle

task: detect
"""
    (target_root / "bottle_data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    train_count = convert_subset(source_root, target_root, "train")
    val_count = convert_subset(source_root, target_root, "val")
    write_yaml(target_root)

    print("Single-class position dataset ready.")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Target root: {target_root}")


if __name__ == "__main__":
    main()
