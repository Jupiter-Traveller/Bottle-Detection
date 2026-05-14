import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset_position" / "bottle_data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "weights"


AUGMENT_PRESETS = {
    "base": {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "light": {
        "degrees": 3.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a bottle-position YOLO model.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model or weights path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name under weights/. Example: yolov8n_img640_base",
    )
    parser.add_argument(
        "--augment-preset",
        choices=sorted(AUGMENT_PRESETS),
        default="base",
        help="Augmentation preset for comparison experiments.",
    )
    parser.add_argument(
        "--device",
        help="Training device. Defaults to mps on Apple Silicon when available, otherwise cpu.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing experiment directory.",
    )
    return parser.parse_args()


def count_images(image_dir):
    image_dir = Path(image_dir)
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(1 for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def validate_dataset():
    train_images = PROJECT_ROOT / "dataset_position" / "images" / "train"
    val_images = PROJECT_ROOT / "dataset_position" / "images" / "val"
    train_labels = PROJECT_ROOT / "dataset_position" / "labels" / "train"
    val_labels = PROJECT_ROOT / "dataset_position" / "labels" / "val"

    required_dirs = [train_images, val_images, train_labels, val_labels, DATA_YAML]
    missing_dirs = [str(path) for path in required_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing dataset directories: {missing_dirs}")

    train_count = count_images(train_images)
    val_count = count_images(val_images)
    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            "Position dataset is empty. Run python3 make_position_dataset.py before training."
        )

    print(f"Position dataset ready: {train_count} train images, {val_count} val images.")


def main():
    args = parse_args()
    validate_dataset()

    import torch
    from ultralytics import YOLO

    ava_device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {ava_device}")
    print(f"Experiment: {args.name}")

    model = YOLO(args.model)
    augment_kwargs = AUGMENT_PRESETS[args.augment_preset]
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,

        patience=30,
        workers=4,
        optimizer="AdamW",
        save=True,
        save_period=-1,

        device=ava_device,
        project=str(OUTPUT_DIR),
        name=args.name,
        exist_ok=args.exist_ok,
        **augment_kwargs,
    )


if __name__ == "__main__":
    main()
