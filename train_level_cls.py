"""
SCRIPT_GROUP: _train
PURPOSE: Train YOLOv8-cls for 1/2/3 water-level classification.

Input dataset:
  dataset_level/train/1|2|3
  dataset_level/val/1|2|3

Output:
  weights_level/<experiment_name>/

Examples:
python3 train_level_cls.py \
  --name yolov8n_cls_img224_base \
  --model yolov8n-cls.pt \
  --imgsz 224 \
  --epochs 80 \
  --batch 32 \
  --augment-preset base

python3 train_level_cls.py \
  --name yolov8n_cls_img224_color \
  --model yolov8n-cls.pt \
  --imgsz 224 \
  --epochs 80 \
  --batch 32 \
  --augment-preset color
"""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset_level"
OUTPUT_DIR = PROJECT_ROOT / "weights_level"


AUGMENT_PRESETS = {
    "base": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
    },
    "color": {
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.25,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
    },
    "light": {
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.25,
        "degrees": 3.0,
        "translate": 0.04,
        "scale": 0.15,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1/2/3 water-level classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_ROOT), help="Classification dataset root.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="YOLO classification model.")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name under weights_level/. Example: yolov8n_cls_level_img224_base",
    )
    parser.add_argument(
        "--augment-preset",
        choices=sorted(AUGMENT_PRESETS),
        default="base",
        help="Augmentation preset.",
    )
    parser.add_argument(
        "--device",
        help="Training device. Defaults to cuda when available, then mps, then cpu.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing experiment directory.",
    )
    parser.add_argument(
        "--resume",
        help="Resume from a YOLO checkpoint, for example weights_level/name/weights/last.pt.",
    )
    return parser.parse_args()


def count_images(directory):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for path in Path(directory).iterdir() if path.suffix.lower() in suffixes)


def validate_dataset(data_root):
    data_root = Path(data_root)
    missing = []
    counts = {}
    for subset in ("train", "val"):
        for class_id in ("1", "2", "3"):
            class_dir = data_root / subset / class_id
            if not class_dir.exists():
                missing.append(str(class_dir))
                continue
            counts[(subset, class_id)] = count_images(class_dir)

    if missing:
        raise FileNotFoundError(f"Missing classification dataset directories: {missing}")

    empty = [f"{subset}/{class_id}" for (subset, class_id), count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(f"Empty classification dataset classes: {empty}")

    print("Water-level classification dataset ready:")
    for subset in ("train", "val"):
        total = sum(counts[(subset, class_id)] for class_id in ("1", "2", "3"))
        detail = ", ".join(f"{class_id}={counts[(subset, class_id)]}" for class_id in ("1", "2", "3"))
        print(f"  {subset}: {total} ({detail})")


def select_device(device_arg):
    if device_arg:
        return device_arg

    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    validate_dataset(args.data)
    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Experiment: {args.name}")

    model = YOLO(args.model)
    augment_kwargs = AUGMENT_PRESETS[args.augment_preset]
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=4,
        optimizer="AdamW",
        save=True,
        save_period=-1,
        device=device,
        project=str(OUTPUT_DIR),
        name=args.name,
        exist_ok=args.exist_ok,
        **augment_kwargs,
    )


if __name__ == "__main__":
    main()
