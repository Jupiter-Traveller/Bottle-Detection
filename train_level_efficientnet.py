"""
Train EfficientNet-B0 for water-level classification.

Example:
python3 train_level_efficientnet.py \
  --name efficientnet_b0_level_img224_base \
  --data dataset_level \
  --imgsz 224 \
  --epochs 80 \
  --batch 32

Try batch 64 if GPU memory is enough:
python3 train_level_efficientnet.py \
  --name efficientnet_b0_level_img224_b64 \
  --data dataset_level \
  --imgsz 224 \
  --epochs 80 \
  --batch 64
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset_level"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "weights_level"


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on dataset_level.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_ROOT), help="Classification dataset root.")
    parser.add_argument("--name", required=True, help="Experiment name under weights_level/.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", help="cuda, cpu, or cuda:0. Defaults to auto.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train without ImageNet weights.")
    parser.add_argument("--light-aug", action="store_true", help="Use light train-time augmentation.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into existing output dir.")
    return parser.parse_args()


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(imgsz, light_aug):
    train_ops = [transforms.Resize((imgsz, imgsz))]
    if light_aug:
        train_ops.extend(
            [
                transforms.RandomAffine(degrees=3, translate=(0.04, 0.04), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_ops = [
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_dataloaders(data_root, imgsz, batch, workers, light_aug):
    data_root = Path(data_root)
    train_tfms, eval_tfms = build_transforms(imgsz, light_aug)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_tfms)
    if train_dataset.classes != ["1", "2", "3"] or val_dataset.classes != ["1", "2", "3"]:
        raise RuntimeError(f"Expected classes ['1', '2', '3'], got {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes, pretrained):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_seen += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return {
        "loss": total_loss / total_seen,
        "acc": total_correct / total_seen,
        "targets": all_targets,
        "preds": all_preds,
    }


def save_checkpoint(path, model, optimizer, epoch, best_acc, class_to_idx, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "model_name": "efficientnet_b0",
        },
        path,
    )


def plot_confusion_matrix(targets, preds, class_names, output_path):
    matrix = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    output_dir = DEFAULT_OUTPUT_ROOT / args.name
    if output_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Output dir exists: {output_dir}. Use --exist-ok to overwrite files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_root=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        light_aug=args.light_aug,
    )

    model = build_model(num_classes=3, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    results_path = output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)}, val images: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Output: {output_dir}")

    best_acc = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_acc": f"{train_metrics['acc']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_acc": f"{val_metrics['acc']:.6f}",
        }
        with results_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_acc, train_dataset.class_to_idx, args)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            epochs_without_improve = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_acc, train_dataset.class_to_idx, args)
            plot_confusion_matrix(
                val_metrics["targets"],
                val_metrics["preds"],
                train_dataset.classes,
                output_dir / "confusion_matrix.png",
            )
            report = classification_report(
                val_metrics["targets"],
                val_metrics["preds"],
                target_names=train_dataset.classes,
                digits=4,
            )
            (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping: no val_acc improvement in {args.patience} epochs.")
            break

    minutes = (time.time() - started) / 60
    print(f"Best val_acc={best_acc:.4f} at epoch {best_epoch}. Finished in {minutes:.2f} min.")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
