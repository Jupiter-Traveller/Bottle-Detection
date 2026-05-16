import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append({key.strip(): value for key, value in row.items()})
    return rows


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_row(rows, keys):
    for key in keys:
        if rows and key in rows[0]:
            return key, max(rows, key=lambda row: to_float(row.get(key)))
    return None, rows[-1] if rows else None


def epoch_of(row, rows):
    if row is None:
        return ""
    if "epoch" in row:
        return row["epoch"]
    return str(rows.index(row))


def summarize_detection(results_path):
    rows = read_rows(results_path)
    key, row = best_row(rows, ["metrics/mAP50-95(B)", "metrics/mAP50(B)"])
    if row is None:
        return None
    return {
        "task": "detect",
        "experiment": results_path.parent.name,
        "best_by": key,
        "best_epoch": epoch_of(row, rows),
        "precision": row.get("metrics/precision(B)", ""),
        "recall": row.get("metrics/recall(B)", ""),
        "mAP50": row.get("metrics/mAP50(B)", ""),
        "mAP50-95": row.get("metrics/mAP50-95(B)", ""),
    }


def summarize_classification(results_path):
    rows = read_rows(results_path)
    key, row = best_row(rows, ["metrics/accuracy_top1", "val_acc"])
    if row is None:
        return None

    if key == "metrics/accuracy_top1":
        return {
            "task": "cls",
            "experiment": results_path.parent.name,
            "best_by": key,
            "best_epoch": epoch_of(row, rows),
            "top1_acc": row.get("metrics/accuracy_top1", ""),
            "top5_acc": row.get("metrics/accuracy_top5", ""),
            "val_loss": row.get("val/loss", ""),
        }

    return {
        "task": "cls",
        "experiment": results_path.parent.name,
        "best_by": key,
        "best_epoch": epoch_of(row, rows),
        "val_acc": row.get("val_acc", ""),
        "train_acc": row.get("train_acc", ""),
        "val_loss": row.get("val_loss", ""),
    }


def print_table(title, rows, columns):
    print(title)
    if not rows:
        print("  (none)")
        return
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  " + "  ".join(column.ljust(widths[column]) for column in columns))
    print("  " + "  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  " + "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main():
    parser = argparse.ArgumentParser(description="Summarize YOLO and PyTorch experiment results.")
    parser.add_argument("--detect-root", default="weights", help="YOLO detection weights root.")
    parser.add_argument("--cls-root", default="weights_level", help="Classification weights root.")
    args = parser.parse_args()

    detect_rows = []
    for results_path in sorted(Path(args.detect_root).glob("*/results.csv")):
        summary = summarize_detection(results_path)
        if summary:
            detect_rows.append(summary)

    cls_rows = []
    for results_path in sorted(Path(args.cls_root).glob("*/results.csv")):
        summary = summarize_classification(results_path)
        if summary:
            cls_rows.append(summary)

    print_table(
        "Detection experiments",
        detect_rows,
        ["experiment", "best_epoch", "precision", "recall", "mAP50", "mAP50-95"],
    )
    print()
    print_table(
        "Classification experiments",
        cls_rows,
        ["experiment", "best_epoch", "top1_acc", "val_acc", "train_acc", "val_loss"],
    )


if __name__ == "__main__":
    main()
