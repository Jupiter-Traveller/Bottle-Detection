"""Train and evaluate qualitative liquid-level classifiers from YOLO labels."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import OUTPUT_DIR
from features_traditional import extract_traditional_features
from preprocess import crop_bbox, read_image, save_image


DEFAULT_CLASS_NAMES = {
    0: "empty",
    1: "low",
    2: "medium",
    3: "high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test qualitative liquid-level recognition from YOLO labels."
    )
    parser.add_argument("--train_image_dir", default="../../train/images")
    parser.add_argument("--train_label_dir", default="../../train/labels")
    parser.add_argument("--test_image_dir", default="../../test/test_images")
    parser.add_argument("--test_label_dir", default="../../test/test_label")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--random_state", type=int, default=2026)
    parser.add_argument(
        "--combined_cv_splits",
        type=int,
        default=5,
        help="Optional stratified train/test resampling on train+test for domain-mix diagnostics.",
    )
    parser.add_argument(
        "--class_names",
        default="empty,low,medium,high",
        help="Comma-separated names ordered by class id.",
    )
    return parser.parse_args()


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    xmin = max(0.0, (x_center - width / 2.0) * image_w)
    ymin = max(0.0, (y_center - height / 2.0) * image_h)
    xmax = min(float(image_w), (x_center + width / 2.0) * image_w)
    ymax = min(float(image_h), (y_center + height / 2.0) * image_h)
    return xmin, ymin, xmax, ymax


def read_yolo_label(label_path: Path) -> Tuple[int, float, float, float, float]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label_path} should contain exactly one label line, found {len(lines)}")
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"{label_path} should have 5 YOLO fields, found {len(parts)}")
    class_id = int(parts[0])
    x_center, y_center, width, height = [float(value) for value in parts[1:]]
    return class_id, x_center, y_center, width, height


def find_image_for_label(image_dir: Path, stem: str) -> Optional[Path]:
    matches = sorted(image_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    return None


def load_yolo_samples(
    split: str,
    image_dir: Path,
    label_dir: Path,
    crop_dir: Path,
    save_crops: bool = True,
) -> Tuple[List[dict], List[str]]:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    warnings = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image_for_label(image_dir, label_path.stem)
        if image_path is None:
            warnings.append(f"{split}: missing image for {label_path.name}")
            continue

        try:
            class_id, x_center, y_center, box_w, box_h = read_yolo_label(label_path)
        except Exception as exc:
            warnings.append(f"{split}: skip {label_path.name}: {exc}")
            continue

        try:
            with Image.open(image_path) as image:
                image_w, image_h = image.size
        except Exception as exc:
            warnings.append(f"{split}: cannot open {image_path.name}: {exc}")
            continue

        bbox = yolo_to_xyxy(x_center, y_center, box_w, box_h, image_w, image_h)
        image = read_image(image_path)
        if image is None:
            warnings.append(f"{split}: cannot read {image_path.name}")
            continue
        crop = crop_bbox(image, bbox)
        if crop is None:
            warnings.append(f"{split}: invalid bbox for {image_path.name}: {bbox}")
            continue

        crop_path = crop_dir / f"{label_path.stem}.jpg"
        if save_crops and not save_image(crop_path, crop):
            warnings.append(f"{split}: cannot save crop for {image_path.name}")

        samples.append(
            {
                "split": split,
                "image_id": label_path.stem,
                "image_path": image_path,
                "label_path": label_path,
                "bbox": bbox,
                "crop": crop,
                "crop_path": crop_path,
                "class_id": int(class_id),
                "yolo_x_center": x_center,
                "yolo_y_center": y_center,
                "yolo_width": box_w,
                "yolo_height": box_h,
            }
        )

    return samples, warnings


def extract_feature_table(samples: List[dict]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    rows = []
    vectors = []
    feature_names: List[str] = []
    for sample in samples:
        vector, names = extract_traditional_features(sample["crop"])
        feature_names = names
        vectors.append(vector)
        rows.append(
            {
                "split": sample["split"],
                "image_id": sample["image_id"],
                "image_path": str(sample["image_path"]),
                "label_path": str(sample["label_path"]),
                "class_id": sample["class_id"],
                "xmin": sample["bbox"][0],
                "ymin": sample["bbox"][1],
                "xmax": sample["bbox"][2],
                "ymax": sample["bbox"][3],
                "crop_path": str(sample["crop_path"]),
                "yolo_x_center": sample["yolo_x_center"],
                "yolo_y_center": sample["yolo_y_center"],
                "yolo_width": sample["yolo_width"],
                "yolo_height": sample["yolo_height"],
            }
        )

    if not vectors:
        raise ValueError("No valid samples available for feature extraction.")
    X = np.vstack(vectors).astype(np.float32)
    feature_df = pd.DataFrame(X, columns=feature_names)
    table = pd.concat([pd.DataFrame(rows), feature_df], axis=1)
    return X, feature_names, table


def make_models(random_state: int) -> Dict[str, object]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def class_count_text(y: Iterable[int], class_names: Dict[int, str]) -> str:
    counts = pd.Series(list(y)).value_counts().sort_index()
    return ", ".join(f"{cid}:{class_names.get(int(cid), str(cid))}={int(count)}" for cid, count in counts.items())


def plot_confusion(
    cm: np.ndarray,
    class_ids: List[int],
    class_names: Dict[int, str],
    out_path: Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cm, cmap="Blues")
    tick_labels = [class_names.get(cid, str(cid)) for cid in class_ids]
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_ids)), tick_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(class_ids)), tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_metric_comparison(metrics: pd.DataFrame, out_path: Path) -> Path:
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    models = metrics["model"].astype(str).tolist()
    values = metrics[metric_cols].to_numpy(dtype=float)
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, metric in enumerate(metric_cols):
        ax.bar(x + (idx - 1.5) * width, values[:, idx], width=width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Liquid-Level Model Comparison")
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def run_combined_cv_diagnostic(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_splits: int,
) -> pd.DataFrame:
    if n_splits <= 0:
        return pd.DataFrame()
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        return pd.DataFrame()

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=random_state)
    rows = []
    for model_name in make_models(random_state):
        acc = []
        balanced_acc = []
        macro_f1 = []
        weighted_f1 = []
        for train_idx, test_idx in splitter.split(X, y):
            model = make_models(random_state)[model_name]
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            acc.append(accuracy_score(y[test_idx], pred))
            balanced_acc.append(balanced_accuracy_score(y[test_idx], pred))
            macro_f1.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
            weighted_f1.append(f1_score(y[test_idx], pred, average="weighted", zero_division=0))
        rows.append(
            {
                "model": model_name,
                "splits": n_splits,
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_std": float(np.std(acc)),
                "balanced_accuracy_mean": float(np.mean(balanced_acc)),
                "balanced_accuracy_std": float(np.std(balanced_acc)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1)),
                "weighted_f1_mean": float(np.mean(weighted_f1)),
                "weighted_f1_std": float(np.std(weighted_f1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False).reset_index(drop=True)


def draw_prediction_panel(samples: List[dict], pred_df: pd.DataFrame, out_path: Path, max_rows: int = 18) -> Optional[Path]:
    errors = pred_df[pred_df["class_id"] != pred_df["pred_class_id"]]
    chosen_ids = errors["image_id"].head(max_rows).tolist()
    if not chosen_ids:
        chosen_ids = pred_df["image_id"].head(max_rows).tolist()
    chosen = [sample for sample in samples if sample["image_id"] in set(chosen_ids)]
    if not chosen:
        return None

    thumbs = []
    for sample in chosen[:max_rows]:
        pred_row = pred_df[pred_df["image_id"] == sample["image_id"]].iloc[0]
        image = read_image(sample["image_path"])
        if image is None:
            continue
        pil = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = [int(round(value)) for value in sample["bbox"]]
        color = (0, 180, 0) if pred_row["class_id"] == pred_row["pred_class_id"] else (230, 40, 40)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, int(min(pil.size) * 0.005)))
        title = f'{sample["image_id"]} true={pred_row["class_name"]} pred={pred_row["pred_class_name"]}'
        draw.rectangle([0, 0, min(pil.width, 620), 34], fill=(255, 255, 255))
        draw.text((6, 8), title, fill=(0, 0, 0))
        pil.thumbnail((250, 230))
        canvas = Image.new("RGB", (250, 255), "white")
        canvas.paste(pil, ((250 - pil.width) // 2, 24 + (230 - pil.height) // 2))
        ImageDraw.Draw(canvas).text((5, 5), title[:42], fill=(0, 0, 0))
        thumbs.append(canvas)

    if not thumbs:
        return None
    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 250, rows * 255), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 250, (idx // cols) * 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path


def safe_run_name(run_name: Optional[str]) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("level_%Y%m%d_%H%M%S")
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_name))
    return clean or datetime.now().strftime("level_%Y%m%d_%H%M%S")


def unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    for idx in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{idx:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise FileExistsError(f"Could not create unique output directory for {base}")


def write_class_distribution(path: Path, y_train: np.ndarray, y_test: np.ndarray, class_names: Dict[int, str]) -> None:
    class_ids = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "train_count", "test_count"])
        writer.writeheader()
        for cid in class_ids:
            writer.writerow(
                {
                    "class_id": cid,
                    "class_name": class_names.get(cid, str(cid)),
                    "train_count": int(np.sum(y_train == cid)),
                    "test_count": int(np.sum(y_test == cid)),
                }
            )


def run_level_experiment(
    train_image_dir: Path,
    train_label_dir: Path,
    test_image_dir: Path,
    test_label_dir: Path,
    output_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    random_state: int = 2026,
    class_names: Optional[Dict[int, str]] = None,
    combined_cv_splits: int = 5,
) -> Dict[str, object]:
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    root = Path(output_root) if output_root is not None else OUTPUT_DIR / "level_runs"
    run_dir = unique_dir(root / safe_run_name(run_name))
    dirs = {
        "run": run_dir,
        "crops_train": run_dir / "crops" / "train",
        "crops_test": run_dir / "crops" / "test",
        "features": run_dir / "features",
        "figures": run_dir / "figures",
        "models": run_dir / "models",
        "reports": run_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    train_samples, train_warnings = load_yolo_samples(
        "train", train_image_dir, train_label_dir, dirs["crops_train"]
    )
    test_samples, test_warnings = load_yolo_samples("test", test_image_dir, test_label_dir, dirs["crops_test"])
    if not train_samples:
        raise ValueError("No valid training samples were loaded.")
    if not test_samples:
        raise ValueError("No valid test samples were loaded.")

    X_train, feature_names, train_features = extract_feature_table(train_samples)
    X_test, _, test_features = extract_feature_table(test_samples)
    y_train = np.array([sample["class_id"] for sample in train_samples], dtype=int)
    y_test = np.array([sample["class_id"] for sample in test_samples], dtype=int)
    class_ids = sorted(set(y_train.tolist()) | set(y_test.tolist()))

    train_features.to_csv(dirs["features"] / "train_features_traditional.csv", index=False, encoding="utf-8-sig")
    test_features.to_csv(dirs["features"] / "test_features_traditional.csv", index=False, encoding="utf-8-sig")
    write_class_distribution(dirs["reports"] / "class_distribution.csv", y_train, y_test, class_names)

    rows = []
    predictions = {}
    trained = {}
    for model_name, model in make_models(random_state).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=class_ids)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
                "confusion_matrix": cm.tolist(),
            }
        )
        predictions[model_name] = pred
        trained[model_name] = model
        plot_confusion(
            cm,
            class_ids,
            class_names,
            dirs["figures"] / f"confusion_matrix_{model_name}.png",
            f"Confusion Matrix - {model_name}",
        )

    metrics = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_pred = predictions[best_name]
    metrics.to_csv(dirs["reports"] / "metrics.csv", index=False, encoding="utf-8-sig")
    plot_metric_comparison(metrics, dirs["figures"] / "model_metric_comparison.png")

    combined_cv = run_combined_cv_diagnostic(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        random_state=random_state,
        n_splits=combined_cv_splits,
    )
    if not combined_cv.empty:
        combined_cv.to_csv(dirs["reports"] / "combined_cv_metrics.csv", index=False, encoding="utf-8-sig")

    target_names = [class_names.get(cid, str(cid)) for cid in class_ids]
    report = classification_report(
        y_test,
        best_pred,
        labels=class_ids,
        target_names=target_names,
        zero_division=0,
    )
    (dirs["reports"] / "classification_report.txt").write_text(report, encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "image_id": [sample["image_id"] for sample in test_samples],
            "image_path": [str(sample["image_path"]) for sample in test_samples],
            "class_id": y_test,
            "class_name": [class_names.get(int(value), str(value)) for value in y_test],
            "pred_class_id": best_pred,
            "pred_class_name": [class_names.get(int(value), str(value)) for value in best_pred],
            "is_correct": y_test == best_pred,
        }
    )
    pred_df.to_csv(dirs["reports"] / "predictions.csv", index=False, encoding="utf-8-sig")
    draw_prediction_panel(test_samples, pred_df, dirs["figures"] / "prediction_examples.jpg")

    joblib.dump(
        {
            "model": trained[best_name],
            "model_name": best_name,
            "feature_mode": "traditional",
            "feature_names": feature_names,
            "class_names": class_names,
            "class_ids": class_ids,
        },
        dirs["models"] / "best_level_model.joblib",
    )
    joblib.dump(
        {
            "models": trained,
            "best_model_name": best_name,
            "feature_mode": "traditional",
            "feature_names": feature_names,
            "class_names": class_names,
            "class_ids": class_ids,
        },
        dirs["models"] / "all_level_models.joblib",
    )

    warnings = train_warnings + test_warnings
    summary_lines = [
        "Qualitative liquid-level experiment summary",
        f"run_dir: {run_dir}",
        f"train_image_dir: {Path(train_image_dir)}",
        f"test_image_dir: {Path(test_image_dir)}",
        f"train_samples: {len(train_samples)} ({class_count_text(y_train, class_names)})",
        f"test_samples: {len(test_samples)} ({class_count_text(y_test, class_names)})",
        f"feature_mode: traditional",
        f"feature_dim: {X_train.shape[1]}",
        f"models: {', '.join(make_models(random_state).keys())}",
        f"best_model: {best_name}",
        f"best_accuracy: {float(metrics.iloc[0]['accuracy']):.6f}",
        f"best_macro_f1: {float(metrics.iloc[0]['macro_f1']):.6f}",
        "class_mapping: " + ", ".join(f"{cid}={class_names.get(cid, str(cid))}" for cid in class_ids),
    ]
    if not combined_cv.empty:
        summary_lines.extend(
            [
                f"combined_cv_splits: {combined_cv_splits}",
                f"combined_cv_best_model: {combined_cv.iloc[0]['model']}",
                f"combined_cv_best_macro_f1_mean: {float(combined_cv.iloc[0]['macro_f1_mean']):.6f}",
                f"combined_cv_best_accuracy_mean: {float(combined_cv.iloc[0]['accuracy_mean']):.6f}",
            ]
        )
    if int(np.min(np.bincount(y_train, minlength=max(class_ids) + 1)[class_ids])) < 5:
        summary_lines.append(
            "warning: at least one training class has fewer than 5 samples; results may be unstable."
        )
    if warnings:
        summary_lines.append("")
        summary_lines.append("Warnings:")
        summary_lines.extend(f"- {warning}" for warning in warnings)
    (dirs["reports"] / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "best_model": best_name,
        "summary_path": dirs["reports"] / "run_summary.txt",
        "metrics_path": dirs["reports"] / "metrics.csv",
        "predictions_path": dirs["reports"] / "predictions.csv",
    }


def main() -> int:
    args = parse_args()
    names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    class_names = {idx: name for idx, name in enumerate(names)}
    result = run_level_experiment(
        train_image_dir=Path(args.train_image_dir),
        train_label_dir=Path(args.train_label_dir),
        test_image_dir=Path(args.test_image_dir),
        test_label_dir=Path(args.test_label_dir),
        output_root=Path(args.output_root) if args.output_root else None,
        run_name=args.run_name,
        random_state=args.random_state,
        class_names=class_names,
        combined_cv_splits=args.combined_cv_splits,
    )
    print(f"run_dir: {result['run_dir']}")
    print(f"best_model: {result['best_model']}")
    print(f"metrics_path: {result['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
