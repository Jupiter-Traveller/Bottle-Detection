"""Visualization helpers for the pretest workflow."""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

from features_traditional import hough_liquid_level_features
from preprocess import read_image, resize_for_features


def _draw_rectangle(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    line_width = max(3, int(min(pil_image.size) * 0.004))
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=line_width)
    return np.asarray(pil_image)


def _draw_line(image: np.ndarray, line: tuple, color: tuple, width: int) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    draw.line(line, fill=color, width=width)
    return np.asarray(pil_image)


def plot_sample_bboxes(
    samples: List[dict],
    figure_dir: Path,
    max_samples: int = 6,
    random_state: int = 42,
) -> Optional[Path]:
    """Save a random panel of original image + bbox + crop."""
    if not samples:
        return None
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(samples, size=min(max_samples, len(samples)), replace=False)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for row_idx, sample in enumerate(chosen):
        original = read_image(sample["image_path"])
        bbox = sample["bbox"]
        crop = sample["crop"]
        label = sample["label"]

        original_vis = _draw_rectangle(original, bbox)
        axes[row_idx, 0].imshow(original_vis)
        axes[row_idx, 0].set_title(f"{sample['image_id']} bbox label={label}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(crop)
        axes[row_idx, 1].set_title("Crop")
        axes[row_idx, 1].axis("off")

    fig.tight_layout()
    out_path = figure_dir / "sample_bboxes_and_crops.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: np.ndarray, figure_dir: Path, model_name: str) -> Path:
    """Save a confusion matrix figure."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], ["empty", "has_liquid"])
    ax.set_yticks([0, 1], ["empty", "has_liquid"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = figure_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_all_confusion_matrices(metrics, figure_dir: Path) -> Path:
    """Save confusion matrices for all evaluated models in one figure."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4))
    if n_models == 1:
        axes = np.array([axes])

    for ax, (_, row) in zip(axes, metrics.iterrows()):
        cm = np.array(row["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(str(row["model"]))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], ["empty", "has_liquid"])
        ax.set_yticks([0, 1], ["empty", "has_liquid"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    out_path = figure_dir / "confusion_matrices_all_models.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_metric_comparison(metrics, figure_dir: Path) -> Path:
    """Save a grouped bar chart comparing model metrics."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    models = metrics["model"].astype(str).tolist()
    values = metrics[metric_cols].to_numpy(dtype=float)

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, metric in enumerate(metric_cols):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, values[:, idx], width=width, label=metric)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metric Comparison")
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figure_dir / "model_metric_comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_feature_2d(
    X: np.ndarray,
    y: np.ndarray,
    figure_dir: Path,
    title: str = "Feature PCA 2D",
) -> Optional[Path]:
    """Project features to two dimensions with PCA and save a scatter plot."""
    if X is None or len(X) < 2:
        return None
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if X.shape[1] >= 2:
        points = PCA(n_components=2, random_state=42).fit_transform(X)
        x_label, y_label = "PC1", "PC2"
    else:
        points = np.column_stack([X[:, 0], np.zeros(len(X))])
        x_label, y_label = "feature_0", "zero"

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=y, cmap="coolwarm", s=45, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["empty", "has_liquid"])
    fig.tight_layout()
    out_path = figure_dir / "feature_2d.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_hough_level_examples(
    samples: List[dict],
    output_dir: Path,
    max_samples: int = 8,
    random_state: int = 42,
) -> List[Path]:
    """Visualize candidate liquid-level lines for selected crops."""
    if not samples:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(samples, size=min(max_samples, len(samples)), replace=False)
    saved = []

    for sample in chosen:
        crop = sample["crop"]
        try:
            fixed = resize_for_features(crop, (128, 256))
            _, _, detail = hough_liquid_level_features(fixed)
            vis = fixed.copy()
            for item in detail.get("lines", [])[:30]:
                vis = _draw_line(vis, item["line"], (180, 180, 180), 1)
            best = detail.get("best_line")
            if best is not None:
                vis = _draw_line(vis, best["line"], (255, 0, 0), 3)

            out_path = output_dir / f"{sample['image_id']}_hough_level.png"
            fig, ax = plt.subplots(figsize=(4, 6))
            ax.imshow(vis)
            score = best["score"] if best is not None else 0.0
            ax.set_title(f"{sample['image_id']} score={score:.3f}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved.append(out_path)
        except Exception:
            continue

    return saved
