"""Dataset loading, validation, and crop generation."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import REQUIRED_COLUMNS
from preprocess import crop_bbox, read_image, save_image


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """Load annotations.csv and validate required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in annotations.csv: {missing_cols}")

    if df.empty:
        raise ValueError("annotations.csv is empty. Add labeled bottle images first.")

    df["has_liquid"] = pd.to_numeric(df["has_liquid"], errors="coerce").fillna(-1).astype(int)
    invalid_labels = sorted(set(df["has_liquid"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"has_liquid must be 0 or 1, found: {invalid_labels}")

    return df


def resolve_image_path(image_root: Path, image_path: str) -> Path:
    """Resolve image_path from annotations.csv against image_root."""
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return Path(image_root) / path


def find_missing_images(df: pd.DataFrame, image_root: Path) -> List[Path]:
    """Return image paths listed in the CSV that do not exist on disk."""
    missing = []
    for _, row in df.iterrows():
        path = resolve_image_path(image_root, row["image_path"])
        if not path.exists():
            missing.append(path)
    return missing


def load_valid_crops(
    df: pd.DataFrame,
    image_root: Path,
    crop_dir: Path,
    save_crops: bool = True,
) -> Tuple[List[dict], List[str]]:
    """Load images, crop bottle bboxes, and optionally save crop files.

    Returns a list of sample dictionaries and a list of warning messages.
    Invalid images or invalid bboxes are skipped with warnings.
    """
    samples = []
    warnings = []
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_path = resolve_image_path(image_root, row["image_path"])
        image = read_image(image_path)
        if image is None:
            warnings.append(f"Skip {row['image_id']}: cannot read image {image_path}")
            continue

        bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        crop = crop_bbox(image, bbox)
        if crop is None:
            warnings.append(f"Skip {row['image_id']}: invalid or empty bbox {bbox}")
            continue

        crop_path = crop_dir / f"{row['image_id']}.jpg"
        if save_crops:
            ok = save_image(crop_path, crop)
            if not ok:
                warnings.append(f"Could not save crop for {row['image_id']} to {crop_path}")

        samples.append(
            {
                "image_id": row["image_id"],
                "image_path": image_path,
                "bbox": bbox,
                "label": int(row["has_liquid"]),
                "crop": crop,
                "crop_path": crop_path,
                "metadata": row.to_dict(),
            }
        )

    return samples, warnings
