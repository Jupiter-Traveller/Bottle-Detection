"""Training and evaluation workflow for the bottle liquid pretest."""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import OUTPUT_DIR, ensure_output_dirs
from dataset import find_missing_images, load_annotations, load_valid_crops
from features_deep import extract_deep_feature_matrix, is_torch_available
from features_traditional import extract_traditional_features
from visualize import (
    plot_all_confusion_matrices,
    plot_confusion_matrix,
    plot_feature_2d,
    plot_model_metric_comparison,
    plot_sample_bboxes,
    save_hough_level_examples,
)


def _bool_can_stratify(y: np.ndarray, test_size: float) -> bool:
    """Check whether train_test_split can use stratification safely."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        return False
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    return n_test >= len(classes) and n_train >= len(classes)


def _build_feature_matrix(
    samples: List[dict],
    feature_mode: str,
    deep_model: str = "resnet18",
) -> Tuple[np.ndarray, List[str], str]:
    """Extract features according to feature_mode."""
    crops = [sample["crop"] for sample in samples]
    status_messages = []

    X_trad = None
    trad_names = []
    if feature_mode in {"traditional", "fused"}:
        rows = []
        for crop in crops:
            vector, names = extract_traditional_features(crop)
            rows.append(vector)
            trad_names = names
        X_trad = np.vstack(rows).astype(np.float32)
        status_messages.append(f"Traditional features: {X_trad.shape[1]} dims.")

    X_deep = None
    deep_names = []
    if feature_mode in {"deep", "fused"}:
        if not is_torch_available():
            msg = "torch/torchvision not available; deep features skipped."
            if feature_mode == "deep":
                raise RuntimeError(msg + " Use --feature_mode traditional or install torch/torchvision.")
            status_messages.append(msg)
        else:
            try:
                X_deep, deep_names, msg = extract_deep_feature_matrix(crops, model_name=deep_model)
                status_messages.append(msg)
            except Exception as exc:
                msg = f"Deep feature extraction failed and was skipped: {exc}"
                if feature_mode == "deep":
                    raise RuntimeError(msg)
                status_messages.append(msg)

    if feature_mode == "traditional":
        return X_trad, trad_names, " ".join(status_messages)
    if feature_mode == "deep":
        return X_deep, deep_names, " ".join(status_messages)
    if feature_mode == "fused":
        if X_deep is None:
            return X_trad, trad_names, " ".join(status_messages) + " Fused mode fell back to traditional only."
        return np.hstack([X_trad, X_deep]).astype(np.float32), trad_names + deep_names, " ".join(status_messages)
    raise ValueError("feature_mode must be traditional, deep, or fused.")


def _save_feature_csv(
    samples: List[dict],
    X: np.ndarray,
    feature_names: List[str],
    feature_mode: str,
    feature_dir: Path,
) -> Path:
    """Save feature matrix with identifiers and labels."""
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(
        {
            "image_id": [sample["image_id"] for sample in samples],
            "has_liquid": [sample["label"] for sample in samples],
            "image_path": [str(sample["image_path"]) for sample in samples],
        }
    )
    feature_df = pd.DataFrame(X, columns=feature_names)
    out_df = pd.concat([base, feature_df], axis=1)
    out_path = feature_dir / f"features_{feature_mode}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def _safe_run_name(name: str) -> str:
    """Return a filesystem-friendly run name."""
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    return clean or datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_unique_run_dir(base_dir: Path) -> Path:
    """Create a unique run directory without overwriting earlier experiments."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    for idx in range(2, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}_{idx:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise FileExistsError(f"Could not create a unique run directory for {base_dir}")


def _build_run_dirs(run_name: Optional[str], output_root: Optional[Path]) -> Dict[str, Path]:
    """Build per-run output directories."""
    root = Path(output_root) if output_root is not None else OUTPUT_DIR / "runs"
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _make_unique_run_dir(root / _safe_run_name(run_name))
    dirs = {
        "run": run_dir,
        "crops": run_dir / "crops",
        "features": run_dir / "features",
        "figures": run_dir / "figures",
        "models": run_dir / "models",
        "reports": run_dir / "reports",
        "level_lines": run_dir / "figures" / "level_line_examples",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_to_csv(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a CSV, falling back to a timestamped file if the target is locked."""
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    except PermissionError:
        stamped = out_path.with_name(f"{out_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{out_path.suffix}")
        df.to_csv(stamped, index=False, encoding="utf-8-sig")
        return stamped


def _candidate_models(random_state: int) -> Dict[str, object]:
    """Return model candidates for the pretest."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    model_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, object, Dict[str, object], StandardScaler, np.ndarray, np.ndarray]:
    """Train/evaluate candidate classifiers and return metrics and best model."""
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes in has_liquid to train classifiers.")
    if len(y) < 4:
        raise ValueError("Need at least 4 valid samples for a train/test split.")

    stratify = y if _bool_can_stratify(y, test_size) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    rows = []
    trained_models = {}
    best_name = None
    best_f1 = -1.0

    classes, counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(counts))
    do_cv = len(y) >= 10 and min_class_count >= 5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if do_cv else None

    candidates = _candidate_models(random_state)
    if model_names is not None:
        candidates = {name: candidates[name] for name in model_names}
    if not candidates:
        raise ValueError("No classifier was selected.")

    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        f1 = f1_score(y_test, pred, zero_division=0)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "cv_f1_mean": np.nan,
            "cv_f1_std": np.nan,
        }
        if do_cv:
            scores = cross_val_score(model, X_all_scaled, y, cv=cv, scoring="f1", n_jobs=None)
            row["cv_f1_mean"] = float(np.mean(scores))
            row["cv_f1_std"] = float(np.std(scores))

        rows.append(row)
        trained_models[name] = model
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    metrics = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    return metrics, best_name, trained_models[best_name], trained_models, scaler, X_test_scaled, y_test


def run_experiment(
    csv_path: Path,
    image_root: Path,
    feature_mode: str = "traditional",
    test_size: float = 0.25,
    random_state: int = 42,
    save_crops: bool = True,
    deep_model: str = "resnet18",
    model_names: Optional[List[str]] = None,
    run_name: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Run the complete pretest workflow and save outputs."""
    ensure_output_dirs()
    run_dirs = _build_run_dirs(run_name=run_name, output_root=output_root)
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    df = load_annotations(csv_path)

    missing = find_missing_images(df, image_root)
    if missing:
        preview = "\n".join([f"  - {p}" for p in missing[:10]])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Some images listed in annotations.csv do not exist. "
            "Please add images or update image_path.\n"
            f"{preview}{more}"
        )

    samples, warnings = load_valid_crops(df, image_root, run_dirs["crops"], save_crops=save_crops)
    if not samples:
        raise ValueError("No valid crops were produced. Check image files and bbox annotations.")

    X, feature_names, feature_status = _build_feature_matrix(samples, feature_mode, deep_model=deep_model)
    y = np.array([sample["label"] for sample in samples], dtype=int)

    feature_csv = _save_feature_csv(samples, X, feature_names, feature_mode, run_dirs["features"])
    metrics, best_name, best_model, trained_models, scaler, X_test_scaled, y_test = _evaluate_models(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        model_names=model_names,
    )

    run_dirs["reports"].mkdir(parents=True, exist_ok=True)
    metrics_path = run_dirs["reports"] / "metrics.csv"
    metrics_path = _safe_to_csv(metrics, metrics_path)

    best_cm = np.array(metrics.iloc[0]["confusion_matrix"])
    cm_path = plot_confusion_matrix(best_cm, run_dirs["figures"], best_name)
    all_cm_path = plot_all_confusion_matrices(metrics, run_dirs["figures"])
    metric_comparison_path = plot_model_metric_comparison(metrics, run_dirs["figures"])
    feature_2d_path = plot_feature_2d(StandardScaler().fit_transform(X), y, run_dirs["figures"])
    sample_plot_path = plot_sample_bboxes(samples, run_dirs["figures"], random_state=random_state)
    hough_paths = save_hough_level_examples(samples, run_dirs["level_lines"], random_state=random_state)

    run_dirs["models"].mkdir(parents=True, exist_ok=True)
    best_model_path = run_dirs["models"] / "best_model.joblib"
    all_models_path = run_dirs["models"] / "all_models.joblib"
    scaler_path = run_dirs["models"] / "scaler.joblib"
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_mode": feature_mode,
            "feature_names": feature_names,
            "deep_model": deep_model,
        },
        best_model_path,
    )
    joblib.dump(
        {
            "models": trained_models,
            "best_model_name": best_name,
            "feature_mode": feature_mode,
            "feature_names": feature_names,
            "deep_model": deep_model,
        },
        all_models_path,
    )
    joblib.dump(scaler, scaler_path)

    summary_path = run_dirs["reports"] / "run_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Bottle liquid residue pretest summary\n")
        f.write(f"run_dir: {run_dirs['run']}\n")
        f.write(f"feature_mode: {feature_mode}\n")
        f.write(f"test_size: {test_size}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"model_names: {', '.join(model_names) if model_names else 'all'}\n")
        f.write(f"feature_status: {feature_status}\n")
        f.write(f"valid_samples: {len(samples)}\n")
        f.write(f"feature_dim: {X.shape[1]}\n")
        f.write(f"best_model: {best_name}\n")
        f.write(f"feature_csv: {feature_csv}\n")
        f.write(f"metrics_csv: {metrics_path}\n")
        if warnings:
            f.write("\nWarnings:\n")
            for warning in warnings:
                f.write(f"- {warning}\n")

    return {
        "samples": samples,
        "warnings": warnings,
        "feature_status": feature_status,
        "feature_csv": feature_csv,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "run_dir": run_dirs["run"],
        "best_model": best_name,
        "best_model_path": best_model_path,
        "all_models_path": all_models_path,
        "scaler_path": scaler_path,
        "summary_path": summary_path,
        "figures": {
            "confusion_matrix": cm_path,
            "confusion_matrices_all_models": all_cm_path,
            "model_metric_comparison": metric_comparison_path,
            "feature_2d": feature_2d_path,
            "sample_bboxes": sample_plot_path,
            "hough_examples": hough_paths,
        },
    }
