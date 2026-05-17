"""Traditional-feature liquid-presence predictor used by predict_all.py."""

from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage
from scipy.stats import kurtosis, skew


IMAGE_SIZE_FOR_FEATURES = (128, 256)


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image array.")
    return array.astype(np.uint8)


def _resize_for_features(image: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE_FOR_FEATURES) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized)


def _safe_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty image for feature extraction.")
    return _resize_for_features(image)


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def hsv_histogram_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    hsv = rgb_to_hsv(image.astype(np.float32) / 255.0)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(8, 8, 8),
        range=((0, 1), (0, 1), (0, 1)),
    )
    hist = hist.astype(np.float32).ravel()
    hist /= max(float(np.linalg.norm(hist)), 1.0)
    return hist, [f"hsv_hist_{i}" for i in range(hist.size)]


def rgb_mean_std_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    pixels = image.astype(np.float32).reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    vector = np.concatenate([means, stds]).astype(np.float32)
    names = ["rgb_r_mean", "rgb_g_mean", "rgb_b_mean", "rgb_r_std", "rgb_g_std", "rgb_b_std"]
    return vector, names


def gray_statistics_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image).astype(np.float32).ravel()
    vector = np.array(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(skew(gray)) if gray.size > 2 else 0.0,
            float(kurtosis(gray)) if gray.size > 3 else 0.0,
        ],
        dtype=np.float32,
    )
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector, ["gray_mean", "gray_std", "gray_skew", "gray_kurtosis"]


def region_difference_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image)
    h, w = gray.shape
    vector = np.array(
        [
            float(np.mean(gray[: h // 2, :]) - np.mean(gray[h // 2 :, :])),
            float(np.mean(gray[:, : w // 2]) - np.mean(gray[:, w // 2 :])),
        ],
        dtype=np.float32,
    )
    return vector, ["gray_top_bottom_mean_diff", "gray_left_right_mean_diff"]


def gradient_histogram_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image).astype(np.float32) / 255.0
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    h, w = gray.shape
    rows = []
    for iy in range(8):
        y0 = int(round(iy * h / 8))
        y1 = int(round((iy + 1) * h / 8))
        for ix in range(4):
            x0 = int(round(ix * w / 4))
            x1 = int(round((ix + 1) * w / 4))
            hist, _ = np.histogram(
                angle[y0:y1, x0:x1].ravel(),
                bins=9,
                range=(0, np.pi),
                weights=mag[y0:y1, x0:x1].ravel(),
            )
            rows.append(hist.astype(np.float32))
    vector = np.concatenate(rows)
    vector /= max(float(np.linalg.norm(vector)), 1.0)
    return vector.astype(np.float32), [f"grad_hist_{i}" for i in range(vector.size)]


def lbp_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image).astype(np.float32)
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= (neighbor >= center).astype(np.uint8) << bit
    hist, _ = np.histogram(code.ravel(), bins=32, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return hist, [f"lbp32_{i}" for i in range(hist.size)]


def edge_density_feature(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image).astype(np.float32)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    threshold = np.percentile(mag, 82)
    return np.array([float(np.mean(mag > threshold))], dtype=np.float32), ["edge_density"]


def hough_liquid_level_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = _gray(image).astype(np.float32)
    smooth = ndimage.gaussian_filter(gray, sigma=1.0)
    h, w = smooth.shape
    row_contrast = np.abs(np.diff(smooth, axis=0)).mean(axis=1)
    col_contrast = np.abs(np.diff(smooth, axis=1)).mean(axis=0)

    best_row = int(np.argmax(row_contrast)) if row_contrast.size else 0
    best_col = int(np.argmax(col_contrast)) if col_contrast.size else 0
    row_score = float(row_contrast[best_row] / 255.0) if row_contrast.size else 0.0
    col_score = float(col_contrast[best_col] / 255.0) if col_contrast.size else 0.0

    if row_score >= col_score:
        best_angle = 0.0
        best_ratio = float(best_row) / max(float(h - 1), 1.0)
        best_score = row_score
    else:
        best_angle = 90.0
        best_ratio = float(best_col) / max(float(w - 1), 1.0)
        best_score = col_score

    vector = np.array(
        [2.0, float(max(w, h)), float((w + h) / 2.0), best_score, best_angle, best_ratio],
        dtype=np.float32,
    )
    names = [
        "candidate_line_count",
        "candidate_max_line_length",
        "candidate_mean_line_length",
        "level_score",
        "best_line_angle",
        "best_line_position_ratio",
    ]
    return vector, names


def get_traditional_feature_names() -> List[str]:
    dummy = np.zeros((IMAGE_SIZE_FOR_FEATURES[1], IMAGE_SIZE_FOR_FEATURES[0], 3), dtype=np.uint8)
    names = []
    for func in [
        hsv_histogram_features,
        rgb_mean_std_features,
        gray_statistics_features,
        region_difference_features,
        gradient_histogram_features,
        lbp_features,
        edge_density_feature,
    ]:
        _, part_names = func(dummy)
        names.extend(part_names)
    _, line_names = hough_liquid_level_features(dummy)
    names.extend(line_names)
    return names


def extract_traditional_features(image: Image.Image | np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Extract the same traditional features used by the binary ML model."""
    try:
        fixed = _safe_image(_as_rgb_array(image))
        parts = []
        names = []
        for func in [
            hsv_histogram_features,
            rgb_mean_std_features,
            gray_statistics_features,
            region_difference_features,
            gradient_histogram_features,
            lbp_features,
            edge_density_feature,
        ]:
            vector, part_names = func(fixed)
            parts.append(vector)
            names.extend(part_names)

        line_vector, line_names = hough_liquid_level_features(fixed)
        parts.append(line_vector)
        names.extend(line_names)
        feature_vector = np.concatenate(parts).astype(np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vector, names
    except Exception:
        names = get_traditional_feature_names()
        return np.zeros(len(names), dtype=np.float32), names


def predict_liquid_presence(
    crop: Image.Image | np.ndarray,
    model_path: str | Path,
    scaler_path: str | Path,
) -> dict:
    """Predict empty / has_liquid from a detected bottle crop."""
    model_bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model = model_bundle["model"]
    feature_mode = model_bundle.get("feature_mode", "traditional")
    if feature_mode != "traditional":
        raise ValueError(f"Unsupported binary feature mode: {feature_mode}")

    features, feature_names = extract_traditional_features(crop)
    expected_names: Sequence[str] | None = model_bundle.get("feature_names")
    if expected_names and list(expected_names) != feature_names:
        raise ValueError("Feature names do not match the saved binary model.")

    x = scaler.transform(features.reshape(1, -1))
    label = int(model.predict(x)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(x)[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))

    return {
        "prediction": label,
        "prediction_text": "has_liquid" if label == 1 else "empty",
        "probability_has_liquid": probability,
        "model_name": model_bundle.get("model_name", "unknown"),
    }
