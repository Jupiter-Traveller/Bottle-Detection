"""Traditional visual features for bottle liquid-residue recognition."""

from typing import List, Tuple

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage
from scipy.stats import kurtosis, skew

from config import IMAGE_SIZE_FOR_FEATURES
from preprocess import resize_for_features


def _safe_image(image: np.ndarray) -> np.ndarray:
    """Return a fixed-size RGB image; raise ValueError for empty input."""
    if image is None or image.size == 0:
        raise ValueError("Empty image for feature extraction.")
    return resize_for_features(image, IMAGE_SIZE_FOR_FEATURES)


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def hsv_histogram_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute a normalized HSV color histogram."""
    hsv = rgb_to_hsv(image.astype(np.float32) / 255.0)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(8, 8, 8),
        range=((0, 1), (0, 1), (0, 1)),
    )
    hist = hist.astype(np.float32).ravel()
    hist /= max(float(np.linalg.norm(hist)), 1.0)
    names = [f"hsv_hist_{i}" for i in range(hist.size)]
    return hist, names


def rgb_mean_std_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute per-channel mean and standard deviation in RGB color space."""
    pixels = image.astype(np.float32).reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    vector = np.concatenate([means, stds]).astype(np.float32)
    names = ["rgb_r_mean", "rgb_g_mean", "rgb_b_mean", "rgb_r_std", "rgb_g_std", "rgb_b_std"]
    return vector, names


def gray_statistics_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute mean, std, skewness, and kurtosis of grayscale intensities."""
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
    names = ["gray_mean", "gray_std", "gray_skew", "gray_kurtosis"]
    return vector, names


def region_difference_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute coarse brightness differences between image halves."""
    gray = _gray(image)
    h, w = gray.shape
    top = gray[: h // 2, :]
    bottom = gray[h // 2 :, :]
    left = gray[:, : w // 2]
    right = gray[:, w // 2 :]
    vector = np.array(
        [
            float(np.mean(top) - np.mean(bottom)),
            float(np.mean(left) - np.mean(right)),
        ],
        dtype=np.float32,
    )
    names = ["gray_top_bottom_mean_diff", "gray_left_right_mean_diff"]
    return vector, names


def gradient_histogram_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute a compact grid histogram of gradient orientations."""
    gray = _gray(image).astype(np.float32) / 255.0
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    h, w = gray.shape
    grid_y, grid_x, bins = 8, 4, 9
    rows = []
    for iy in range(grid_y):
        y0 = int(round(iy * h / grid_y))
        y1 = int(round((iy + 1) * h / grid_y))
        for ix in range(grid_x):
            x0 = int(round(ix * w / grid_x))
            x1 = int(round((ix + 1) * w / grid_x))
            local_angle = angle[y0:y1, x0:x1].ravel()
            local_mag = mag[y0:y1, x0:x1].ravel()
            hist, _ = np.histogram(local_angle, bins=bins, range=(0, np.pi), weights=local_mag)
            rows.append(hist.astype(np.float32))
    vector = np.concatenate(rows)
    vector /= max(float(np.linalg.norm(vector)), 1.0)
    names = [f"grad_hist_{i}" for i in range(vector.size)]
    return vector.astype(np.float32), names


def lbp_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute a simple local binary pattern texture histogram."""
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
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(code.ravel(), bins=32, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    names = [f"lbp32_{i}" for i in range(hist.size)]
    return hist, names


def edge_density_feature(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute the fraction of strong gradient pixels."""
    gray = _gray(image).astype(np.float32)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    threshold = np.percentile(mag, 82)
    density = float(np.mean(mag > threshold))
    return np.array([density], dtype=np.float32), ["edge_density"]


def hough_liquid_level_features(image: np.ndarray) -> Tuple[np.ndarray, List[str], dict]:
    """Estimate simple liquid-level candidates from row-wise edge signals."""
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
        best_line = (0, best_row, w - 1, best_row)
        best_angle = 0.0
        best_ratio = float(best_row) / max(float(h - 1), 1.0)
        best_score = row_score
    else:
        best_line = (best_col, 0, best_col, h - 1)
        best_angle = 90.0
        best_ratio = float(best_col) / max(float(w - 1), 1.0)
        best_score = col_score

    vector = np.array(
        [
            2.0,
            float(max(w, h)),
            float((w + h) / 2.0),
            best_score,
            best_angle,
            best_ratio,
        ],
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
    detail = {
        "best_line": {"line": best_line, "score": best_score},
        "lines": [
            {"line": (0, best_row, w - 1, best_row), "score": row_score},
            {"line": (best_col, 0, best_col, h - 1), "score": col_score},
        ],
    }
    return vector, names, detail


def extract_traditional_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Extract all traditional features as a one-dimensional vector."""
    try:
        fixed = _safe_image(image)
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

        line_vector, line_names, _ = hough_liquid_level_features(fixed)
        parts.append(line_vector)
        names.extend(line_names)

        feature_vector = np.concatenate(parts).astype(np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vector, names
    except Exception:
        names = get_traditional_feature_names()
        return np.zeros(len(names), dtype=np.float32), names


def get_traditional_feature_names() -> List[str]:
    """Return traditional feature names by running extraction on a dummy image."""
    dummy = np.zeros((IMAGE_SIZE_FOR_FEATURES[1], IMAGE_SIZE_FOR_FEATURES[0], 3), dtype=np.uint8)
    parts = []
    for func in [
        hsv_histogram_features,
        rgb_mean_std_features,
        gray_statistics_features,
        region_difference_features,
        gradient_histogram_features,
        lbp_features,
        edge_density_feature,
    ]:
        _, names = func(dummy)
        parts.extend(names)
    _, line_names, _ = hough_liquid_level_features(dummy)
    parts.extend(line_names)
    return parts
