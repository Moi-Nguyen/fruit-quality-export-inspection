"""Manual grayscale histogram utilities."""

import numpy as np


def to_uint8_gray(gray: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to uint8 in the range 0 to 255."""
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    clipped = np.clip(gray, 0, 255)
    return clipped.astype(np.uint8)


def compute_gray_histogram(gray: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """Compute a grayscale histogram without OpenCV or matplotlib."""
    if num_bins != 256:
        raise ValueError("This project uses 256 bins for uint8 grayscale images.")

    gray_uint8 = to_uint8_gray(gray)
    histogram = np.bincount(gray_uint8.ravel(), minlength=256)
    return histogram[:256]


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """Compute a normalized cumulative distribution function."""
    if histogram.ndim != 1:
        raise ValueError("Histogram must be a 1D array.")
    if histogram.size == 0:
        raise ValueError("Histogram must not be empty.")
    if np.any(histogram < 0):
        raise ValueError("Histogram counts must not be negative.")

    total_count = np.sum(histogram)
    if total_count <= 0:
        raise ValueError("Histogram must contain at least one pixel count.")

    cdf = np.cumsum(histogram).astype(np.float32)
    cdf = cdf / float(total_count)
    return cdf.astype(np.float32)


def histogram_equalization_gray(gray: np.ndarray) -> np.ndarray:
    """Improve grayscale contrast using manual histogram equalization."""
    gray_uint8 = to_uint8_gray(gray)
    histogram = compute_gray_histogram(gray_uint8)
    cdf = compute_cdf(histogram)

    nonzero_cdf = cdf[histogram > 0]
    if nonzero_cdf.size <= 1:
        return gray_uint8.astype(np.float32)

    cdf_min = nonzero_cdf[0]
    denominator = 1.0 - cdf_min
    if denominator <= 0:
        return gray_uint8.astype(np.float32)

    mapping = np.round((cdf - cdf_min) / denominator * 255.0)
    mapping = np.clip(mapping, 0, 255).astype(np.uint8)
    equalized = mapping[gray_uint8]

    return equalized.astype(np.float32)


def compute_histogram(gray_image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Legacy wrapper for grayscale histogram computation."""
    return compute_gray_histogram(gray_image, num_bins=bins)


def histogram_equalization(gray_image: np.ndarray) -> np.ndarray:
    """Legacy wrapper for grayscale histogram equalization."""
    return histogram_equalization_gray(gray_image)
