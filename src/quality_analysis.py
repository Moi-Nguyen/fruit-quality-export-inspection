"""Image quality analysis using NumPy-based image processing."""

import numpy as np

from src.config import (
    DARK_BRIGHTNESS_THRESHOLD,
    HIGH_NOISE_THRESHOLD,
    LOW_CONTRAST_THRESHOLD,
)
from src.io_utils import validate_image_array


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA or grayscale image data to 2D grayscale."""
    validate_image_array(image)

    if image.ndim == 2:
        return image.astype(np.float32)

    rgb_image = image[:, :, :3].astype(np.float32)
    red = rgb_image[:, :, 0]
    green = rgb_image[:, :, 1]
    blue = rgb_image[:, :, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return gray.astype(np.float32)


def compute_brightness(gray: np.ndarray) -> float:
    """Return the mean grayscale intensity."""
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    """Return the standard deviation of grayscale intensity."""
    return float(np.std(gray))


def mean_filter_3x3(gray: np.ndarray) -> np.ndarray:
    """Apply a simple 3x3 mean filter with edge padding."""
    gray_float = gray.astype(np.float32)
    padded = np.pad(gray_float, pad_width=1, mode="edge")
    smoothed = np.zeros_like(gray_float, dtype=np.float32)

    for row_offset in range(3):
        for col_offset in range(3):
            smoothed += padded[
                row_offset : row_offset + gray_float.shape[0],
                col_offset : col_offset + gray_float.shape[1],
            ]

    return smoothed / 9.0


def estimate_noise_level(gray: np.ndarray) -> float:
    """Estimate noise using residuals from a 3x3 mean filter."""
    smoothed = mean_filter_3x3(gray)
    residual = gray.astype(np.float32) - smoothed
    return float(np.std(residual))


def classify_image_quality(brightness: float, contrast: float, noise_level: float) -> str:
    """Classify basic image quality using empirical thresholds."""
    if brightness < DARK_BRIGHTNESS_THRESHOLD:
        return "dark"
    if contrast < LOW_CONTRAST_THRESHOLD:
        return "low_contrast"
    if noise_level > HIGH_NOISE_THRESHOLD:
        return "noisy"
    return "normal"


def analyze_image_quality(image: np.ndarray) -> dict[str, float | str]:
    """Compute brightness, contrast, noise level, and quality label."""
    gray = rgb_to_grayscale(image)
    brightness = compute_brightness(gray)
    contrast = compute_contrast(gray)
    noise_level = estimate_noise_level(gray)
    quality_label = classify_image_quality(brightness, contrast, noise_level)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise_level": noise_level,
        "quality_label": quality_label,
    }


def estimate_brightness(image_array: np.ndarray) -> float:
    """Estimate average image brightness for legacy callers."""
    return compute_brightness(rgb_to_grayscale(image_array))


def estimate_contrast(image_array: np.ndarray) -> float:
    """Estimate image contrast for legacy callers."""
    return compute_contrast(rgb_to_grayscale(image_array))


def estimate_noise(image_array: np.ndarray) -> float:
    """Estimate image noise level for legacy callers."""
    return estimate_noise_level(rgb_to_grayscale(image_array))
