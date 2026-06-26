"""Adaptive preprocessing pipeline for Step 2."""

from typing import Any

import numpy as np

from src.config import (
    DEFAULT_GAUSSIAN_KERNEL_SIZE,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MEDIAN_KERNEL_SIZE,
)
from src.filters import gaussian_filter_gray, median_filter_gray
from src.histogram import histogram_equalization_gray
from src.quality_analysis import analyze_image_quality, rgb_to_grayscale


def choose_preprocessing_method(quality_label: str) -> str:
    """Choose a preprocessing method from the Step 1 quality label."""
    if quality_label == "dark":
        return "histogram_equalization"
    if quality_label == "low_contrast":
        return "histogram_equalization"
    if quality_label == "noisy":
        return "median_filter"
    if quality_label == "normal":
        return "gaussian_filter"
    raise ValueError(f"Unknown quality label: {quality_label}")


def preprocess_grayscale_by_quality(
    gray: np.ndarray,
    quality_label: str,
) -> tuple[np.ndarray, str]:
    """Apply the selected preprocessing method to a grayscale image."""
    method_name = choose_preprocessing_method(quality_label)

    if method_name == "histogram_equalization":
        preprocessed = histogram_equalization_gray(gray)
    elif method_name == "median_filter":
        preprocessed = median_filter_gray(gray, kernel_size=DEFAULT_MEDIAN_KERNEL_SIZE)
    else:
        preprocessed = gaussian_filter_gray(
            gray,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
            sigma=DEFAULT_GAUSSIAN_SIGMA,
        )

    return preprocessed.astype(np.float32), method_name


def run_preprocessing_pipeline(image: np.ndarray) -> dict[str, object]:
    """Run grayscale conversion, quality analysis, and adaptive preprocessing."""
    grayscale = rgb_to_grayscale(image)
    quality = analyze_image_quality(image)
    quality_label = str(quality["quality_label"])
    preprocessed_gray, method_name = preprocess_grayscale_by_quality(grayscale, quality_label)

    return {
        "original_image": image,
        "grayscale": grayscale,
        "quality": quality,
        "preprocessing_method": method_name,
        "preprocessed_gray": preprocessed_gray,
    }


def run_adaptive_pipeline(image_array: np.ndarray) -> dict[str, Any]:
    """Legacy wrapper for the current adaptive preprocessing pipeline."""
    return run_preprocessing_pipeline(image_array)
