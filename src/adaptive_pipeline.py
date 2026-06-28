"""Adaptive preprocessing and segmentation pipelines."""

from typing import Any

import numpy as np

from src.components import keep_largest_reasonable_component
from src.config import (
    DEFAULT_COMPONENT_CONNECTIVITY,
    DEFAULT_GAUSSIAN_KERNEL_SIZE,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MEDIAN_KERNEL_SIZE,
    DEFAULT_MORPH_KERNEL_SIZE,
)
from src.filters import gaussian_filter_gray, median_filter_gray
from src.histogram import histogram_equalization_gray
from src.morphology import clean_binary_mask, fill_holes_binary
from src.quality_analysis import analyze_image_quality, rgb_to_grayscale
from src.thresholding import compute_otsu_threshold, create_combined_fruit_candidate_mask, create_otsu_mask


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


def run_segmentation_pipeline(image: np.ndarray) -> dict[str, object]:
    """Run preprocessing, thresholding, morphology, and largest component selection."""
    preprocessing_results = run_preprocessing_pipeline(image)
    preprocessed_gray = preprocessing_results["preprocessed_gray"]

    otsu_threshold = compute_otsu_threshold(preprocessed_gray)
    initial_mask = create_otsu_mask(preprocessed_gray)
    combined_mask = create_combined_fruit_candidate_mask(image, preprocessed_gray, initial_mask)
    cleaned_mask = clean_binary_mask(combined_mask, kernel_size=DEFAULT_MORPH_KERNEL_SIZE)
    cleaned_mask = fill_holes_binary(cleaned_mask, connectivity=DEFAULT_COMPONENT_CONNECTIVITY)
    fruit_mask = keep_largest_reasonable_component(cleaned_mask, connectivity=DEFAULT_COMPONENT_CONNECTIVITY)

    return {
        "original_image": preprocessing_results["original_image"],
        "grayscale": preprocessing_results["grayscale"],
        "quality": preprocessing_results["quality"],
        "preprocessing_method": preprocessing_results["preprocessing_method"],
        "preprocessed_gray": preprocessed_gray,
        "initial_mask": initial_mask,
        "combined_mask": combined_mask,
        "cleaned_mask": cleaned_mask,
        "fruit_mask": fruit_mask,
        "otsu_threshold": otsu_threshold,
    }


def run_adaptive_pipeline(image_array: np.ndarray) -> dict[str, Any]:
    """Legacy wrapper for the current adaptive preprocessing pipeline."""
    return run_preprocessing_pipeline(image_array)
