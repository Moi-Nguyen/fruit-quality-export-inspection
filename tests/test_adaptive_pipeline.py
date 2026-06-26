"""Tests for adaptive preprocessing pipeline."""

import numpy as np

from src.adaptive_pipeline import (
    choose_preprocessing_method,
    preprocess_grayscale_by_quality,
    run_preprocessing_pipeline,
)


def test_choose_preprocessing_method_returns_expected_methods() -> None:
    assert choose_preprocessing_method("dark") == "histogram_equalization"
    assert choose_preprocessing_method("low_contrast") == "histogram_equalization"
    assert choose_preprocessing_method("noisy") == "median_filter"
    assert choose_preprocessing_method("normal") == "gaussian_filter"


def test_preprocess_grayscale_by_quality_preserves_image_shape() -> None:
    gray = np.arange(25, dtype=np.float32).reshape(5, 5)

    preprocessed, method_name = preprocess_grayscale_by_quality(gray, "normal")

    assert preprocessed.shape == gray.shape
    assert method_name == "gaussian_filter"


def test_run_preprocessing_pipeline_returns_required_keys() -> None:
    image = np.full((6, 6, 3), 120, dtype=np.uint8)

    results = run_preprocessing_pipeline(image)

    expected_keys = {
        "original_image",
        "grayscale",
        "quality",
        "preprocessing_method",
        "preprocessed_gray",
    }
    assert expected_keys.issubset(results.keys())
