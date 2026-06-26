"""Tests for Step 1 image quality analysis."""

import numpy as np

from src.quality_analysis import (
    analyze_image_quality,
    classify_image_quality,
    compute_brightness,
    compute_contrast,
    estimate_noise_level,
    mean_filter_3x3,
    rgb_to_grayscale,
)


def test_rgb_to_grayscale_returns_correct_shape():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    gray = rgb_to_grayscale(image)

    assert gray.shape == (3, 4)
    assert gray.dtype == np.float32


def test_rgb_to_grayscale_gives_expected_value_for_simple_rgb_pixel():
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)

    gray = rgb_to_grayscale(image)

    expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
    assert np.isclose(gray[0, 0], expected)


def test_compute_brightness_equals_mean_intensity():
    gray = np.array([[10, 20], [30, 40]], dtype=np.float32)

    assert compute_brightness(gray) == float(np.mean(gray))


def test_compute_contrast_equals_standard_deviation():
    gray = np.array([[10, 20], [30, 40]], dtype=np.float32)

    assert compute_contrast(gray) == float(np.std(gray))


def test_mean_filter_3x3_keeps_constant_image_unchanged():
    gray = np.full((5, 5), 80, dtype=np.float32)

    filtered = mean_filter_3x3(gray)

    assert np.allclose(filtered, gray)


def test_estimate_noise_level_is_near_zero_for_constant_image():
    gray = np.full((5, 5), 80, dtype=np.float32)

    noise_level = estimate_noise_level(gray)

    assert np.isclose(noise_level, 0.0)


def test_estimate_noise_level_is_higher_for_image_with_impulse_noise():
    constant_gray = np.full((5, 5), 80, dtype=np.float32)
    noisy_gray = constant_gray.copy()
    noisy_gray[2, 2] = 255

    constant_noise = estimate_noise_level(constant_gray)
    impulse_noise = estimate_noise_level(noisy_gray)

    assert impulse_noise > constant_noise


def test_classify_image_quality_returns_all_expected_labels():
    assert classify_image_quality(brightness=30, contrast=50, noise_level=5) == "dark"
    assert classify_image_quality(brightness=100, contrast=5, noise_level=5) == "low_contrast"
    assert classify_image_quality(brightness=100, contrast=50, noise_level=30) == "noisy"
    assert classify_image_quality(brightness=100, contrast=50, noise_level=5) == "normal"


def test_analyze_image_quality_returns_all_required_keys():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    results = analyze_image_quality(image)

    assert set(results) == {"brightness", "contrast", "noise_level", "quality_label"}
