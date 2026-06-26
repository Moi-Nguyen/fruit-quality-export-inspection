"""Tests for manual grayscale histogram utilities."""

import numpy as np
import pytest

from src.histogram import (
    compute_cdf,
    compute_gray_histogram,
    histogram_equalization_gray,
    to_uint8_gray,
)


def test_to_uint8_gray_clips_values_correctly() -> None:
    gray = np.array([[-10, 0, 128, 300]], dtype=np.float32)

    converted = to_uint8_gray(gray)

    assert converted.dtype == np.uint8
    assert converted.tolist() == [[0, 0, 128, 255]]


def test_compute_gray_histogram_returns_256_bins() -> None:
    gray = np.array([[0, 1], [2, 255]], dtype=np.uint8)

    histogram = compute_gray_histogram(gray)

    assert histogram.shape == (256,)


def test_histogram_count_sum_equals_number_of_pixels() -> None:
    gray = np.array([[0, 1], [2, 255]], dtype=np.uint8)

    histogram = compute_gray_histogram(gray)

    assert np.sum(histogram) == gray.size


def test_compute_cdf_starts_non_negative_and_ends_at_one() -> None:
    histogram = np.array([1, 2, 1], dtype=np.int64)

    cdf = compute_cdf(histogram)

    assert cdf[0] >= 0
    assert cdf[-1] == pytest.approx(1.0)


def test_histogram_equalization_gray_preserves_image_shape() -> None:
    gray = np.array([[0, 50], [200, 255]], dtype=np.float32)

    equalized = histogram_equalization_gray(gray)

    assert equalized.shape == gray.shape


def test_histogram_equalization_gray_handles_constant_image_safely() -> None:
    gray = np.full((4, 4), 80, dtype=np.float32)

    equalized = histogram_equalization_gray(gray)

    assert equalized.shape == gray.shape
    assert np.all(np.isfinite(equalized))


def test_histogram_equalization_gray_returns_float32_values_in_range() -> None:
    gray = np.array([[0, 50], [200, 255]], dtype=np.float32)

    equalized = histogram_equalization_gray(gray)

    assert equalized.dtype == np.float32
    assert float(np.min(equalized)) >= 0.0
    assert float(np.max(equalized)) <= 255.0
