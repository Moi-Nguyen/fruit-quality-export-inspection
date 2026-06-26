"""Tests for manual NumPy filtering functions."""

import numpy as np
import pytest

from src.filters import (
    convolve2d_gray,
    create_gaussian_kernel,
    gaussian_filter_gray,
    median_filter_gray,
    validate_kernel_size,
)


def test_gaussian_kernel_has_correct_shape() -> None:
    kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0)

    assert kernel.shape == (5, 5)


def test_gaussian_kernel_sum_is_one() -> None:
    kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0)

    assert np.sum(kernel) == pytest.approx(1.0)


@pytest.mark.parametrize("kernel_size", [0, -3, 2, 4])
def test_invalid_kernel_size_raises_value_error(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        validate_kernel_size(kernel_size)


def test_convolve2d_gray_preserves_shape() -> None:
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    filtered = convolve2d_gray(image, kernel)

    assert filtered.shape == image.shape


def test_gaussian_filter_gray_keeps_constant_image_nearly_unchanged() -> None:
    image = np.full((5, 5), 100, dtype=np.float32)

    filtered = gaussian_filter_gray(image, kernel_size=5, sigma=1.0)

    assert np.allclose(filtered, image, atol=0.01)


def test_median_filter_gray_reduces_single_impulse_noise_pixel() -> None:
    image = np.full((5, 5), 50, dtype=np.float32)
    image[2, 2] = 255

    filtered = median_filter_gray(image, kernel_size=3)

    assert filtered[2, 2] == pytest.approx(50.0)


def test_median_filter_gray_preserves_shape() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    filtered = median_filter_gray(image, kernel_size=3)

    assert filtered.shape == image.shape
