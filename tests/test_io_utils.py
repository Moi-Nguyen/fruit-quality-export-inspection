"""Tests for image input/output utilities."""

import numpy as np
import pytest
from PIL import Image

from src.io_utils import (
    ensure_uint8_image,
    is_supported_image_file,
    load_image,
    normalize_to_float,
    save_image,
    validate_image_array,
)


def test_supported_image_extension_returns_true(tmp_path):
    image_path = tmp_path / "fruit.JPG"

    assert is_supported_image_file(image_path) is True


def test_unsupported_image_extension_returns_false(tmp_path):
    image_path = tmp_path / "fruit.txt"

    assert is_supported_image_file(image_path) is False


def test_load_image_reads_temporary_rgb_image(tmp_path):
    image_path = tmp_path / "fruit.png"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)

    image = load_image(image_path)

    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[0, 0], np.array([10, 20, 30], dtype=np.uint8))


def test_save_image_writes_image_file(tmp_path):
    output_path = tmp_path / "nested" / "fruit.png"
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    save_image(image, output_path)

    assert output_path.exists()


def test_validate_image_array_rejects_invalid_arrays():
    with pytest.raises(ValueError):
        validate_image_array(np.array([]))

    with pytest.raises(ValueError):
        validate_image_array(np.zeros((2, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError):
        validate_image_array("not an image")


def test_normalize_to_float_converts_uint8_image_to_zero_one_range():
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    normalized = normalize_to_float(image)

    assert normalized.dtype == np.float32
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_ensure_uint8_image_converts_float_image_to_uint8():
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    uint8_image = ensure_uint8_image(image)

    assert uint8_image.dtype == np.uint8
    assert np.array_equal(uint8_image, np.array([[0, 127, 255]], dtype=np.uint8))
