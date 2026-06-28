import numpy as np
import pytest

from src.morphology import (
    clean_binary_mask,
    closing_binary,
    create_structuring_element,
    dilate_binary,
    erode_binary,
    fill_holes_binary,
    opening_binary,
    validate_binary_mask,
)


def test_create_structuring_element_square():
    element = create_structuring_element(3, "square")

    np.testing.assert_array_equal(element, np.ones((3, 3), dtype=np.uint8))


def test_create_structuring_element_cross():
    element = create_structuring_element(3, "cross")
    expected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(element, expected)


def test_erode_binary_shrinks_simple_object():
    mask = np.ones((5, 5), dtype=np.uint8)

    eroded = erode_binary(mask, 3)

    assert int(np.sum(eroded)) == 9


def test_dilate_binary_expands_single_pixel():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    dilated = dilate_binary(mask, 3)

    assert int(np.sum(dilated)) == 9


def test_opening_binary_removes_isolated_noise():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    opened = opening_binary(mask, 3)

    assert int(np.sum(opened)) == 0


def test_closing_binary_fills_small_hole_when_possible():
    mask = np.ones((5, 5), dtype=np.uint8)
    mask[2, 2] = 0

    closed = closing_binary(mask, 3)

    assert closed[2, 2] == 1


def test_clean_binary_mask_preserves_shape():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    cleaned = clean_binary_mask(mask)

    assert cleaned.shape == mask.shape


def test_invalid_mask_raises_value_error():
    with pytest.raises(ValueError):
        validate_binary_mask(np.array([[0, 2]], dtype=np.uint8))


def test_fill_holes_binary_fills_hole_inside_square_object():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    mask[3, 3] = 0

    filled = fill_holes_binary(mask)

    assert filled[3, 3] == 1
    assert int(np.sum(filled)) == 25
