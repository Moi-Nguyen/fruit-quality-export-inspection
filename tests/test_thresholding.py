import numpy as np
import pytest

from src.thresholding import (
    apply_threshold,
    compute_otsu_threshold,
    create_background_distance_mask,
    create_color_difference_mask,
    create_combined_fruit_candidate_mask,
    create_non_white_background_mask,
    create_otsu_mask,
    estimate_background_color_from_border,
    estimate_border_foreground_ratio,
    remove_black_border_pixels,
)


def test_compute_otsu_threshold_between_two_groups():
    gray = np.array([[20, 20, 20], [200, 200, 200]], dtype=np.float32)

    threshold = compute_otsu_threshold(gray)

    assert 20 < threshold < 200


def test_apply_threshold_bright_foreground():
    gray = np.array([[10, 100], [150, 200]], dtype=np.float32)

    mask = apply_threshold(gray, 120, foreground="bright")

    expected = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_apply_threshold_dark_foreground():
    gray = np.array([[10, 100], [150, 200]], dtype=np.float32)

    mask = apply_threshold(gray, 120, foreground="dark")

    expected = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_apply_threshold_invalid_foreground():
    with pytest.raises(ValueError):
        apply_threshold(np.zeros((2, 2)), 10, foreground="middle")


def test_estimate_border_foreground_ratio_simple_mask():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, :] = 1
    mask[1, 1] = 1

    ratio = estimate_border_foreground_ratio(mask)

    assert ratio == 3 / 8


def test_create_otsu_mask_preserves_shape_and_binary_values():
    gray = np.zeros((6, 6), dtype=np.float32)
    gray[2:4, 2:4] = 200

    mask = create_otsu_mask(gray)

    assert mask.shape == gray.shape
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_constant_image_does_not_crash():
    gray = np.full((4, 4), 80, dtype=np.float32)

    threshold = compute_otsu_threshold(gray)
    mask = create_otsu_mask(gray, auto_invert=False)

    assert threshold == 80.0
    assert mask.shape == gray.shape

def test_create_non_white_background_mask_detects_non_white_pixels():
    image = np.full((3, 3, 3), 250, dtype=np.uint8)
    image[1, 1] = [220, 210, 40]

    mask = create_non_white_background_mask(image, white_threshold=235)

    assert mask[1, 1] == 1
    assert int(np.sum(mask)) == 1

def test_create_color_difference_mask_detects_colorful_pixels():
    image = np.full((3, 3, 3), 240, dtype=np.uint8)
    image[0, 0] = [230, 220, 20]
    image[1, 1] = [230, 60, 40]
    image[2, 2] = [240, 150, 20]

    mask = create_color_difference_mask(image, color_difference_threshold=20)

    assert mask[0, 0] == 1
    assert mask[1, 1] == 1
    assert mask[2, 2] == 1
    assert int(np.sum(mask)) == 3

def test_create_color_difference_mask_ignores_pure_white_background():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)

    mask = create_color_difference_mask(image, color_difference_threshold=20)

    assert int(np.sum(mask)) == 0

def test_remove_black_border_pixels_removes_black_pixels_from_mask():
    mask = np.ones((3, 3), dtype=np.uint8)
    gray = np.full((3, 3), 100, dtype=np.float32)
    gray[0, 0] = 0

    cleaned = remove_black_border_pixels(mask, gray, black_threshold=15)

    assert cleaned[0, 0] == 0
    assert cleaned[1, 1] == 1

def test_create_combined_fruit_candidate_mask_preserves_shape_and_binary_values():
    image = np.full((4, 5, 3), 250, dtype=np.uint8)
    image[1:3, 2:4] = [230, 180, 20]
    gray = np.mean(image, axis=2)
    otsu_mask = np.zeros(gray.shape, dtype=np.uint8)

    combined = create_combined_fruit_candidate_mask(image, gray, otsu_mask)

    assert combined.shape == gray.shape
    assert set(np.unique(combined).tolist()) <= {0, 1}
    assert int(np.sum(combined)) > 0

def test_combined_candidate_mask_does_not_select_white_background():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    gray = np.mean(image, axis=2)
    otsu_mask = np.ones(gray.shape, dtype=np.uint8)

    combined = create_combined_fruit_candidate_mask(image, gray, otsu_mask)

    assert int(np.sum(combined)) == 0

def test_combined_candidate_mask_selects_colored_object_on_white_background():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    image[4:8, 5:9] = [240, 190, 20]
    gray = np.mean(image, axis=2)
    otsu_mask = np.zeros(gray.shape, dtype=np.uint8)

    combined = create_combined_fruit_candidate_mask(image, gray, otsu_mask)

    assert int(np.sum(combined)) == 16

def test_combined_candidate_mask_removes_black_border_pixels():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    image[0, :] = [0, 0, 0]
    image[3:6, 3:6] = [240, 190, 20]
    gray = np.mean(image, axis=2)
    otsu_mask = np.ones(gray.shape, dtype=np.uint8)

    combined = create_combined_fruit_candidate_mask(image, gray, otsu_mask)

    assert int(np.sum(combined[0, :])) == 0
    assert int(np.sum(combined[3:6, 3:6])) == 9


def test_estimate_background_color_from_border_returns_near_white_background():
    image = np.full((20, 20, 3), 250, dtype=np.uint8)
    image[8:12, 8:12] = [220, 40, 30]
    gray = np.mean(image, axis=2)

    background = estimate_background_color_from_border(image, gray, border_width=3)

    assert background.shape == (3,)
    assert np.all(background > 245)

def test_create_background_distance_mask_detects_red_object_on_white_background():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    image[7:13, 7:13] = [220, 30, 20]
    gray = np.mean(image, axis=2)

    mask = create_background_distance_mask(image, gray, distance_threshold=45.0)

    assert int(np.sum(mask[7:13, 7:13])) == 36
    assert int(np.sum(mask[:3, :])) == 0

def test_create_background_distance_mask_detects_yellow_object_on_white_background():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    image[6:14, 8:12] = [240, 190, 20]
    gray = np.mean(image, axis=2)

    mask = create_background_distance_mask(image, gray, distance_threshold=45.0)

    assert int(np.sum(mask[6:14, 8:12])) == 32

def test_create_background_distance_mask_ignores_black_border_pixels():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[0, :] = [0, 0, 0]
    image[5:11, 5:11] = [240, 190, 20]
    gray = np.mean(image, axis=2)

    mask = create_background_distance_mask(image, gray, distance_threshold=45.0)

    assert int(np.sum(mask[0, :])) == 0
    assert int(np.sum(mask[5:11, 5:11])) == 36
