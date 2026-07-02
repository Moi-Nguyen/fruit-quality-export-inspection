import numpy as np

from src.features import (
    compute_area,
    compute_aspect_ratio,
    compute_bounding_box,
    compute_centroid,
    compute_circularity,
    compute_gradient_magnitude,
    compute_mask_area_ratio,
    compute_perimeter,
    compute_rgb_histogram,
    compute_rgb_mean_std,
    create_defect_map,
    extract_all_features_from_pipeline_result,
    extract_color_features,
    extract_defect_features,
    extract_shape_features,
    extract_texture_features,
    get_masked_pixels,
)


def make_mask():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    return mask


def make_image():
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    image[1:4, 2:5] = np.array([100, 150, 200], dtype=np.uint8)
    return image


def test_compute_area_returns_correct_area():
    assert compute_area(make_mask()) == 9


def test_compute_bounding_box_returns_correct_bbox():
    bbox = compute_bounding_box(make_mask())
    assert bbox == {
        "min_row": 1,
        "min_col": 2,
        "max_row": 3,
        "max_col": 4,
        "height": 3,
        "width": 3,
    }


def test_compute_aspect_ratio_works_correctly():
    assert compute_aspect_ratio(make_mask()) == 1.0


def test_compute_centroid_works_correctly():
    assert compute_centroid(make_mask()) == (2.0, 3.0)


def test_compute_perimeter_returns_positive_value_for_square_object():
    assert compute_perimeter(make_mask()) > 0


def test_compute_circularity_returns_positive_value():
    assert compute_circularity(area=9, perimeter=8) > 0


def test_compute_circularity_never_exceeds_one():
    assert compute_circularity(area=100, perimeter=1) == 1.0


def test_compute_circularity_returns_zero_for_zero_perimeter():
    assert compute_circularity(area=9, perimeter=0) == 0.0


def test_compute_mask_area_ratio_works_correctly():
    assert compute_mask_area_ratio(make_mask()) == 9 / 30


def test_get_masked_pixels_returns_correct_number_of_pixels():
    pixels = get_masked_pixels(make_image(), make_mask())
    assert pixels.shape == (9, 3)


def test_compute_rgb_mean_std_returns_expected_values_on_simple_image():
    features = compute_rgb_mean_std(make_image(), make_mask())
    assert features["mean_r"] == 100.0
    assert features["mean_g"] == 150.0
    assert features["mean_b"] == 200.0
    assert features["std_r"] == 0.0
    assert features["std_g"] == 0.0
    assert features["std_b"] == 0.0


def test_compute_rgb_histogram_returns_expected_number_of_features():
    features = compute_rgb_histogram(make_image(), make_mask(), bins_per_channel=4)
    assert len(features) == 12


def test_histogram_values_sum_to_one_for_each_channel_when_mask_is_not_empty():
    features = compute_rgb_histogram(make_image(), make_mask(), bins_per_channel=4)
    assert sum(features[f"hist_r_{index}"] for index in range(4)) == 1.0
    assert sum(features[f"hist_g_{index}"] for index in range(4)) == 1.0
    assert sum(features[f"hist_b_{index}"] for index in range(4)) == 1.0


def test_compute_gradient_magnitude_preserves_shape():
    gray = np.arange(30, dtype=np.float32).reshape(5, 6)
    gradient = compute_gradient_magnitude(gray)
    assert gradient.shape == gray.shape


def test_extract_texture_features_returns_required_keys():
    gray = np.arange(30, dtype=np.float32).reshape(5, 6)
    features = extract_texture_features(gray, make_mask())
    assert set(features) == {
        "gray_mean_inside_mask",
        "gray_std_inside_mask",
        "gradient_mean_inside_mask",
        "gradient_std_inside_mask",
    }


def test_create_defect_map_returns_binary_mask():
    image = make_image()
    gray = np.full((5, 6), 120, dtype=np.float32)
    gray[2, 3] = 10
    defect_map = create_defect_map(image, gray, make_mask())
    assert defect_map.shape == gray.shape
    assert defect_map.dtype == np.uint8
    assert set(np.unique(defect_map)).issubset({0, 1})


def test_extract_defect_features_returns_defect_area_and_defect_ratio():
    image = make_image()
    gray = np.full((5, 6), 120, dtype=np.float32)
    gray[2, 3] = 10
    features = extract_defect_features(image, gray, make_mask())
    assert "defect_area" in features
    assert "defect_ratio" in features


def test_extract_shape_features_returns_required_keys():
    features = extract_shape_features(make_mask())
    required_keys = {
        "area",
        "perimeter",
        "circularity",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row",
        "bbox_max_col",
        "bbox_height",
        "bbox_width",
        "aspect_ratio",
        "centroid_row",
        "centroid_col",
        "mask_area_ratio",
    }
    assert required_keys.issubset(features.keys())


def test_extract_color_features_returns_required_keys():
    features = extract_color_features(make_image(), make_mask(), bins_per_channel=4)
    required_keys = {
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "hist_r_0",
        "hist_g_0",
        "hist_b_0",
    }
    assert required_keys.issubset(features.keys())


def test_extract_all_features_from_pipeline_result_works_with_synthetic_result():
    pipeline_result = {
        "original_image": make_image(),
        "grayscale": np.full((5, 6), 120, dtype=np.float32),
        "fruit_mask": make_mask(),
        "quality": {
            "brightness": 120.0,
            "contrast": 15.0,
            "noise_level": 2.0,
        },
        "preprocessing_method": "gaussian_filter",
    }
    features = extract_all_features_from_pipeline_result(pipeline_result)
    assert features["area"] == 9.0
    assert features["brightness"] == 120.0
    assert features["contrast"] == 15.0
    assert features["noise_level"] == 2.0
    assert features["preprocessing_method"] == "gaussian_filter"


def test_color_ratio_features_on_synthetic_rgb_images() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, 0] = [220, 20, 20]
    image[:, 1] = [230, 150, 20]
    image[:, 2] = [230, 220, 20]
    image[:, 3] = [20, 180, 40]
    mask = np.ones((2, 4), dtype=np.uint8)

    features = extract_color_features(image, mask, bins_per_channel=4)

    assert features["red_ratio"] == 0.25
    assert features["orange_ratio"] == 0.25
    assert features["yellow_ratio"] == 0.25
    assert features["green_ratio"] == 0.25
    assert "saturation_mean" in features
    assert "saturation_std" in features


def test_highlight_pixels_are_not_counted_as_defects() -> None:
    image = np.full((9, 9, 3), 120, dtype=np.uint8)
    gray = np.full((9, 9), 120, dtype=np.uint8)
    mask = np.ones((9, 9), dtype=np.uint8)
    image[4:6, 4:6] = [255, 255, 255]
    gray[4:6, 4:6] = 255

    defect_map = create_defect_map(image, gray, mask)

    assert int(np.sum(defect_map)) == 0


def test_boundary_only_artifacts_do_not_create_high_defect_ratio() -> None:
    image = np.full((12, 12, 3), 130, dtype=np.uint8)
    gray = np.full((12, 12), 130, dtype=np.uint8)
    mask = np.ones((12, 12), dtype=np.uint8)
    gray[0, :] = 20
    gray[-1, :] = 20
    gray[:, 0] = 20
    gray[:, -1] = 20
    image[gray == 20] = [25, 18, 12]

    features = extract_defect_features(image, gray, mask)

    assert features["defect_ratio"] == 0.0
