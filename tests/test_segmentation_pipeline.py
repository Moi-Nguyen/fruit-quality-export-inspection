import numpy as np

from src.adaptive_pipeline import run_segmentation_pipeline


REQUIRED_KEYS = {
    "original_image",
    "grayscale",
    "quality",
    "preprocessing_method",
    "preprocessed_gray",
    "initial_mask",
    "combined_mask",
    "cleaned_mask",
    "fruit_mask",
    "otsu_threshold",
}


def test_run_segmentation_pipeline_returns_required_keys():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[5:11, 5:11, :] = 220

    results = run_segmentation_pipeline(image)

    assert REQUIRED_KEYS <= set(results.keys())


def test_fruit_mask_has_same_height_and_width_as_input_image():
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    image[3:9, 6:12, :] = 220

    results = run_segmentation_pipeline(image)

    assert results["fruit_mask"].shape == image.shape[:2]

def test_combined_mask_has_same_height_and_width_as_input_image():
    image = np.full((12, 18, 3), 255, dtype=np.uint8)
    image[3:9, 6:12, :] = [230, 190, 30]

    results = run_segmentation_pipeline(image)

    assert results["combined_mask"].shape == image.shape[:2]


def test_segmentation_pipeline_works_on_bright_object():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[6:14, 6:14, :] = 220

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0


def test_segmentation_pipeline_works_on_dark_object_with_bright_background():
    image = np.full((20, 20, 3), 230, dtype=np.uint8)
    image[6:14, 6:14, :] = 40

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0

def test_segmentation_pipeline_works_on_colorful_object_with_white_background():
    image = np.full((24, 24, 3), 255, dtype=np.uint8)
    image[7:17, 8:16] = [240, 190, 20]

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0


def test_synthetic_colored_object_on_white_background_produces_non_empty_fruit_mask():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[10:22, 11:21] = [240, 190, 20]

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0

def test_fruit_mask_area_ratio_is_not_unrealistically_large():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[10:22, 11:21] = [240, 190, 20]

    results = run_segmentation_pipeline(image)
    area_ratio = float(np.sum(results["fruit_mask"])) / results["fruit_mask"].size

    assert 0 < area_ratio < 0.60


def test_synthetic_red_object_on_white_background_produces_non_empty_fruit_mask():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[10:22, 11:21] = [220, 30, 20]

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0

def test_synthetic_yellow_object_on_white_background_produces_non_empty_fruit_mask():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[10:22, 11:21] = [240, 190, 20]

    results = run_segmentation_pipeline(image)

    assert int(np.sum(results["fruit_mask"])) > 0

def test_synthetic_fruit_mask_area_ratio_is_reasonable():
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[12:28, 14:26] = [220, 30, 20]

    results = run_segmentation_pipeline(image)
    area_ratio = float(np.sum(results["fruit_mask"])) / results["fruit_mask"].size

    assert 0.005 <= area_ratio <= 0.80
