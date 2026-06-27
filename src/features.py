"""Handcrafted feature extraction for fruit images."""

from __future__ import annotations

import math

import numpy as np

from src.adaptive_pipeline import run_segmentation_pipeline


def validate_mask(mask: np.ndarray) -> None:
    """Validate that a mask is a 2D binary NumPy array."""
    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be a NumPy array.")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    unique_values = np.unique(mask)
    valid_values = {0, 1, False, True}
    for value in unique_values:
        if value not in valid_values:
            raise ValueError("Mask values must be binary: 0/1 or False/True.")


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a validated boolean mask."""
    validate_mask(mask)
    return mask.astype(bool)


def compute_area(mask: np.ndarray) -> int:
    """Count foreground pixels in a binary mask."""
    binary_mask = _binary_mask(mask)
    return int(np.sum(binary_mask))


def compute_bounding_box(mask: np.ndarray) -> dict[str, int]:
    """Compute the bounding box of the foreground region."""
    binary_mask = _binary_mask(mask)
    rows, cols = np.where(binary_mask)

    if rows.size == 0:
        return {
            "min_row": 0,
            "min_col": 0,
            "max_row": 0,
            "max_col": 0,
            "height": 0,
            "width": 0,
        }

    min_row = int(np.min(rows))
    max_row = int(np.max(rows))
    min_col = int(np.min(cols))
    max_col = int(np.max(cols))

    return {
        "min_row": min_row,
        "min_col": min_col,
        "max_row": max_row,
        "max_col": max_col,
        "height": max_row - min_row + 1,
        "width": max_col - min_col + 1,
    }


def compute_aspect_ratio(mask: np.ndarray) -> float:
    """Compute width divided by height for the mask bounding box."""
    bounding_box = compute_bounding_box(mask)
    height = bounding_box["height"]
    if height == 0:
        return 0.0
    return float(bounding_box["width"] / height)


def compute_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute the center of foreground pixels as row and column."""
    binary_mask = _binary_mask(mask)
    rows, cols = np.where(binary_mask)

    if rows.size == 0:
        return (0.0, 0.0)

    return (float(np.mean(rows)), float(np.mean(cols)))


def compute_perimeter(mask: np.ndarray) -> int:
    """Estimate perimeter by counting foreground boundary pixels."""
    binary_mask = _binary_mask(mask)
    if binary_mask.size == 0:
        return 0

    padded_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=False)
    center = padded_mask[1:-1, 1:-1]
    up = padded_mask[:-2, 1:-1]
    down = padded_mask[2:, 1:-1]
    left = padded_mask[1:-1, :-2]
    right = padded_mask[1:-1, 2:]

    boundary_pixels = center & (~up | ~down | ~left | ~right)
    return int(np.sum(boundary_pixels))


def compute_circularity(area: int, perimeter: int) -> float:
    """Compute circularity from area and perimeter."""
    if perimeter <= 0:
        return 0.0
    circularity = 4.0 * math.pi * area / (perimeter ** 2)
    return float(min(max(circularity, 0.0), 1.0))


def compute_mask_area_ratio(mask: np.ndarray) -> float:
    """Compute foreground area divided by total image pixels."""
    binary_mask = _binary_mask(mask)
    if binary_mask.size == 0:
        return 0.0
    return float(np.sum(binary_mask) / binary_mask.size)


def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    """Extract shape features from a binary fruit mask."""
    area = compute_area(mask)
    perimeter = compute_perimeter(mask)
    bounding_box = compute_bounding_box(mask)
    centroid_row, centroid_col = compute_centroid(mask)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": compute_circularity(area, perimeter),
        "bbox_min_row": float(bounding_box["min_row"]),
        "bbox_min_col": float(bounding_box["min_col"]),
        "bbox_max_row": float(bounding_box["max_row"]),
        "bbox_max_col": float(bounding_box["max_col"]),
        "bbox_height": float(bounding_box["height"]),
        "bbox_width": float(bounding_box["width"]),
        "aspect_ratio": compute_aspect_ratio(mask),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "mask_area_ratio": compute_mask_area_ratio(mask),
    }


def _validate_color_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image must be an RGB or RGBA array.")


def get_masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return RGB pixels inside the fruit mask."""
    _validate_color_image(image)
    binary_mask = _binary_mask(mask)

    if image.shape[:2] != binary_mask.shape:
        raise ValueError("Image and mask must have the same height and width.")

    rgb_image = image[:, :, :3]
    if np.sum(binary_mask) == 0:
        return np.empty((0, 3), dtype=rgb_image.dtype)
    return rgb_image[binary_mask]


def compute_rgb_mean_std(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute RGB mean and standard deviation inside the mask."""
    pixels = get_masked_pixels(image, mask)
    feature_names = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]

    if pixels.shape[0] == 0:
        return {name: 0.0 for name in feature_names}

    pixels_float = pixels.astype(np.float32)
    means = np.mean(pixels_float, axis=0)
    stds = np.std(pixels_float, axis=0)

    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
    }


def _channel_histogram(values: np.ndarray, bins_per_channel: int) -> np.ndarray:
    bin_indices = np.floor(values.astype(np.float32) * bins_per_channel / 256.0).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bins_per_channel - 1)
    counts = np.bincount(bin_indices, minlength=bins_per_channel).astype(np.float32)
    return counts / counts.sum()


def compute_rgb_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    bins_per_channel: int = 8,
) -> dict[str, float]:
    """Compute normalized RGB histograms inside the mask."""
    if bins_per_channel <= 0:
        raise ValueError("bins_per_channel must be positive.")

    pixels = get_masked_pixels(image, mask)
    features: dict[str, float] = {}
    channel_names = ["r", "g", "b"]

    for channel_index, channel_name in enumerate(channel_names):
        if pixels.shape[0] == 0:
            histogram = np.zeros(bins_per_channel, dtype=np.float32)
        else:
            histogram = _channel_histogram(pixels[:, channel_index], bins_per_channel)

        for bin_index, value in enumerate(histogram):
            features[f"hist_{channel_name}_{bin_index}"] = float(value)

    return features


def extract_color_features(
    image: np.ndarray,
    mask: np.ndarray,
    bins_per_channel: int = 8,
) -> dict[str, float]:
    """Extract simple color statistics and histograms."""
    features = compute_rgb_mean_std(image, mask)
    features.update(compute_rgb_histogram(image, mask, bins_per_channel=bins_per_channel))
    return features


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using simple finite differences."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Grayscale image must be a 2D NumPy array.")

    gray_float = gray.astype(np.float32)
    gx = np.zeros_like(gray_float, dtype=np.float32)
    gy = np.zeros_like(gray_float, dtype=np.float32)

    if gray.shape[1] >= 3:
        gx[:, 1:-1] = gray_float[:, 2:] - gray_float[:, :-2]
    if gray.shape[0] >= 3:
        gy[1:-1, :] = gray_float[2:, :] - gray_float[:-2, :]

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gradient_magnitude.astype(np.float32)


def extract_texture_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Extract grayscale and gradient texture features inside the mask."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Grayscale image must be a 2D NumPy array.")

    binary_mask = _binary_mask(mask)
    if gray.shape != binary_mask.shape:
        raise ValueError("Grayscale image and mask must have the same shape.")

    feature_names = [
        "gray_mean_inside_mask",
        "gray_std_inside_mask",
        "gradient_mean_inside_mask",
        "gradient_std_inside_mask",
    ]
    if np.sum(binary_mask) == 0:
        return {name: 0.0 for name in feature_names}

    gray_values = gray.astype(np.float32)[binary_mask]
    gradient = compute_gradient_magnitude(gray)
    gradient_values = gradient[binary_mask]

    return {
        "gray_mean_inside_mask": float(np.mean(gray_values)),
        "gray_std_inside_mask": float(np.std(gray_values)),
        "gradient_mean_inside_mask": float(np.mean(gradient_values)),
        "gradient_std_inside_mask": float(np.std(gradient_values)),
    }


def create_defect_map(image: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a simple heuristic defect map inside the fruit region."""
    _validate_color_image(image)
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Grayscale image must be a 2D NumPy array.")

    binary_mask = _binary_mask(mask)
    if image.shape[:2] != binary_mask.shape or gray.shape != binary_mask.shape:
        raise ValueError("Image, grayscale image, and mask must have matching height and width.")

    defect_map = np.zeros(binary_mask.shape, dtype=np.uint8)
    if np.sum(binary_mask) == 0:
        return defect_map

    gray_float = gray.astype(np.float32)
    gray_inside = gray_float[binary_mask]
    q25 = float(np.percentile(gray_inside, 25))
    q75 = float(np.percentile(gray_inside, 75))
    iqr = q75 - q25

    rgb_image = image[:, :, :3].astype(np.float32)
    color_difference = np.max(rgb_image, axis=2) - np.min(rgb_image, axis=2)

    dark_defects = gray_float < (q25 - 0.5 * iqr)
    bright_low_color_defects = (gray_float > (q75 + 0.5 * iqr)) & (color_difference < 30)
    defects_inside_mask = binary_mask & (dark_defects | bright_low_color_defects)

    defect_map[defects_inside_mask] = 1
    return defect_map


def extract_defect_features(
    image: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Extract defect area and defect ratio features."""
    binary_mask = _binary_mask(mask)
    fruit_area = int(np.sum(binary_mask))
    if fruit_area == 0:
        return {"defect_area": 0.0, "defect_ratio": 0.0}

    defect_map = create_defect_map(image, gray, mask)
    defect_area = int(np.sum(defect_map))

    return {
        "defect_area": float(defect_area),
        "defect_ratio": float(defect_area / fruit_area),
    }


def extract_all_features_from_pipeline_result(
    pipeline_result: dict[str, object],
) -> dict[str, float | str]:
    """Extract all handcrafted features from a segmentation pipeline result."""
    original_image = pipeline_result["original_image"]
    grayscale = pipeline_result["grayscale"]
    fruit_mask = pipeline_result["fruit_mask"]
    quality = pipeline_result["quality"]
    preprocessing_method = pipeline_result["preprocessing_method"]

    if not isinstance(original_image, np.ndarray):
        raise ValueError("pipeline_result['original_image'] must be a NumPy array.")
    if not isinstance(grayscale, np.ndarray):
        raise ValueError("pipeline_result['grayscale'] must be a NumPy array.")
    if not isinstance(fruit_mask, np.ndarray):
        raise ValueError("pipeline_result['fruit_mask'] must be a NumPy array.")
    if not isinstance(quality, dict):
        raise ValueError("pipeline_result['quality'] must be a dictionary.")

    features: dict[str, float | str] = {}
    features.update(extract_shape_features(fruit_mask))
    features.update(extract_color_features(original_image, fruit_mask))
    features.update(extract_texture_features(grayscale, fruit_mask))
    features.update(extract_defect_features(original_image, grayscale, fruit_mask))

    features["brightness"] = float(quality.get("brightness", 0.0))
    features["contrast"] = float(quality.get("contrast", 0.0))
    features["noise_level"] = float(quality.get("noise_level", 0.0))
    features["preprocessing_method"] = str(preprocessing_method)

    return features


def extract_features_from_image(image: np.ndarray) -> dict[str, float | str]:
    """Run segmentation and extract all handcrafted features from an image."""
    pipeline_result = run_segmentation_pipeline(image)
    return extract_all_features_from_pipeline_result(pipeline_result)


def extract_all_features(image_array: np.ndarray, binary_mask: np.ndarray) -> dict[str, float]:
    """Legacy helper for extracting shape and color features from image and mask."""
    features = extract_shape_features(binary_mask)
    features.update(extract_color_features(image_array, binary_mask))
    return features
