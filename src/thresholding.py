"""Manual thresholding algorithms for fruit segmentation."""

import numpy as np

from src.config import (
    BACKGROUND_BORDER_WIDTH,
    BACKGROUND_COLOR_DISTANCE_THRESHOLD,
    BLACK_BORDER_THRESHOLD,
    BORDER_FOREGROUND_RATIO_THRESHOLD,
    COLOR_DIFFERENCE_THRESHOLD,
    MAX_REASONABLE_FRUIT_AREA_RATIO,
    MIN_REASONABLE_FRUIT_AREA_RATIO,
    OTSU_NUM_BINS,
    STRICT_BACKGROUND_COLOR_DISTANCE_THRESHOLD,
    STRICT_COLOR_DIFFERENCE_THRESHOLD,
    WHITE_BACKGROUND_THRESHOLD,
)


def _to_uint8_values(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale values to the range 0 to 255."""
    gray_array = np.asarray(gray, dtype=np.float32)
    if gray_array.ndim != 2:
        raise ValueError("Grayscale image must be a 2D NumPy array.")

    clipped = np.clip(gray_array, 0, 255)
    return np.rint(clipped).astype(np.uint8)


def compute_otsu_threshold(gray: np.ndarray, num_bins: int = OTSU_NUM_BINS) -> float:
    """Compute an Otsu threshold manually from a grayscale image."""
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1.")

    gray_uint8 = _to_uint8_values(gray)
    if gray_uint8.size == 0:
        raise ValueError("Grayscale image must not be empty.")
    if np.min(gray_uint8) == np.max(gray_uint8):
        return float(gray_uint8.flat[0])

    histogram, bin_edges = np.histogram(gray_uint8, bins=num_bins, range=(0, 256))
    probabilities = histogram.astype(np.float64) / gray_uint8.size
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    best_threshold = float(bin_centers[0])
    best_variance = -1.0

    for threshold_index in range(num_bins):
        w0 = np.sum(probabilities[: threshold_index + 1])
        w1 = np.sum(probabilities[threshold_index + 1 :])
        if w0 == 0 or w1 == 0:
            continue

        mu0 = np.sum(probabilities[: threshold_index + 1] * bin_centers[: threshold_index + 1]) / w0
        mu1 = np.sum(probabilities[threshold_index + 1 :] * bin_centers[threshold_index + 1 :]) / w1
        sigma_b_squared = w0 * w1 * (mu0 - mu1) ** 2

        if sigma_b_squared > best_variance:
            best_variance = float(sigma_b_squared)
            best_threshold = float(bin_centers[threshold_index])

    return best_threshold


def apply_threshold(gray: np.ndarray, threshold: float, foreground: str = "bright") -> np.ndarray:
    """Create a binary mask using a threshold."""
    gray_array = np.asarray(gray, dtype=np.float32)
    if foreground == "bright":
        mask = gray_array >= threshold
    elif foreground == "dark":
        mask = gray_array < threshold
    else:
        raise ValueError("foreground must be 'bright' or 'dark'.")

    return mask.astype(np.uint8)


def estimate_border_foreground_ratio(mask: np.ndarray) -> float:
    """Return the fraction of image border pixels that are foreground."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("Mask must be a 2D NumPy array.")
    if mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        raise ValueError("Mask must not be empty.")

    top_row = mask_array[0, :]
    bottom_row = mask_array[-1, :]
    left_column = mask_array[1:-1, 0]
    right_column = mask_array[1:-1, -1]
    border_pixels = np.concatenate([top_row, bottom_row, left_column, right_column])
    return float(np.mean(border_pixels > 0))


def create_otsu_mask(gray: np.ndarray, auto_invert: bool = True) -> np.ndarray:
    """Create a binary Otsu mask and optionally invert bright backgrounds."""
    threshold = compute_otsu_threshold(gray)
    mask = apply_threshold(gray, threshold, foreground="bright")

    if auto_invert and estimate_border_foreground_ratio(mask) > BORDER_FOREGROUND_RATIO_THRESHOLD:
        mask = 1 - mask

    return mask.astype(np.uint8)

def create_non_white_background_mask(
    image: np.ndarray,
    white_threshold: int = WHITE_BACKGROUND_THRESHOLD,
) -> np.ndarray:
    """Detect pixels that are probably not pure white background.

    This mask should not be used alone as a fruit foreground mask because
    shadows, compression artifacts, and rotated white backgrounds can also
    become non-white.
    """
    image_array = np.asarray(image)

    if image_array.ndim == 2:
        background = image_array > white_threshold
    elif image_array.ndim == 3 and image_array.shape[2] >= 3:
        rgb = image_array[:, :, :3]
        background = np.all(rgb > white_threshold, axis=2)
    else:
        raise ValueError("image must be grayscale, RGB, or RGBA.")

    return np.logical_not(background).astype(np.uint8)

def create_color_difference_mask(
    image: np.ndarray,
    color_difference_threshold: int = COLOR_DIFFERENCE_THRESHOLD,
) -> np.ndarray:
    """Detect colorful fruit pixels against white or gray background."""
    image_array = np.asarray(image)

    if image_array.ndim == 2:
        return np.zeros(image_array.shape, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError("image must be grayscale, RGB, or RGBA.")

    rgb = image_array[:, :, :3].astype(np.int16)
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    color_difference = max_channel - min_channel
    return (color_difference >= color_difference_threshold).astype(np.uint8)

def remove_black_border_pixels(
    mask: np.ndarray,
    gray: np.ndarray,
    black_threshold: int = BLACK_BORDER_THRESHOLD,
) -> np.ndarray:
    """Force very dark border-artifact pixels to background in a mask."""
    mask_array = np.asarray(mask, dtype=np.uint8)
    gray_array = np.asarray(gray)
    if mask_array.shape != gray_array.shape:
        raise ValueError("mask and gray must have the same shape.")

    cleaned_mask = mask_array.copy()
    cleaned_mask[gray_array <= black_threshold] = 0
    return (cleaned_mask > 0).astype(np.uint8)

def estimate_background_color_from_border(
    image: np.ndarray,
    gray: np.ndarray,
    border_width: int = BACKGROUND_BORDER_WIDTH,
    black_threshold: int = BLACK_BORDER_THRESHOLD,
) -> np.ndarray:
    """Estimate background RGB color from border pixels, ignoring black artifacts."""
    image_array = np.asarray(image)
    gray_array = np.asarray(gray, dtype=np.float32)
    if gray_array.ndim != 2:
        raise ValueError("gray must be a 2D NumPy array.")

    if image_array.ndim == 2:
        median_gray = float(np.median(image_array.astype(np.float32)))
        return np.array([median_gray, median_gray, median_gray], dtype=np.float32)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError("image must be grayscale, RGB, or RGBA.")
    if image_array.shape[:2] != gray_array.shape:
        raise ValueError("image height and width must match gray.")

    rgb = image_array[:, :, :3].astype(np.float32)
    rows, cols = gray_array.shape
    width = max(1, min(int(border_width), rows, cols))

    top = rgb[:width, :, :]
    bottom = rgb[rows - width :, :, :]
    left = rgb[:, :width, :]
    right = rgb[:, cols - width :, :]
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ])

    top_gray = gray_array[:width, :]
    bottom_gray = gray_array[rows - width :, :]
    left_gray = gray_array[:, :width]
    right_gray = gray_array[:, cols - width :]
    border_gray = np.concatenate([
        top_gray.reshape(-1),
        bottom_gray.reshape(-1),
        left_gray.reshape(-1),
        right_gray.reshape(-1),
    ])

    valid_pixels = border_pixels[border_gray > black_threshold]
    if valid_pixels.size == 0:
        return np.median(rgb.reshape(-1, 3), axis=0).astype(np.float32)
    return np.median(valid_pixels, axis=0).astype(np.float32)

def create_background_distance_mask(
    image: np.ndarray,
    gray: np.ndarray,
    distance_threshold: float = BACKGROUND_COLOR_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Select pixels whose RGB color is sufficiently different from the background."""
    image_array = np.asarray(image)
    gray_array = np.asarray(gray, dtype=np.float32)
    if gray_array.ndim != 2:
        raise ValueError("gray must be a 2D NumPy array.")

    background_color = estimate_background_color_from_border(image_array, gray_array)
    if image_array.ndim == 2:
        rgb = np.repeat(image_array[:, :, np.newaxis], 3, axis=2).astype(np.float32)
    elif image_array.ndim == 3 and image_array.shape[2] >= 3:
        rgb = image_array[:, :, :3].astype(np.float32)
    else:
        raise ValueError("image must be grayscale, RGB, or RGBA.")
    if rgb.shape[:2] != gray_array.shape:
        raise ValueError("image height and width must match gray.")

    color_distance = np.sqrt(np.sum((rgb - background_color) ** 2, axis=2))
    candidate = (color_distance >= distance_threshold) & (gray_array > BLACK_BORDER_THRESHOLD)
    return candidate.astype(np.uint8)

def create_combined_fruit_candidate_mask(
    image: np.ndarray,
    gray: np.ndarray,
    otsu_mask: np.ndarray,
) -> np.ndarray:
    """Create a fruit candidate mask using background color distance."""
    gray_array = np.asarray(gray, dtype=np.float32)
    otsu_array = np.asarray(otsu_mask, dtype=np.uint8)
    if gray_array.shape != otsu_array.shape:
        raise ValueError("gray and otsu_mask must have the same shape.")

    candidate = create_background_distance_mask(
        image,
        gray_array,
        BACKGROUND_COLOR_DISTANCE_THRESHOLD,
    )
    if candidate.shape != otsu_array.shape:
        raise ValueError("image height and width must match gray and otsu_mask.")

    area_ratio = float(np.sum(candidate)) / candidate.size
    if area_ratio > MAX_REASONABLE_FRUIT_AREA_RATIO:
        candidate = create_background_distance_mask(
            image,
            gray_array,
            STRICT_BACKGROUND_COLOR_DISTANCE_THRESHOLD,
        )
        area_ratio = float(np.sum(candidate)) / candidate.size

    if area_ratio < MIN_REASONABLE_FRUIT_AREA_RATIO:
        fallback = (otsu_array > 0) & (gray_array > BLACK_BORDER_THRESHOLD)
        fallback_area_ratio = float(np.sum(fallback)) / fallback.size
        if fallback_area_ratio <= MAX_REASONABLE_FRUIT_AREA_RATIO:
            candidate = fallback

    return candidate.astype(np.uint8)
