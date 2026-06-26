"""Manual thresholding placeholders."""

import numpy as np


def rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to grayscale manually."""
    # TODO: Apply weighted RGB conversion using NumPy operations.
    raise NotImplementedError("RGB to grayscale conversion is not implemented yet.")


def otsu_threshold(gray_image: np.ndarray) -> int:
    """Compute an Otsu threshold manually."""
    # TODO: Search thresholds and maximize between-class variance.
    raise NotImplementedError("Otsu thresholding is not implemented yet.")


def apply_threshold(gray_image: np.ndarray, threshold: int) -> np.ndarray:
    """Create a binary mask from a grayscale image and threshold."""
    # TODO: Return a boolean or 0/1 binary mask consistently.
    raise NotImplementedError("Threshold application is not implemented yet.")
