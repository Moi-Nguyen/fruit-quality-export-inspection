"""Feature extraction placeholders for fruit images and masks."""

from typing import Any

import numpy as np


def extract_shape_features(binary_mask: np.ndarray) -> dict[str, float]:
    """Extract shape features from a binary fruit mask."""
    # TODO: Compute area, perimeter, circularity, bounding box, and aspect ratio.
    raise NotImplementedError("Shape feature extraction is not implemented yet.")


def extract_color_features(image_array: np.ndarray, binary_mask: np.ndarray) -> dict[str, float]:
    """Extract color features from the fruit region."""
    # TODO: Compute manual color histograms and summary color statistics.
    raise NotImplementedError("Color feature extraction is not implemented yet.")


def extract_defect_features(image_array: np.ndarray, binary_mask: np.ndarray) -> dict[str, float]:
    """Extract defect-related features from the fruit region."""
    # TODO: Estimate dark spots or abnormal color regions as defect ratio.
    raise NotImplementedError("Defect feature extraction is not implemented yet.")


def extract_all_features(image_array: np.ndarray, binary_mask: np.ndarray) -> dict[str, Any]:
    """Extract all planned features for machine learning and export rules."""
    # TODO: Combine quality, shape, color, and defect features into one dictionary.
    raise NotImplementedError("Full feature extraction is not implemented yet.")
