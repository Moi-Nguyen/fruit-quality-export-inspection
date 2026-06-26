"""Manual histogram computation and equalization placeholders."""

import numpy as np


def compute_histogram(gray_image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute a grayscale histogram manually."""
    # TODO: Count pixel intensities without using OpenCV histogram functions.
    raise NotImplementedError("Histogram computation is not implemented yet.")


def histogram_equalization(gray_image: np.ndarray) -> np.ndarray:
    """Apply manual histogram equalization to a grayscale image."""
    # TODO: Compute CDF and remap intensities manually.
    raise NotImplementedError("Histogram equalization is not implemented yet.")
