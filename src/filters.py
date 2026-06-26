"""Manual NumPy filter placeholders."""

import numpy as np


def gaussian_filter(image_array: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Apply a manually implemented gaussian filter."""
    # TODO: Build gaussian kernel and convolve manually without OpenCV.
    raise NotImplementedError("Gaussian filtering is not implemented yet.")


def median_filter(image_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a manually implemented median filter."""
    # TODO: Slide a window over the image and compute median values manually.
    raise NotImplementedError("Median filtering is not implemented yet.")
