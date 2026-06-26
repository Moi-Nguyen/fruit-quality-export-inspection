"""Image quality analysis placeholders for brightness, contrast, and noise."""

import numpy as np


def estimate_brightness(image_array: np.ndarray) -> float:
    """Estimate average image brightness."""
    # TODO: Convert to grayscale manually and compute mean intensity.
    raise NotImplementedError("Brightness estimation is not implemented yet.")


def estimate_contrast(image_array: np.ndarray) -> float:
    """Estimate image contrast."""
    # TODO: Compute standard deviation or another contrast metric manually.
    raise NotImplementedError("Contrast estimation is not implemented yet.")


def estimate_noise(image_array: np.ndarray) -> float:
    """Estimate image noise level."""
    # TODO: Use a manual local-difference or high-frequency residual method.
    raise NotImplementedError("Noise estimation is not implemented yet.")
