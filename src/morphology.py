"""Manual binary morphology placeholders."""

import numpy as np


def erode(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply manual binary erosion."""
    # TODO: Implement erosion with a sliding structuring element.
    raise NotImplementedError("Erosion is not implemented yet.")


def dilate(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply manual binary dilation."""
    # TODO: Implement dilation with a sliding structuring element.
    raise NotImplementedError("Dilation is not implemented yet.")


def opening(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply manual binary opening."""
    # TODO: Combine erosion followed by dilation.
    raise NotImplementedError("Opening is not implemented yet.")


def closing(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply manual binary closing."""
    # TODO: Combine dilation followed by erosion.
    raise NotImplementedError("Closing is not implemented yet.")
