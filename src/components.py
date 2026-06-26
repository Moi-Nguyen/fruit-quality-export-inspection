"""Manual connected component labeling placeholders."""

import numpy as np


def connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components in a binary mask manually."""
    # TODO: Implement flood fill or two-pass connected component labeling.
    raise NotImplementedError("Connected component labeling is not implemented yet.")


def largest_component(label_image: np.ndarray) -> np.ndarray:
    """Return a mask for the largest labeled component."""
    # TODO: Count labels and select the largest non-background component.
    raise NotImplementedError("Largest component selection is not implemented yet.")
