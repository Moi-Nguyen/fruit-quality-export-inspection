"""Visualization helper placeholders using matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_mask_figure(mask: np.ndarray, output_path: Path) -> None:
    """Save a segmentation mask visualization."""
    # TODO: Add titles, color maps, and consistent figure formatting.
    plt.imsave(output_path, mask, cmap="gray")


def save_defect_map_figure(defect_map: np.ndarray, output_path: Path) -> None:
    """Save a defect map visualization."""
    # TODO: Create clearer overlays when defect detection is implemented.
    plt.imsave(output_path, defect_map, cmap="hot")
