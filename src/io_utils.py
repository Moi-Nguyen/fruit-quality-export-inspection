"""Image input/output helpers using Pillow for file access only."""

from pathlib import Path

import numpy as np
from PIL import Image


def read_image(image_path: Path) -> Image.Image:
    """Read an image from disk using Pillow."""
    # TODO: Add validation for supported image formats.
    return Image.open(image_path).convert("RGB")


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save a Pillow image to disk."""
    # TODO: Ensure parent directories exist before saving.
    image.save(output_path)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a NumPy array."""
    # TODO: Standardize array dtype and scaling policy.
    return np.asarray(image)


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert a NumPy image array to a Pillow image."""
    # TODO: Clip values safely and support grayscale arrays.
    return Image.fromarray(image_array.astype(np.uint8))
