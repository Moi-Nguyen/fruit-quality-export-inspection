"""Image input/output helpers using Pillow for file access only."""

from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_supported_image_file(path: Path) -> bool:
    """Return True when the file extension is supported."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def validate_image_array(image: np.ndarray) -> None:
    """Validate that an array has a supported image shape."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image array must not be empty.")

    if image.ndim == 2:
        return

    if image.ndim == 3:
        channel_count = image.shape[2]
        if channel_count in (3, 4):
            return
        raise ValueError("Color image must have 3 RGB channels or 4 RGBA channels.")

    raise ValueError("Image must be either 2D grayscale or 3D RGB/RGBA.")


def load_image(path: Path) -> np.ndarray:
    """Load an image file as a NumPy array."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not is_supported_image_file(path):
        raise ValueError(f"Unsupported image file extension: {path.suffix}")

    with Image.open(path) as pil_image:
        image = np.asarray(pil_image)

    validate_image_array(image)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return image


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert a valid image array to uint8 when needed."""
    validate_image_array(image)

    if image.dtype == np.uint8:
        return image

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image array must contain numeric values.")

    if np.issubdtype(image.dtype, np.floating):
        if np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0:
            return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> None:
    """Save a NumPy image array to disk using Pillow."""
    validate_image_array(image)
    uint8_image = ensure_uint8_image(image)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(uint8_image).save(path)


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 values in the range [0, 1]."""
    validate_image_array(image)

    float_image = image.astype(np.float32)
    if float_image.min() >= 0.0 and float_image.max() <= 1.0:
        return float_image

    return np.clip(float_image, 0.0, 255.0) / 255.0


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a NumPy array."""
    return np.asarray(image)


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert a NumPy image array to a Pillow image."""
    return Image.fromarray(ensure_uint8_image(image_array))


def read_image(image_path: Path) -> Image.Image:
    """Read an image from disk using Pillow for legacy callers."""
    return numpy_to_pil(load_image(image_path))
