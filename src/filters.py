"""Manual NumPy filtering functions for grayscale images."""

import numpy as np


def validate_kernel_size(kernel_size: int) -> None:
    """Check that a kernel size is a positive odd integer."""
    if not isinstance(kernel_size, int):
        raise ValueError("Kernel size must be an integer.")
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive.")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")


def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel using NumPy."""
    validate_kernel_size(kernel_size)
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    center = kernel_size // 2
    axis_values = np.arange(-center, center + 1, dtype=np.float32)
    x_values, y_values = np.meshgrid(axis_values, axis_values)

    squared_distance = x_values**2 + y_values**2
    kernel = np.exp(-squared_distance / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def convolve2d_gray(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply manual 2D filtering to a grayscale image using edge padding."""
    if image.ndim != 2:
        raise ValueError("convolve2d_gray expects a 2D grayscale image.")
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square.")

    kernel_size = kernel.shape[0]
    validate_kernel_size(kernel_size)

    gray = image.astype(np.float32)
    kernel_float = kernel.astype(np.float32)
    padding = kernel_size // 2
    padded = np.pad(gray, pad_width=padding, mode="edge")
    filtered = np.zeros_like(gray, dtype=np.float32)

    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            image_slice = padded[
                row_offset : row_offset + gray.shape[0],
                col_offset : col_offset + gray.shape[1],
            ]
            filtered += kernel_float[row_offset, col_offset] * image_slice

    return filtered.astype(np.float32)


def gaussian_filter_gray(
    gray: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Smooth a grayscale image with a manual Gaussian filter."""
    kernel = create_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    return convolve2d_gray(gray, kernel)


def median_filter_gray(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove impulse noise using a manual median filter."""
    validate_kernel_size(kernel_size)
    if gray.ndim != 2:
        raise ValueError("median_filter_gray expects a 2D grayscale image.")

    gray_float = gray.astype(np.float32)
    padding = kernel_size // 2
    padded = np.pad(gray_float, pad_width=padding, mode="edge")
    filtered = np.zeros_like(gray_float, dtype=np.float32)

    for row in range(gray_float.shape[0]):
        for col in range(gray_float.shape[1]):
            window = padded[row : row + kernel_size, col : col + kernel_size]
            filtered[row, col] = np.median(window)

    return filtered.astype(np.float32)


def gaussian_filter(image_array: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Legacy wrapper for Gaussian grayscale filtering."""
    return gaussian_filter_gray(image_array, kernel_size=kernel_size, sigma=sigma)


def median_filter(image_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Legacy wrapper for median grayscale filtering."""
    return median_filter_gray(image_array, kernel_size=kernel_size)
