"""Manual binary morphology operations using NumPy."""

import numpy as np

from src.config import DEFAULT_MORPH_KERNEL_SIZE, DEFAULT_MORPH_SHAPE


def validate_binary_mask(mask: np.ndarray) -> None:
    """Validate that a mask is 2D and contains only binary values."""
    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be a NumPy array.")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D NumPy array.")
    if mask.size == 0:
        raise ValueError("Mask must not be empty.")

    unique_values = np.unique(mask)
    valid_values = {0, 1, False, True}
    if not all(value in valid_values for value in unique_values.tolist()):
        raise ValueError("Mask must contain only 0/1 or False/True values.")


def create_structuring_element(kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE, shape: str = DEFAULT_MORPH_SHAPE) -> np.ndarray:
    """Create a square or cross structuring element."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    if shape == "square":
        return np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if shape == "cross":
        element = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        element[center, :] = 1
        element[:, center] = 1
        return element
    raise ValueError("shape must be 'square' or 'cross'.")


def erode_binary(mask: np.ndarray, kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE, shape: str = DEFAULT_MORPH_SHAPE) -> np.ndarray:
    """Erode a binary mask manually with zero padding."""
    validate_binary_mask(mask)
    binary_mask = mask.astype(np.uint8)
    element = create_structuring_element(kernel_size, shape)
    padding = kernel_size // 2
    padded = np.pad(binary_mask, padding, mode="constant", constant_values=0)
    eroded = np.zeros_like(binary_mask, dtype=np.uint8)
    active_positions = element == 1

    for row in range(binary_mask.shape[0]):
        for col in range(binary_mask.shape[1]):
            window = padded[row : row + kernel_size, col : col + kernel_size]
            if np.all(window[active_positions] == 1):
                eroded[row, col] = 1

    return eroded


def dilate_binary(mask: np.ndarray, kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE, shape: str = DEFAULT_MORPH_SHAPE) -> np.ndarray:
    """Dilate a binary mask manually with zero padding."""
    validate_binary_mask(mask)
    binary_mask = mask.astype(np.uint8)
    element = create_structuring_element(kernel_size, shape)
    padding = kernel_size // 2
    padded = np.pad(binary_mask, padding, mode="constant", constant_values=0)
    dilated = np.zeros_like(binary_mask, dtype=np.uint8)
    active_positions = element == 1

    for row in range(binary_mask.shape[0]):
        for col in range(binary_mask.shape[1]):
            window = padded[row : row + kernel_size, col : col + kernel_size]
            if np.any(window[active_positions] == 1):
                dilated[row, col] = 1

    return dilated


def opening_binary(mask: np.ndarray, kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE, shape: str = DEFAULT_MORPH_SHAPE) -> np.ndarray:
    """Apply erosion followed by dilation."""
    return dilate_binary(erode_binary(mask, kernel_size, shape), kernel_size, shape)


def closing_binary(mask: np.ndarray, kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE, shape: str = DEFAULT_MORPH_SHAPE) -> np.ndarray:
    """Apply dilation followed by erosion."""
    return erode_binary(dilate_binary(mask, kernel_size, shape), kernel_size, shape)


def clean_binary_mask(mask: np.ndarray, kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE) -> np.ndarray:
    """Remove small noise and fill small holes in a binary mask."""
    opened = opening_binary(mask, kernel_size=kernel_size, shape=DEFAULT_MORPH_SHAPE)
    closed = closing_binary(opened, kernel_size=kernel_size, shape=DEFAULT_MORPH_SHAPE)
    return closed.astype(np.uint8)

def fill_holes_binary(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Fill background holes that are fully surrounded by foreground."""
    validate_binary_mask(mask)
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")

    binary_mask = mask.astype(np.uint8)
    rows, cols = binary_mask.shape
    background = binary_mask == 0
    border_background = np.zeros_like(background, dtype=bool)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    stack: list[tuple[int, int]] = []
    for col in range(cols):
        if background[0, col]:
            stack.append((0, col))
        if background[rows - 1, col]:
            stack.append((rows - 1, col))
    for row in range(1, rows - 1):
        if background[row, 0]:
            stack.append((row, 0))
        if background[row, cols - 1]:
            stack.append((row, cols - 1))

    while stack:
        row, col = stack.pop()
        if border_background[row, col]:
            continue
        border_background[row, col] = True

        for row_offset, col_offset in neighbors:
            next_row = row + row_offset
            next_col = col + col_offset
            if 0 <= next_row < rows and 0 <= next_col < cols:
                if background[next_row, next_col] and not border_background[next_row, next_col]:
                    stack.append((next_row, next_col))

    holes = background & ~border_background
    filled = binary_mask.copy()
    filled[holes] = 1
    return filled.astype(np.uint8)
