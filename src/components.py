"""Manual connected component labeling for binary masks."""

from collections import deque

import numpy as np

from src.config import (
    BORDER_MARGIN,
    DEFAULT_COMPONENT_CONNECTIVITY,
    MAX_REASONABLE_FRUIT_AREA_RATIO,
    MIN_REASONABLE_FRUIT_AREA_RATIO,
)
from src.morphology import validate_binary_mask


def get_neighbor_offsets(connectivity: int = DEFAULT_COMPONENT_CONNECTIVITY) -> list[tuple[int, int]]:
    """Return neighbor offsets for 4-connectivity or 8-connectivity."""
    if connectivity == 4:
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8.")


def connected_component_labeling(mask: np.ndarray, connectivity: int = DEFAULT_COMPONENT_CONNECTIVITY) -> tuple[np.ndarray, int]:
    """Label foreground components in a binary mask using BFS."""
    validate_binary_mask(mask)
    binary_mask = mask.astype(np.uint8)
    labels = np.zeros(binary_mask.shape, dtype=np.int32)
    neighbor_offsets = get_neighbor_offsets(connectivity)
    current_label = 0
    rows, cols = binary_mask.shape

    for row in range(rows):
        for col in range(cols):
            if binary_mask[row, col] == 0 or labels[row, col] != 0:
                continue

            current_label += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = current_label

            while queue:
                current_row, current_col = queue.popleft()
                for row_offset, col_offset in neighbor_offsets:
                    next_row = current_row + row_offset
                    next_col = current_col + col_offset
                    inside_image = 0 <= next_row < rows and 0 <= next_col < cols

                    if not inside_image:
                        continue
                    if binary_mask[next_row, next_col] == 0:
                        continue
                    if labels[next_row, next_col] != 0:
                        continue

                    labels[next_row, next_col] = current_label
                    queue.append((next_row, next_col))

    return labels, current_label


def compute_component_stats(labels: np.ndarray, num_components: int) -> list[dict[str, int]]:
    """Compute area and bounding box information for each component."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D NumPy array.")

    stats = []
    for label_value in range(1, num_components + 1):
        component_rows, component_cols = np.where(labels == label_value)
        if component_rows.size == 0:
            continue

        min_row = int(np.min(component_rows))
        max_row = int(np.max(component_rows))
        min_col = int(np.min(component_cols))
        max_col = int(np.max(component_cols))
        stats.append(
            {
                "label": int(label_value),
                "area": int(component_rows.size),
                "min_row": min_row,
                "min_col": min_col,
                "max_row": max_row,
                "max_col": max_col,
                "height": int(max_row - min_row + 1),
                "width": int(max_col - min_col + 1),
            }
        )

    return stats


def keep_largest_component(mask: np.ndarray, connectivity: int = DEFAULT_COMPONENT_CONNECTIVITY) -> np.ndarray:
    """Keep only the largest foreground component in a binary mask."""
    labels, num_components = connected_component_labeling(mask, connectivity)
    if num_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    stats = compute_component_stats(labels, num_components)
    largest = max(stats, key=lambda component: component["area"])
    return (labels == largest["label"]).astype(np.uint8)


def keep_largest_non_border_component(
    mask: np.ndarray,
    connectivity: int = DEFAULT_COMPONENT_CONNECTIVITY,
    border_margin: int = BORDER_MARGIN,
) -> np.ndarray:
    """Keep the largest component that does not touch the image border."""
    labels, num_components = connected_component_labeling(mask, connectivity)
    if num_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    rows, cols = mask.shape
    stats = compute_component_stats(labels, num_components)
    non_border_components = []

    for component in stats:
        touches_border = (
            component["min_row"] <= border_margin
            or component["min_col"] <= border_margin
            or component["max_row"] >= rows - 1 - border_margin
            or component["max_col"] >= cols - 1 - border_margin
        )
        if not touches_border:
            non_border_components.append(component)

    if len(non_border_components) == 0:
        return keep_largest_component(mask, connectivity=connectivity)

    largest = max(non_border_components, key=lambda component: component["area"])
    return (labels == largest["label"]).astype(np.uint8)

def keep_largest_reasonable_component(
    mask: np.ndarray,
    connectivity: int = DEFAULT_COMPONENT_CONNECTIVITY,
    min_area_ratio: float = MIN_REASONABLE_FRUIT_AREA_RATIO,
    max_area_ratio: float = MAX_REASONABLE_FRUIT_AREA_RATIO,
) -> np.ndarray:
    """Keep the largest component with a reasonable area ratio."""
    labels, num_components = connected_component_labeling(mask, connectivity)
    if num_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    total_pixels = mask.size
    stats = compute_component_stats(labels, num_components)
    valid_components = []
    for component in stats:
        area_ratio = float(component["area"]) / total_pixels
        if min_area_ratio <= area_ratio <= max_area_ratio:
            valid_components.append(component)

    if len(valid_components) == 0:
        return keep_largest_component(mask, connectivity=connectivity)

    largest = max(valid_components, key=lambda component: component["area"])
    return (labels == largest["label"]).astype(np.uint8)

def connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Legacy wrapper for connected component labeling."""
    return connected_component_labeling(binary_mask)


def largest_component(label_image: np.ndarray) -> np.ndarray:
    """Legacy wrapper that returns the largest label as a binary mask."""
    if label_image.ndim != 2:
        raise ValueError("label_image must be a 2D NumPy array.")
    num_components = int(np.max(label_image))
    if num_components == 0:
        return np.zeros_like(label_image, dtype=np.uint8)
    stats = compute_component_stats(label_image, num_components)
    largest = max(stats, key=lambda component: component["area"])
    return (label_image == largest["label"]).astype(np.uint8)
