"""Visualization helpers using matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.io_utils import ensure_uint8_image


def save_quality_analysis_figure(
    original: np.ndarray,
    grayscale: np.ndarray,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Save original and grayscale images side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    if title is not None:
        figure.suptitle(title)

    axes[0].imshow(ensure_uint8_image(original), cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_preprocessing_figure(
    original: np.ndarray,
    grayscale: np.ndarray,
    preprocessed: np.ndarray,
    output_path: Path,
    method_name: str,
    title: str | None = None,
) -> None:
    """Save original, grayscale, and preprocessed images side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    if title is not None:
        figure.suptitle(title)

    axes[0].imshow(ensure_uint8_image(original), cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(preprocessed, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(method_name)
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_parameter_sweep_figure(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    main_title: str | None = None,
) -> None:
    """Save grayscale parameter sweep results side by side."""
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length.")
    if len(images) == 0:
        raise ValueError("At least one image is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    if main_title is not None:
        figure.suptitle(main_title)

    for axis, image, title_text in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title_text)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_mask_figure(mask: np.ndarray, output_path: Path) -> None:
    """Save a segmentation mask visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, mask, cmap="gray")


def save_defect_map_figure(defect_map: np.ndarray, output_path: Path) -> None:
    """Save a defect map visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, defect_map, cmap="hot")
