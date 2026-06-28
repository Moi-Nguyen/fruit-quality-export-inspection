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

def save_feature_extraction_figure(
    original: np.ndarray,
    fruit_mask: np.ndarray,
    defect_map: np.ndarray,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Save original image, fruit mask, and defect map side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    if title is not None:
        figure.suptitle(title)

    axes[0].imshow(ensure_uint8_image(original), cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(fruit_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Fruit mask")
    axes[1].axis("off")

    axes[2].imshow(defect_map, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Defect map")
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)



def save_prediction_figure(
    original: np.ndarray,
    fruit_mask: np.ndarray,
    defect_map: np.ndarray,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Save original image, fruit mask, and defect map for prediction."""
    save_feature_extraction_figure(
        original=original,
        fruit_mask=fruit_mask,
        defect_map=defect_map,
        output_path=output_path,
        title=title,
    )

def save_segmentation_figure(
    original: np.ndarray,
    grayscale: np.ndarray,
    preprocessed: np.ndarray,
    initial_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    fruit_mask: np.ndarray,
    output_path: Path,
    combined_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Save the main segmentation stages in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if combined_mask is None:
        images = [original, grayscale, preprocessed, initial_mask, cleaned_mask, fruit_mask]
        titles = [
            "Original",
            "Grayscale",
            "Preprocessed",
            "Initial Otsu Mask",
            "Cleaned Mask",
            "Fruit Mask",
        ]
    else:
        images = [original, grayscale, preprocessed, initial_mask, combined_mask, cleaned_mask, fruit_mask]
        titles = [
            "Original",
            "Grayscale",
            "Preprocessed",
            "Initial Otsu Mask",
            "Combined Candidate Mask",
            "Cleaned Mask",
            "Fruit Mask",
        ]

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    flat_axes = axes.ravel()
    if title is not None:
        figure.suptitle(title)

    for axis, image, image_title in zip(flat_axes, images, titles):
        if image is original:
            axis.imshow(ensure_uint8_image(image), cmap="gray" if image.ndim == 2 else None)
        elif "Mask" in image_title:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(image_title)
        axis.axis("off")

    for axis in flat_axes[len(images) :]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
