"""Entry point for the fruit quality inspection project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.adaptive_pipeline import run_preprocessing_pipeline, run_segmentation_pipeline
from src.config import FIGURES_DIR
from src.dataset_sampling import check_datasets, create_sample_dataset
from src.features import create_defect_map, extract_all_features_from_pipeline_result
from src.io_utils import load_image
from src.quality_analysis import analyze_image_quality, rgb_to_grayscale
from src.visualization import (
    save_feature_extraction_figure,
    save_preprocessing_figure,
    save_quality_analysis_figure,
    save_segmentation_figure,
)

STARTUP_MESSAGE = "Fruit Quality Inspection project skeleton is ready."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for project utility commands."""
    parser = argparse.ArgumentParser(description="Fruit quality inspection project utilities.")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create a balanced sampled dataset under data/sample/.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Print class image counts for data/raw/ and data/sample/.",
    )
    parser.add_argument(
        "--analyze-image",
        type=Path,
        help="Analyze brightness, contrast, and noise for one image.",
    )
    parser.add_argument(
        "--preprocess-image",
        type=Path,
        help="Run adaptive preprocessing for one image and save a figure.",
    )
    parser.add_argument(
        "--segment-image",
        type=Path,
        help="Run fruit segmentation for one image and save a figure.",
    )
    parser.add_argument(
        "--extract-features",
        type=Path,
        help="Extract handcrafted features for one image and save a figure.",
    )
    return parser.parse_args()


def run_image_quality_analysis(image_path: Path) -> None:
    """Run Step 1 image loading and quality analysis for one image."""
    image = load_image(image_path)
    grayscale = rgb_to_grayscale(image)
    results = analyze_image_quality(image)

    figure_name = f"quality_analysis_{image_path.stem}.png"
    figure_path = FIGURES_DIR / figure_name
    save_quality_analysis_figure(
        original=image,
        grayscale=grayscale,
        output_path=figure_path,
        title=image_path.name,
    )

    print("Image quality analysis:")
    print(f"- Brightness: {results['brightness']:.2f}")
    print(f"- Contrast: {results['contrast']:.2f}")
    print(f"- Noise level: {results['noise_level']:.2f}")
    print(f"- Quality label: {results['quality_label']}")
    print(f"Saved figure: {figure_path}")


def run_image_preprocessing(image_path: Path) -> None:
    """Run Step 2 adaptive preprocessing for one image."""
    image = load_image(image_path)
    results = run_preprocessing_pipeline(image)
    quality = results["quality"]

    figure_name = f"preprocessing_{image_path.stem}.png"
    figure_path = FIGURES_DIR / figure_name
    save_preprocessing_figure(
        original=results["original_image"],
        grayscale=results["grayscale"],
        preprocessed=results["preprocessed_gray"],
        output_path=figure_path,
        method_name=str(results["preprocessing_method"]),
        title=image_path.name,
    )

    print("Preprocessing analysis:")
    print(f"- Brightness: {quality['brightness']:.2f}")
    print(f"- Contrast: {quality['contrast']:.2f}")
    print(f"- Noise level: {quality['noise_level']:.2f}")
    print(f"- Quality label: {quality['quality_label']}")
    print(f"- Selected method: {results['preprocessing_method']}")
    print(f"Saved figure: {figure_path}")


def run_image_segmentation(image_path: Path) -> None:
    """Run Step 3 segmentation for one image."""
    image = load_image(image_path)
    results = run_segmentation_pipeline(image)
    quality = results["quality"]
    fruit_mask = results["fruit_mask"]
    combined_mask = results["combined_mask"]

    figure_name = f"segmentation_{image_path.stem}.png"
    figure_path = FIGURES_DIR / figure_name
    save_segmentation_figure(
        original=results["original_image"],
        grayscale=results["grayscale"],
        preprocessed=results["preprocessed_gray"],
        initial_mask=results["initial_mask"],
        cleaned_mask=results["cleaned_mask"],
        fruit_mask=fruit_mask,
        output_path=figure_path,
        combined_mask=combined_mask,
        title=image_path.name,
    )

    print("Segmentation analysis:")
    print(f"- Brightness: {quality['brightness']:.2f}")
    print(f"- Contrast: {quality['contrast']:.2f}")
    print(f"- Noise level: {quality['noise_level']:.2f}")
    print(f"- Quality label: {quality['quality_label']}")
    print(f"- Selected preprocessing method: {results['preprocessing_method']}")
    print(f"- Otsu threshold: {results['otsu_threshold']:.2f}")
    print(f"- Combined mask area: {int(np.sum(combined_mask))} pixels")
    print(f"- Fruit mask area: {int(np.sum(fruit_mask))} pixels")
    print(f"- Fruit area ratio: {float(np.sum(fruit_mask)) / fruit_mask.size:.4f}")
    print(f"Saved figure: {figure_path}")

def run_feature_extraction(image_path: Path) -> None:
    """Run Step 4 handcrafted feature extraction for one image."""
    image = load_image(image_path)
    results = run_segmentation_pipeline(image)
    features = extract_all_features_from_pipeline_result(results)
    defect_map = create_defect_map(
        results["original_image"],
        results["grayscale"],
        results["fruit_mask"],
    )

    figure_name = f"features_{image_path.stem}.png"
    figure_path = FIGURES_DIR / figure_name
    save_feature_extraction_figure(
        original=results["original_image"],
        fruit_mask=results["fruit_mask"],
        defect_map=defect_map,
        output_path=figure_path,
        title=image_path.name,
    )

    print("Feature extraction:")
    print(f"- Area: {features['area']:.0f}")
    print(f"- Perimeter: {features['perimeter']:.0f}")
    print(f"- Circularity: {features['circularity']:.2f}")
    print(f"- Aspect ratio: {features['aspect_ratio']:.2f}")
    print(f"- Mask area ratio: {features['mask_area_ratio']:.2f}")
    print(
        "- Mean RGB: "
        f"({features['mean_r']:.2f}, {features['mean_g']:.2f}, {features['mean_b']:.2f})"
    )
    print(f"- Brightness: {features['brightness']:.2f}")
    print(f"- Contrast: {features['contrast']:.2f}")
    print(f"- Noise level: {features['noise_level']:.2f}")
    print(f"- Defect ratio: {features['defect_ratio']:.2f}")
    print(f"Saved figure: {figure_path}")


def main() -> None:
    """Run the requested project utility command or print the startup message."""
    args = parse_args()

    if args.sample_data:
        create_sample_dataset()
        return

    if args.check_data:
        check_datasets()
        return

    if args.analyze_image is not None:
        run_image_quality_analysis(args.analyze_image)
        return

    if args.preprocess_image is not None:
        run_image_preprocessing(args.preprocess_image)
        return

    if args.segment_image is not None:
        run_image_segmentation(args.segment_image)
        return

    if args.extract_features is not None:
        run_feature_extraction(args.extract_features)
        return

    print(STARTUP_MESSAGE)


if __name__ == "__main__":
    main()
