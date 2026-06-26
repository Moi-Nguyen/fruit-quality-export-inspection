"""Entry point for the fruit quality inspection project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import FIGURES_DIR
from src.adaptive_pipeline import run_preprocessing_pipeline
from src.dataset_sampling import check_datasets, create_sample_dataset
from src.io_utils import load_image
from src.quality_analysis import analyze_image_quality, rgb_to_grayscale
from src.visualization import save_preprocessing_figure, save_quality_analysis_figure

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

    print(STARTUP_MESSAGE)


if __name__ == "__main__":
    main()
