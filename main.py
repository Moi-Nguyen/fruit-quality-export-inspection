"""Entry point for the fruit quality inspection project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.adaptive_pipeline import run_preprocessing_pipeline, run_segmentation_pipeline
from src.config import FIGURES_DIR
from src.dataset_sampling import check_datasets, create_sample_dataset
from src.features import create_defect_map, extract_all_features_from_pipeline_result
from src.features_dataset import export_sample_features
from src.io_utils import load_image
from src.ml_train import train_all_models
from src.predict import predict_image
from src.quality_analysis import analyze_image_quality, rgb_to_grayscale
from src.system_evaluation import evaluate_system_on_test_set
from src.visualization import (
    save_feature_extraction_figure,
    save_preprocessing_figure,
    save_prediction_figure,
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
    parser.add_argument(
        "--export-features",
        action="store_true",
        help="Extract features for data/sample/ and save CSV files.",
    )
    parser.add_argument(
        "--train-models",
        action="store_true",
        help="Train and evaluate classical ML models from feature CSV files.",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Predict fruit type and quality for one image.",
    )
    parser.add_argument(
        "--evaluate-system",
        action="store_true",
        help="Run end-to-end prediction evaluation on data/sample/test/.",
    )
    parser.add_argument(
        "--eval-max-per-class",
        type=int,
        default=None,
        help="Limit end-to-end evaluation to N images per class.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Start the Tkinter desktop GUI.",
    )
    return parser.parse_args()


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file, excluding the header."""
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        line_count = sum(1 for _ in csv_file)
    return max(line_count - 1, 0)


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


def run_feature_csv_export() -> None:
    """Run Step 5 batch feature extraction and CSV export."""
    train_csv_path, test_csv_path = export_sample_features(
        sample_root=Path("data/sample"),
        output_dir=Path("outputs/features"),
    )

    print("Feature CSV export complete:")
    print(f"- Train CSV: {train_csv_path}")
    print(f"- Train rows: {count_csv_rows(train_csv_path)}")
    print(f"- Test CSV: {test_csv_path}")
    print(f"- Test rows: {count_csv_rows(test_csv_path)}")



def print_target_training_summary(title: str, target_results: dict[str, object]) -> None:
    """Print model metrics for one classification target."""
    print(f"{title}:")
    for model_name, result in target_results["results"].items():
        metrics = result["metrics"]
        print(
            f"  {model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
    print(f"  best: {target_results['best_model_name']}")

def run_model_training() -> None:
    """Run Step 6 machine learning training and evaluation."""
    train_csv = Path("outputs/features/train_features.csv")
    test_csv = Path("outputs/features/test_features.csv")
    model_dir = Path("models")
    report_dir = Path("outputs/reports")

    all_results = train_all_models(
        train_csv=train_csv,
        test_csv=test_csv,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    print(f"Feature count: {len(all_results['feature_columns'])}")
    print_target_training_summary(
        "Fruit type classification",
        all_results["fruit_type"],
    )
    print_target_training_summary(
        "Quality classification",
        all_results["quality"],
    )
    print("Saved models:")
    print(f"  {model_dir / 'fruit_type_model.pkl'}")
    print(f"  {model_dir / 'quality_model.pkl'}")
    print("Saved report:")
    print(f"  {report_dir / 'ml_evaluation_report.txt'}")

def run_single_image_prediction(image_path: Path) -> None:
    """Run Step 7 single-image ML prediction."""
    fruit_model_path = Path("models/fruit_type_model.pkl")
    quality_model_path = Path("models/quality_model.pkl")

    try:
        result = predict_image(
            image_path=image_path,
            fruit_model_path=fruit_model_path,
            quality_model_path=quality_model_path,
        )
    except FileNotFoundError as error:
        print(error)
        print("Please run: python main.py --train-models")
        return

    figure_path = FIGURES_DIR / f"prediction_{image_path.stem}.png"
    save_prediction_figure(
        original=result["original_image"],
        fruit_mask=result["fruit_mask"],
        defect_map=result["defect_map"],
        output_path=figure_path,
        title=(
            f"Prediction: {result['fruit_type']} | {result['quality']} | "
            f"{result['market_grade']}"
        ),
    )

    print("Prediction result:")
    print(f"- Image: {result['image_path']}")
    print(f"- Fruit type: {result['fruit_type']}")
    print(f"- Quality: {result['quality']}")
    print(f"- Export suitability: {result['export_suitability']}")
    print(f"- Final market grade: {result['market_grade']}")
    print("")
    print("Export suitability reasons:")
    export_reasons = result["export_reasons"]
    if isinstance(export_reasons, list):
        for reason_index, reason in enumerate(export_reasons, start=1):
            print(f"- {reason_index}. {reason}")
    print("")
    print("Market grade reasons:")
    market_grade_reasons = result["market_grade_reasons"]
    if isinstance(market_grade_reasons, list):
        for reason_index, reason in enumerate(market_grade_reasons, start=1):
            print(f"- {reason_index}. {reason}")
    print("")
    print("Important features:")
    print(f"- Area: {result['area']:.2f}")
    print(f"- Perimeter: {result['perimeter']:.2f}")
    print(f"- Circularity: {result['circularity']:.2f}")
    print(f"- Aspect ratio: {result['aspect_ratio']:.2f}")
    print(f"- Mask area ratio: {result['mask_area_ratio']:.2f}")
    print(
        "- Mean RGB: "
        f"({result['mean_r']:.2f}, {result['mean_g']:.2f}, {result['mean_b']:.2f})"
    )
    print(f"- Brightness: {result['brightness']:.2f}")
    print(f"- Contrast: {result['contrast']:.2f}")
    print(f"- Noise level: {result['noise_level']:.2f}")
    print(f"- Defect ratio: {result['defect_ratio']:.2f}")
    print("")
    print("Saved figure:")
    print(figure_path)

def run_system_evaluation(max_per_class: int | None = None) -> None:
    """Run Step 11 end-to-end batch prediction evaluation."""
    try:
        result = evaluate_system_on_test_set(
            test_dir=Path("data/sample/test"),
            fruit_model_path=Path("models/fruit_type_model.pkl"),
            quality_model_path=Path("models/quality_model.pkl"),
            csv_output_path=Path("outputs/reports/end_to_end_test_predictions.csv"),
            report_output_path=Path("outputs/reports/end_to_end_evaluation_report.txt"),
            max_per_class=max_per_class,
        )
    except FileNotFoundError as error:
        print(error)
        print("Please run: python main.py --train-models")
        return

    print("End-to-end system evaluation:")
    print(f"- Total images tested: {result.total_images}")
    print(f"- Failed images: {result.failed_images}")
    print(f"- Fruit type accuracy: {result.fruit_type_accuracy:.4f}")
    print(f"- Quality accuracy: {result.quality_accuracy:.4f}")
    print("Saved reports:")
    print(f"  {result.csv_output_path}")
    print(f"  {result.report_output_path}")

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

    if args.export_features:
        run_feature_csv_export()
        return

    if args.train_models:
        run_model_training()
        return

    if args.predict_image is not None:
        run_single_image_prediction(args.predict_image)
        return

    if args.evaluate_system:
        run_system_evaluation(args.eval_max_per_class)
        return

    if args.gui:
        from src.gui_app import run_gui

        run_gui()
        return

    print(STARTUP_MESSAGE)


if __name__ == "__main__":
    main()
