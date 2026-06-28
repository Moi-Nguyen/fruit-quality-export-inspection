"""End-to-end batch evaluation for the fruit inspection system."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import CLASS_NAMES, FRUIT_TYPES, QUALITY_LABELS
from src.predict import IMPORTANT_FEATURES, predict_image

CLASS_LABEL_MAP: dict[str, tuple[str, str]] = {
    "freshapples": ("apple", "fresh"),
    "freshbanana": ("banana", "fresh"),
    "freshoranges": ("orange", "fresh"),
    "rottenapples": ("apple", "rotten"),
    "rottenbanana": ("banana", "rotten"),
    "rottenoranges": ("orange", "rotten"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

CSV_COLUMNS = [
    "image_path",
    "class_name",
    "true_fruit_type",
    "predicted_fruit_type",
    "true_quality",
    "predicted_quality",
    "export_suitability",
    "market_grade",
    "error",
    *IMPORTANT_FEATURES,
    "is_fruit_type_correct",
    "is_quality_correct",
]


@dataclass(frozen=True)
class EvaluationResult:
    """Summary values returned after end-to-end test evaluation."""

    total_images: int
    failed_images: int
    fruit_type_accuracy: float
    quality_accuracy: float
    csv_output_path: Path
    report_output_path: Path


def infer_ground_truth_labels(class_name: str) -> tuple[str, str]:
    """Infer fruit type and quality labels from a test-set class folder name."""
    try:
        return CLASS_LABEL_MAP[class_name]
    except KeyError as error:
        raise ValueError(f"Unsupported test class folder: {class_name}") from error


def iter_test_images(test_dir: Path, max_per_class: Optional[int] = None) -> Iterable[Path]:
    """Yield supported image files under known class folders in deterministic order."""
    if max_per_class is not None and max_per_class <= 0:
        return

    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        class_count = 0
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path
                class_count += 1
                if max_per_class is not None and class_count >= max_per_class:
                    break


def build_evaluation_row(
    image_path: Path,
    prediction: dict[str, object],
) -> dict[str, object]:
    """Convert one image prediction into a flat CSV row with correctness flags."""
    class_name = image_path.parent.name
    true_fruit_type, true_quality = infer_ground_truth_labels(class_name)
    predicted_fruit_type = str(prediction.get("fruit_type", ""))
    predicted_quality = str(prediction.get("quality", ""))

    row: dict[str, object] = {
        "image_path": str(image_path),
        "class_name": class_name,
        "true_fruit_type": true_fruit_type,
        "predicted_fruit_type": predicted_fruit_type,
        "true_quality": true_quality,
        "predicted_quality": predicted_quality,
        "export_suitability": prediction.get("export_suitability", ""),
        "market_grade": prediction.get("market_grade", ""),
        "error": "",
        "is_fruit_type_correct": predicted_fruit_type == true_fruit_type,
        "is_quality_correct": predicted_quality == true_quality,
    }
    for feature_name in IMPORTANT_FEATURES:
        row[feature_name] = float(prediction.get(feature_name, 0.0))
    return row

def build_error_row(image_path: Path, error: Exception) -> dict[str, object]:
    """Convert a failed image prediction into a CSV row that preserves the error."""
    class_name = image_path.parent.name
    true_fruit_type, true_quality = infer_ground_truth_labels(class_name)
    row: dict[str, object] = {
        "image_path": str(image_path),
        "class_name": class_name,
        "true_fruit_type": true_fruit_type,
        "predicted_fruit_type": "",
        "true_quality": true_quality,
        "predicted_quality": "",
        "export_suitability": "",
        "market_grade": "",
        "error": f"{type(error).__name__}: {error}",
        "is_fruit_type_correct": False,
        "is_quality_correct": False,
    }
    for feature_name in IMPORTANT_FEATURES:
        row[feature_name] = 0.0
    return row


def write_prediction_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write per-image end-to-end predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_confusion_matrix(matrix, labels: list[str]) -> str:
    """Format a confusion matrix with labels for a readable text report."""
    lines = ["labels: " + ", ".join(labels)]
    for label, values in zip(labels, matrix):
        counts = " ".join(str(int(value)) for value in values)
        lines.append(f"{label}: {counts}")
    return "\n".join(lines)


def build_text_report(rows: list[dict[str, object]]) -> tuple[str, float, float]:
    """Build a readable evaluation report and return report text plus accuracies."""
    successful_rows = [row for row in rows if not str(row.get("error", ""))]
    failed_rows = [row for row in rows if str(row.get("error", ""))]
    true_fruit = [str(row["true_fruit_type"]) for row in successful_rows]
    predicted_fruit = [str(row["predicted_fruit_type"]) for row in successful_rows]
    true_quality = [str(row["true_quality"]) for row in successful_rows]
    predicted_quality = [str(row["predicted_quality"]) for row in successful_rows]

    fruit_accuracy = float(accuracy_score(true_fruit, predicted_fruit)) if successful_rows else 0.0
    quality_accuracy = float(accuracy_score(true_quality, predicted_quality)) if successful_rows else 0.0
    fruit_report = classification_report(
        true_fruit,
        predicted_fruit,
        labels=FRUIT_TYPES,
        zero_division=0,
    ) if successful_rows else "No successful images evaluated."
    quality_report = classification_report(
        true_quality,
        predicted_quality,
        labels=QUALITY_LABELS,
        zero_division=0,
    ) if successful_rows else "No successful images evaluated."
    fruit_matrix = confusion_matrix(true_fruit, predicted_fruit, labels=FRUIT_TYPES) if successful_rows else []
    quality_matrix = confusion_matrix(true_quality, predicted_quality, labels=QUALITY_LABELS) if successful_rows else []
    market_grade_counts = Counter(str(row["market_grade"]) for row in successful_rows)
    misclassified_rows = [
        row for row in successful_rows
        if not bool(row["is_fruit_type_correct"]) or not bool(row["is_quality_correct"])
    ]

    lines = [
        "End-to-End Test Evaluation Report",
        "",
        f"Total images tested: {len(rows)}",
        f"Successful images: {len(successful_rows)}",
        f"Failed images: {len(failed_rows)}",
        f"Fruit type accuracy: {fruit_accuracy:.4f}",
        f"Quality accuracy: {quality_accuracy:.4f}",
        "",
        "Classification report for fruit_type:",
        fruit_report,
        "Classification report for quality:",
        quality_report,
        "Confusion matrix for fruit_type:",
        format_confusion_matrix(fruit_matrix, FRUIT_TYPES) if successful_rows else "No successful images evaluated.",
        "",
        "Confusion matrix for quality:",
        format_confusion_matrix(quality_matrix, QUALITY_LABELS) if successful_rows else "No successful images evaluated.",
        "",
        "Market grade counts:",
    ]

    if market_grade_counts:
        for grade, count in sorted(market_grade_counts.items()):
            lines.append(f"- {grade}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "Misclassified images:"])
    if misclassified_rows:
        for row in misclassified_rows:
            lines.append(
                "- "
                f"{row['image_path']} | "
                f"fruit_type {row['true_fruit_type']} -> {row['predicted_fruit_type']} | "
                f"quality {row['true_quality']} -> {row['predicted_quality']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "Failed images:"])
    if failed_rows:
        for row in failed_rows:
            lines.append(f"- {row['image_path']} | {row['error']}")
    else:
        lines.append("- None")

    return "\n".join(lines), fruit_accuracy, quality_accuracy


def write_text_report(rows: list[dict[str, object]], output_path: Path) -> tuple[float, float]:
    """Write the readable end-to-end evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text, fruit_accuracy, quality_accuracy = build_text_report(rows)
    output_path.write_text(report_text, encoding="utf-8")
    return fruit_accuracy, quality_accuracy


def evaluate_system_on_test_set(
    test_dir: Path,
    fruit_model_path: Path,
    quality_model_path: Path,
    csv_output_path: Path,
    report_output_path: Path,
    max_per_class: Optional[int] = None,
) -> EvaluationResult:
    """Run prediction for every test image and save CSV plus text evaluation reports."""
    rows: list[dict[str, object]] = []
    image_paths = list(iter_test_images(test_dir, max_per_class=max_per_class))
    total_images = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        print(f"Processing {index}/{total_images}: {image_path.parent.name}/{image_path.name}", flush=True)
        try:
            prediction = predict_image(
                image_path=image_path,
                fruit_model_path=fruit_model_path,
                quality_model_path=quality_model_path,
                save_figure=False,
            )
            rows.append(build_evaluation_row(image_path, prediction))
        except Exception as error:
            rows.append(build_error_row(image_path, error))

    write_prediction_csv(rows, csv_output_path)
    fruit_accuracy, quality_accuracy = write_text_report(rows, report_output_path)

    return EvaluationResult(
        total_images=len(rows),
        failed_images=sum(1 for row in rows if str(row.get("error", ""))),
        fruit_type_accuracy=fruit_accuracy,
        quality_accuracy=quality_accuracy,
        csv_output_path=csv_output_path,
        report_output_path=report_output_path,
    )
