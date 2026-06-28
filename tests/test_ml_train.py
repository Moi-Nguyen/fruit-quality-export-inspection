"""Tests for Step 6 machine learning training helpers."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from src.ml_train import (
    EXCLUDED_COLUMNS,
    build_feature_matrix,
    build_label_vector,
    create_models,
    evaluate_model,
    get_numeric_feature_columns,
    save_metrics_report,
    save_model_bundle,
    train_all_models,
    train_and_evaluate_for_target,
)


def make_rows() -> list[dict[str, str]]:
    """Create a tiny labeled feature table."""
    return [
        make_row("apple", "fresh", 1.0, 10.0),
        make_row("apple", "rotten", 1.2, 11.0),
        make_row("banana", "fresh", 5.0, 20.0),
        make_row("banana", "rotten", 5.2, 21.0),
        make_row("orange", "fresh", 9.0, 30.0),
        make_row("orange", "rotten", 9.2, 31.0),
    ]


def make_row(fruit_type: str, quality: str, area: float, brightness: float) -> dict[str, str]:
    """Create one synthetic feature row."""
    class_name = f"{quality}_{fruit_type}"
    return {
        "image_path": f"data/{class_name}.jpg",
        "file_name": f"{class_name}.jpg",
        "class_name": class_name,
        "fruit_type": fruit_type,
        "quality": quality,
        "area": str(area),
        "brightness": str(brightness),
        "preprocessing_method": "median",
    }


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Write synthetic rows to CSV."""
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_get_numeric_feature_columns_excludes_metadata_targets_and_non_numeric() -> None:
    rows = make_rows()

    columns = get_numeric_feature_columns(rows, EXCLUDED_COLUMNS)

    assert columns == ["area", "brightness"]
    assert "fruit_type" not in columns
    assert "quality" not in columns
    assert "preprocessing_method" not in columns


def test_build_feature_matrix_returns_correct_shape() -> None:
    rows = make_rows()

    matrix = build_feature_matrix(rows, ["area", "brightness"])

    assert matrix.shape == (6, 2)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0


def test_build_label_vector_returns_expected_labels() -> None:
    labels = build_label_vector(make_rows(), "fruit_type")

    assert labels.tolist() == ["apple", "apple", "banana", "banana", "orange", "orange"]


def test_create_models_returns_expected_models() -> None:
    models = create_models()

    assert set(models.keys()) == {"random_forest", "knn", "svm"}


def test_evaluate_model_returns_required_metric_keys() -> None:
    rows = make_rows()
    X = build_feature_matrix(rows, ["area", "brightness"])
    y = build_label_vector(rows, "quality")
    model = create_models()["random_forest"]
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert set(metrics.keys()) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "classification_report",
    }


def test_train_and_evaluate_for_target_works_on_tiny_dataset() -> None:
    rows = make_rows()

    result = train_and_evaluate_for_target(rows, rows, "fruit_type", ["area", "brightness"])

    assert result["target"] == "fruit_type"
    assert set(result["results"].keys()) == {"random_forest", "knn", "svm"}
    assert result["best_model_name"] in result["results"]


def test_save_model_bundle_creates_pickle_file(tmp_path: Path) -> None:
    output_path = tmp_path / "model.pkl"
    model = create_models()["random_forest"]

    save_model_bundle(model, output_path, ["area"], "quality", "random_forest", ["fresh", "rotten"])

    assert output_path.exists()
    with output_path.open("rb") as model_file:
        bundle = pickle.load(model_file)
    assert bundle["target_column"] == "quality"
    assert bundle["feature_columns"] == ["area"]


def test_save_metrics_report_creates_text_file(tmp_path: Path) -> None:
    rows = make_rows()
    result = train_and_evaluate_for_target(rows, rows, "quality", ["area", "brightness"])
    all_results = {"fruit_type": result, "quality": result, "feature_columns": ["area", "brightness"]}
    output_path = tmp_path / "report.txt"

    save_metrics_report(all_results, output_path)

    assert output_path.exists()
    assert "Best model" in output_path.read_text(encoding="utf-8")


def test_train_all_models_works_on_tiny_csv_files(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    rows = make_rows()
    write_rows(train_csv, rows)
    write_rows(test_csv, rows)

    all_results = train_all_models(train_csv, test_csv, model_dir, report_dir)

    assert len(all_results["feature_columns"]) == 2
    assert (model_dir / "fruit_type_model.pkl").exists()
    assert (model_dir / "quality_model.pkl").exists()
    assert (report_dir / "ml_evaluation_report.txt").exists()
