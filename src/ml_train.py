"""Machine learning training and evaluation helpers."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

EXCLUDED_COLUMNS = {"image_path", "file_name", "class_name", "fruit_type", "quality"}


def load_feature_csv(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load feature rows from a CSV file."""
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def get_numeric_feature_columns(
    rows: list[dict[str, str]],
    excluded_columns: set[str],
) -> list[str]:
    """Return CSV columns that contain numeric feature values."""
    if not rows:
        return []

    candidate_columns = set(rows[0].keys()) - excluded_columns
    numeric_columns: list[str] = []

    for column in candidate_columns:
        is_numeric = True
        has_value = False
        for row in rows:
            value = row.get(column, "")
            if value == "" or value is None:
                continue
            try:
                float(value)
                has_value = True
            except ValueError:
                is_numeric = False
                break
        if is_numeric and has_value:
            numeric_columns.append(column)

    return sorted(numeric_columns)


def build_feature_matrix(
    rows: list[dict[str, str]],
    feature_columns: list[str],
) -> np.ndarray:
    """Convert selected feature columns to a float32 matrix."""
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(feature_columns):
            try:
                matrix[row_index, column_index] = float(row.get(column, ""))
            except (TypeError, ValueError):
                matrix[row_index, column_index] = 0.0
    return matrix


def build_label_vector(
    rows: list[dict[str, str]],
    target_column: str,
) -> np.ndarray:
    """Return labels for one target column."""
    return np.array([row[target_column] for row in rows])


def create_models() -> dict[str, object]:
    """Create the classical machine learning models."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                    ),
                ),
            ]
        ),
    }

def model_supports_confidence(model: object) -> bool:
    """Return True when the fitted model can expose class probabilities."""
    return hasattr(model, "predict_proba")

def model_selection_score(model_name: str, model: object, metrics: dict[str, object]) -> tuple[float, float, int, int]:
    """Score models by accuracy first, then confidence support for tied metrics."""
    confidence_capable = 1 if model_supports_confidence(model) else 0
    random_forest_preference = 1 if model_name == "random_forest" else 0
    return (
        float(metrics["f1_macro"]),
        float(metrics["accuracy"]),
        confidence_capable,
        random_forest_preference,
    )


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    """Evaluate a fitted model with common classification metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def train_and_evaluate_for_target(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    target_column: str,
    feature_columns: list[str],
) -> dict[str, object]:
    """Train and evaluate all models for one target label."""
    X_train = build_feature_matrix(train_rows, feature_columns)
    X_test = build_feature_matrix(test_rows, feature_columns)
    y_train = build_label_vector(train_rows, target_column)
    y_test = build_label_vector(test_rows, target_column)

    results: dict[str, dict[str, object]] = {}
    for model_name, model in create_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = {"model": model, "metrics": metrics}

    best_model_name = max(
        results,
        key=lambda name: model_selection_score(
            name,
            results[name]["model"],
            results[name]["metrics"],
        ),
    )

    return {
        "target": target_column,
        "feature_columns": feature_columns,
        "results": results,
        "best_model_name": best_model_name,
        "best_model": results[best_model_name]["model"],
        "best_metrics": results[best_model_name]["metrics"],
    }


def save_model_bundle(
    model,
    output_path: Path,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    labels: list[str],
) -> None:
    """Save a trained model and its metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_name": model_name,
        "labels": labels,
    }
    with output_path.open("wb") as model_file:
        pickle.dump(bundle, model_file)


def save_metrics_report(all_results: dict[str, object], output_path: Path) -> None:
    """Save a readable text evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["Machine Learning Evaluation Report", ""]

    for target_name in ["fruit_type", "quality"]:
        target_results = all_results[target_name]
        lines.append(f"Target: {target_results['target']}")
        lines.append(f"Feature count: {len(target_results['feature_columns'])}")
        lines.append("")

        for model_name, result in target_results["results"].items():
            metrics = result["metrics"]
            lines.append(f"Model: {model_name}")
            lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"Precision macro: {metrics['precision_macro']:.4f}")
            lines.append(f"Recall macro: {metrics['recall_macro']:.4f}")
            lines.append(f"F1 macro: {metrics['f1_macro']:.4f}")
            lines.append("Confusion matrix:")
            lines.append(str(metrics["confusion_matrix"]))
            lines.append("Classification report:")
            lines.append(str(metrics["classification_report"]))
            lines.append("")

        lines.append(f"Best model: {target_results['best_model_name']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_all_models(
    train_csv: Path,
    test_csv: Path,
    model_dir: Path,
    report_dir: Path,
) -> dict[str, object]:
    """Train fruit type and quality models from feature CSV files."""
    train_rows, _ = load_feature_csv(train_csv)
    test_rows, _ = load_feature_csv(test_csv)
    feature_columns = get_numeric_feature_columns(train_rows, EXCLUDED_COLUMNS)

    all_results = {
        "feature_columns": feature_columns,
        "fruit_type": train_and_evaluate_for_target(
            train_rows, test_rows, "fruit_type", feature_columns
        ),
        "quality": train_and_evaluate_for_target(
            train_rows, test_rows, "quality", feature_columns
        ),
    }

    save_model_bundle(
        model=all_results["fruit_type"]["best_model"],
        output_path=model_dir / "fruit_type_model.pkl",
        feature_columns=feature_columns,
        target_column="fruit_type",
        model_name=all_results["fruit_type"]["best_model_name"],
        labels=sorted(set(build_label_vector(train_rows, "fruit_type"))),
    )
    save_model_bundle(
        model=all_results["quality"]["best_model"],
        output_path=model_dir / "quality_model.pkl",
        feature_columns=feature_columns,
        target_column="quality",
        model_name=all_results["quality"]["best_model_name"],
        labels=sorted(set(build_label_vector(train_rows, "quality"))),
    )

    save_metrics_report(all_results, report_dir / "ml_evaluation_report.txt")
    return all_results
