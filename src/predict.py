"""Single-image prediction helpers for trained fruit models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.adaptive_pipeline import run_segmentation_pipeline
from src.export_rules import assess_export_suitability
from src.features import create_defect_map, extract_all_features_from_pipeline_result
from src.io_utils import load_image


IMPORTANT_FEATURES = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "mask_area_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "brightness",
    "contrast",
    "noise_level",
    "defect_ratio",
]


def load_model_bundle(model_path: Path) -> dict[str, object]:
    """Load a trained model bundle from a pickle file."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please run: python main.py --train-models"
        )

    with model_path.open("rb") as model_file:
        bundle = pickle.load(model_file)

    if not isinstance(bundle, dict):
        raise ValueError(f"Model file does not contain a valid bundle: {model_path}")
    return bundle


def build_single_feature_vector(
    features: dict[str, object],
    feature_columns: list[str],
) -> np.ndarray:
    """Build one 2D feature matrix in the trained column order."""
    values: list[float] = []
    for column in feature_columns:
        try:
            value = float(features.get(column, 0.0))
        except (TypeError, ValueError):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0
        values.append(value)

    return np.array([values], dtype=np.float32)


def predict_with_bundle(
    features: dict[str, object],
    model_bundle: dict[str, object],
) -> str:
    """Predict one label using a trained model bundle."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    if not isinstance(feature_columns, list):
        raise ValueError("model_bundle['feature_columns'] must be a list.")

    feature_vector = build_single_feature_vector(features, feature_columns)
    prediction = model.predict(feature_vector)
    return str(prediction[0])


def predict_image(
    image_path: Path,
    fruit_model_path: Path,
    quality_model_path: Path,
) -> dict[str, object]:
    """Run segmentation, features, and ML prediction for one image."""
    image = load_image(image_path)
    pipeline_result = run_segmentation_pipeline(image)
    features = extract_all_features_from_pipeline_result(pipeline_result)

    fruit_model_bundle = load_model_bundle(fruit_model_path)
    quality_model_bundle = load_model_bundle(quality_model_path)

    fruit_type = predict_with_bundle(features, fruit_model_bundle)
    quality = predict_with_bundle(features, quality_model_bundle)
    fruit_feature_columns = fruit_model_bundle.get("feature_columns", [])

    original_image = pipeline_result["original_image"]
    grayscale = pipeline_result["grayscale"]
    fruit_mask = pipeline_result["fruit_mask"]
    if not isinstance(original_image, np.ndarray):
        raise ValueError("pipeline_result['original_image'] must be a NumPy array.")
    if not isinstance(grayscale, np.ndarray):
        raise ValueError("pipeline_result['grayscale'] must be a NumPy array.")
    if not isinstance(fruit_mask, np.ndarray):
        raise ValueError("pipeline_result['fruit_mask'] must be a NumPy array.")

    result: dict[str, object] = {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "fruit_type": fruit_type,
        "quality": quality,
        "feature_count": len(fruit_feature_columns) if isinstance(fruit_feature_columns, list) else 0,
        "fruit_mask": fruit_mask,
        "defect_map": create_defect_map(original_image, grayscale, fruit_mask),
        "original_image": original_image,
    }
    for feature_name in IMPORTANT_FEATURES:
        result[feature_name] = float(features.get(feature_name, 0.0))

    export_result = assess_export_suitability(
        fruit_type=fruit_type,
        quality=quality,
        features=result,
    )
    result["export_suitability"] = export_result["suitability"]
    result["export_reasons"] = export_result["reasons"]
    result["export_rule_flags"] = export_result["rule_flags"]

    return result
