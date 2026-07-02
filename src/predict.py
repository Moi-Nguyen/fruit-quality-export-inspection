"""Single-image prediction helpers for trained fruit models."""

from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from src.adaptive_pipeline import run_segmentation_pipeline
from src.config import MAX_PROCESSING_SIDE
from src.export_rules import (
    LOW_CONFIDENCE_THRESHOLD,
    LOW_DEFECT_ROTTEN_RECHECK_REASON,
    LOW_FRUIT_TYPE_CONFIDENCE_REASON,
    LOW_MODEL_CONFIDENCE_REASON,
    assess_export_suitability,
    decide_market_grade,
)
from src.features import create_defect_map, extract_all_features_from_pipeline_result
from src.io_utils import ensure_uint8_image, load_image


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
    "brown_dark_ratio",
    "red_ratio",
    "yellow_ratio",
    "orange_ratio",
    "green_ratio",
]

_MODEL_CACHE: dict[Path, dict[str, object]] = {}


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

def clear_model_cache() -> None:
    """Clear cached model bundles, mainly useful for tests or retraining sessions."""
    _MODEL_CACHE.clear()

def get_model_bundle(model_path: Path) -> dict[str, object]:
    """Load a model bundle once per process and reuse it for later predictions."""
    cache_key = model_path.resolve()
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model_bundle(model_path)
    return _MODEL_CACHE[cache_key]

def resize_image_for_processing(
    image: np.ndarray,
    max_side: int = MAX_PROCESSING_SIDE,
) -> tuple[np.ndarray, bool]:
    """Resize large images proportionally for faster traditional processing."""
    if max_side <= 0:
        return image, False

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image, False

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(ensure_uint8_image(image))
    resized = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized), True


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

def predict_label_and_confidence(
    features: dict[str, object],
    model_bundle: dict[str, object],
) -> tuple[str, float | None]:
    """Predict one label and return predict_proba confidence when available."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    if not isinstance(feature_columns, list):
        raise ValueError("model_bundle['feature_columns'] must be a list.")

    feature_vector = build_single_feature_vector(features, feature_columns)
    prediction = str(model.predict(feature_vector)[0])
    if not hasattr(model, "predict_proba"):
        return prediction, None

    probabilities = model.predict_proba(feature_vector)[0]
    classes = getattr(model, "classes_", [])
    if len(classes) == len(probabilities):
        matching_indices = np.where(np.asarray(classes).astype(str) == prediction)[0]
        if matching_indices.size > 0:
            return prediction, float(probabilities[int(matching_indices[0])])
    return prediction, float(np.max(probabilities))


def predict_image(
    image_path: Path,
    fruit_model_path: Path,
    quality_model_path: Path,
    save_figure: bool = True,
    use_model_cache: bool = True,
    max_processing_side: int = MAX_PROCESSING_SIDE,
) -> dict[str, object]:
    """Run segmentation, features, and ML prediction for one image."""
    _ = save_figure
    start_time = perf_counter()
    image = load_image(image_path)
    image, resized_for_processing = resize_image_for_processing(image, max_processing_side)
    pipeline_result = run_segmentation_pipeline(image)
    features = extract_all_features_from_pipeline_result(pipeline_result)

    model_loader = get_model_bundle if use_model_cache else load_model_bundle
    fruit_model_bundle = model_loader(fruit_model_path)
    quality_model_bundle = model_loader(quality_model_path)

    fruit_type, fruit_type_confidence = predict_label_and_confidence(features, fruit_model_bundle)
    quality, quality_confidence = predict_label_and_confidence(features, quality_model_bundle)
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
        "fruit_type_confidence": fruit_type_confidence,
        "quality_confidence": quality_confidence,
        "feature_count": len(fruit_feature_columns) if isinstance(fruit_feature_columns, list) else 0,
        "processing_time_seconds": perf_counter() - start_time,
        "resized_for_processing": resized_for_processing,
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

    market_grade, market_grade_reasons = decide_market_grade(
        quality=quality,
        defect_ratio=float(result["defect_ratio"]),
        mask_area_ratio=float(result["mask_area_ratio"]),
        circularity=float(result["circularity"]),
        brown_dark_ratio=float(result["brown_dark_ratio"]),
        quality_confidence=quality_confidence,
    )
    result["market_grade"] = market_grade
    result["market_grade_reasons"] = market_grade_reasons
    result["consistency_warnings"] = []
    if LOW_DEFECT_ROTTEN_RECHECK_REASON in market_grade_reasons:
        result["consistency_warnings"] = [LOW_DEFECT_ROTTEN_RECHECK_REASON]

    if fruit_type_confidence is not None and fruit_type_confidence < LOW_CONFIDENCE_THRESHOLD:
        result["export_reasons"] = list(result["export_reasons"]) + [
            LOW_FRUIT_TYPE_CONFIDENCE_REASON
        ]
        result["market_grade_reasons"] = list(result["market_grade_reasons"]) + [
            LOW_FRUIT_TYPE_CONFIDENCE_REASON
        ]

    low_confidence_values = [
        confidence for confidence in (fruit_type_confidence, quality_confidence) if confidence is not None
    ]
    has_low_confidence = bool(low_confidence_values) and min(low_confidence_values) < LOW_CONFIDENCE_THRESHOLD
    if has_low_confidence and str(quality).strip().lower() != "rotten":
        result["export_suitability"] = "Need Recheck"
        result["market_grade"] = "Need Recheck"
        confidence_reason = LOW_MODEL_CONFIDENCE_REASON
        result["export_reasons"] = list(result["export_reasons"]) + [confidence_reason]
        result["market_grade_reasons"] = list(result["market_grade_reasons"]) + [confidence_reason]

    return result



