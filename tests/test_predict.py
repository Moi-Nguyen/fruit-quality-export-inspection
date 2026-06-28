"""Tests for single-image prediction helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src import predict
from src.predict import build_single_feature_vector, load_model_bundle, predict_image, predict_with_bundle


def test_build_single_feature_vector_returns_2d_float32_array() -> None:
    features = {"area": 12, "perimeter": "5.5"}
    feature_columns = ["area", "perimeter"]

    vector = build_single_feature_vector(features, feature_columns)

    assert vector.shape == (1, 2)
    assert vector.dtype == np.float32
    assert vector.tolist() == [[12.0, 5.5]]


def test_build_single_feature_vector_uses_zero_for_missing_value() -> None:
    vector = build_single_feature_vector({"area": 12}, ["area", "contrast"])

    assert vector.tolist() == [[12.0, 0.0]]


def test_build_single_feature_vector_uses_zero_for_invalid_value() -> None:
    vector = build_single_feature_vector(
        {"area": "bad", "contrast": float("nan")},
        ["area", "contrast"],
    )

    assert vector.tolist() == [[0.0, 0.0]]


def test_predict_with_bundle_uses_trained_sklearn_model() -> None:
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(np.array([[1.0, 0.0], [10.0, 1.0]], dtype=np.float32), ["apple", "banana"])
    bundle = {"model": model, "feature_columns": ["area", "defect_ratio"]}

    prediction = predict_with_bundle({"area": 9.5, "defect_ratio": 1.0}, bundle)

    assert prediction == "banana"


def test_load_model_bundle_raises_helpful_error_for_missing_model(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="python main.py --train-models"):
        load_model_bundle(missing_path)


def test_load_model_bundle_reads_pickle_bundle(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    bundle = {"model": object(), "feature_columns": ["area"]}
    with model_path.open("wb") as model_file:
        pickle.dump(bundle, model_file)

    loaded = load_model_bundle(model_path)

    assert loaded["feature_columns"] == ["area"]

def test_predict_image_includes_export_suitability_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)

    def fake_load_image(image_path: Path) -> np.ndarray:
        return image

    def fake_run_segmentation_pipeline(input_image: np.ndarray) -> dict[str, object]:
        return {
            "original_image": input_image,
            "grayscale": np.zeros((4, 4), dtype=np.uint8),
            "fruit_mask": mask,
        }

    def fake_extract_features(pipeline_result: dict[str, object]) -> dict[str, object]:
        return {
            "area": 16.0,
            "perimeter": 16.0,
            "circularity": 0.80,
            "aspect_ratio": 1.0,
            "mask_area_ratio": 0.50,
            "mean_r": 120.0,
            "mean_g": 130.0,
            "mean_b": 110.0,
            "brightness": 120.0,
            "contrast": 30.0,
            "noise_level": 5.0,
            "defect_ratio": 0.02,
        }

    def fake_load_model_bundle(model_path: Path) -> dict[str, object]:
        return {"model": object(), "feature_columns": ["area"]}

    predictions = iter(["orange", "fresh"])

    def fake_predict_with_bundle(
        features: dict[str, object],
        model_bundle: dict[str, object],
    ) -> str:
        return next(predictions)

    monkeypatch.setattr(predict, "load_image", fake_load_image)
    monkeypatch.setattr(predict, "run_segmentation_pipeline", fake_run_segmentation_pipeline)
    monkeypatch.setattr(predict, "extract_all_features_from_pipeline_result", fake_extract_features)
    monkeypatch.setattr(predict, "load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(predict, "predict_with_bundle", fake_predict_with_bundle)
    monkeypatch.setattr(predict, "create_defect_map", lambda original, grayscale, fruit_mask: mask)

    result = predict_image(
        image_path=tmp_path / "orange.png",
        fruit_model_path=tmp_path / "fruit.pkl",
        quality_model_path=tmp_path / "quality.pkl",
    )

    assert result["export_suitability"] == "Suitable"
    assert isinstance(result["export_reasons"], list)
    assert isinstance(result["export_rule_flags"], dict)
