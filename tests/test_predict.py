"""Tests for single-image prediction helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.predict import build_single_feature_vector, load_model_bundle, predict_with_bundle


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
