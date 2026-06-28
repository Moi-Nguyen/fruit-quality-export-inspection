"""Tests for end-to-end system evaluation reporting."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from src import system_evaluation


def create_tiny_image(path: Path) -> None:
    """Create a tiny RGB image for path discovery tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(120, 40, 30)).save(path)


def fake_prediction(fruit_type: str, quality: str) -> dict[str, object]:
    """Build a prediction payload matching predict_image output."""
    return {
        "fruit_type": fruit_type,
        "quality": quality,
        "export_suitability": "Suitable",
        "market_grade": "Export Grade",
        "area": 4.0,
        "perimeter": 8.0,
        "circularity": 0.75,
        "aspect_ratio": 1.0,
        "mask_area_ratio": 0.5,
        "mean_r": 120.0,
        "mean_g": 40.0,
        "mean_b": 30.0,
        "brightness": 63.0,
        "contrast": 5.0,
        "noise_level": 1.0,
        "defect_ratio": 0.01,
    }


def test_infer_labels_from_known_class_names() -> None:
    assert system_evaluation.infer_ground_truth_labels("freshapples") == ("apple", "fresh")
    assert system_evaluation.infer_ground_truth_labels("rottenbanana") == ("banana", "rotten")
    assert system_evaluation.infer_ground_truth_labels("freshoranges") == ("orange", "fresh")


@pytest.mark.parametrize("class_name", ["unknown", "freshkiwi", "applefresh"])
def test_infer_labels_rejects_unknown_class_names(class_name: str) -> None:
    with pytest.raises(ValueError, match="Unsupported test class folder"):
        system_evaluation.infer_ground_truth_labels(class_name)


def test_evaluate_system_writes_csv_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    apple_image = test_dir / "freshapples" / "apple.png"
    banana_image = test_dir / "rottenbanana" / "banana.jpg"
    create_tiny_image(apple_image)
    create_tiny_image(banana_image)

    predictions = {
        apple_image.resolve(): fake_prediction("apple", "fresh"),
        banana_image.resolve(): fake_prediction("apple", "rotten"),
    }

    def fake_predict_image(
        image_path: Path,
        fruit_model_path: Path,
        quality_model_path: Path,
        save_figure: bool = True,
    ) -> dict[str, object]:
        assert save_figure is False
        return predictions[image_path.resolve()]

    monkeypatch.setattr(system_evaluation, "predict_image", fake_predict_image)

    csv_path = tmp_path / "reports" / "predictions.csv"
    report_path = tmp_path / "reports" / "summary.txt"
    result = system_evaluation.evaluate_system_on_test_set(
        test_dir=test_dir,
        fruit_model_path=tmp_path / "fruit.pkl",
        quality_model_path=tmp_path / "quality.pkl",
        csv_output_path=csv_path,
        report_output_path=report_path,
    )

    assert result.total_images == 2
    assert result.failed_images == 0
    assert result.fruit_type_accuracy == 0.5
    assert result.quality_accuracy == 1.0
    assert csv_path.exists()
    assert report_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["class_name"] == "freshapples"
    assert rows[0]["true_fruit_type"] == "apple"
    assert rows[0]["predicted_fruit_type"] == "apple"
    assert rows[0]["is_fruit_type_correct"] == "True"
    assert rows[1]["true_fruit_type"] == "banana"
    assert rows[1]["predicted_fruit_type"] == "apple"
    assert rows[1]["is_fruit_type_correct"] == "False"
    assert rows[1]["is_quality_correct"] == "True"

    report_text = report_path.read_text(encoding="utf-8")
    assert "End-to-End Test Evaluation Report" in report_text
    assert "Total images tested: 2" in report_text
    assert "Failed images: 0" in report_text
    assert "Fruit type accuracy: 0.5000" in report_text
    assert "Quality accuracy: 1.0000" in report_text
    assert "Classification report for fruit_type:" in report_text
    assert "Confusion matrix for quality:" in report_text
    assert "Market grade counts:" in report_text
    assert "Misclassified images:" in report_text
    assert "rottenbanana" in report_text

def test_evaluate_system_supports_max_per_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    for index in range(3):
        create_tiny_image(test_dir / "freshapples" / f"apple_{index}.png")
        create_tiny_image(test_dir / "rottenbanana" / f"banana_{index}.png")

    seen_paths: list[Path] = []

    def fake_predict_image(
        image_path: Path,
        fruit_model_path: Path,
        quality_model_path: Path,
        save_figure: bool = True,
    ) -> dict[str, object]:
        assert save_figure is False
        seen_paths.append(image_path)
        if image_path.parent.name == "freshapples":
            return fake_prediction("apple", "fresh")
        return fake_prediction("banana", "rotten")

    monkeypatch.setattr(system_evaluation, "predict_image", fake_predict_image)

    result = system_evaluation.evaluate_system_on_test_set(
        test_dir=test_dir,
        fruit_model_path=tmp_path / "fruit.pkl",
        quality_model_path=tmp_path / "quality.pkl",
        csv_output_path=tmp_path / "reports" / "predictions.csv",
        report_output_path=tmp_path / "reports" / "summary.txt",
        max_per_class=2,
    )

    assert result.total_images == 4
    assert len(seen_paths) == 4
    assert Counter(path.parent.name for path in seen_paths) == {"freshapples": 2, "rottenbanana": 2}

def test_evaluate_system_continues_when_prediction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    bad_image = test_dir / "freshapples" / "bad.png"
    good_image = test_dir / "freshapples" / "good.png"
    create_tiny_image(bad_image)
    create_tiny_image(good_image)

    def fake_predict_image(
        image_path: Path,
        fruit_model_path: Path,
        quality_model_path: Path,
        save_figure: bool = True,
    ) -> dict[str, object]:
        assert save_figure is False
        if image_path.name == "bad.png":
            raise RuntimeError("broken image")
        return fake_prediction("apple", "fresh")

    monkeypatch.setattr(system_evaluation, "predict_image", fake_predict_image)
    csv_path = tmp_path / "reports" / "predictions.csv"
    report_path = tmp_path / "reports" / "summary.txt"

    result = system_evaluation.evaluate_system_on_test_set(
        test_dir=test_dir,
        fruit_model_path=tmp_path / "fruit.pkl",
        quality_model_path=tmp_path / "quality.pkl",
        csv_output_path=csv_path,
        report_output_path=report_path,
    )

    assert result.total_images == 2
    assert result.failed_images == 1
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["error"] == "RuntimeError: broken image"
    assert rows[1]["error"] == ""

    report_text = report_path.read_text(encoding="utf-8")
    assert "Failed images: 1" in report_text
    assert "RuntimeError: broken image" in report_text
