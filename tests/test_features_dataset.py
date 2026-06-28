from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from src.features_dataset import (
    export_sample_features,
    extract_features_for_split,
    iter_image_paths,
    parse_class_name,
    write_feature_csv,
)


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("freshapples", ("apple", "fresh")),
        ("freshbanana", ("banana", "fresh")),
        ("freshoranges", ("orange", "fresh")),
        ("rottenapples", ("apple", "rotten")),
        ("rottenbanana", ("banana", "rotten")),
        ("rottenoranges", ("orange", "rotten")),
    ],
)
def test_parse_class_name_returns_fruit_type_and_quality(class_name: str, expected: tuple[str, str]) -> None:
    assert parse_class_name(class_name) == expected


def test_parse_class_name_raises_for_unknown_class() -> None:
    with pytest.raises(ValueError):
        parse_class_name("freshpears")


def test_iter_image_paths_returns_only_sorted_image_files(tmp_path: Path) -> None:
    class_dir = tmp_path / "freshapples"
    class_dir.mkdir()
    (class_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    _save_rgb_image(class_dir / "b.png")
    _save_rgb_image(class_dir / "a.jpg")
    _save_rgb_image(class_dir / "c.bmp")

    paths = iter_image_paths(tmp_path)

    assert [path.name for path in paths] == ["a.jpg", "b.png", "c.bmp"]


def test_write_feature_csv_creates_file_with_metadata_first(tmp_path: Path) -> None:
    output_path = tmp_path / "features" / "train_features.csv"
    rows = [
        {
            "mean_r": 120.0,
            "quality": "fresh",
            "image_path": "image.png",
            "class_name": "freshapples",
            "fruit_type": "apple",
            "file_name": "image.png",
        }
    ]

    write_feature_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

    assert output_path.exists()
    assert header[:5] == ["image_path", "file_name", "class_name", "fruit_type", "quality"]


def test_extract_features_for_split_returns_rows_for_tiny_dataset(tmp_path: Path) -> None:
    class_dir = tmp_path / "freshapples"
    class_dir.mkdir()
    _save_rgb_image(class_dir / "apple1.png", color=(220, 30, 30))
    _save_rgb_image(class_dir / "apple2.png", color=(210, 40, 40))

    rows = extract_features_for_split(tmp_path)

    assert len(rows) == 2
    assert rows[0]["fruit_type"] == "apple"
    assert rows[0]["quality"] == "fresh"
    assert "area" in rows[0]
    assert "mean_r" in rows[0]
    assert "defect_ratio" in rows[0]


def test_export_sample_features_creates_train_and_test_csvs(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    for split in ["train", "test"]:
        class_dir = sample_root / split / "freshbanana"
        class_dir.mkdir(parents=True)
        _save_rgb_image(class_dir / f"{split}.png", color=(230, 220, 40))

    output_dir = tmp_path / "outputs" / "features"

    train_csv_path, test_csv_path = export_sample_features(sample_root, output_dir)

    assert train_csv_path.exists()
    assert test_csv_path.exists()
    assert train_csv_path.name == "train_features.csv"
    assert test_csv_path.name == "test_features.csv"


def _save_rgb_image(path: Path, color: tuple[int, int, int] = (200, 50, 50)) -> None:
    image = Image.new("RGB", (12, 12), color)
    image.save(path)
