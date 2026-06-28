"""Batch feature extraction and CSV export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from src.features import extract_features_from_image
from src.io_utils import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

_CLASS_LABELS = {
    "freshapples": ("apple", "fresh"),
    "freshbanana": ("banana", "fresh"),
    "freshoranges": ("orange", "fresh"),
    "rottenapples": ("apple", "rotten"),
    "rottenbanana": ("banana", "rotten"),
    "rottenoranges": ("orange", "rotten"),
}

_METADATA_COLUMNS = ["image_path", "file_name", "class_name", "fruit_type", "quality"]


def parse_class_name(class_name: str) -> tuple[str, str]:
    """Convert a dataset folder name into fruit type and quality labels."""
    if class_name not in _CLASS_LABELS:
        raise ValueError(f"Unknown class name: {class_name}")
    return _CLASS_LABELS[class_name]


def iter_image_paths(split_dir: Path) -> list[Path]:
    """Return sorted image paths from class folders in a split directory."""
    image_paths: list[Path] = []
    if not split_dir.exists():
        return image_paths

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    return sorted(image_paths)


def extract_features_for_image_path(image_path: Path, class_name: str) -> dict[str, object]:
    """Load one image and return metadata, labels, and handcrafted features."""
    fruit_type, quality = parse_class_name(class_name)
    image = load_image(image_path)
    extracted_features = extract_features_from_image(image)

    row: dict[str, object] = {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "class_name": class_name,
        "fruit_type": fruit_type,
        "quality": quality,
    }
    row.update(extracted_features)
    return row


def extract_features_for_split(
    split_dir: Path,
    continue_on_error: bool = True,
) -> list[dict[str, object]]:
    """Extract feature rows for every image in one dataset split."""
    rows: list[dict[str, object]] = []
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for index, image_path in enumerate(image_paths, start=1):
            print(f"Processing {class_name}: {index}/{len(image_paths)}", flush=True)
            try:
                rows.append(extract_features_for_image_path(image_path, class_name))
            except Exception as error:
                if not continue_on_error:
                    raise
                print(f"Warning: skipped {image_path}: {error}", flush=True)

    return rows


def write_feature_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write feature rows to a CSV file with stable column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    feature_columns = sorted(key for key in all_keys if key not in _METADATA_COLUMNS)
    fieldnames = _METADATA_COLUMNS + feature_columns

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_sample_features(sample_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Export train and test sample feature CSV files."""
    train_csv_path = output_dir / "train_features.csv"
    test_csv_path = output_dir / "test_features.csv"

    train_rows = extract_features_for_split(sample_root / "train")
    test_rows = extract_features_for_split(sample_root / "test")

    write_feature_csv(train_rows, train_csv_path)
    write_feature_csv(test_rows, test_csv_path)

    return train_csv_path, test_csv_path

