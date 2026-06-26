"""Dataset sampling utilities for creating a balanced smaller dataset."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]
RAW_DATA_DIR = Path("data/raw")
SAMPLE_DATA_DIR = Path("data/sample")
TRAIN_SAMPLE_COUNT = 150
TEST_SAMPLE_COUNT = 50
RANDOM_SEED = 42


def list_image_files(directory: Path) -> list[Path]:
    """Return supported image files in a directory sorted by file name."""
    if not directory.exists() or not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def ensure_directory(path: Path) -> None:
    """Create a directory and its parents when they do not exist."""
    path.mkdir(parents=True, exist_ok=True)


def clean_directory(path: Path) -> None:
    """Remove all files and folders inside a directory without deleting it."""
    ensure_directory(path)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def copy_random_samples(
    source_dir: Path,
    destination_dir: Path,
    samples_per_class: int,
    class_names: list[str],
    seed: int,
) -> dict[str, int]:
    """Copy a fixed number of random images per class into destination folders."""
    counts: dict[str, int] = {}

    for class_name in class_names:
        class_source_dir = source_dir / class_name
        class_destination_dir = destination_dir / class_name
        clean_directory(class_destination_dir)

        image_files = list_image_files(class_source_dir)
        if len(image_files) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' in '{class_source_dir}' has "
                f"{len(image_files)} images, but {samples_per_class} are required."
            )

        class_random = random.Random(f"{seed}-{class_name}")
        selected_files = class_random.sample(image_files, samples_per_class)

        for source_file in selected_files:
            shutil.copy2(source_file, class_destination_dir / source_file.name)

        counts[class_name] = len(selected_files)

    return counts


def validate_dataset_structure(dataset_dir: Path, class_names: list[str]) -> bool:
    """Return True when a split directory contains every required class folder."""
    return dataset_dir.exists() and all((dataset_dir / class_name).is_dir() for class_name in class_names)


def count_images_by_class(split_dir: Path, class_names: list[str]) -> dict[str, int]:
    """Count supported image files for each class folder in a split directory."""
    return {class_name: len(list_image_files(split_dir / class_name)) for class_name in class_names}


def print_counts(dataset_name: str, dataset_dir: Path, class_names: list[str]) -> None:
    """Print image counts for train and test splits in a dataset directory."""
    print(f"\n{dataset_name}: {dataset_dir}")
    if not dataset_dir.exists():
        print("  Dataset directory does not exist.")
        return

    for split_name in ("train", "test"):
        split_dir = dataset_dir / split_name
        print(f"  {split_name}:")
        if not split_dir.exists():
            print("    Split directory does not exist.")
            continue

        is_valid = validate_dataset_structure(split_dir, class_names)
        if not is_valid:
            print("    Missing one or more class folders.")

        counts = count_images_by_class(split_dir, class_names)
        for class_name, count in counts.items():
            print(f"    {class_name}: {count}")


def check_datasets() -> None:
    """Print current class counts for raw and sampled datasets."""
    print_counts("Raw dataset", RAW_DATA_DIR, CLASS_NAMES)
    print_counts("Sample dataset", SAMPLE_DATA_DIR, CLASS_NAMES)


def create_sample_dataset() -> None:
    """Create the balanced sample dataset from the raw train and test folders."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Place the original dataset under 'data/raw/' first."
        )

    train_counts = copy_random_samples(
        source_dir=RAW_DATA_DIR / "train",
        destination_dir=SAMPLE_DATA_DIR / "train",
        samples_per_class=TRAIN_SAMPLE_COUNT,
        class_names=CLASS_NAMES,
        seed=RANDOM_SEED,
    )
    test_counts = copy_random_samples(
        source_dir=RAW_DATA_DIR / "test",
        destination_dir=SAMPLE_DATA_DIR / "test",
        samples_per_class=TEST_SAMPLE_COUNT,
        class_names=CLASS_NAMES,
        seed=RANDOM_SEED,
    )

    print("Sample dataset created successfully.")
    print(f"Source: {RAW_DATA_DIR}")
    print(f"Destination: {SAMPLE_DATA_DIR}")
    print("\nCopied train images:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count}")
    print("\nCopied test images:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count}")


# Backward-compatible wrapper for the original skeleton function name.
def create_balanced_sample(raw_data_dir: Path, sample_data_dir: Path, seed: int) -> None:
    """Create a balanced sample using the project default class names and counts."""
    copy_random_samples(
        source_dir=raw_data_dir / "train",
        destination_dir=sample_data_dir / "train",
        samples_per_class=TRAIN_SAMPLE_COUNT,
        class_names=CLASS_NAMES,
        seed=seed,
    )
    copy_random_samples(
        source_dir=raw_data_dir / "test",
        destination_dir=sample_data_dir / "test",
        samples_per_class=TEST_SAMPLE_COUNT,
        class_names=CLASS_NAMES,
        seed=seed,
    )
