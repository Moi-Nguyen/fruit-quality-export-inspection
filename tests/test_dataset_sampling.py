"""Tests for dataset sampling utilities."""

from pathlib import Path

from src.dataset_sampling import (
    copy_random_samples,
    count_images_by_class,
    list_image_files,
    validate_dataset_structure,
)


def create_file(path: Path) -> None:
    """Create a small placeholder file for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test file", encoding="utf-8")


def test_list_image_files_includes_supported_extensions(tmp_path: Path) -> None:
    """Image listing should include common supported image extensions."""
    create_file(tmp_path / "apple.JPG")
    create_file(tmp_path / "banana.jpeg")
    create_file(tmp_path / "orange.png")
    create_file(tmp_path / "fruit.bmp")
    create_file(tmp_path / "fruit.webp")

    files = list_image_files(tmp_path)

    assert [file.name for file in files] == [
        "apple.JPG",
        "banana.jpeg",
        "fruit.bmp",
        "fruit.webp",
        "orange.png",
    ]


def test_list_image_files_ignores_unsupported_files(tmp_path: Path) -> None:
    """Image listing should ignore text files and unsupported extensions."""
    create_file(tmp_path / "apple.jpg")
    create_file(tmp_path / "notes.txt")
    create_file(tmp_path / "archive.zip")

    files = list_image_files(tmp_path)

    assert [file.name for file in files] == ["apple.jpg"]


def test_sampling_copies_correct_number_of_files(tmp_path: Path) -> None:
    """Sampling should copy the requested number of files for each class."""
    source_dir = tmp_path / "raw" / "train"
    destination_dir = tmp_path / "sample" / "train"
    class_names = ["freshapples", "rottenapples"]

    for class_name in class_names:
        for index in range(5):
            create_file(source_dir / class_name / f"image_{index}.jpg")

    counts = copy_random_samples(
        source_dir=source_dir,
        destination_dir=destination_dir,
        samples_per_class=3,
        class_names=class_names,
        seed=42,
    )

    assert counts == {"freshapples": 3, "rottenapples": 3}
    assert count_images_by_class(destination_dir, class_names) == counts


def test_sampling_preserves_class_folder_structure(tmp_path: Path) -> None:
    """Sampling should create destination folders using the original class names."""
    source_dir = tmp_path / "raw" / "test"
    destination_dir = tmp_path / "sample" / "test"
    class_names = ["freshbanana", "rottenbanana"]

    for class_name in class_names:
        for index in range(2):
            create_file(source_dir / class_name / f"image_{index}.png")

    copy_random_samples(
        source_dir=source_dir,
        destination_dir=destination_dir,
        samples_per_class=1,
        class_names=class_names,
        seed=42,
    )

    assert (destination_dir / "freshbanana").is_dir()
    assert (destination_dir / "rottenbanana").is_dir()


def test_validation_detects_missing_class_folders(tmp_path: Path) -> None:
    """Validation should fail when any required class folder is missing."""
    split_dir = tmp_path / "sample" / "train"
    (split_dir / "freshoranges").mkdir(parents=True)

    is_valid = validate_dataset_structure(
        split_dir,
        ["freshoranges", "rottenoranges"],
    )

    assert is_valid is False
