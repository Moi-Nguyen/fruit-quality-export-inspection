"""Dataset sampling utilities for creating a balanced smaller dataset."""

from pathlib import Path


def create_balanced_sample(raw_data_dir: Path, sample_data_dir: Path, seed: int) -> None:
    """Create a balanced train/test sample dataset from the raw dataset."""
    # TODO: Randomly copy a fixed number of images per class using the given seed.
    raise NotImplementedError("Dataset sampling will be implemented in a later step.")
