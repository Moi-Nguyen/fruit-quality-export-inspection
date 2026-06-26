"""Machine learning training placeholders for fruit classification."""

from pathlib import Path

import pandas as pd


def build_feature_table(sample_data_dir: Path) -> pd.DataFrame:
    """Build a feature table from the sampled dataset."""
    # TODO: Run the image pipeline for each image and collect labels/features.
    raise NotImplementedError("Feature table generation is not implemented yet.")


def train_models(feature_table: pd.DataFrame, models_dir: Path) -> None:
    """Train Random Forest, KNN, and SVM models."""
    # TODO: Train and save baseline scikit-learn models.
    raise NotImplementedError("Model training is not implemented yet.")
