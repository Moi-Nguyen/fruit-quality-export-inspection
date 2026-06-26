"""Prediction placeholders for trained fruit quality models."""

from pathlib import Path
from typing import Any

import numpy as np


def load_model(model_path: Path) -> Any:
    """Load a trained model from disk."""
    # TODO: Load joblib or pickle model files after training is implemented.
    raise NotImplementedError("Model loading is not implemented yet.")


def predict_image(image_array: np.ndarray, models_dir: Path) -> dict[str, Any]:
    """Predict fruit type and Fresh/Rotten quality for one image."""
    # TODO: Extract features, load models, and return prediction results.
    raise NotImplementedError("Image prediction is not implemented yet.")
