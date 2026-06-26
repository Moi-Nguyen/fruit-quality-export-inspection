"""Configuration constants for the fruit quality inspection project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
SAMPLE_TRAIN_DIR: Path = SAMPLE_DATA_DIR / "train"
SAMPLE_TEST_DIR: Path = SAMPLE_DATA_DIR / "test"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MASKS_DIR: Path = OUTPUTS_DIR / "masks"
DEFECT_MAPS_DIR: Path = OUTPUTS_DIR / "defect_maps"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

RANDOM_SEED: int = 42
TRAIN_SAMPLES_PER_CLASS: int = 150
TEST_SAMPLES_PER_CLASS: int = 50
IMAGE_SIZE: tuple[int, int] = (128, 128)

CLASS_NAMES: list[str] = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

FRUIT_TYPES: list[str] = ["apple", "banana", "orange"]
QUALITY_LABELS: list[str] = ["fresh", "rotten"]

DARK_BRIGHTNESS_THRESHOLD: float = 70.0
LOW_CONTRAST_THRESHOLD: float = 20.0
HIGH_NOISE_THRESHOLD: float = 15.0

# Initial empirical image quality thresholds.
# These values will be adjusted later after parameter sweep experiments.
LOW_DEFECT_RATIO_THRESHOLD: float = 0.10
HIGH_DEFECT_RATIO_THRESHOLD: float = 0.30
MIN_CIRCULARITY_THRESHOLD: float = 0.40
MAX_CIRCULARITY_THRESHOLD: float = 1.20
