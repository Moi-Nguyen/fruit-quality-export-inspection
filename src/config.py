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

# Export suitability rule thresholds.
HIGH_DEFECT_RATIO_THRESHOLD: float = 0.20
MEDIUM_DEFECT_RATIO_THRESHOLD: float = 0.10
LOW_CIRCULARITY_THRESHOLD: float = 0.35
MIN_MASK_AREA_RATIO_THRESHOLD: float = 0.05

# Final market grading thresholds.
EXPORT_MAX_DEFECT_RATIO: float = 0.05
DOMESTIC_MAX_DEFECT_RATIO: float = 0.15
MIN_REASONABLE_MASK_RATIO: float = 0.05
MAX_REASONABLE_MASK_RATIO: float = 0.85

# Initial preprocessing values.
# These will be explored later in parameter sweep experiments.
DEFAULT_GAUSSIAN_KERNEL_SIZE: int = 5
DEFAULT_GAUSSIAN_SIGMA: float = 1.0
DEFAULT_MEDIAN_KERNEL_SIZE: int = 3


# Initial segmentation defaults.
# These values will be explored later in parameter sweep experiments.
OTSU_NUM_BINS: int = 256
DEFAULT_MORPH_KERNEL_SIZE: int = 3
DEFAULT_MORPH_SHAPE: str = "square"
DEFAULT_COMPONENT_CONNECTIVITY: int = 8
BORDER_FOREGROUND_RATIO_THRESHOLD: float = 0.5

# Initial empirical dataset-specific segmentation values.
# These values will be explored later in parameter sweep experiments.
WHITE_BACKGROUND_THRESHOLD: int = 240
COLOR_DIFFERENCE_THRESHOLD: int = 20
STRICT_COLOR_DIFFERENCE_THRESHOLD: int = 35
BACKGROUND_BORDER_WIDTH: int = 10
BACKGROUND_COLOR_DISTANCE_THRESHOLD: float = 45.0
STRICT_BACKGROUND_COLOR_DISTANCE_THRESHOLD: float = 65.0
BLACK_BORDER_THRESHOLD: int = 15
MAX_REASONABLE_FRUIT_AREA_RATIO: float = 0.80
MIN_REASONABLE_FRUIT_AREA_RATIO: float = 0.005
BORDER_MARGIN: int = 2

# Initial empirical image quality thresholds.
# These values will be adjusted later after parameter sweep experiments.
LOW_DEFECT_RATIO_THRESHOLD: float = MEDIUM_DEFECT_RATIO_THRESHOLD
MIN_CIRCULARITY_THRESHOLD: float = 0.40
MAX_CIRCULARITY_THRESHOLD: float = 1.20
