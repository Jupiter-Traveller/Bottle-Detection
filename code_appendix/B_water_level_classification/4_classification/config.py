"""Central configuration for the liquid residue pretest workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
DEFAULT_ANNOTATION_CSV = DATA_DIR / "annotations.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
CROP_DIR = OUTPUT_DIR / "crops"
FEATURE_DIR = OUTPUT_DIR / "features"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
LEVEL_LINE_DIR = FIGURE_DIR / "level_line_examples"

REQUIRED_COLUMNS = [
    "image_id",
    "image_path",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "has_liquid",
    "liquid_level",
    "pose",
    "liquid_type",
    "source",
]

IMAGE_SIZE_FOR_FEATURES = (128, 256)  # width, height for fixed-size traditional features
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    """Create the unified per-run output root."""
    for path in [OUTPUT_DIR, RUNS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
