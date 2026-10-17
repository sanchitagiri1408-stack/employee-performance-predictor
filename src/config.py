"""Project-wide configuration and artifact locations."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_NAME = "Employee Performance Predictor using Data Analytics"
DEFAULT_RANDOM_SEED = 42
TARGET_COLUMN = "performance_label"
ID_COLUMN = "employee_id"
POSITIVE_LABEL = "High"
NEGATIVE_LABEL = "Needs Improvement"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one reproducible end-to-end pipeline run."""

    n_samples: int = 1000
    test_size: float = 0.20
    random_seed: int = DEFAULT_RANDOM_SEED
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def __post_init__(self) -> None:
        if self.n_samples < 30:
            raise ValueError("n_samples must be at least 30 for a stratified train/test split")
        if not 0.05 <= self.test_size < 0.50:
            raise ValueError("test_size must be between 0.05 and 0.49")

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def images_dir(self) -> Path:
        return self.project_root / "images"

    def ensure_directories(self) -> None:
        """Create the artifact directories when running from a clean checkout."""

        for directory in (self.data_dir, self.models_dir, self.outputs_dir, self.images_dir):
            directory.mkdir(parents=True, exist_ok=True)
