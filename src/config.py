"""
Configuration management for the project.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import yaml


# Find project root dynamically
# This file is in src/config.py, so project root is one level up
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Reports directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Config files
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class DataConfig:
    """Data configuration."""

    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    lob_depth: int = 20


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    use_basic_features: bool = True
    use_book_shape: bool = True
    use_trade_features: bool = True
    use_hawkes: bool = True
    pca_n_components: int = 10


@dataclass
class ModelConfig:
    """Model configuration."""

    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 20
    rf_min_samples_split: int = 10

    # Gradient Boosting
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 5

    # General
    random_state: int = 42
    n_cv_folds: int = 5


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Debug: Print paths when config is imported (optional, remove later)
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
