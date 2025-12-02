"""Data loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.config import (
    COMPLETE_TEAM_DATA_FILE,
    PCA_TRANSFORMED_FILE,
    RAW_DATA_FILE,
    TEAM_DATA_FILE,
    TEAM_METRICS_FILE,
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw LoL esports match data from Oracle's Elixir.

    Args:
        filepath: Path to the raw CSV file. If None, uses default from config.

    Returns:
        DataFrame containing raw match data.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if filepath is None:
        filepath = RAW_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {filepath}. "
            f"Please download from https://oracleselixir.com/tools/downloads"
        )

    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


def load_processed_data(
    data_type: Literal["team", "complete", "metrics", "pca"] = "pca"
) -> pd.DataFrame:
    """
    Load processed data at various stages of the pipeline.

    Args:
        data_type: Type of processed data to load. Options:
            - 'team': Team-level filtered data
            - 'complete': Complete team data (no missing values)
            - 'metrics': Selected team metrics
            - 'pca': PCA-transformed features (default)

    Returns:
        DataFrame containing processed data.

    Raises:
        ValueError: If data_type is invalid.
        FileNotFoundError: If the data file doesn't exist.
    """
    file_map = {
        'team': TEAM_DATA_FILE,
        'complete': COMPLETE_TEAM_DATA_FILE,
        'metrics': TEAM_METRICS_FILE,
        'pca': PCA_TRANSFORMED_FILE
    }

    if data_type not in file_map:
        raise ValueError(
            f"Invalid data_type '{data_type}'. "
            f"Choose from: {list(file_map.keys())}"
        )

    filepath = file_map[data_type]

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data file not found at {filepath}. "
            f"Run the preprocessing pipeline first."
        )

    logger.info(f"Loading {data_type} data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    return df
