"""Data preprocessing utilities."""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    COMPLETE_TEAM_DATA_FILE,
    FEATURE_COLUMNS,
    TEAM_DATA_FILE,
    TEAM_METRICS_FILE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_team_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only team-level observations.

    Individual player statistics are removed, keeping only aggregated
    team performance metrics.

    Args:
        df: Raw DataFrame containing both player and team observations.

    Returns:
        DataFrame filtered to team-level data only.
    """
    team_data = df[df["position"] == "team"].copy()
    team_data.reset_index(drop=True, inplace=True)

    logger.info(f"Original rows: {len(df):,}")
    logger.info(f"Team-level rows: {len(team_data):,}")
    logger.info(f"Removed {len(df) - len(team_data):,} player-level observations")

    return team_data


def filter_complete_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only complete observations.

    Removes rows with incomplete data to ensure data quality.

    Args:
        df: DataFrame to filter.

    Returns:
        DataFrame containing only complete observations.
    """
    complete_data = df[df["datacompleteness"] == "complete"].copy()
    complete_data.reset_index(drop=True, inplace=True)

    logger.info(f"Original rows: {len(df):,}")
    logger.info(f"Complete rows: {len(complete_data):,}")
    logger.info(f"Removed {len(df) - len(complete_data):,} incomplete observations")

    return complete_data


def filter_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant team-wide statistics for modeling.

    Filters to essential features including game objectives, economy,
    vision control, and teamfight metrics.

    Args:
        df: DataFrame containing team data.

    Returns:
        DataFrame with selected feature columns only.
    """
    # Keep only columns that exist in the dataframe
    available_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_columns = set(FEATURE_COLUMNS) - set(available_columns)

    if missing_columns:
        logger.warning(f"Missing columns in dataset: {missing_columns}")

    team_data = df[available_columns].copy()
    team_data.reset_index(drop=True, inplace=True)

    logger.info(f"Selected {len(available_columns)} feature columns")

    return team_data


def preprocess_pipeline(
    input_filepath: Path, save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Pipeline steps:
    1. Load raw data
    2. Filter to team-level observations
    3. Filter to complete observations only
    4. Select relevant features

    Args:
        input_filepath: Path to raw data CSV file.
        save_intermediate: Whether to save intermediate outputs.

    Returns:
        Preprocessed DataFrame ready for feature engineering.
    """
    from src.data.loader import load_raw_data

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    # Step 1: Load raw data
    df = load_raw_data(input_filepath)

    # Step 2: Filter to team data
    logger.info("\nStep 1: Filtering to team-level data")
    team_data = filter_team_data(df)
    if save_intermediate:
        team_data.to_csv(TEAM_DATA_FILE, index=False)
        logger.info(f"Saved to {TEAM_DATA_FILE}")

    # Step 3: Filter to complete data
    logger.info("\nStep 2: Filtering to complete observations")
    complete_data = filter_complete_data(team_data)
    if save_intermediate:
        complete_data.to_csv(COMPLETE_TEAM_DATA_FILE, index=False)
        logger.info(f"Saved to {COMPLETE_TEAM_DATA_FILE}")

    # Step 4: Select relevant features
    logger.info("\nStep 3: Selecting relevant features")
    metrics_data = filter_team_features(complete_data)
    if save_intermediate:
        metrics_data.to_csv(TEAM_METRICS_FILE, index=False)
        logger.info(f"Saved to {TEAM_METRICS_FILE}")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing pipeline completed successfully")
    logger.info(
        f"Final dataset: {len(metrics_data):,} rows × "
        f"{len(metrics_data.columns)} columns"
    )
    logger.info("=" * 60)

    return metrics_data
