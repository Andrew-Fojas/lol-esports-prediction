"""Tests for data loading utilities."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.data.loader import load_raw_data, load_processed_data


class TestLoadRawData:
    """Tests for load_raw_data function."""

    def test_load_raw_data_file_not_found(self, temp_dir: Path):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        non_existent_file = temp_dir / "non_existent.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_raw_data(non_existent_file)

        assert "Raw data file not found" in str(exc_info.value)

    def test_load_raw_data_success(
        self, temp_dir: Path, sample_raw_data: pd.DataFrame
    ):
        """Test successful data loading."""
        test_file = temp_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)

        df = load_raw_data(test_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_data)
        assert list(df.columns) == list(sample_raw_data.columns)


class TestLoadProcessedData:
    """Tests for load_processed_data function."""

    def test_load_processed_data_invalid_type(self):
        """Test that ValueError is raised for invalid data type."""
        with pytest.raises(ValueError) as exc_info:
            load_processed_data("invalid_type")  # type: ignore

        assert "Invalid data_type" in str(exc_info.value)

    def test_load_processed_data_file_not_found(self, monkeypatch, tmp_path):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        # Create a non-existent file path
        non_existent = tmp_path / "non_existent.csv"

        # Mock the filepath in the loader module where it's used
        monkeypatch.setattr(
            "src.data.loader.PCA_TRANSFORMED_FILE",
            non_existent
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            load_processed_data("pca")

        assert "Processed data file not found" in str(exc_info.value)
