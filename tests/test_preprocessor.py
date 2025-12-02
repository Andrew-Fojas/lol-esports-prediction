"""Tests for data preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from src.data.preprocessor import (
    filter_complete_data,
    filter_team_data,
    filter_team_features,
)


class TestFilterTeamData:
    """Tests for filter_team_data function."""

    def test_filter_team_data_removes_players(self, sample_raw_data: pd.DataFrame):
        """Test that only team-level data is retained."""
        # Add some player rows
        player_data = sample_raw_data.copy()
        player_data["position"] = "top"
        combined = pd.concat([sample_raw_data, player_data], ignore_index=True)

        result = filter_team_data(combined)

        assert len(result) == len(sample_raw_data)
        assert all(result["position"] == "team")

    def test_filter_team_data_preserves_columns(self, sample_raw_data: pd.DataFrame):
        """Test that all columns are preserved."""
        result = filter_team_data(sample_raw_data)

        assert set(result.columns) == set(sample_raw_data.columns)


class TestFilterCompleteData:
    """Tests for filter_complete_data function."""

    def test_filter_complete_data_removes_incomplete(self):
        """Test that incomplete data is filtered out."""
        df = pd.DataFrame(
            {
                "datacompleteness": ["complete", "partial", "complete", "partial"],
                "value": [1, 2, 3, 4],
            }
        )

        result = filter_complete_data(df)

        assert len(result) == 2
        assert all(result["datacompleteness"] == "complete")
        assert list(result["value"]) == [1, 3]


class TestFilterTeamFeatures:
    """Tests for filter_team_features function."""

    def test_filter_team_features_selects_columns(self, sample_raw_data: pd.DataFrame):
        """Test that only relevant features are selected."""
        from src.config import FEATURE_COLUMNS

        result = filter_team_features(sample_raw_data)

        # Check that all columns in result are in FEATURE_COLUMNS
        assert all(col in FEATURE_COLUMNS for col in result.columns)

    def test_filter_team_features_handles_missing_columns(self):
        """Test that function handles missing expected columns gracefully."""
        df = pd.DataFrame({"result": [0, 1, 0], "teamkills": [10, 15, 12]})

        # Should not raise an error
        result = filter_team_features(df)

        assert isinstance(result, pd.DataFrame)
        assert "result" in result.columns
        assert "teamkills" in result.columns
