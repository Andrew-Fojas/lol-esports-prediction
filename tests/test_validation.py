"""Tests for data validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation import DataValidator


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_check_missing_values_no_issues(self):
        """Test that no issues are found when data is complete."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        validator = DataValidator(df)
        validator.check_missing_values()

        assert len(validator.issues) == 0

    def test_check_missing_values_finds_issues(self):
        """Test that issues are found when missing values exceed threshold."""
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [np.nan, np.nan, np.nan, 40, 50]}  # 60% missing
        )

        validator = DataValidator(df)
        validator.check_missing_values(threshold=0.5)

        assert len(validator.issues) == 1
        assert validator.issues[0]["type"] == "missing_values"
        assert validator.issues[0]["severity"] == "high"

    def test_check_duplicates_no_issues(self):
        """Test that no issues are found when there are no duplicates."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        validator = DataValidator(df)
        validator.check_duplicates()

        assert len(validator.issues) == 0

    def test_check_duplicates_finds_issues(self):
        """Test that duplicates are detected."""
        df = pd.DataFrame({"a": [1, 2, 2, 3, 3], "b": [10, 20, 20, 30, 30]})

        validator = DataValidator(df)
        validator.check_duplicates()

        assert len(validator.issues) == 1
        assert validator.issues[0]["type"] == "duplicates"
        assert validator.issues[0]["details"]["count"] == 2

    def test_check_required_columns_all_present(self):
        """Test when all required columns are present."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})

        validator = DataValidator(df)
        validator.check_required_columns(["a", "b"])

        assert len(validator.issues) == 0

    def test_check_required_columns_missing(self):
        """Test when required columns are missing."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        validator = DataValidator(df)
        validator.check_required_columns(["a", "b", "c", "d"])

        assert len(validator.issues) == 1
        assert validator.issues[0]["type"] == "missing_columns"
        assert set(validator.issues[0]["details"]) == {"c", "d"}

    def test_check_class_balance_balanced(self):
        """Test with balanced classes."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1]})

        validator = DataValidator(df)
        validator.check_class_balance("target", imbalance_threshold=0.2)

        assert len(validator.issues) == 0

    def test_check_class_balance_imbalanced(self):
        """Test with imbalanced classes."""
        df = pd.DataFrame({"target": [0] * 90 + [1] * 10})  # 10% minority class

        validator = DataValidator(df)
        validator.check_class_balance("target", imbalance_threshold=0.2)

        assert len(validator.issues) == 1
        assert validator.issues[0]["type"] == "class_imbalance"

    def test_get_report_structure(self):
        """Test that report has correct structure."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        validator = DataValidator(df)
        report = validator.get_report()

        assert "total_rows" in report
        assert "total_columns" in report
        assert "issues_found" in report
        assert "issues" in report
        assert "is_valid" in report
        assert report["total_rows"] == 3
        assert report["total_columns"] == 1

    def test_method_chaining(self):
        """Test that validator methods can be chained."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        report = (
            DataValidator(df)
            .check_missing_values()
            .check_duplicates()
            .check_required_columns(["a", "b"])
            .get_report()
        )

        assert isinstance(report, dict)
        assert report["is_valid"] is True
