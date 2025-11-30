"""Data validation utilities for quality checks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with a DataFrame.

        Args:
            df: DataFrame to validate.
        """
        self.df = df
        self.issues: List[Dict[str, Any]] = []

    def check_missing_values(
        self, threshold: float = 0.5
    ) -> DataValidator:
        """
        Check for columns with excessive missing values.

        Args:
            threshold: Maximum allowed proportion of missing values (0-1).

        Returns:
            Self for method chaining.
        """
        missing_pct = self.df.isnull().sum() / len(self.df)
        problematic_cols = missing_pct[missing_pct > threshold]

        if not problematic_cols.empty:
            self.issues.append({
                "type": "missing_values",
                "severity": "high",
                "message": f"{len(problematic_cols)} columns exceed missing value threshold",
                "details": problematic_cols.to_dict()
            })
            logger.warning(
                f"Found {len(problematic_cols)} columns with >{threshold*100}% missing values"
            )

        return self

    def check_duplicates(self, subset: Optional[List[str]] = None) -> DataValidator:
        """
        Check for duplicate rows.

        Args:
            subset: List of columns to check for duplicates. If None, checks all.

        Returns:
            Self for method chaining.
        """
        n_duplicates = self.df.duplicated(subset=subset).sum()

        if n_duplicates > 0:
            self.issues.append({
                "type": "duplicates",
                "severity": "medium",
                "message": f"Found {n_duplicates} duplicate rows",
                "details": {"count": int(n_duplicates)}
            })
            logger.warning(f"Found {n_duplicates} duplicate rows")

        return self

    def check_data_types(
        self, expected_types: Dict[str, type]
    ) -> DataValidator:
        """
        Verify column data types match expectations.

        Args:
            expected_types: Dictionary mapping column names to expected types.

        Returns:
            Self for method chaining.
        """
        mismatches = []
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = self.df[col].dtype
                if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                    mismatches.append({
                        "column": col,
                        "expected": str(expected_type),
                        "actual": str(actual_type)
                    })

        if mismatches:
            self.issues.append({
                "type": "type_mismatch",
                "severity": "medium",
                "message": f"{len(mismatches)} columns have unexpected types",
                "details": mismatches
            })
            logger.warning(f"Found {len(mismatches)} type mismatches")

        return self

    def check_value_ranges(
        self, ranges: Dict[str, tuple[float, float]]
    ) -> DataValidator:
        """
        Check if numeric values fall within expected ranges.

        Args:
            ranges: Dictionary mapping column names to (min, max) tuples.

        Returns:
            Self for method chaining.
        """
        out_of_range = []
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                out_of_bounds = (
                    (self.df[col] < min_val) | (self.df[col] > max_val)
                ).sum()
                if out_of_bounds > 0:
                    out_of_range.append({
                        "column": col,
                        "expected_range": (min_val, max_val),
                        "violations": int(out_of_bounds)
                    })

        if out_of_range:
            self.issues.append({
                "type": "value_range",
                "severity": "medium",
                "message": f"{len(out_of_range)} columns have out-of-range values",
                "details": out_of_range
            })
            logger.warning(
                f"Found {len(out_of_range)} columns with out-of-range values"
            )

        return self

    def check_required_columns(
        self, required: List[str]
    ) -> DataValidator:
        """
        Verify all required columns are present.

        Args:
            required: List of required column names.

        Returns:
            Self for method chaining.
        """
        missing_cols = set(required) - set(self.df.columns)

        if missing_cols:
            self.issues.append({
                "type": "missing_columns",
                "severity": "high",
                "message": f"Missing {len(missing_cols)} required columns",
                "details": list(missing_cols)
            })
            logger.error(f"Missing required columns: {missing_cols}")

        return self

    def check_class_balance(
        self, target_col: str, imbalance_threshold: float = 0.2
    ) -> DataValidator:
        """
        Check for severe class imbalance in target variable.

        Args:
            target_col: Name of the target column.
            imbalance_threshold: Minimum proportion for smallest class.

        Returns:
            Self for method chaining.
        """
        if target_col not in self.df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return self

        class_counts = self.df[target_col].value_counts(normalize=True)
        min_proportion = class_counts.min()

        if min_proportion < imbalance_threshold:
            self.issues.append({
                "type": "class_imbalance",
                "severity": "low",
                "message": f"Severe class imbalance detected (smallest class: {min_proportion:.2%})",
                "details": class_counts.to_dict()
            })
            logger.warning(
                f"Class imbalance detected. Smallest class: {min_proportion:.2%}"
            )

        return self

    def get_report(self) -> Dict[str, Any]:
        """
        Get validation report.

        Returns:
            Dictionary containing validation results and issues.
        """
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "issues_found": len(self.issues),
            "issues": self.issues,
            "is_valid": len(self.issues) == 0
        }

    def validate_all(
        self,
        required_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all validation checks.

        Args:
            required_cols: List of required columns.
            target_col: Name of target column for class balance check.

        Returns:
            Validation report dictionary.
        """
        logger.info("Running comprehensive data validation...")

        if required_cols:
            self.check_required_columns(required_cols)

        self.check_missing_values()
        self.check_duplicates()

        if target_col:
            self.check_class_balance(target_col)

        report = self.get_report()
        logger.info(f"Validation complete. Found {report['issues_found']} issues.")
        return report
