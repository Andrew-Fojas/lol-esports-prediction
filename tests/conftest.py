"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Create sample raw esports data for testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame(
        {
            "gameid": [f"game_{i}" for i in range(n_rows)],
            "position": ["team"] * n_rows,
            "datacompleteness": ["complete"] * n_rows,
            "teamname": [f"Team_{i % 10}" for i in range(n_rows)],
            "result": np.random.randint(0, 2, n_rows),
            "gamelength": np.random.uniform(1500, 3000, n_rows),
            "teamkills": np.random.randint(5, 30, n_rows),
            "teamdeaths": np.random.randint(5, 30, n_rows),
            "dragons": np.random.randint(0, 5, n_rows),
            "barons": np.random.randint(0, 3, n_rows),
            "towers": np.random.randint(0, 11, n_rows),
            "totalgold": np.random.uniform(40000, 80000, n_rows),
            "golddiffat15": np.random.uniform(-3000, 3000, n_rows),
        }
    )


@pytest.fixture
def sample_team_data() -> pd.DataFrame:
    """Create sample team-level data."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame(
        {
            "result": np.random.randint(0, 2, n_rows),
            "teamkills": np.random.randint(5, 30, n_rows),
            "teamdeaths": np.random.randint(5, 30, n_rows),
            "dragons": np.random.randint(0, 5, n_rows),
            "barons": np.random.randint(0, 3, n_rows),
            "towers": np.random.randint(0, 11, n_rows),
            "totalgold": np.random.uniform(40000, 80000, n_rows),
        }
    )


@pytest.fixture
def sample_pca_data() -> pd.DataFrame:
    """Create sample PCA-transformed data."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame(
        {
            "result": np.random.randint(0, 2, n_rows),
            "PC1": np.random.randn(n_rows),
            "PC2": np.random.randn(n_rows),
            "PC3": np.random.randn(n_rows),
        }
    )


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    return tmp_path
