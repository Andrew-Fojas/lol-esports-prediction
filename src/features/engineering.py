"""Feature engineering utilities including PCA transformation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional

from src.config import (
    EXCLUDE_COLUMNS,
    N_PCA_COMPONENTS,
    PCA_COMPONENT_NAMES,
    PCA_TRANSFORMED_FILE,
    RANDOM_STATE,
    RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_pca_with_permutation_test(
    data: pd.DataFrame,
    n_permutations: int = 1000,
    plot: bool = True,
    save_path: Optional[Path] = None
) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Fit PCA and perform permutation testing to validate components.

    Args:
        data: DataFrame with features (result column will be excluded).
        n_permutations: Number of permutation iterations.
        plot: Whether to create visualization of results.
        save_path: Path to save the plot (if None, saves to results/).

    Returns:
        Tuple of (fitted PCA object, fitted scaler, p-values array).
    """
    logger.info("Starting PCA with permutation testing")

    exclude_cols = [col for col in EXCLUDE_COLUMNS if col in data.columns]
    if 'result' in data.columns:
        exclude_cols.append('result')

    X = data.drop(columns=exclude_cols, errors='ignore')
    X = X.fillna(X.mean())
    logger.info(f"Feature matrix shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    real_variance = pca.explained_variance_ratio_

    logger.info(f"Explained variance by first {N_PCA_COMPONENTS} components: "
                f"{real_variance[:N_PCA_COMPONENTS].sum():.3f}")

    logger.info(f"Running {n_permutations} permutations...")
    permuted_variance = np.zeros((n_permutations, X_scaled.shape[1]))

    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            logger.info(f"  Permutation {i + 1}/{n_permutations}")

        shuffled = X_scaled.copy()
        for col in shuffled.columns:
            shuffled[col] = shuffled[col].sample(len(shuffled)).values

        pca_perm = PCA(random_state=RANDOM_STATE)
        pca_perm.fit(shuffled)
        permuted_variance[i, :] = pca_perm.explained_variance_ratio_

    p_values = np.sum(permuted_variance > real_variance, axis=0) / n_permutations

    sig_components = np.sum(p_values < 0.05)
    logger.info(f"Number of significant components (p < 0.05): {sig_components}")

    if plot:
        if save_path is None:
            save_path = RESULTS_DIR / "pca_permutation_test.png"

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(p_values) + 1), p_values, 'bo-', label='P-values')
        plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        plt.xlabel('Principal Component')
        plt.ylabel('p-value')
        plt.title('PCA Permutation Test Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved permutation test plot to {save_path}")
        plt.close()

    return pca, scaler, p_values


def apply_pca_transformation(
    data: pd.DataFrame,
    pca: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
    n_components: int = N_PCA_COMPONENTS
) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Apply PCA transformation to features.

    Args:
        data: DataFrame with features (result column will be preserved).
        pca: Pre-fitted PCA object (if None, will fit new one).
        scaler: Pre-fitted scaler (if None, will fit new one).
        n_components: Number of components to keep.

    Returns:
        Tuple of (transformed DataFrame, PCA object, scaler object).
    """
    logger.info("Applying PCA transformation")

    # Preserve result column if it exists
    result_col = data['result'] if 'result' in data.columns else None

    # Prepare features
    exclude_cols = [col for col in EXCLUDE_COLUMNS if col in data.columns]
    if 'result' in data.columns:
        exclude_cols.append('result')

    X = data.drop(columns=exclude_cols, errors='ignore')
    X = X.fillna(X.mean())

    # Fit scaler and PCA if not provided
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if pca is None:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_transformed = pca.fit_transform(X_scaled)
    else:
        X_transformed = pca.transform(X_scaled)[:, :n_components]

    # Create DataFrame with named components
    component_names = [PCA_COMPONENT_NAMES[f'PC{i+1}'] for i in range(n_components)]
    df_transformed = pd.DataFrame(X_transformed, columns=component_names)

    # Add result column back
    if result_col is not None:
        df_transformed.insert(0, 'result', result_col.values)

    logger.info(f"PCA transformation complete: {df_transformed.shape}")

    return df_transformed, pca, scaler


def create_pca_features(
    data: pd.DataFrame,
    run_permutation_test: bool = True,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Complete PCA feature engineering pipeline.

    Args:
        data: Input DataFrame with raw features.
        run_permutation_test: Whether to run permutation testing.
        save_output: Whether to save transformed data.

    Returns:
        DataFrame with PCA-transformed features.
    """
    logger.info("=" * 60)
    logger.info("Starting PCA feature engineering")
    logger.info("=" * 60)

    # Fit PCA with optional permutation test
    if run_permutation_test:
        pca, scaler, p_values = fit_pca_with_permutation_test(data)
    else:
        # Prepare data
        exclude_cols = [col for col in EXCLUDE_COLUMNS if col in data.columns]
        if 'result' in data.columns:
            exclude_cols.append('result')
        X = data.drop(columns=exclude_cols, errors='ignore').fillna(data.mean())

        # Fit scaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(random_state=RANDOM_STATE)
        pca.fit(X_scaled)

    # Apply transformation
    df_transformed, pca, scaler = apply_pca_transformation(
        data, pca=pca, scaler=scaler, n_components=N_PCA_COMPONENTS
    )

    # Display top features per component
    logger.info("\nTop 5 features per component:")
    feature_names = [col for col in data.columns if col not in EXCLUDE_COLUMNS and col != 'result']
    for i in range(N_PCA_COMPONENTS):
        loadings = np.abs(pca.components_[i])
        top_5_idx = loadings.argsort()[-5:][::-1]
        logger.info(f"\n  PC{i+1} ({PCA_COMPONENT_NAMES[f'PC{i+1}']}):")
        for idx in top_5_idx:
            logger.info(f"    - {feature_names[idx]}")

    # Save output
    if save_output:
        df_transformed.to_csv(PCA_TRANSFORMED_FILE, index=False)
        logger.info(f"\nSaved PCA-transformed data to {PCA_TRANSFORMED_FILE}")

    logger.info("\n" + "=" * 60)
    logger.info("PCA feature engineering complete")
    logger.info("=" * 60)

    return df_transformed
