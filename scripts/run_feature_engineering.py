"""Run PCA feature engineering pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed_data
from src.features.engineering import create_pca_features

if __name__ == "__main__":
    print("Starting feature engineering pipeline...")

    # Load preprocessed data
    data = load_processed_data('metrics')

    # Create PCA features
    pca_data = create_pca_features(
        data,
        run_permutation_test=True,
        save_output=True
    )

    print("\nFeature engineering complete!")
    print(f"PCA-transformed data shape: {pca_data.shape}")
