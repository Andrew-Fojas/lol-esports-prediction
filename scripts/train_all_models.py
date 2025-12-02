"""Train all models and compare performance."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import compare_models, train_all_models  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip hyperparameter tuning"
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")

    args = parser.parse_args()

    print("Training all models...")

    trained_models = train_all_models(
        tune_hyperparameters=not args.no_tune, mlflow_tracking=args.mlflow
    )

    print("\nComparing model performance...")
    comparison = compare_models(trained_models)

    print("\nAll models trained successfully!")
