"""Train a single model."""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import train_model, MODEL_CONFIGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a single model')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help='Type of model to train'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )

    args = parser.parse_args()

    print(f"Training {args.model_type}...")

    model = train_model(
        model_type=args.model_type,
        tune_hyperparameters=not args.no_tune,
        mlflow_tracking=not args.no_mlflow
    )

    print(f"\n{args.model_type} training complete!")
