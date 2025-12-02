"""Model training utilities and configurations."""

import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_STATE
from src.data import load_processed_data
from src.models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "naive_bayes": {
        "model": GaussianNB(),
        "param_grid": None,  # No hyperparameters to tune for Gaussian NB
        "use_scaler": False,
    },
    "logistic_regression": {
        "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
        "use_scaler": False,
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "use_scaler": False,
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "use_scaler": False,
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 5, 7],
            "subsample": [0.75, 1.0],
        },
        "use_scaler": False,
    },
    "bagging": {
        "model": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        "param_grid": {
            "n_estimators": [50, 75, 100, 150],
            "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
        },
        "use_scaler": False,
    },
}


def train_model(
    model_type: str,
    data: Optional[pd.DataFrame] = None,
    tune_hyperparameters: bool = True,
    mlflow_tracking: bool = True,
    save_model: bool = True,
    save_visualizations: bool = True,
) -> BaseModel:
    """
    Train a single model.

    Args:
        model_type: Type of model to train (key from MODEL_CONFIGS).
        data: Input DataFrame (if None, loads PCA data).
        tune_hyperparameters: Whether to perform hyperparameter tuning.
        mlflow_tracking: Whether to log to MLflow.
        save_model: Whether to save trained model.
        save_visualizations: Whether to create evaluation plots.

    Returns:
        Trained BaseModel instance.

    Raises:
        ValueError: If model_type is invalid.
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    # Load data
    if data is None:
        data = load_processed_data("pca")

    # Prepare features and target
    X = data.drop("result", axis=1)
    y = data["result"]

    # Get model configuration
    config = MODEL_CONFIGS[model_type]
    model_name = model_type.replace("_", " ").title()

    logger.info("=" * 60)
    logger.info(f"Training {model_name}")
    logger.info("=" * 60)

    # Create model instance
    base_model = BaseModel(
        model=config["model"],
        param_grid=config["param_grid"],
        model_name=model_name,
        use_scaler=config["use_scaler"],
        mlflow_tracking=mlflow_tracking,
    )

    # Prepare data (includes train/test split and scaling)
    X_train, X_test, y_train, y_test = base_model.prepare_data(X, y)

    # Train model
    base_model.train(X_train, y_train, tune_hyperparameters=tune_hyperparameters)

    # Evaluate model
    base_model.evaluate(X_test, y_test, save_visualizations=save_visualizations)

    # Log to MLflow
    if mlflow_tracking:
        base_model.log_to_mlflow(save_model=save_model)

    # Save model to disk
    if save_model:
        base_model.save()

    logger.info("=" * 60)
    logger.info(f"{model_name} training complete")
    logger.info("=" * 60)

    return base_model


def train_all_models(
    data: Optional[pd.DataFrame] = None,
    tune_hyperparameters: bool = True,
    mlflow_tracking: bool = False,
    save_models: bool = True,
) -> Dict[str, BaseModel]:
    """
    Train all configured models.

    Args:
        data: Input DataFrame (if None, loads PCA data).
        tune_hyperparameters: Whether to perform hyperparameter tuning.
        mlflow_tracking: Whether to log to MLflow.
        save_models: Whether to save trained models.

    Returns:
        Dictionary mapping model names to trained BaseModel instances.
    """
    if mlflow_tracking:
        mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
        mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=" * 60)
    logger.info("Training All Models")
    logger.info("=" * 60)

    trained_models = {}

    for model_type in MODEL_CONFIGS.keys():
        try:
            model = train_model(
                model_type=model_type,
                data=data,
                tune_hyperparameters=tune_hyperparameters,
                mlflow_tracking=mlflow_tracking,
                save_model=save_models,
            )
            trained_models[model_type] = model
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("All models trained successfully")
    logger.info("=" * 60)

    return trained_models


def compare_models(trained_models: Dict[str, BaseModel]) -> pd.DataFrame:
    """
    Compare performance of all trained models.

    Args:
        trained_models: Dictionary of trained model instances.

    Returns:
        DataFrame with model comparison results.
    """
    comparison_data = []

    for model_type, model in trained_models.items():
        if model.test_metrics:
            row = {
                "Model": model.model_name,
                "Accuracy": model.test_metrics.get("accuracy", np.nan),
                "Precision": model.test_metrics.get("precision", np.nan),
                "Recall": model.test_metrics.get("recall", np.nan),
                "F1 Score": model.test_metrics.get("f1_score", np.nan),
                "ROC AUC": model.test_metrics.get("roc_auc", np.nan),
                "Log Loss": model.test_metrics.get("log_loss", np.nan),
            }

            if model.cv_results:
                row["Best CV F1"] = model.cv_results.get("best_score", np.nan)

            comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    if len(comparison_df) == 0:
        logger.warning("No models were successfully trained. Cannot create comparison.")
        return comparison_df

    comparison_df = comparison_df.sort_values("F1 Score", ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    print(comparison_df.to_string(index=False))
    logger.info("=" * 80)

    comparison_path = Path("results") / "model_comparison.csv"
    comparison_path.parent.mkdir(exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved comparison to {comparison_path}")

    return comparison_df
