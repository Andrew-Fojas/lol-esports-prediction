"""Base model class for consistent training and evaluation."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.evaluation import calculate_baseline_accuracy, create_evaluation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for all ML models providing consistent interface.

    This class handles:
    - Train/test splitting
    - Data scaling (when needed)
    - Hyperparameter tuning
    - Model training
    - Evaluation with proper metrics (fixes log_loss bug)
    - MLflow experiment tracking
    - Model persistence
    """

    def __init__(
        self,
        model,
        param_grid: Optional[Dict] = None,
        model_name: str = "Model",
        use_scaler: bool = False,
        mlflow_tracking: bool = True,
    ):
        """
        Initialize base model.

        Args:
            model: Scikit-learn compatible model instance.
            param_grid: Parameter grid for GridSearchCV (optional).
            model_name: Name of the model for logging and saving.
            use_scaler: Whether to apply StandardScaler to features.
            mlflow_tracking: Whether to log to MLflow.
        """
        self.model = model
        self.param_grid = param_grid
        self.model_name = model_name
        self.use_scaler = use_scaler
        self.mlflow_tracking = mlflow_tracking

        self.best_model = None
        self.scaler = None
        self.baseline_metrics = None
        self.test_metrics = None
        self.cv_results = None

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with train/test split and optional scaling.

        FIXES DATA LEAKAGE: Scaler is fit only on training data.

        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Calculate baseline metrics
        self.baseline_metrics = calculate_baseline_accuracy(y_train)

        # Apply scaling if needed (FIX: fit only on train, transform both)
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)  # Fit on train only
            X_test = self.scaler.transform(X_test)  # Transform test
            logger.info("Applied StandardScaler to features")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = True,
    ):
        """
        Train the model with optional hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training labels.
            tune_hyperparameters: Whether to perform GridSearchCV.
        """
        logger.info(f"Training {self.model_name}...")

        if tune_hyperparameters and self.param_grid:
            logger.info(f"Performing GridSearchCV with {CV_FOLDS}-fold CV")
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=CV_FOLDS,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            self.best_model = grid_search.best_estimator_
            self.cv_results = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "cv_results": grid_search.cv_results_,
            }

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score (F1): {grid_search.best_score_:.4f}")
        else:
            self.best_model = self.model
            self.best_model.fit(X_train, y_train)
            logger.info("Model training complete (no hyperparameter tuning)")

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix.
            return_proba: Whether to return probabilities.

        Returns:
            Predictions (binary labels or probabilities).
        """
        if return_proba:
            if hasattr(self.best_model, "predict_proba"):
                return self.best_model.predict_proba(X)[:, 1]
            else:
                logger.warning(f"{self.model_name} doesn't support predict_proba")
                return self.best_model.predict(X)
        else:
            return self.best_model.predict(X)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, save_visualizations: bool = True
    ) -> Dict:
        """
        Evaluate model on test set.

        FIXES LOG_LOSS BUG: Uses probabilities instead of binary predictions.

        Args:
            X_test: Test features.
            y_test: Test labels.
            save_visualizations: Whether to create and save plots.

        Returns:
            Dictionary with evaluation results.
        """
        logger.info(f"Evaluating {self.model_name} on test set...")

        # Get predictions and probabilities
        y_pred = self.predict(X_test, return_proba=False)
        y_pred_proba = self.predict(X_test, return_proba=True)

        # Get feature importance if available
        feature_importance = None
        if hasattr(self.best_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": range(X_test.shape[1]),
                    "importance": self.best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

        # Create comprehensive evaluation report
        report = create_evaluation_report(
            y_test,
            y_pred,
            y_pred_proba,
            model_name=self.model_name,
            feature_importance=feature_importance,
            save_visualizations=save_visualizations,
        )

        self.test_metrics = report["metrics"]
        return report

    def log_to_mlflow(self, params: Optional[Dict] = None, save_model: bool = True):
        """
        Log model, parameters, and metrics to MLflow.

        Args:
            params: Additional parameters to log.
            save_model: Whether to save model artifact.
        """
        if not self.mlflow_tracking:
            return

        with mlflow.start_run(run_name=self.model_name):
            # Log parameters
            if self.cv_results:
                mlflow.log_params(self.cv_results["best_params"])
                mlflow.log_metric("cv_best_f1", self.cv_results["best_score"])

            if params:
                mlflow.log_params(params)

            # Log baseline metrics
            if self.baseline_metrics:
                mlflow.log_metric(
                    "baseline_accuracy", self.baseline_metrics["baseline_accuracy"]
                )

            # Log test metrics
            if self.test_metrics:
                for metric_name, metric_value in self.test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)

            # Save model
            if save_model:
                mlflow.sklearn.log_model(self.best_model, "model")

            logger.info(f"Logged {self.model_name} to MLflow")

    def save(self, filepath: Optional[Path] = None):
        """
        Save model to disk.

        Args:
            filepath: Path to save model (if None, auto-generates).
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name.lower().replace(' ', '_')}.pkl"

        model_data = {
            "model": self.best_model,
            "scaler": self.scaler,
            "model_name": self.model_name,
            "baseline_metrics": self.baseline_metrics,
            "test_metrics": self.test_metrics,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Saved {self.model_name} to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            filepath: Path to saved model.

        Returns:
            Loaded BaseModel instance.
        """
        model_data = joblib.load(filepath)

        instance = cls(model=model_data["model"], model_name=model_data["model_name"])
        instance.best_model = model_data["model"]
        instance.scaler = model_data.get("scaler")
        instance.baseline_metrics = model_data.get("baseline_metrics")
        instance.test_metrics = model_data.get("test_metrics")

        logger.info(f"Loaded {instance.model_name} from {filepath}")
        return instance
