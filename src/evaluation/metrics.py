"""Model evaluation metrics and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_baseline_accuracy(y_train: np.ndarray) -> Dict[str, float]:
    """
    Calculate baseline accuracy metrics.

    The baseline is the accuracy achieved by always predicting
    the most frequent class.

    Args:
        y_train: Training labels.

    Returns:
        Dictionary with baseline metrics.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    baseline_acc = counts.max() / len(y_train)

    baseline_metrics = {
        "majority_class": int(majority_class),
        "baseline_accuracy": baseline_acc,
        "class_distribution": dict(
            zip([int(u) for u in unique], [int(c) for c in counts])
        ),
    }

    logger.info(f"Baseline (majority class): {baseline_acc:.4f}")
    logger.info(f"Class distribution: {baseline_metrics['class_distribution']}")

    return baseline_metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (binary).
        y_pred_proba: Predicted probabilities for positive class.
                      Required for log_loss and roc_auc.
        model_name: Name of the model for logging.

    Returns:
        Dictionary containing all metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    # Metrics that require probabilities
    if y_pred_proba is not None:
        # FIXED: Use probabilities for log_loss and roc_auc, not binary predictions
        metrics["log_loss"] = log_loss(y_true, y_pred_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    else:
        logger.warning(
            f"{model_name}: No probabilities provided. "
            "Skipping log_loss and roc_auc calculations."
        )

    # Log results
    logger.info(f"\n{model_name} Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    if "log_loss" in metrics:
        logger.info(f"  Log Loss:  {metrics['log_loss']:.4f}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create and save confusion matrix visualization.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model for the title.
        save_path: Path to save figure (if None, auto-generates).
        show: Whether to display the plot.

    Returns:
        Path where the figure was saved.
    """
    if save_path is None:
        save_path = (
            RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Loss", "Win"],
        yticklabels=["Loss", "Win"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create and save ROC curve visualization.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        model_name: Name of the model for the title.
        save_path: Path to save figure (if None, auto-generates).
        show: Whether to display the plot.

    Returns:
        Path where the figure was saved.
    """
    if save_path is None:
        save_path = (
            RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    logger.info(f"Saved ROC curve to {save_path}")
    return save_path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create and save Precision-Recall curve visualization.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        model_name: Name of the model for the title.
        save_path: Path to save figure (if None, auto-generates).
        show: Whether to display the plot.

    Returns:
        Path where the figure was saved.
    """
    if save_path is None:
        save_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"{model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    logger.info(f"Saved Precision-Recall curve to {save_path}")
    return save_path


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    model_name: str = "Model",
    feature_importance: Optional[pd.DataFrame] = None,
    save_visualizations: bool = True,
) -> Dict:
    """
    Create comprehensive evaluation report with metrics and visualizations.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for positive class.
        model_name: Name of the model.
        feature_importance: DataFrame with feature importance (optional).
        save_visualizations: Whether to save visualization plots.

    Returns:
        Dictionary containing all evaluation results.
    """
    logger.info("=" * 60)
    logger.info(f"Generating evaluation report for {model_name}")
    logger.info("=" * 60)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba, model_name)

    # Create visualizations
    viz_paths = {}
    if save_visualizations:
        viz_paths["confusion_matrix"] = plot_confusion_matrix(
            y_true, y_pred, model_name
        )

        if y_pred_proba is not None:
            viz_paths["roc_curve"] = plot_roc_curve(y_true, y_pred_proba, model_name)
            viz_paths["pr_curve"] = plot_precision_recall_curve(
                y_true, y_pred_proba, model_name
            )

    # Classification report
    class_report = classification_report(
        y_true, y_pred, target_names=["Loss", "Win"], output_dict=True
    )

    report = {
        "model_name": model_name,
        "metrics": metrics,
        "classification_report": class_report,
        "visualization_paths": viz_paths,
    }

    if feature_importance is not None:
        report["feature_importance"] = feature_importance

    logger.info("=" * 60)
    logger.info(f"Evaluation report completed for {model_name}")
    logger.info("=" * 60)

    return report
