"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.evaluation.metrics import (
    calculate_baseline_accuracy,
    calculate_metrics,
)


class TestCalculateBaselineAccuracy:
    """Tests for calculate_baseline_accuracy function."""

    def test_balanced_classes(self):
        """Test baseline with balanced classes."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        result = calculate_baseline_accuracy(y)

        assert result['baseline_accuracy'] == 0.5
        assert result['majority_class'] in [0, 1]
        assert result['class_distribution'][0] == 4
        assert result['class_distribution'][1] == 4

    def test_imbalanced_classes(self):
        """Test baseline with imbalanced classes."""
        y = np.array([0] * 8 + [1] * 2)
        result = calculate_baseline_accuracy(y)

        assert result['baseline_accuracy'] == 0.8
        assert result['majority_class'] == 0
        assert result['class_distribution'][0] == 8
        assert result['class_distribution'][1] == 2


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics

    def test_metrics_without_probabilities(self):
        """Test that metrics work without probabilities."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' not in metrics
        assert 'log_loss' not in metrics

    def test_all_metrics_present_with_proba(self):
        """Test that all expected metrics are calculated with probabilities."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_proba = np.array([0.2, 0.6, 0.8, 0.3, 0.1, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_metrics_range(self):
        """Test that metric values are in valid ranges."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_proba = np.array([0.2, 0.6, 0.8, 0.3, 0.1, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # All metrics except log_loss should be between 0 and 1
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            assert 0 <= metrics[key] <= 1

        # Log loss should be non-negative
        assert metrics['log_loss'] >= 0
