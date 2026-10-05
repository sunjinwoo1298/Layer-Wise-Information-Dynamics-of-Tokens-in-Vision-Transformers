"""
Unit tests for evaluation metrics and marginal efficiency calculations.
"""

import pytest
import torch
import pandas as pd
from src.metrics.prediction import compute_prediction_metrics, compute_sample_level_metrics
from src.metrics.information import calculate_marginal_efficiency


def test_prediction_metrics_identity():
    """When logits are identical, agreement should be 1.0, KL should be 0.0."""
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, 3.2, 1.1]])
    targets = torch.tensor([0, 1])

    metrics = compute_prediction_metrics(logits, logits, targets)
    assert metrics.prediction_agreement == 1.0
    assert metrics.top1_accuracy == 1.0
    assert abs(metrics.kl_divergence) < 1e-6
    assert abs(metrics.cosine_distance) < 1e-6
    assert abs(metrics.confidence_retention - 1.0) < 1e-5


def test_marginal_efficiency_calculation():
    df = pd.DataFrame([
        {"layer": 4, "retained_tokens": 50, "prediction_agreement": 0.60, "normalized_flops": 0.50},
        {"layer": 4, "retained_tokens": 100, "prediction_agreement": 0.85, "normalized_flops": 0.75},
        {"layer": 4, "retained_tokens": 196, "prediction_agreement": 1.00, "normalized_flops": 1.00},
    ])

    intervals = calculate_marginal_efficiency(df, "prediction_agreement", "normalized_flops")
    assert len(intervals) == 2
    # Interval 1: 50 -> 100: delta_p = 0.25, delta_c = 0.25 -> gain = 1.0
    assert abs(intervals[0].marginal_gain - 1.0) < 1e-4
    # Interval 2: 100 -> 196: delta_p = 0.15, delta_c = 0.25 -> gain = 0.6
    assert abs(intervals[1].marginal_gain - 0.6) < 1e-4
