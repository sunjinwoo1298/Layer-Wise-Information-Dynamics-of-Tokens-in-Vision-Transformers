"""
Unit tests for theoretical FLOP estimation and latency measurement.
"""

import pytest
import torch
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper


def test_theoretical_flops_scaling():
    estimator = ComputeEstimator(
        num_layers=12,
        hidden_dim=192,
        num_heads=3,
        num_original_tokens=196,
        num_classes=1000,
    )

    # Baseline (k=196) normalized flops should be exactly 1.0
    res_base = estimator.compute_flops(layer_idx=12, k=196)
    assert abs(res_base.normalized_flops - 1.0) < 1e-6

    # Pruning earlier (layer 2) to k=50 should reduce FLOPs more than pruning at layer 10
    res_early = estimator.compute_flops(layer_idx=2, k=50)
    res_late = estimator.compute_flops(layer_idx=10, k=50)

    assert res_early.normalized_flops < res_late.normalized_flops
    assert res_early.normalized_flops < 1.0


def test_latency_measurement_smoke():
    model = ViTWrapper(model_name="deit_tiny_patch16_224", pretrained=False, device="cpu")
    estimator = ComputeEstimator(
        num_layers=model.num_layers,
        hidden_dim=model.hidden_dim,
        num_heads=model.num_heads,
        num_original_tokens=model.num_original_tokens,
    )
    dummy_x = torch.randn(1, 3, 224, 224)
    lat_res = estimator.measure_latency_and_memory(model, dummy_x, layer_idx=4, k=50, num_runs=2, warmup_runs=1)
    assert lat_res.mean_latency_ms > 0.0
