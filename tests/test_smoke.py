"""
End-to-End Smoke Test verifying the complete Phase 1 experimental pipeline.
"""

import os
import shutil
import pytest
from src.experiments.runner import ExperimentRunner, ExperimentConfig
from src.visualization.plots import generate_all_plots


def test_full_pipeline_smoke_run(tmp_path):
    output_dir = str(tmp_path / "smoke_results")

    config = ExperimentConfig(
        model_name="deit_tiny_patch16_224",
        pretrained=False,  # Use uninitialized weights for rapid offline testing
        dataset_name="synthetic",
        num_samples=10,
        batch_size=5,
        seed=42,
        layers=[4, 8],
        retention_ratios=[1.0, 0.5],
        scorers=["random", "prediction_kl_approximate", "attention_cls"],
        measure_latency=True,
        output_dir=output_dir,
        device="cpu",
    )

    runner = ExperimentRunner(config)
    res = runner.run()

    # Verify structured files were generated
    assert os.path.isfile(os.path.join(output_dir, "config.json"))
    assert os.path.isfile(os.path.join(output_dir, "layer_metrics.csv"))
    assert os.path.isfile(os.path.join(output_dir, "marginal_efficiency.csv"))
    assert os.path.isfile(os.path.join(output_dir, "summary.txt"))

    # Generate plots
    generate_all_plots(output_dir)
    plots_dir = os.path.join(output_dir, "plots")
    assert os.path.isdir(plots_dir)
    assert os.path.isfile(os.path.join(plots_dir, "information_vs_tokens_all_strategies.png"))
    assert os.path.isfile(os.path.join(plots_dir, "information_vs_compute_all_strategies.png"))
    assert os.path.isfile(os.path.join(plots_dir, "compute_vs_tokens.png"))
    assert os.path.isfile(os.path.join(plots_dir, "accuracy_vs_compute.png"))
