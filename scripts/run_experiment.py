#!/usr/bin/env python3
"""
CLI Entry Point: Run Layer-Wise Token Dynamics Experiment.
"""

import argparse
import sys
import os
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.runner import ExperimentRunner, ExperimentConfig
from src.visualization.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Layer-Wise Information Dynamics Experiment on Vision Transformers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke_test.yaml",
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of dataset samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (cuda or cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip automatic plot generation.",
    )
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> ExperimentConfig:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    exp_cfg = cfg.get("experiment", {})

    return ExperimentConfig(
        model_name=model_cfg.get("name", "deit_tiny_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        dataset_name=ds_cfg.get("name", "synthetic"),
        data_dir=ds_cfg.get("data_dir", "./data"),
        num_samples=ds_cfg.get("num_samples", 100),
        batch_size=ds_cfg.get("batch_size", 16),
        seed=exp_cfg.get("seed", 42),
        layers=exp_cfg.get("layers", [2, 4, 6, 8, 10, 12]),
        retention_ratios=exp_cfg.get("retention_ratios", [1.0, 0.75, 0.5, 0.25, 0.1]),
        scorers=exp_cfg.get("scorers", ["random", "prediction_kl_approximate", "attention_cls"]),
        measure_latency=exp_cfg.get("measure_latency", True),
        output_dir=exp_cfg.get("output_dir", "./results/experiment_001"),
    )


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)

    # CLI Overrides
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    runner = ExperimentRunner(config)
    res = runner.run()

    if not args.no_plots:
        generate_all_plots(config.output_dir)


if __name__ == "__main__":
    main()
