#!/usr/bin/env python3
"""
CLI Entry Point: Analyze Results and Re-generate Research Plots.
"""

import argparse
import sys
import os
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.layerwise_analysis import format_research_summary
from src.visualization.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Layer-Wise Token Dynamics Results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the directory containing experiment results.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="prediction_agreement",
        help="Primary metric to display in summary report.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Re-generate all visualization plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_csv = os.path.join(args.results_dir, "layer_metrics.csv")

    if not os.path.isfile(metrics_csv):
        print(f"Error: {metrics_csv} does not exist.")
        sys.exit(1)

    df = pd.read_csv(metrics_csv)
    summary_text = format_research_summary(df, primary_metric=args.primary_metric)
    print(summary_text)

    if args.generate_plots:
        generate_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
