"""
Publication-Quality Visualization Suite for Layer-Wise Token Dynamics.
Generates research plots strictly separated per strategy (never averaged across strategies).
"""

import os
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_plotting_theme():
    """Configures clean, modern aesthetic for publication figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def plot_information_vs_tokens_per_strategy(
    df: pd.DataFrame,
    plots_dir: str,
    metric_name: str = "prediction_agreement",
    metric_label: str = "Prediction Agreement (P_l(k))",
):
    """
    Generates multi-panel figure and per-strategy plots for Information Retention vs. Token Count.
    Never averages across strategies.
    """
    setup_plotting_theme()
    scorers = df["scorer"].unique()
    layers = sorted(df["layer"].unique())
    palette = sns.color_palette("viridis", n_colors=len(layers))

    # 1. Multi-panel side-by-side figure (1 row x N_scorers cols)
    fig, axes = plt.subplots(1, len(scorers), figsize=(5.5 * len(scorers), 5), sharey=True)
    if len(scorers) == 1:
        axes = [axes]

    for s_idx, scorer in enumerate(scorers):
        ax = axes[s_idx]
        sub_df = df[df["scorer"] == scorer]

        for l_idx, layer in enumerate(layers):
            layer_df = sub_df[sub_df["layer"] == layer].sort_values("retained_tokens")
            x = layer_df["retained_tokens"]
            y = layer_df[metric_name]
            ci_low = layer_df.get(f"{metric_name}_ci_lower", None)
            ci_high = layer_df.get(f"{metric_name}_ci_upper", None)

            line = ax.plot(x, y, label=f"Layer {layer}", marker="o", color=palette[l_idx], linewidth=2)[0]
            if ci_low is not None and ci_high is not None and not ci_low.isna().all():
                ax.fill_between(x, ci_low, ci_high, color=palette[l_idx], alpha=0.15)

        ax.set_xlabel("Number of Retained Tokens (k)")
        ax.set_title(f"Strategy: {scorer}")
        if s_idx == 0:
            ax.set_ylabel(metric_label)
        ax.set_ylim(-0.05, 1.05)

    axes[-1].legend(title="Depth (Layer)", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.suptitle("Layer-Wise Token Information Retention across Strategies", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "information_vs_tokens_all_strategies.png"), bbox_inches="tight")
    plt.close()

    # 2. Dedicated single-strategy plots
    for scorer in scorers:
        plt.figure(figsize=(7.5, 5))
        sub_df = df[df["scorer"] == scorer]
        for l_idx, layer in enumerate(layers):
            layer_df = sub_df[sub_df["layer"] == layer].sort_values("retained_tokens")
            x = layer_df["retained_tokens"]
            y = layer_df[metric_name]
            ci_low = layer_df.get(f"{metric_name}_ci_lower", None)
            ci_high = layer_df.get(f"{metric_name}_ci_upper", None)

            plt.plot(x, y, label=f"Layer {layer}", marker="o", color=palette[l_idx], linewidth=2)
            if ci_low is not None and ci_high is not None and not ci_low.isna().all():
                plt.fill_between(x, ci_low, ci_high, color=palette[l_idx], alpha=0.15)

        plt.xlabel("Number of Retained Visual Tokens (k)")
        plt.ylabel(metric_label)
        plt.title(f"Token Information Retention: {scorer}")
        plt.ylim(-0.05, 1.05)
        plt.legend(title="Depth (Layer)", bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"information_vs_tokens_{scorer}.png"), bbox_inches="tight")
        plt.close()


def plot_information_vs_compute_per_strategy(
    df: pd.DataFrame,
    plots_dir: str,
    metric_name: str = "prediction_agreement",
    metric_label: str = "Prediction Agreement (P_l(k))",
):
    """Plots Information Retention vs. Normalized Compute separately per strategy."""
    setup_plotting_theme()
    scorers = df["scorer"].unique()
    layers = sorted(df["layer"].unique())
    palette = sns.color_palette("magma", n_colors=len(layers))

    fig, axes = plt.subplots(1, len(scorers), figsize=(5.5 * len(scorers), 5), sharey=True)
    if len(scorers) == 1:
        axes = [axes]

    for s_idx, scorer in enumerate(scorers):
        ax = axes[s_idx]
        sub_df = df[df["scorer"] == scorer]

        for l_idx, layer in enumerate(layers):
            layer_df = sub_df[sub_df["layer"] == layer].sort_values("normalized_flops")
            x = layer_df["normalized_flops"] * 100.0
            y = layer_df[metric_name]

            ax.plot(x, y, label=f"Layer {layer}", marker="s", color=palette[l_idx], linewidth=2)

        ax.set_xlabel("Normalized End-to-End FLOPs (%)")
        ax.set_title(f"Strategy: {scorer}")
        if s_idx == 0:
            ax.set_ylabel(metric_label)
        ax.set_ylim(-0.05, 1.05)

    axes[-1].legend(title="Depth (Layer)", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.suptitle("Information Retention vs. Normalized Compute", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "information_vs_compute_all_strategies.png"), bbox_inches="tight")
    plt.close()


def plot_compute_vs_tokens(df: pd.DataFrame, output_path: str):
    """Plot 3: Theoretical End-to-End Compute vs. Retained Tokens."""
    setup_plotting_theme()
    plt.figure(figsize=(7.5, 5))

    sub_df = df.drop_duplicates(subset=["layer", "retained_tokens"])
    layers = sorted(sub_df["layer"].unique())
    palette = sns.color_palette("coolwarm", n_colors=len(layers))

    for idx, layer in enumerate(layers):
        layer_df = sub_df[sub_df["layer"] == layer].sort_values("retained_tokens")
        x = layer_df["retained_tokens"]
        y = layer_df["normalized_flops"] * 100.0

        plt.plot(x, y, label=f"Layer {layer}", marker="^", color=palette[idx], linewidth=2)

    plt.xlabel("Number of Retained Visual Tokens (k)")
    plt.ylabel("Normalized End-to-End FLOPs (%)")
    plt.title("End-to-End Compute Scaling vs. Pruning Depth & Token Count")
    plt.legend(title="Pruned Layer", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_marginal_efficiency_per_strategy(marginal_df: pd.DataFrame, output_path: str):
    """Plot 4: Marginal Information Gain per Compute (Delta P / Delta C) faceted per strategy."""
    if marginal_df.empty:
        return

    setup_plotting_theme()
    scorers = marginal_df["scorer"].unique() if "scorer" in marginal_df.columns else ["default"]

    fig, axes = plt.subplots(1, len(scorers), figsize=(6 * len(scorers), 5), sharey=True)
    if len(scorers) == 1:
        axes = [axes]

    for s_idx, scorer in enumerate(scorers):
        ax = axes[s_idx]
        sub_df = marginal_df[marginal_df["scorer"] == scorer] if "scorer" in marginal_df.columns else marginal_df

        sns.barplot(
            data=sub_df,
            x="layer_idx",
            y="marginal_gain",
            hue="k_interval",
            palette="crest",
            ax=ax,
        )
        ax.set_xlabel("Transformer Layer (l)")
        ax.set_title(f"Strategy: {scorer}")
        if s_idx == 0:
            ax.set_ylabel("Marginal Efficiency (ΔP / ΔC)")
        if s_idx < len(scorers) - 1:
            ax.get_legend().remove() if ax.get_legend() else None

    if axes[-1].get_legend():
        axes[-1].legend(title="Token Interval", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.suptitle("Marginal Information Gain per Unit Compute by Strategy", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_pareto_accuracy_vs_compute(df: pd.DataFrame, output_path: str):
    """
    Plot 5: Pareto Accuracy / Agreement vs. Compute across Pruning Strategies.
    Strictly keeps individual strategy points and curves distinct (never averaged).
    """
    setup_plotting_theme()
    plt.figure(figsize=(8.5, 5.5))

    scorers = df["scorer"].unique()
    palette = sns.color_palette("tab10", n_colors=len(scorers))
    markers = ["o", "s", "^", "D", "v"]

    metric_to_use = "top1_accuracy" if "top1_accuracy" in df.columns and df["top1_accuracy"].max() > 0 else "prediction_agreement"
    metric_title = "Top-1 Accuracy" if metric_to_use == "top1_accuracy" else "Prediction Agreement"

    for idx, scorer in enumerate(scorers):
        scorer_df = df[df["scorer"] == scorer].sort_values("normalized_flops")
        plt.scatter(
            scorer_df["normalized_flops"] * 100.0,
            scorer_df[metric_to_use],
            label=f"Strategy: {scorer}",
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            s=55,
            alpha=0.85,
        )

    plt.xlabel("Normalized Compute (% Baseline FLOPs)")
    plt.ylabel(metric_title)
    plt.title("Compute-Accuracy Pareto Frontier: Individual Strategies")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_flops_vs_latency_discrepancy(df: pd.DataFrame, output_path: str):
    """Plot 6: Full Forward Pass Latency vs Theoretical FLOPs Discrepancy."""
    if "mean_latency_ms" not in df.columns or df["mean_latency_ms"].isna().all():
        return

    setup_plotting_theme()
    plt.figure(figsize=(8, 5.5))

    sns.scatterplot(
        data=df,
        x="normalized_flops",
        y="mean_latency_ms",
        hue="scorer",
        style="layer",
        s=70,
    )

    plt.xlabel("Theoretical Normalized FLOPs")
    plt.ylabel("Full Forward Pass Latency (ms)")
    plt.title("Hardware Reality: Theoretical FLOPs vs. Full Forward Pass Latency")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_all_plots(results_dir: str):
    """Generates all publication-ready figures from saved CSV results."""
    metrics_path = os.path.join(results_dir, "layer_metrics.csv")
    marginal_path = os.path.join(results_dir, "marginal_efficiency.csv")

    if not os.path.isfile(metrics_path):
        print(f"[Plotting Error] Metrics file not found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    marginal_df = pd.read_csv(marginal_path) if os.path.isfile(marginal_path) else pd.DataFrame()

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_information_vs_tokens_per_strategy(df, plots_dir)
    plot_information_vs_compute_per_strategy(df, plots_dir)
    plot_compute_vs_tokens(df, os.path.join(plots_dir, "compute_vs_tokens.png"))
    if not marginal_df.empty:
        plot_marginal_efficiency_per_strategy(marginal_df, os.path.join(plots_dir, "marginal_information.png"))
    plot_pareto_accuracy_vs_compute(df, os.path.join(plots_dir, "accuracy_vs_compute.png"))
    plot_flops_vs_latency_discrepancy(df, os.path.join(plots_dir, "flops_vs_latency.png"))

    print(f"[Visualizations] All research figures generated in {plots_dir}")
