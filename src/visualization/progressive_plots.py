"""Plots for the Phase 2 progressive-pruning evaluation. Discrete measured points only (no fitted/smoothed lines)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MARK = {"baseline": ("*", "black", "unpruned"), "single": ("o", "#1f77b4", "single-layer"),
         "progressive": ("s", "#d62728", "progressive")}


def _panels(metrics: pd.DataFrame):
    scorers = [s for s in metrics.scorer.unique() if s != "none"]
    fig, axes = plt.subplots(1, len(scorers), figsize=(5.2 * len(scorers), 4.4), sharey=True, squeeze=False)
    return fig, axes[0], scorers


def _metric_vs_flops(metrics, metric, ylabel, path, seed):
    fig, axes, scorers = _panels(metrics)
    base = metrics[(metrics.scorer == "none") & (metrics.seed == seed)]
    for ax, sc in zip(axes, scorers):
        d = pd.concat([base, metrics[(metrics.scorer == sc) & (metrics.seed == seed)]])
        for typ, (mk, col, lab) in _MARK.items():
            s = d[d.schedule_type == typ]
            if s.empty:
                continue
            x, y = 100 * s.normalized_flops, 100 * s[metric]
            err = np.clip(np.vstack([y - 100 * s[f"{metric}_ci_low"], 100 * s[f"{metric}_ci_high"] - y]), 0, None)
            ax.errorbar(x, y, yerr=err, fmt=mk, color=col, label=lab, ms=6, capsize=2, lw=0.8, alpha=0.85, ls="none")
        ax.set_title(sc)
        ax.set_xlabel("analytic FLOPs (% of unpruned)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs analytic FLOPs (seed {seed}; bars = Wilson 95% CI; each marker = one schedule)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _latency(metrics, path, seed):
    if "latency_total_ms_median" not in metrics:
        return
    fig, axes, scorers = _panels(metrics)
    dev = metrics.latency_device.dropna().iloc[0] if "latency_device" in metrics else "?"
    base = metrics[(metrics.scorer == "none") & (metrics.seed == seed)]
    for ax, sc in zip(axes, scorers):
        d = pd.concat([base, metrics[(metrics.scorer == sc) & (metrics.seed == seed)]])
        for typ, (mk, col, lab) in _MARK.items():
            s = d[d.schedule_type == typ]
            if s.empty:
                continue
            ax.plot(100 * s.normalized_flops, s.latency_total_ms_median, mk, color=col, ms=6, ls="none",
                    label=f"{lab}: total (incl. selection)")
            if typ != "baseline":
                ax.plot(100 * s.normalized_flops, s.scorer_overhead_ms_median, "x", color=col, ms=6, ls="none",
                        label=f"{lab}: scorer overhead only")
        ax.set_title(sc)
        ax.set_xlabel("analytic FLOPs (% of unpruned)")
        ax.grid(alpha=0.3)
        if sc == "prediction_kl_approx":
            ax.set_yscale("log")
    axes[0].set_ylabel(f"median latency per batch (ms, {dev})")
    axes[0].legend(fontsize=7)
    fig.suptitle("Measured latency vs analytic FLOPs (separate quantities; not a speedup claim)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _matched(matched, path, seed):
    if matched.empty:
        return
    m = matched[matched.seed == seed]
    scorers = list(m.scorer.unique())
    fig, axes = plt.subplots(len(scorers), 2, figsize=(11, max(3.0, 0.32 * len(m[m.scorer == scorers[0]]) + 1.5) * len(scorers)),
                             squeeze=False)
    for r, sc in enumerate(scorers):
        s = m[m.scorer == sc].reset_index(drop=True)
        labels = [f"{a} vs {b}" for a, b in zip(s.progressive, s.single)]
        y = np.arange(len(s))
        for c, (metric, title) in enumerate((("agreement", "agreement"), ("top1", "top-1 accuracy"))):
            ax = axes[r][c]
            d = 100 * s[f"{metric}_diff"]
            err = np.clip(np.vstack([d - 100 * s[f"{metric}_boot_ci_low"], 100 * s[f"{metric}_boot_ci_high"] - d]), 0, None)
            ax.errorbar(d, y, xerr=err, fmt="o", color="#444", capsize=2, lw=0.8, ms=4)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels if c == 0 else [], fontsize=6)
            ax.set_xlabel(f"Δ{title}: progressive − single (pp)")
            ax.set_title(f"{sc}: {title}", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Matched-FLOPs pairs (seed {seed}; bars = paired bootstrap 95% CI; >0 favours progressive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all_plots(metrics: pd.DataFrame, matched: pd.DataFrame, out_dir: str, seed: int = 42):
    pd_ = os.path.join(out_dir, "plots")
    os.makedirs(pd_, exist_ok=True)
    seed = int(metrics.seed.min()) if seed not in set(metrics.seed) else seed
    _metric_vs_flops(metrics, "agreement", "prediction agreement with unpruned (%)",
                     os.path.join(pd_, "agreement_vs_flops.png"), seed)
    _metric_vs_flops(metrics, "top1", "top-1 accuracy (%)", os.path.join(pd_, "top1_vs_flops.png"), seed)
    _latency(metrics, os.path.join(pd_, "latency_vs_flops.png"), seed)
    _matched(matched, os.path.join(pd_, "matched_progressive_vs_single.png"), seed)
