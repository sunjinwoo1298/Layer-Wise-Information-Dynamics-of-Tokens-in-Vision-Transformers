"""Plots for the pruning-stage ablation. Discrete measured points with uncertainty only; no fitted/smoothed lines."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_STY = {0: ("*", "black", "unpruned"), 1: ("o", "#1f77b4", "1 stage"), 2: ("s", "#2ca02c", "2 stages"),
        3: ("^", "#d62728", "3 stages")}


def _err(y, lo, hi):
    return np.clip(np.vstack([y - lo, hi - y]), 0, None)


def _vs_flops(m, first_layer, metric, ylabel, path, seed):
    sc_list = [s for s in m[m.first_layer == first_layer].scorer.unique()]
    if not sc_list:
        return
    fig, axes = plt.subplots(1, len(sc_list), figsize=(5.2 * len(sc_list), 4.4), sharey=True, squeeze=False)
    base = m[(m.scorer == "none") & (m.seed == seed)]
    for ax, sc in zip(axes[0], sc_list):
        d = pd.concat([base, m[(m.scorer == sc) & (m.seed == seed) & (m.first_layer == first_layer)]])
        for sc_count, (mk, col, lab) in _STY.items():
            s = d[d.stage_count == sc_count]
            if s.empty:
                continue
            y = 100 * s[metric]
            ax.errorbar(100 * s.normalized_flops, y, yerr=100 * _err(s[metric], s[f"{metric}_ci_low"], s[f"{metric}_ci_high"]),
                        fmt=mk, color=col, label=lab, ms=6, capsize=2, lw=0.8, alpha=0.85, ls="none")
        ax.set_title(sc); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend(fontsize=8)
    fig.suptitle(f"First pruning layer {first_layer}: {ylabel} vs analytic FLOPs (seed {seed}; bars = Wilson 95% CI; each marker one schedule)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _matched(m, scorer, first_layer, path, seed):
    d = m[(m.scorer == scorer) & (m.seed == seed) & (m.first_layer == first_layer)]
    if d.empty:
        return
    groups = sorted(d.group.unique(), key=lambda g: d[d.group == g].budget_target.iloc[0])
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 4.2), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        s = d[d.group == g].sort_values(["stage_count", "structure"]).reset_index(drop=True)
        x = np.arange(len(s))
        for sc_count, (mk, col, lab) in _STY.items():
            t = s[s.stage_count == sc_count]
            if t.empty:
                continue
            ax.errorbar(t.index, 100 * t.agreement, yerr=100 * _err(t.agreement, t.agreement_ci_low, t.agreement_ci_high),
                        fmt=mk, color=col, capsize=2, ms=6, label=lab, ls="none")
        ax.set_xticks(x); ax.set_xticklabels(["1-stage" if r.stage_count == 1 else r.structure for r in s.itertuples()],
                                             rotation=60, fontsize=7)
        ax.set_title(f"{g}\n(FLOPs {100 * s.normalized_flops.min():.1f}–{100 * s.normalized_flops.max():.1f}%)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("agreement with unpruned (%)"); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Matched-FLOPs budgets, `{scorer}`, first layer {first_layer}, seed {seed} (bars = Wilson 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _latency(m, first_layer, path, seed):
    if "latency_total_ms_median" not in m:
        return
    sc_list = list(m[m.first_layer == first_layer].scorer.unique())
    if not sc_list:
        return
    dev = m.latency_device.dropna().iloc[0]
    fig, axes = plt.subplots(1, len(sc_list), figsize=(5.2 * len(sc_list), 4.4), squeeze=False)
    base = m[(m.scorer == "none") & (m.seed == seed)]
    for ax, sc in zip(axes[0], sc_list):
        d = pd.concat([base, m[(m.scorer == sc) & (m.seed == seed) & (m.first_layer == first_layer)]])
        for sc_count, (mk, col, lab) in _STY.items():
            s = d[d.stage_count == sc_count]
            if s.empty:
                continue
            ax.plot(100 * s.normalized_flops, s.latency_total_ms_median, mk, color=col, ls="none", ms=6, label=f"{lab}: total")
            if sc_count:
                ax.plot(100 * s.normalized_flops, s.scorer_overhead_ms_median, "x", color=col, ls="none", ms=6, label=f"{lab}: scorer only")
        ax.set_title(sc); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
        if sc == "prediction_kl_approx":
            ax.set_yscale("log")
    axes[0][0].set_ylabel(f"median latency per batch (ms, {dev})"); axes[0][0].legend(fontsize=6)
    fig.suptitle(f"First layer {first_layer}: measured latency vs analytic FLOPs (separate quantities; not a speedup claim)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all_plots(metrics: pd.DataFrame, stats: pd.DataFrame, out_dir: str, seed: int = 42):
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    for f in sorted(int(x) for x in metrics.first_layer.dropna().unique()):
        _vs_flops(metrics, f, "agreement", "agreement with unpruned (%)", os.path.join(pdir, f"agreement_vs_flops_L{f}.png"), seed)
        _vs_flops(metrics, f, "top1", "top-1 accuracy (%)", os.path.join(pdir, f"top1_vs_flops_L{f}.png"), seed)
        _latency(metrics, f, os.path.join(pdir, f"latency_vs_flops_L{f}.png"), seed)
        for sc in metrics[metrics.first_layer == f].scorer.unique():
            _matched(metrics, sc, f, os.path.join(pdir, f"matched_budget_stages_{sc}_L{f}.png"), seed)
