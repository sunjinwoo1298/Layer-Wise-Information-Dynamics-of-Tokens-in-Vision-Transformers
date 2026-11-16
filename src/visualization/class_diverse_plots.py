"""Plots for the class-diverse validation study. Discrete measured points with uncertainty only; no fitted/smoothed lines."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_STY = {0: ("*", "black", "unpruned"), 1: ("o", "#1f77b4", "1 stage"), 2: ("s", "#2ca02c", "2 stages"), 3: ("^", "#d62728", "3 stages")}
_FILL = {"geometric": "full", "equal_drop": "none", "shared": "full", "none": "full"}


def _err(y, lo, hi):
    return np.clip(np.vstack([np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]), 0, None)


def _vs_flops(m, first_layer, metric, ylabel, path, seed):
    scs = list(m[m.first_layer == first_layer].scorer.unique())
    if not scs:
        return
    fig, axes = plt.subplots(1, len(scs), figsize=(5.4 * len(scs), 4.6), sharey=True, squeeze=False)
    base = m[(m.scorer == "none") & (m.seed == seed)]
    for ax, sc in zip(axes[0], scs):
        d = pd.concat([base, m[(m.scorer == sc) & (m.seed == seed) & (m.first_layer == first_layer)]])
        for k, fam in [(0, "none"), (1, "shared"), (2, "geometric"), (2, "equal_drop"), (3, "geometric"), (3, "equal_drop")]:
            mk, col, lab = _STY[k]
            s = d[d.stage_count == k] if k < 2 else d[(d.stage_count == k) & (d.allocation_family == fam)]
            if s.empty:
                continue
            ax.errorbar(100 * s.normalized_flops, 100 * s[metric], yerr=100 * _err(s[metric], s[f"{metric}_ci_low"], s[f"{metric}_ci_high"]),
                        fmt=mk, color=col, mfc=("white" if fam == "equal_drop" else col), ms=6, capsize=2, lw=0.8, ls="none",
                        label=lab + ("" if k < 2 else f" ({fam})"))
        ax.set_title(sc); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"First layer {first_layer}: {ylabel} vs analytic FLOPs (seed {seed}; bars = Wilson 95% CI; one marker per schedule; open = equal_drop)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _matched(m, scorer, first_layer, path, seed):
    d = m[(m.scorer == scorer) & (m.seed == seed) & (m.first_layer == first_layer)]
    if d.empty:
        return
    groups = sorted(d.group.unique(), key=lambda g: d[d.group == g].budget_target.iloc[0])
    fig, axes = plt.subplots(1, len(groups), figsize=(4.0 * len(groups), 4.6), sharey=False, squeeze=False)
    for ax, g in zip(axes[0], groups):
        s = d[d.group == g].sort_values(["stage_count", "structure", "allocation_family"]).reset_index(drop=True)
        for k, fam in [(1, "shared"), (2, "geometric"), (2, "equal_drop"), (3, "geometric"), (3, "equal_drop")]:
            mk, col, lab = _STY[k]
            t = s[s.stage_count == k] if k < 2 else s[(s.stage_count == k) & (s.allocation_family == fam)]
            if t.empty:
                continue
            ax.errorbar(t.index, 100 * t.agreement, yerr=100 * _err(t.agreement, t.agreement_ci_low, t.agreement_ci_high), fmt=mk, color=col,
                        mfc=("white" if fam == "equal_drop" else col), capsize=2, ms=6, ls="none", label=lab + ("" if k < 2 else f" ({fam})"))
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(["1-stage" if r.stage_count == 1 else f"{r.structure}\n{r.allocation_family[:4]}" for r in s.itertuples()], rotation=70, fontsize=6)
        ax.set_title(f"{g}\n(FLOPs {100 * s.normalized_flops.min():.1f}–{100 * s.normalized_flops.max():.1f}%)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("agreement with unpruned (%)"); axes[0][0].legend(fontsize=6)
    fig.suptitle(f"Matched-FLOPs budgets, `{scorer}`, first layer {first_layer}, seed {seed} (bars = Wilson 95% CI; open = equal_drop)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _allocation(st, path):
    if st is None or st.empty:
        return
    d = st[(st.comparison_type == "geometric_vs_equal_drop") & (st.metric == "agreement") & (st.scorer == "cls_attention") &
           (st.seed == 42) & (st.first_layer == 2)].sort_values(["budget_target", "stage_count_a", "structure_a"]).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(d) + 2), 4.4))
    col = np.where(d.stage_count_a == 2, "#2ca02c", "#d62728")
    ax.errorbar(d.index, 100 * d["diff"], yerr=100 * _err(d["diff"], d.ci_low, d.ci_high), fmt="none", ecolor="gray", capsize=2)
    ax.scatter(d.index, 100 * d["diff"], c=col, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(d.index); ax.set_xticklabels([f"B{int(100 * r.budget_target)}\n{r.structure_a}" for r in d.itertuples()], rotation=70, fontsize=7)
    ax.set_ylabel("agreement: geometric − equal_drop (pp)"); ax.grid(alpha=0.3)
    ax.set_title("Allocation-family comparison at matched FLOPs, cls_attention, layer 2 (bars = paired-bootstrap 95% CI; green 2-stage, red 3-stage)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _stage_effect(st, path):
    if st is None or st.empty:
        return
    d = st[(st.comparison_type == "multi_vs_one_stage") & (st.metric == "agreement") & (st.scorer == "cls_attention") &
           (st.seed == 42) & (st.first_layer == 2)].sort_values(["family_a", "budget_target", "stage_count_a", "structure_a"]).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(d) + 2), 4.6))
    col = np.where(d.family_a == "geometric", "#1f77b4", "#ff7f0e")
    ax.errorbar(d.index, 100 * d["diff"], yerr=100 * _err(d["diff"], d.ci_low, d.ci_high), fmt="none", ecolor="gray", capsize=2)
    ax.scatter(d.index, 100 * d["diff"], c=col, marker="o", zorder=3)
    sig = d["significant_holm_0.05"].to_numpy()
    ax.scatter(d.index[sig], 100 * d["diff"][sig], facecolors="none", edgecolors="k", s=110, zorder=4, label="Holm p<0.05")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(d.index)
    ax.set_xticklabels([f"{r.family_a[:3]} B{int(100 * r.budget_target)} {r.structure_a}" for r in d.itertuples()], rotation=80, fontsize=6)
    ax.set_ylabel("agreement: multi-stage − one-stage (pp)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("Primary family: stage-count effect at matched FLOPs (blue geometric, orange equal_drop; bars = paired-bootstrap 95% CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _latency(m, first_layer, path, seed):
    if "latency_total_ms_median" not in m:
        return
    scs = [s for s in m[m.first_layer == first_layer].scorer.unique() if m[(m.scorer == s) & m.latency_total_ms_median.notna()].shape[0]]
    if not scs:
        return
    dev = m.latency_device.dropna().iloc[0]
    fig, axes = plt.subplots(1, len(scs), figsize=(5.4 * len(scs), 4.6), squeeze=False)
    base = m[(m.scorer == "none") & (m.seed == seed)]
    for ax, sc in zip(axes[0], scs):
        d = pd.concat([base, m[(m.scorer == sc) & (m.seed == seed) & (m.first_layer == first_layer)]])
        d = d[d.latency_total_ms_median.notna()]
        for k, (mk, col, lab) in _STY.items():
            s = d[d.stage_count == k]
            if s.empty:
                continue
            yerr = np.clip(np.vstack([s.latency_total_ms_median - s.latency_total_ms_min_batch, s.latency_total_ms_max_batch - s.latency_total_ms_median]), 0, None)
            ax.errorbar(100 * s.normalized_flops, s.latency_total_ms_median, yerr=yerr, fmt=mk, color=col, ls="none", ms=6, capsize=2, label=f"{lab}: total")
            if k:
                ax.plot(100 * s.normalized_flops, s.scorer_overhead_ms_median, "x", color=col, ls="none", ms=6, label=f"{lab}: scorer only")
        ax.set_title(sc); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
        if sc == "prediction_kl_approx":
            ax.set_yscale("log")
    axes[0][0].set_ylabel(f"median latency per batch (ms, {dev}; bars = range over batches)"); axes[0][0].legend(fontsize=6)
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
    _allocation(stats, os.path.join(pdir, "allocation_family_comparison_cls_attention_L2.png"))
    _stage_effect(stats, os.path.join(pdir, "stage_effect_primary_cls_attention_L2.png"))
