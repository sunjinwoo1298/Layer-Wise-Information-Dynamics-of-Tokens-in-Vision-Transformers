"""Plots for the minimum final-token study. Discrete measured points with uncertainty only; no fitted lines."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COL = {"none": "black", "shared": "#7f7f7f", "geometric": "#1f77b4", "equal_drop": "#ff7f0e", "constrained_equal_drop": "#2ca02c"}
_MK = {0: "*", 1: "o", 2: "s", 3: "^"}
_FLOOR_COL = {8: "#bcbd22", 16: "#2ca02c", 24: "#17becf", 32: "#9467bd"}


def _err(y, lo, hi):
    return np.clip(np.vstack([np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]), 0, None)


def _prim(m, seed):
    return m[(m.scorer == "cls_attention") & (m.seed == seed) & (m.first_layer == 2)]


def _vs_flops(m, metric, ylabel, path, seed):
    d = pd.concat([m[(m.scorer == "none") & (m.seed == seed)], _prim(m, seed)])
    fig, ax = plt.subplots(figsize=(8, 5))
    for (fam, k), s in d.groupby(["allocation_family", "stage_count"]):
        if fam == "constrained_equal_drop":
            for F, t in s.groupby("floor"):
                ax.errorbar(100 * t.normalized_flops, 100 * t[metric], yerr=100 * _err(t[metric], t[f"{metric}_ci_low"], t[f"{metric}_ci_high"]),
                            fmt=_MK[k], color=_FLOOR_COL[int(F)], mfc="none", ms=5, capsize=1.5, lw=0.7, ls="none", alpha=0.85,
                            label=f"constrained f{int(F)} ({k}-st)")
        else:
            ax.errorbar(100 * s.normalized_flops, 100 * s[metric], yerr=100 * _err(s[metric], s[f"{metric}_ci_low"], s[f"{metric}_ci_high"]),
                        fmt=_MK[k], color=_COL.get(fam, "k"), ms=6, capsize=1.5, lw=0.7, ls="none", alpha=0.85,
                        label=f"{fam} ({k}-st)" if k else "unpruned")
    ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=5, ncol=2)
    ax.set_title(f"cls_attention, first layer 2, seed {seed}: {ylabel} vs FLOPs (bars = Wilson 95% CI; one marker per schedule; hollow = constrained)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _vs_final(m, path, seed):
    d = _prim(m, seed)
    d = d[d.stage_count >= 2]
    if d.empty:
        return
    budgets = sorted(d.budget_target.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 4.4), squeeze=False)
    for ax, B in zip(axes[0], budgets):
        s = d[d.budget_target == B]
        for fam in ("geometric", "equal_drop", "constrained_equal_drop"):
            t = s[s.allocation_family == fam]
            if t.empty:
                continue
            ax.errorbar(t.final_patch_tokens, 100 * t.agreement, yerr=100 * _err(t.agreement, t.agreement_ci_low, t.agreement_ci_high), fmt="o",
                        color=_COL[fam], ms=4, capsize=1.5, lw=0.6, ls="none", alpha=0.8, label=fam)
        ax.set_xscale("log"); ax.set_xlabel("final patch tokens (after last prune)"); ax.set_title(f"budget {int(100 * B)}% FLOPs", fontsize=9); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("agreement with unpruned (%)"); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Agreement vs final patch-token count (cls_attention, layer 2, seed {seed}; bars = Wilson 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _paired(st, comparison, title, path, infeasible):
    if st is None or st.empty:
        return
    d = st[(st.comparison_type == comparison) & (st.metric == "agreement") & (st.scorer == "cls_attention") & (st.seed == 42)].sort_values(
        ["floor_a", "stage_count", "budget_target", "structure"]).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(d) + 2), 5))
    ax.errorbar(d.index, 100 * d["diff"], yerr=100 * _err(d["diff"], d.ci_low, d.ci_high), fmt="none", ecolor="gray", capsize=2, lw=0.8)
    ax.scatter(d.index, 100 * d["diff"], c=[_FLOOR_COL[int(f)] for f in d.floor_a], zorder=3, s=22)
    sig = d["significant_holm_0.05"].to_numpy()
    ax.scatter(d.index[sig], 100 * d["diff"][sig], facecolors="none", edgecolors="k", s=90, zorder=4, label="Holm p<0.05 (within its family)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(d.index)
    ax.set_xticklabels([f"f{int(r.floor_a)} B{int(100 * r.budget_target)} {r.structure}" for r in d.itertuples()], rotation=85, fontsize=5)
    ax.set_ylabel("agreement difference (pp)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    inf = "; ".join(f"f{u['floor']} B{int(100 * u['budget'])} {u['structure']}" for u in infeasible) or "none"
    ax.set_title(title + f"\n(colours = floor 8/16/24/32; bars = paired-bootstrap 95% CI; INFEASIBLE, not plotted: {inf})", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _latency(m, path, seed):
    if "latency_total_ms_median" not in m:
        return
    d = m[(m.seed == seed) & m.latency_total_ms_median.notna() & ~m.is_alias]
    scs = [s for s in d.scorer.unique() if s != "none"]
    if not scs:
        return
    fig, axes = plt.subplots(1, len(scs), figsize=(5.2 * len(scs), 4.5), squeeze=False)
    dev = d.latency_device.iloc[0]
    for ax, sc in zip(axes[0], scs):
        s = pd.concat([d[d.scorer == "none"], d[d.scorer == sc]])
        for (fam, k), t in s.groupby(["allocation_family", "stage_count"]):
            ax.plot(100 * t.normalized_flops, t.latency_total_ms_median, _MK[k], color=_COL.get(fam, "k"), ls="none", ms=5, alpha=0.8,
                    label=f"{fam} {k}-st: total")
            if k:
                ax.plot(100 * t.normalized_flops, t.scorer_overhead_ms_median, "x", color=_COL.get(fam, "k"), ls="none", ms=5, alpha=0.8)
        ax.set_title(sc + " (x = scorer overhead only)"); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
        if sc == "prediction_kl_approx":
            ax.set_yscale("log")
    axes[0][0].set_ylabel(f"median latency per batch (ms, {dev})"); axes[0][0].legend(fontsize=5)
    fig.suptitle("Measured latency vs analytic FLOPs (separate quantities; not a speedup claim)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all_plots(metrics, stats, infeasible, out_dir, seed=42):
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    m = metrics.copy()
    if "is_alias" not in m:
        m["is_alias"] = False
    _vs_flops(m, "agreement", "agreement with unpruned (%)", os.path.join(pdir, "agreement_vs_flops.png"), seed)
    _vs_flops(m, "top1", "top-1 accuracy (%)", os.path.join(pdir, "top1_vs_flops.png"), seed)
    _vs_final(m, os.path.join(pdir, "performance_vs_final_tokens.png"), seed)
    _paired(stats, "constrained_vs_unconstrained", "Constrained minus unconstrained equal-drop (cls_attention, layer 2)",
            os.path.join(pdir, "constrained_vs_unconstrained.png"), infeasible)
    _paired(stats, "constrained_vs_geometric", "Constrained equal-drop minus geometric (cls_attention, layer 2)",
            os.path.join(pdir, "constrained_vs_geometric.png"), infeasible)
    _latency(m, os.path.join(pdir, "latency_vs_flops.png"), seed)
