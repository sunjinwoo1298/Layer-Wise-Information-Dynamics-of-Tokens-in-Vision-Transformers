"""Plots for the final-token / trajectory isolation experiment. Discrete measured points with uncertainty only; no fitted lines."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COL = {"none": "black", "shared": "#7f7f7f", "geometric": "#1f77b4", "equal_drop": "#ff7f0e", "constrained_equal_drop": "#2ca02c",
        "final_count_matched_geometric": "#d62728"}
_NAME = {"geometric": "geometric", "equal_drop": "unconstrained equal-drop", "constrained_equal_drop": "constrained equal-drop",
         "final_count_matched_geometric": "final-count-matched geo. control", "shared": "one-stage anchor"}
_MK = {"geometric": "o", "equal_drop": "s", "constrained_equal_drop": "^", "final_count_matched_geometric": "D", "shared": "x"}


def _err(y, lo, hi):
    return np.clip(np.vstack([np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]), 0, None)


def _pr(m):
    return m[(m.role != "baseline")]


def plot_tokens(m, entries, path):
    ents = [e for e in entries if e["role"] == "member" and not e["alias_of"]]
    cells = sorted({(e["budget_target"], e["structure"]) for e in ents})
    nc = 4
    nr = int(np.ceil(len(cells) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for ax, (b, s) in zip(axes.ravel(), cells):
        for e in [x for x in ents if x["budget_target"] == b and x["structure"] == s]:
            ls = list(e["schedule"].keys()); ks = list(e["schedule"].values())
            ax.plot([0] + ls, [196] + ks, marker=_MK.get(e["family"], "o"), ms=3, lw=1.6 if e["family"] == "geometric" else 0.8,
                    color=_COL[e["family"]], alpha=0.9 if e["family"] != "final_count_matched_geometric" else 0.6,
                    label=_NAME[e["family"]] if e["name"] == next(x["name"] for x in ents if x["family"] == e["family"] and x["budget_target"] == b and x["structure"] == s) else None)
        ax.set_title(f"B{int(100 * b)}  pattern {s}", fontsize=9); ax.set_xlabel("layer (after block l)"); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    axes[0][0].set_ylabel("patch tokens kept"); axes[0][0].legend(fontsize=6)
    fig.suptitle("Patch-token count by pruning stage, all executed feasible schedules (layer 0 = 196 input patches)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _vs_flops(m, metric, ylabel, path):
    d = _pr(m)
    cells = sorted(d.structure.unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(6.5 * len(cells), 5), squeeze=False)
    for ax, s in zip(axes[0], cells):
        for fam, t in d[(d.structure == s)].groupby("allocation_family"):
            ax.errorbar(t.flops_pct, 100 * t[metric], yerr=100 * _err(t[metric], t[f"{metric}_ci_low"], t[f"{metric}_ci_high"]),
                        fmt=_MK.get(fam, "o"), color=_COL.get(fam, "k"), ms=5, capsize=1.5, lw=0.7, ls="none", alpha=0.8, label=_NAME.get(fam, fam))
        ax.set_title(f"pattern {s}", fontsize=9); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{ylabel} vs FLOPs by allocation family (cls_attention, seed 42; bars = Wilson 95% CI; one marker per schedule)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_paired(st, path):
    d = st[(st.metric == "agreement")].copy()
    d = d[d.comparison.isin(["geo_vs_ctrl", "geo_vs_cons", "geo_vs_unc", "cons_vs_unc"])]
    if d.empty:
        return
    d["cell"] = [f"B{int(100 * b)} {s}" for b, s in zip(d.budget_target, d.structure)]
    d = d.sort_values(["comparison", "budget_target", "structure", "schedule_b"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(d) + 2))
    cm = {"geo_vs_ctrl": "#d62728", "geo_vs_cons": "#2ca02c", "geo_vs_unc": "#ff7f0e", "cons_vs_unc": "#9467bd"}
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        ax.errorbar(100 * r.diff, yi, xerr=[[100 * (r.diff - r.ci_low)], [100 * (r.ci_high - r.diff)]], fmt="o", ms=3, lw=0.8,
                    color=cm[r.comparison], mfc="none" if r.p_holm >= 0.05 else cm[r.comparison])
    ax.set_yticks(y); ax.set_yticklabels([f"{r.cell} | {r.family_a} vs {r.family_b}" + (f" {r.vector_b}" if r.comparison == 'geo_vs_ctrl' else "")
                                          for r in d.itertuples()], fontsize=5)
    ax.axvline(0, color="k", lw=0.6)
    for k, c in cm.items():
        ax.plot([], [], "o", color=c, label=k)
    ax.legend(fontsize=7); ax.set_xlabel("agreement difference A − B (pp), 95% paired-bootstrap CI; filled = Holm-significant")
    ax.set_title("Paired agreement differences (geometric comparisons: positive = geometric better)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_final(m, path):
    d = _pr(m)
    budgets = sorted(d.budget_target.unique())
    fig, axes = plt.subplots(2, len(budgets), figsize=(4.2 * len(budgets), 7), squeeze=False, sharey="row")
    for r_i, s in enumerate(sorted(d.structure.unique())[:2]):
        for ax, B in zip(axes[r_i], budgets):
            t0 = d[(d.budget_target == B) & (d.structure == s)]
            for fam, t in t0.groupby("allocation_family"):
                ax.errorbar(t.final_patch_tokens, 100 * t.agreement, yerr=100 * _err(t.agreement, t.agreement_ci_low, t.agreement_ci_high),
                            fmt=_MK.get(fam, "o"), color=_COL.get(fam, "k"), ms=5, capsize=1.5, lw=0.6, ls="none", alpha=0.8, label=_NAME.get(fam, fam))
            ax.set_xscale("log"); ax.set_title(f"B{int(100 * B)} {s}", fontsize=9); ax.grid(alpha=0.3)
            ax.set_xlabel("final patch tokens")
        axes[r_i][0].set_ylabel("agreement with unpruned (%)")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Agreement vs final patch-token count (cls_attention, seed 42; bars = Wilson 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all_plots(metrics, stats, entries, out_dir):
    d = os.path.join(out_dir, "plots"); os.makedirs(d, exist_ok=True)
    plot_tokens(metrics, entries, os.path.join(d, "1_token_count_by_stage.png"))
    _vs_flops(metrics, "agreement", "agreement with unpruned (%)", os.path.join(d, "2_agreement_vs_flops.png"))
    _vs_flops(metrics, "top1", "top-1 accuracy (%)", os.path.join(d, "3_top1_vs_flops.png"))
    if stats is not None and len(stats):
        plot_paired(stats, os.path.join(d, "4_paired_agreement_differences.png"))
    plot_final(metrics, os.path.join(d, "5_final_tokens_vs_agreement.png"))
