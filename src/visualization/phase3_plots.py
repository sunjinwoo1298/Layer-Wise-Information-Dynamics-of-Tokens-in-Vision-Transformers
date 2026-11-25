"""Phase 3 plots. Discrete measured points with uncertainty only; no fitted lines."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COL = {"none": "black", "shared": "#7f7f7f", "geometric": "#1f77b4", "equal_drop": "#ff7f0e", "constrained_equal_drop": "#2ca02c", "proposed": "#d62728"}
_NAME = {"geometric": "geometric", "equal_drop": "unconstrained equal-drop", "constrained_equal_drop": "constrained equal-drop",
         "proposed": "proposed", "shared": "single-stage", "none": "unpruned"}
_MK = {"geometric": "o", "equal_drop": "s", "constrained_equal_drop": "^", "proposed": "D", "shared": "x", "none": "*"}


def _err(y, lo, hi):
    return np.clip(np.vstack([np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]), 0, None)


def plot_tokens(entries, path):
    ents = [e for e in entries if e.get("role") == "member" and not e.get("alias_of")]
    cells = sorted({(e["budget_target"], e["structure"]) for e in ents})
    nc = 4
    nr = int(np.ceil(len(cells) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.4 * nr), squeeze=False)
    for ax, (b, s) in zip(axes.ravel(), cells):
        seen = set()
        for e in [x for x in ents if x["budget_target"] == b and x["structure"] == s]:
            ls, ks = list(e["schedule"].keys()), list(e["schedule"].values())
            lab = _NAME[e["family"]] if e["family"] not in seen else None
            seen.add(e["family"])
            ax.plot([0] + ls, [196] + ks, marker=_MK[e["family"]], ms=3, lw=1.8 if e["family"] in ("proposed", "geometric") else 0.8,
                    color=_COL[e["family"]], alpha=0.9 if e["family"] == "proposed" else 0.6, label=lab)
        ax.set_title(f"B{int(100 * b)}  {s}", fontsize=9); ax.set_xlabel("layer (after block l)"); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    axes[0][0].set_ylabel("patch tokens kept"); axes[0][0].legend(fontsize=6)
    fig.suptitle("Token allocation by stage and budget (layer 0 = 196 input patches; proposed shown for all floors)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _vs_flops(m, metric, ylabel, path):
    d = m[m.family != "none"]
    cells = sorted(d.structure.unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(6.3 * len(cells), 5), squeeze=False)
    for ax, s in zip(axes[0], cells):
        for fam, t in d[d.structure == s].groupby("family"):
            ax.errorbar(t.flops_pct, 100 * t[metric], yerr=100 * _err(t[metric], t[f"{metric}_ci_low"], t[f"{metric}_ci_high"]), fmt=_MK.get(fam, "o"),
                        color=_COL.get(fam, "k"), ms=5, capsize=1.5, lw=0.7, ls="none", alpha=0.8, label=_NAME.get(fam, fam))
        ax.set_title(f"pattern {s}", fontsize=9); ax.set_xlabel("analytic FLOPs (% of unpruned)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{ylabel} vs FLOPs (bars = Wilson 95% CI; single-stage anchors appear in every panel)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_paired(st, path):
    d = st[(st.comparison == "prop_vs_geo") & (st.metric == "agreement")].sort_values(["budget_target", "structure", "floor"]).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(d) + 1.8))
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        sig = r.p_holm < 0.05
        ax.errorbar(100 * r.diff, yi, xerr=[[100 * (r.diff - r.ci_low)], [100 * (r.ci_high - r.diff)]], fmt="o", ms=4, lw=0.9, color="#d62728",
                    mfc="#d62728" if sig else "none")
    ax.set_yticks(y); ax.set_yticklabels([f"B{int(100 * r.budget_target)} {r.structure} f{int(r.floor)} {r.vector_a}" for r in d.itertuples()], fontsize=6)
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("agreement: proposed − geometric (pp), 95% paired-bootstrap CI; filled = Holm-significant")
    ax.set_title("Proposed vs geometric at matched FLOPs (positive = proposed better)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cal_val(m, path):
    d = m[(m.family != "none") & m.cal_agreement.notna()]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for fam, t in d.groupby("family"):
        ax.errorbar(100 * t.cal_agreement, 100 * t.agreement, yerr=100 * _err(t.agreement, t.agreement_ci_low, t.agreement_ci_high), fmt=_MK.get(fam, "o"),
                    color=_COL.get(fam, "k"), ms=4, capsize=1, lw=0.5, ls="none", alpha=0.75, label=_NAME.get(fam, fam))
    lo, hi = 100 * min(d.cal_agreement.min(), d.agreement.min()) - 2, 100 * max(d.cal_agreement.max(), d.agreement.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
    ax.set_xlabel("calibration agreement (%)  [500 images]"); ax.set_ylabel("evaluation-split agreement (%)  [bars = Wilson 95% CI]")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Calibration vs held-out agreement (dashed = equality)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_latency(lat, path):
    if lat is None or len(lat) == 0:
        return
    d = lat[lat.family != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for fam, t in d.groupby("family"):
        axes[0].scatter(t.flops_pct, t.end_to_end_ms_median, s=22, color=_COL.get(fam, "k"), marker=_MK.get(fam, "o"), alpha=0.8, label=_NAME.get(fam, fam))
        axes[1].scatter(t.scorer_overhead_ms_median, t.end_to_end_ms_median, s=22, color=_COL.get(fam, "k"), marker=_MK.get(fam, "o"), alpha=0.8)
    axes[0].axhline(d.unpruned_forward_ms_median.iloc[0], color="k", ls=":", lw=0.8, label="unpruned forward")
    axes[0].set_xlabel("analytic FLOPs (%)"); axes[0].set_ylabel("end-to-end CPU latency (ms, batch 25)"); axes[0].legend(fontsize=6); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("scorer overhead (ms, inside the end-to-end time)"); axes[1].set_ylabel("end-to-end CPU latency (ms)"); axes[1].grid(alpha=0.3)
    fig.suptitle("FLOPs, scorer overhead and measured latency are separate quantities (CPU only; calibration/search cost reported separately)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all_plots(out, cfg, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    m, st, ent, lat = out["metrics"], out["stats"], out["entries"], out["latency"]
    plot_tokens(ent, os.path.join(plots_dir, "1_token_allocation.png"))
    _vs_flops(m, "agreement", "agreement with unpruned (%)", os.path.join(plots_dir, "2_agreement_vs_flops.png"))
    _vs_flops(m, "top1", "top-1 accuracy (%)", os.path.join(plots_dir, "3_top1_vs_flops.png"))
    if st is not None and len(st):
        plot_paired(st, os.path.join(plots_dir, "4_proposed_vs_geometric_paired.png"))
    plot_cal_val(m, os.path.join(plots_dir, "5_calibration_vs_validation.png"))
    plot_latency(lat, os.path.join(plots_dir, "6_overhead_vs_latency.png"))
