#!/usr/bin/env python3
"""
Phase 4 Gate 2: analytic (label-free, prediction-free) FLOPs feasibility of each method's NATIVE budget controls.
Uses the existing ComputeEstimator read-only. Writes only to results/phase4_adaptive_method_benchmark/.
FLOPs convention: attention (+ norm1 half) charged on tokens ENTERING a block, MLP (+ norm2 half) on tokens LEAVING the block's
compression step; patch-embed + head fixed; identical to the Phase 1-3 ComputeEstimator when no compression happens (verified below).
Merge/fusion/scoring cost is NOT in this FLOPs figure; it is reported separately as auxiliary overhead.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.metrics.compute import ComputeEstimator  # noqa: E402
from src.optimization.schedule_search import fixed_flops  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
D, MLP, H, L = 192, 768, 3, 12
est = ComputeEstimator(num_layers=L, hidden_dim=D, num_heads=H, num_original_tokens=196)


def attn_f(T): return 6 * T * D * D + 2 * T * T * D + 3 * H * T * T + 2 * T * T * D + 2 * T * D * D + 2 * T * D


def mlp_f(T): return 2 * T * D * MLP + 8 * T * MLP + 2 * T * MLP * D + 2 * T * D


def flops_pct(entering, leaving):
    tot = fixed_flops(est) + sum(attn_f(a) + mlp_f(m) for a, m in zip(entering, leaving))
    return 100.0 * tot / est.baseline_flops


assert abs(flops_pct([197] * L, [197] * L) - 100.0) < 1e-9

BUDGETS, TOL = [40, 50, 60, 70], 1.0


def tome_r_list(r, infl):
    mn = int(r * (1 - infl)); mx = 2 * r - mn
    return [r] * L if infl == 0 else [int(mn + (mx - mn) * i / (L - 1)) for i in range(L)]  # as read in tome/utils.py parse_r (to be parity-checked)


def tome_counts(rs):
    n, ent, out = 197, [], []
    for r in rs:
        ent.append(n); n -= max(min(r, (n - 1) // 2), 0); out.append(n)
    return ent, out


def evit_counts(kappa, locs=(4, 7, 10)):
    n, ent, out = 197, [], []
    for i in range(1, L + 1):
        ent.append(n)
        if i in locs:
            n = int(np.ceil(kappa * (n - 1))) + 1 + 1  # CLS + top-k + one fused token
        out.append(n)
    return ent, out


def diffrate_counts(prune, merge):
    n, ent, out = 197, [], []
    for p, m in zip(prune, merge):
        ent.append(n); n = min(n, p); n = min(n, m); out.append(n)  # as released; realized counts are re-checked when executed
    return ent, out


rows = []
for r in range(1, 41):
    for infl in (-1.0, -0.5, 0.0, 0.5, 1.0):
        e, o = tome_counts(tome_r_list(r, infl))
        rows.append(dict(method="ToMe", config=f"r={r},inflection={infl}", flops_pct=flops_pct(e, o), final_tokens=o[-1], counts=json.dumps(o)))
for k in np.arange(0.20, 1.0001, 0.005):
    e, o = evit_counts(float(k))
    rows.append(dict(method="EViT-style", config=f"kappa={k:.3f}", flops_pct=flops_pct(e, o), final_tokens=o[-1], counts=json.dumps(o)))
rates = json.load(open(os.path.join(OUT, "diffrate_deit_tiny_rates_as_published.json")))
for key, v in rates.items():
    if key.startswith("_"):
        continue
    e, o = diffrate_counts(v["prune_kept_num"], v["merge_kept_num"])
    rows.append(dict(method="DiffRate(released)", config=f"target={key}", flops_pct=flops_pct(e, o), final_tokens=o[-1], counts=json.dumps(o)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "analytic_candidates.csv"), index=False)

summ = []
for m in ("ToMe", "EViT-style", "DiffRate(released)"):
    t = df[df.method == m]
    for b in BUDGETS:
        ok = t[(t.flops_pct - b).abs() <= TOL]
        best = t.iloc[(t.flops_pct - b).abs().argsort()[:1]].iloc[0]
        summ.append(dict(method=m, budget=b, n_feasible_configs=len(ok), nearest_config=best.config, nearest_flops_pct=round(best.flops_pct, 2),
                         abs_gap_pp=round(abs(best.flops_pct - b), 2), feasible=bool(len(ok))))
s = pd.DataFrame(summ)
s.to_csv(os.path.join(OUT, "analytic_feasibility_summary.csv"), index=False)
print(s.to_string(index=False))
print(df[df.method == "DiffRate(released)"][["config", "flops_pct", "final_tokens"]].round(2).to_string(index=False))
