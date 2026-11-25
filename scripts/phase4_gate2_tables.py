#!/usr/bin/env python3
"""
Phase 4 Gate 2 corrections (read-only on Phase 3 artifacts; writes only under results/phase4_adaptive_method_benchmark/):
  A. diffrate_flops_interpretations.csv : published lists, literal requested counts, candidate execution interpretations, analytic FLOPs.
     Executed counts are NOT known yet (Gate 3); every FLOPs value here is an analytic prediction, not a verified realized value.
  B. phase3_replication_mapping.csv     : the 24 statistical comparisons proposed-vs-geometric, with the exact frozen schedule used for each floor.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from phase4_analytic_feasibility import flops_pct  # noqa: E402  (re-runs the feasibility script's module-level code; deterministic)

OUT = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
rates = json.load(open(os.path.join(OUT, "diffrate_deit_tiny_rates_as_published.json")))
rows = []
for key, v in rates.items():
    if key.startswith("_"):
        continue
    p, m = v["prune_kept_num"], v["merge_kept_num"]
    # I1: per-block literal request, each block's output = min(prune_i, merge_i) irrespective of the block's actual input (can rise)
    lit = [min(a, b) for a, b in zip(p, m)]
    ent1 = [197] + lit[:-1]
    # I2: monotone-clamped: a block can never output more tokens than it received
    n, ent2, out2 = 197, [], []
    for a, b in zip(p, m):
        ent2.append(n); n = min(n, a, b); out2.append(n)
    rises = [i for i in range(1, len(lit)) if lit[i] > lit[i - 1]]
    # literal merge requests that exceed the tokens available (merge target larger than current count, i.e. an ill-defined merge)
    ill = [i for i in range(len(m)) if m[i] < p[i] and m[i] > ent2[i]]
    rows.append(dict(target=key, published_prune_kept_num=json.dumps(p), published_merge_kept_num=json.dumps(m),
                     literal_requested_counts_min_per_block=json.dumps(lit), literal_has_rise_at_blocks=json.dumps(rises),
                     ill_defined_merge_blocks_under_monotone=json.dumps(ill), I1_literal_flops_pct=round(flops_pct(ent1, lit), 4),
                     I2_monotone_clamped_counts=json.dumps(out2), I2_monotone_clamped_flops_pct=round(flops_pct(ent2, out2), 4),
                     executed_token_counts="PENDING (Gate 3 execution)", executed_flops_pct="PENDING (Gate 3 execution)",
                     status="analytic prediction only; not verified until executed"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "diffrate_flops_interpretations.csv"), index=False)
print(pd.DataFrame(rows)[["target", "literal_has_rise_at_blocks", "ill_defined_merge_blocks_under_monotone", "I1_literal_flops_pct", "I2_monotone_clamped_flops_pct"]].to_string(index=False))

fz = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
prop = {(round(s["budget_target"], 2), s["structure"], s["floor"]): s for s in fz["proposed_schedules"]}
geo = {(round(s["budget_target"], 2), s["structure"]): s for s in fz["baseline_schedules"] if s["family"] == "geometric"}
m = []
for b in (0.4, 0.5, 0.6, 0.7):
    for st in ("2-6-10", "2-8-10"):
        for fl in (16, 24, 32):
            p, g = prop[(b, st, fl)], geo[(b, st)]
            m.append(dict(comparison_id=f"B{int(b * 100)}_{st}_f{fl}", budget_pct=int(b * 100), pattern=st, floor=fl, proposed_schedule=p["name"],
                          proposed_layer_to_patch_count=json.dumps({int(k): v for k, v in sorted(p["schedule"].items(), key=lambda kv: int(kv[0]))}),
                          proposed_alias_of=p.get("alias_of"), geometric_schedule=g["name"],
                          geometric_layer_to_patch_count=json.dumps({int(k): v for k, v in sorted(g["schedule"].items(), key=lambda kv: int(kv[0]))})))
pd.DataFrame(m).to_csv(os.path.join(OUT, "phase3_replication_mapping.csv"), index=False)
print(len(m), "comparisons;", len({r["proposed_schedule"] for r in m}), "distinct proposed schedule names;",
      len({r["proposed_layer_to_patch_count"] + r["pattern"] for r in m}), "distinct (pattern, vector)")
print(pd.DataFrame(m)[["comparison_id", "proposed_schedule", "proposed_layer_to_patch_count", "geometric_layer_to_patch_count", "proposed_alias_of"]].head(7).to_string(index=False))
