#!/usr/bin/env python3
"""Phase 4 Gate 4 bookkeeping (P4-DEV image 0 only, label-free): record the EXACT candidate grids (ToMe 200, EViT-style 161) with realized analytic FLOPs from the
Phase 4 engine, and list every grid point where the Gate 2 analytic table (np.arange kappa, float drift) differs. Writes gate4/grid_realized_flops.csv and
gate4/candidate_grid_record.json. P4-CONFIRM is never loaded."""
import hashlib
import json
import os
import sys

import pandas as pd
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.phase4.data import load_split  # noqa: E402
from src.phase4.engine import flops_pct, run_compressed  # noqa: E402
from src.phase4.evit_style import EViTStyle  # noqa: E402
from src.phase4.selection import kappa_grid, tome_grid  # noqa: E402
from src.phase4.tome import ToMe  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
G4 = os.path.join(P4, "gate4")
X = load_split("dev")["images"][:1]
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
rows = []
for r, i in tome_grid():
    _, info = run_compressed(model, X, ToMe(r if i == 0 else (r, float(i))))
    rows.append({"method": "tome", "config": json.dumps([r, i]), "realized_flops_pct": round(flops_pct(info["entering"], info["leaving"]), 6), "final_tokens": info["leaving"][-1]})
for k in kappa_grid():
    _, info = run_compressed(model, X, EViTStyle(k))
    rows.append({"method": "evit", "config": json.dumps(k), "realized_flops_pct": round(flops_pct(info["entering"], info["leaving"]), 6), "final_tokens": info["leaving"][-1]})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(G4, "grid_realized_flops.csv"), index=False)
an = pd.read_csv(os.path.join(P4, "analytic_candidates.csv"))
diffs = []
for _, a in an.iterrows():
    if a.method == "EViT-style":
        k = round(float(a.config.split("=")[1]), 3)
        r = df[(df.method == "evit") & (df.config == json.dumps(k))].iloc[0]
    elif a.method == "ToMe":
        rr, ii = a.config.replace("r=", "").replace("inflection=", "").split(",")
        r = df[(df.method == "tome") & (df.config == json.dumps([int(rr), float(ii)]))].iloc[0]
    else:
        continue
    if abs(a.flops_pct - r.realized_flops_pct) > 1e-4:
        diffs.append({"method": a.method, "config": a.config, "gate2_analytic_flops_pct": round(float(a.flops_pct), 6), "gate4_realized_flops_pct": float(r.realized_flops_pct)})
rec = {"tome_grid": "r in 1..40 x inflection in {-1,-0.5,0,0.5,1} (200); r int if inflection==0 else (r, inflection)",
       "evit_grid": "kappa = round(0.20 + 0.005*i, 3), i = 0..160 (161)",
       "grid_realized_flops_csv_sha256": hashlib.sha256(open(os.path.join(G4, "grid_realized_flops.csv"), "rb").read()).hexdigest(),
       "n_grid_points": len(df), "differences_vs_gate2_analytic_table": diffs,
       "explanation": "Gate 2 analytic script generated kappa with np.arange(0.20,1.0001,0.005); values such as 0.7500000000000004 push ceil(kappa*(N-1)) up by one token, whereas the protocol's rounded grid (used at Gate 4) gives the exact decimal. Realized engine counts on the rounded grid are authoritative."}
json.dump(rec, open(os.path.join(G4, "candidate_grid_record.json"), "w"), indent=1)
print(json.dumps(diffs, indent=1))
