#!/usr/bin/env python3
"""
Phase 4 Gate 3: execute the RELEASED DiffRate DeiT-Tiny lists (as published, unrepaired) on P4-DEV only and record, separately:
published lists, literal requested counts, EXECUTED token counts, resulting analytic FLOPs, executed events (incl. ill-defined merges),
and dev agreement / top-1 (validation-vs-paper plausibility check; P4-DEV only; not a benchmark result).
"""
import json
import os
import sys
import time

import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.phase4.data import load_split  # noqa: E402
from src.phase4.diffrate import DiffRateReleased  # noqa: E402
from src.phase4.engine import Compressor, flops_pct, run_compressed  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
RATES = os.path.join(P4, "diffrate_deit_tiny_rates_as_published.json")
torch.set_num_threads(4)
d = load_split("dev")
X, y = d["images"], d["labels"]
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
t0 = time.time()
ref = torch.cat([run_compressed(model, X[i:i + 25], Compressor())[0].argmax(1) for i in range(0, len(X), 25)])
out = {"n": len(X), "unpruned_top1_count": int((ref == y).sum()), "unpruned_seconds": round(time.time() - t0, 1),
       "paper_reference_table8": {"baseline_top1": 72.13, "offtheshelf_top1": {"0.6": 70.36, "0.7": 71.16, "0.8": 71.74, "0.9": 71.91, "1.0": 72.12},
                                  "note": "full ImageNet validation, paper's counting; not comparable in absolute terms to this 995-class dev subset"},
       "targets": {}}
published = json.load(open(RATES))
for tgt in ("0.6", "0.7", "0.8", "0.9", "1.0"):
    pub = published[tgt]
    literal = [min(a, b) for a, b in zip(pub["prune_kept_num"], pub["merge_kept_num"])]
    t1 = time.time()
    preds, info0 = [], None
    for i in range(0, len(X), 25):
        lg, info = run_compressed(model, X[i:i + 25], DiffRateReleased.from_published(RATES, tgt))
        preds.append(lg.argmax(1))
        info0 = info0 or info
    pr = torch.cat(preds)
    out["targets"][tgt] = {
        "published_prune_kept_num": pub["prune_kept_num"], "published_merge_kept_num": pub["merge_kept_num"],
        "literal_requested_counts_min_per_block": literal,
        "executed_entering": info0["entering"], "executed_leaving": info0["leaving"],
        "executed_analytic_flops_pct": round(flops_pct(info0["entering"], info0["leaving"]), 4),
        "events_ill_defined": [e for e in info0["events"] if "ill_defined" in e],
        "dev_agreement_count": int((pr == ref).sum()), "dev_top1_count": int((pr == y).sum()), "n": len(X),
        "executed_equals_literal_requested": info0["leaving"] == literal, "seconds": round(time.time() - t1, 1)}
    print(tgt, out["targets"][tgt]["executed_analytic_flops_pct"], "agree", out["targets"][tgt]["dev_agreement_count"], "top1",
          out["targets"][tgt]["dev_top1_count"], "ill", len(out["targets"][tgt]["events_ill_defined"]), flush=True)
json.dump(out, open(os.path.join(P4, "diffrate_dev_validation.json"), "w"), indent=1)
