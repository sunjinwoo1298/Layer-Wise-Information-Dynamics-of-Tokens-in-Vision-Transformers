#!/usr/bin/env python3
"""
Phase 4 Gate 4 (P4-DEV ONLY): regenerate the Phase 3 comparators at the primary budgets on all 1,990 P4-DEV images, using the FROZEN Phase 3 schedules
(no re-selection) through the unchanged Phase 1-3 progressive pruner. Schedules are read as {layer: patch_count} dictionaries (see Phase 3 errata).
Writes results/phase4_adaptive_method_benchmark/gate4/phase3_comparators_dev.csv and phase3_comparators_dev_predictions.npz. P4-CONFIRM never loaded.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.experiments.budget_aware_allocation import AllocConfig, DataSet, Runner  # noqa: E402  (read-only reuse)
from src.phase4.data import load_split  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
G4 = os.path.join(P4, "gate4")
torch.set_num_threads(int(os.environ.get("P4_THREADS", "4")))
fz = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
hp = dict(fz["hyperparameters"]); hp.update(output_dir=G4, device=None, max_cal=None, max_val=None, max_conf=None)
cfg = AllocConfig(**hp)
d = load_split("dev")
man = pd.read_csv(os.path.join(P4, "splits", "p4_dev_manifest.csv"))
ds = DataSet("p4dev_full", d["images"], d["labels"], man.image_id.to_numpy(), man)
runner = Runner(cfg)
runner.prepare(ds)
ref = runner.baseline_pred(ds)
entries = fz["proposed_schedules"] + fz["baseline_schedules"]
rows, preds, cache, t0 = [], {}, {}, time.time()
uniq = {}
for e in entries:
    if e.get("alias_of"):
        continue
    sched = {int(k): int(v) for k, v in e["schedule"].items()}
    uniq.setdefault(tuple(sorted(sched.items())), []).append(e["name"])
print("unique schedules to run:", len(uniq), flush=True)
for j, (key, names) in enumerate(uniq.items()):
    t1 = time.time()
    r = runner.run(dict(key), ds)
    cache[key] = r
    print(f"[{j + 1}/{len(uniq)}] {names[0]} agree={int(r['agree'].sum())} top1={int(r['correct'].sum())} ({time.time() - t1:.0f}s)", flush=True)
for e in entries:
    src_name = e.get("alias_of") or e["name"]
    base = next(x for x in entries if x["name"] == src_name)
    sched = {int(k): int(v) for k, v in base["schedule"].items()}
    r = cache[tuple(sorted(sched.items()))]
    preds[e["name"]] = r["pred"].astype(np.int16)
    rows.append({"schedule": e["name"], "family": e["family"], "floor": e.get("floor"), "structure": e["structure"], "budget_target": e["budget_target"],
                 "vector_by_layer": json.dumps(dict(sorted(sched.items()))), "flops_pct": round(100 * e["normalized_flops"], 4),
                 "agree_count": int(r["agree"].sum()), "top1_count": int(r["correct"].sum()), "n": len(ds), "alias_of": e.get("alias_of")})
pd.DataFrame(rows).to_csv(os.path.join(G4, "phase3_comparators_dev.csv"), index=False)
np.savez_compressed(os.path.join(G4, "phase3_comparators_dev_predictions.npz"), ref=ref, labels=ds.labels.numpy(), **preds)
json.dump({"n": len(ds), "unique_schedules": len(uniq), "seconds": round(time.time() - t0, 1), "frozen_sha256": open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen.sha256")).read().strip()},
          open(os.path.join(G4, "phase3_comparators_dev_meta.json"), "w"), indent=1)
print("DONE", round(time.time() - t0), "s")
