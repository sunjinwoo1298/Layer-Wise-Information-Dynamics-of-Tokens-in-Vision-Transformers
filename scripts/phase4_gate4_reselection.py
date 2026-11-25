#!/usr/bin/env python3
"""
Phase 4 Gate 4 (P4-DEV ONLY): re-run the UNCHANGED Phase 3 selection pipeline (src.experiments.budget_aware_allocation.run_calibration) with new FLOPs caps equal
to DiffRate's executed analytic FLOPs for targets 0.6-0.9, using the fixed 500-image P4-DEV calibration subset (manifest/hash fixed before this run).
Geometric stays outside the candidate pool (Phase 3 code unchanged). Then evaluate the resulting schedules and baselines on the 1,490 held-out P4-DEV images.
Writes only under results/phase4_adaptive_method_benchmark/supplementary_phase3_reselection/. P4-CONFIRM is never loaded.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.experiments.budget_aware_allocation import AllocConfig, DataSet, Runner, run_calibration  # noqa: E402  (read-only reuse)
from src.phase4.data import load_split  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
OUT = os.path.join(P4, "supplementary_phase3_reselection")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("P4_THREADS", "4")))

fz3 = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
nat = json.load(open(os.path.join(P4, "diffrate_dev_validation.json")))["targets"]
budgets = [round(nat[t]["executed_analytic_flops_pct"] / 100.0, 6) for t in ("0.6", "0.7", "0.8", "0.9")]
hp = dict(fz3["hyperparameters"])
hp.update(budgets=budgets, output_dir=OUT, resume=True, make_plots=False, max_cal=None, max_val=None, max_conf=None, device=None)
cfg = AllocConfig(**hp)

d = load_split("dev")
cal_m = pd.read_csv(os.path.join(P4, "splits", "p4_dev_cal500_manifest.csv"))
held_m = pd.read_csv(os.path.join(P4, "splits", "p4_dev_heldout1490_manifest.csv"))
summ = json.load(open(os.path.join(P4, "splits", "p4_dev_cal500_summary.json")))
assert hashlib.sha256(",".join(cal_m.image_id).encode()).hexdigest() == summ["calibration_ids_sha256"], "calibration subset changed after being fixed"
assert hashlib.sha256(",".join(held_m.image_id).encode()).hexdigest() == summ["heldout_ids_sha256"]


def mk(tag, m):
    idx = m.dev_position.to_numpy()
    return DataSet(tag, d["images"][idx], d["labels"][idx], m.image_id.to_numpy(), m)


data = {"calibration": mk("calibration", cal_m), "validation": mk("validation", held_m)}
runner = Runner(cfg)
t0 = time.time()
out = run_calibration(cfg, runner=runner, data=data)
cal_seconds = time.time() - t0
fz = out["frozen"]
print("calibration done", round(cal_seconds), "s;", len(fz["proposed_schedules"]), "proposed;",
      [(s["pattern"], s["budget"], s["floor"], s["status"]) for s in fz["selections"] if s["status"] != "selected"], flush=True)

held = data["validation"]
runner.prepare(held)
rows, per_image = [], {}
entries = [dict(e, family=e["family"]) for e in fz["proposed_schedules"]] + fz["baseline_schedules"]
cache = {}
for e in entries:
    if e.get("alias_of"):
        continue
    sched = {int(k): int(v) for k, v in e["schedule"].items()}
    key = (tuple(sorted(sched.items())))
    if key not in cache:
        cache[key] = runner.run(sched, held)
    r = cache[key]
    per_image[e["name"]] = r["pred"].astype(np.int16)
    rows.append({"schedule": e["name"], "family": e["family"], "floor": e.get("floor"), "structure": e["structure"], "budget_target": e["budget_target"],
                 "layers": json.dumps(e["layers"]), "vector_by_layer": json.dumps(dict(sorted(sched.items()))),
                 "flops_pct": round(100 * e["normalized_flops"], 4), "agree_count": int(r["agree"].sum()), "top1_count": int(r["correct"].sum()),
                 "n": len(held), "alias_of": None})
for e in entries:
    if e.get("alias_of"):
        src = next(r for r in rows if r["schedule"] == e["alias_of"])
        rows.append({**src, "schedule": e["name"], "family": e["family"], "floor": e.get("floor"), "alias_of": e["alias_of"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "heldout_dev_results.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "heldout_dev_predictions.npz"), ref=runner.baseline_pred(held), labels=held.labels.numpy(), **per_image)
json.dump({"native_caps_fraction": budgets, "cal_seconds": cal_seconds, "heldout_seconds": time.time() - t0 - cal_seconds, "n_calibration": 500,
           "n_heldout": len(held), "frozen_sha256": open(os.path.join(OUT, "frozen.sha256")).read().strip(),
           "calibration_ids_sha256": summ["calibration_ids_sha256"], "threads": torch.get_num_threads(),
           "note": "selection used the 500-image calibration subset only; held-out 1,490 images used for development reporting"},
          open(os.path.join(OUT, "run_meta.json"), "w"), indent=1)
print("DONE", round(time.time() - t0), "s")
