#!/usr/bin/env python3
"""
Phase 4 Gate 4 (P4-DEV ONLY, SECONDARY): CPU latency and auxiliary time. Batch 25, 1 warm-up + 5 timed runs on each of 3 fixed P4-DEV batches,
median / IQR of per-image ms. Phase 4 engine (ToMe / EViT-style / DiffRate, plus its own unpruned reference) and the Phase 1-3 ProgressivePruner
(Phase 3 schedules, plus its own unpruned reference) are timed separately; ratios to each engine's own reference are the comparable quantity.
Equal analytic FLOPs are not equal latency. P4-CONFIRM never loaded. Run alone (no other CPU jobs).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.experiments.budget_aware_allocation import AllocConfig, Runner  # noqa: E402
from src.compression.progressive import ProgressivePruner  # noqa: E402
from src.phase4.data import load_split  # noqa: E402
from src.phase4.diffrate import DiffRateReleased  # noqa: E402
from src.phase4.engine import run_compressed  # noqa: E402
from src.phase4.evit_style import EViTStyle  # noqa: E402
from src.phase4.tome import ToMe  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
G4 = os.path.join(P4, "gate4")
torch.set_num_threads(4)
d = load_split("dev")
BATCHES = [d["images"][i * 25:(i + 1) * 25] for i in (0, 20, 40)]
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()


def timeit(fn):
    per = []
    with torch.no_grad():
        for b in BATCHES:
            fn(b)                                           # warm-up
            for _ in range(5):
                t = time.perf_counter(); fn(b); per.append((time.perf_counter() - t) / len(b) * 1000)
    q = np.percentile(per, [25, 50, 75])
    return {"median_ms_per_img": q[1], "iqr_ms": q[2] - q[0], "n_runs": len(per)}


def aux(comp_factory):
    tot = []
    with torch.no_grad():
        for b in BATCHES:
            _, info = run_compressed(model, b, comp_factory())
            tot.append(info["aux_seconds"] / len(b) * 1000)
    return float(np.median(tot))


rows = []
ref4 = timeit(lambda b: run_compressed(model, b))
rows.append({"engine": "phase4_explicit_attention", "method": "unpruned_reference", "config": "-", **ref4, "ratio_to_engine_ref": 1.0, "aux_ms_per_img": 0.0})
sel = json.load(open(os.path.join(G4, "methods_selection.json")))
for key, v in sel.items():
    tag, m = key.split("|")
    c = v["chosen"]["config"]
    mk = (lambda c=c: ToMe(c[0] if c[1] == 0 else (c[0], float(c[1])))) if m == "tome" else (lambda c=c: EViTStyle(c))
    r = timeit(lambda b, mk=mk: run_compressed(model, b, mk()))
    rows.append({"engine": "phase4_explicit_attention", "method": m, "config": f"{tag}:{c}", **r, "ratio_to_engine_ref": r["median_ms_per_img"] / ref4["median_ms_per_img"], "aux_ms_per_img": aux(mk)})
    print(rows[-1]["method"], rows[-1]["config"], round(r["median_ms_per_img"], 3), flush=True)
dv = os.path.join(P4, "diffrate_deit_tiny_rates_as_published.json")
for t in ("0.6", "0.7", "0.8", "0.9"):
    mk = lambda t=t: DiffRateReleased.from_published(dv, t)
    r = timeit(lambda b, mk=mk: run_compressed(model, b, mk()))
    rows.append({"engine": "phase4_explicit_attention", "method": "diffrate", "config": f"DR{t}", **r, "ratio_to_engine_ref": r["median_ms_per_img"] / ref4["median_ms_per_img"], "aux_ms_per_img": aux(mk)})
    print("diffrate", t, round(r["median_ms_per_img"], 3), flush=True)

fz = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
hp = dict(fz["hyperparameters"]); hp.update(output_dir=G4, device=None, max_cal=None, max_val=None, max_conf=None)
runner = Runner(AllocConfig(**hp))
w = runner.wrapper
ref3 = timeit(lambda b: w(b))
rows.append({"engine": "phase1_3_progressive_pruner", "method": "unpruned_reference", "config": "-", **ref3, "ratio_to_engine_ref": 1.0, "aux_ms_per_img": 0.0})
seen = set()
for e in fz["proposed_schedules"] + fz["baseline_schedules"]:
    if e.get("alias_of"):
        continue
    s = {int(k): int(x) for k, x in e["schedule"].items()}
    key = tuple(sorted(s.items()))
    if key in seen:
        continue
    seen.add(key)
    pr = ProgressivePruner(w, s, scorer=hp["scorer"], seed=hp["seed"])
    r = timeit(lambda b, pr=pr: pr.forward(b))
    rows.append({"engine": "phase1_3_progressive_pruner", "method": "phase3", "config": e["name"], **r, "ratio_to_engine_ref": r["median_ms_per_img"] / ref3["median_ms_per_img"], "aux_ms_per_img": float("nan")})
    print("phase3", e["name"], round(r["median_ms_per_img"], 3), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(G4, "latency_dev.csv"), index=False)
json.dump({"threads": torch.get_num_threads(), "batch": 25, "warmup": 1, "runs_per_batch": 5, "batches": 3, "torch": torch.__version__,
           "note": "secondary measurement; CPU variability applies; engines not comparable in absolute ms"}, open(os.path.join(G4, "latency_dev_meta.json"), "w"), indent=1)
print("DONE")
