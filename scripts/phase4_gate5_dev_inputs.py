#!/usr/bin/env python3
"""
Analysis-validation helper (P4-DEV ONLY): produce, on ALL 1,990 P4-DEV images, per-image arrays in exactly the format and naming that the frozen
Gate 5 runner writes (reference_unpruned.npz, <method>_<tag>.npz, diffrate_<t>.npz, p3__<name>.npz, p3sup__<name>.npz), so the analysis script can be
developed and validated end to end without touching P4-CONFIRM. The logic mirrors scripts/phase4_gate5_confirmatory.py (not modified). Never loads P4-CONFIRM.
"""
import json
import os
import sys
import time

import numpy as np
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
OUT = os.path.join(P4, "analysis_dev_validation", "inputs")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
from src.compression.progressive import ProgressivePruner  # noqa: E402
from src.experiments.budget_aware_allocation import AllocConfig, Runner  # noqa: E402
from src.phase4.data import load_split  # noqa: E402
from src.phase4.diffrate import DiffRateReleased  # noqa: E402
from src.phase4.engine import run_compressed  # noqa: E402
from src.phase4.evit_style import EViTStyle  # noqa: E402
from src.phase4.tome import ToMe  # noqa: E402

fcfg = json.load(open(os.path.join(P4, "frozen_method_configs.json")))
d = load_split("dev")
X, y = d["images"], d["labels"].numpy()
n, BS = len(X), 25
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
t0 = time.time()


def save(name, logits, extra=None):
    if os.path.exists(os.path.join(OUT, f"{name}.npz")):
        return
    lg = torch.cat(logits)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), pred=lg.argmax(1).numpy().astype(np.int16), top5=lg.topk(5, 1).indices.numpy().astype(np.int16),
                        logits=lg.half().numpy(), **(extra or {}))
    print(name, round(time.time() - t0), "s", flush=True)


def run_phase4(name, make):
    if os.path.exists(os.path.join(OUT, f"{name}.npz")):
        return
    lgs, af, asec = [], 0.0, 0.0
    with torch.no_grad():
        for i in range(0, n, BS):
            lg, info = run_compressed(model, X[i:i + BS], make())
            lgs.append(lg); af += info["aux_flops"]; asec += info["aux_seconds"]
    save(name, lgs, {"aux_flops_per_image": af / n, "aux_seconds_total": asec})


with torch.no_grad():
    ref = [model(X[i:i + BS]) for i in range(0, n, BS)]
save("reference_unpruned", ref, {"labels": y, "ids": np.asarray(d["ids"])})
for key, v in fcfg["selected"].items():
    tag, m = key.split("|")
    c = v["config"]
    make = (lambda c=c: ToMe(c[0] if c[1] == 0 else (c[0], float(c[1])))) if m == "tome" else (lambda c=c: EViTStyle(c))
    run_phase4(f"{m}_{tag}", make)
rates = os.path.join(P4, "diffrate_deit_tiny_rates_as_published.json")
for t in fcfg["diffrate_released"]:
    run_phase4(f"diffrate_{t}", lambda t=t: DiffRateReleased.from_published(rates, t))
f3 = os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")
hp = json.load(open(f3))["hyperparameters"]
runner = Runner(AllocConfig(**{**hp, "output_dir": OUT, "device": None, "max_cal": None, "max_val": None, "max_conf": None}))
seen = {}
for src_name, path in (("p3", f3), ("p3sup", os.path.join(P4, "supplementary_phase3_reselection", "frozen_schedules.json"))):
    fz = json.load(open(path))
    for e in fz["proposed_schedules"] + fz["baseline_schedules"]:
        if e.get("alias_of"):
            continue
        sched = {int(k): int(v) for k, v in e["schedule"].items()}
        key, fn = (src_name, tuple(sorted(sched.items()))), f"{src_name}__{e['name']}"
        if key in seen:
            np.savez_compressed(os.path.join(OUT, fn + ".npz"), **dict(np.load(os.path.join(OUT, seen[key] + ".npz"))))
            continue
        if os.path.exists(os.path.join(OUT, fn + ".npz")):
            seen[key] = fn
            continue
        pr = ProgressivePruner(runner.wrapper, sched, scorer=hp["scorer"], seed=hp["seed"])
        lgs = []
        with torch.no_grad():
            for i in range(0, n, BS):
                lgs.append(pr.forward(X[i:i + BS].to(runner.dev)).logits.cpu())
        save(fn, lgs, {"vector": json.dumps(dict(sorted(sched.items())))})
        seen[key] = fn
print("DONE", round(time.time() - t0), "s")
