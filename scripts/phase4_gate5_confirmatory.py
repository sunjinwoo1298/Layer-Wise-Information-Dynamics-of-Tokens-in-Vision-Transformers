#!/usr/bin/env python3
"""
Phase 4 Gate 5: the SINGLE confirmatory inference run on P4-CONFIRM (2,985 images) with the frozen configurations. INFERENCE ONLY: it writes per-image
predictions / top-5 / float16 logits; statistics are computed afterwards by a separate analysis step from the saved arrays with the protocol-fixed families.

    set PHASE4_CONFIRMATORY=1 && conda run --no-capture-output -n minor python -u scripts/phase4_gate5_confirmatory.py

--dry-run : exercises the identical code path on the first 50 P4-DEV images (never P4-CONFIRM), writes under gate5_dry_run/, no marker.

Safeguards: loader guard (env var + verified frozen_method_configs.json bytes and dependency hashes + protocol freeze + no marker); the marker
`confirmatory_run_done.marker` is written BEFORE inference ("started") and finalized "completed"/"failed", so a failed run cannot be silently re-run.
No tuning, substitution, or filtering: configurations come only from frozen_method_configs.json and the frozen Phase 3 schedule files.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
OUT = os.path.join(P4, "gate5_dry_run" if args.dry_run else "per_image_results")
MARK = os.path.join(P4, "confirmatory_run_done.marker")
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
if args.dry_run:
    d = load_split("dev")
    d = {k: (v[:50] if hasattr(v, "__len__") else v) for k, v in d.items()}
else:
    if os.path.exists(MARK):
        sys.exit("confirmatory marker exists; refusing a second run")
    d = load_split("confirm")                                       # guarded; verifies frozen config bytes + dependency hashes + protocol freeze
    json.dump({"status": "started", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "frozen_method_configs_sha256": open(os.path.join(P4, "frozen_method_configs.sha256")).read().strip()},
              open(MARK, "w"))
X, y = d["images"], d["labels"].numpy()
n = len(X)
BS = 25
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
t_all = time.time()
log = []


def save(name, logits, extra=None):
    lg = torch.cat(logits)
    pred = lg.argmax(1).numpy().astype(np.int16)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), pred=pred, top5=lg.topk(5, 1).indices.numpy().astype(np.int16), logits=lg.half().numpy(), **(extra or {}))
    log.append({"name": name, "n": n, "elapsed_s": round(time.time() - t_all, 1)})
    print(f"{name} n={n} agree_with_ref=NA elapsed={time.time() - t_all:.0f}s", flush=True)


def run_phase4(name, make):
    lgs, aux_f, aux_s = [], 0.0, 0.0
    with torch.no_grad():
        for i in range(0, n, BS):
            lg, info = run_compressed(model, X[i:i + BS], make())
            lgs.append(lg); aux_f += info["aux_flops"]; aux_s += info["aux_seconds"]
    save(name, lgs, {"aux_flops_per_image": aux_f / n, "aux_seconds_total": aux_s})


status = "failed"
try:
    with torch.no_grad():
        ref = [model(X[i:i + BS]) for i in range(0, n, BS)]
    save("reference_unpruned", ref, {"labels": y, "ids": np.asarray(d["ids"])})
    for key, v in fcfg["selected"].items():                            # ToMe / EViT-style at primary + native targets (frozen)
        tag, m = key.split("|")
        c = v["config"]
        make = (lambda c=c: ToMe(c[0] if c[1] == 0 else (c[0], float(c[1])))) if m == "tome" else (lambda c=c: EViTStyle(c))
        run_phase4(f"{m}_{tag}", make)
    rates = os.path.join(P4, "diffrate_deit_tiny_rates_as_published.json")
    for t in fcfg["diffrate_released"]:                                # DiffRate 0.6-0.9 only (1.0 is not applicable as released)
        run_phase4(f"diffrate_{t}", lambda t=t: DiffRateReleased.from_published(rates, t))
    runner = Runner(AllocConfig(**{**json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))["hyperparameters"],
                                   "output_dir": OUT, "device": None, "max_cal": None, "max_val": None, "max_conf": None}))
    w = runner.wrapper
    seen = {}
    for src_name, path in (("p3", os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")),
                           ("p3sup", os.path.join(P4, "supplementary_phase3_reselection", "frozen_schedules.json"))):
        fz = json.load(open(path))
        for e in fz["proposed_schedules"] + fz["baseline_schedules"]:
            sched = {int(k): int(v) for k, v in e["schedule"].items()}
            key = (src_name, tuple(sorted(sched.items())))
            fn = f"{src_name}__{e['name']}"
            if e.get("alias_of"):
                continue
            if key in seen:                                            # identical vector under another name: reuse the saved array (deterministic scorer)
                np.savez_compressed(os.path.join(OUT, fn + ".npz"), **dict(np.load(os.path.join(OUT, seen[key] + ".npz"))))
                continue
            pr = ProgressivePruner(w, sched, scorer=json.load(open(path))["hyperparameters"]["scorer"], seed=json.load(open(path))["hyperparameters"]["seed"])
            lgs = []
            with torch.no_grad():
                for i in range(0, n, BS):
                    lgs.append(pr.forward(X[i:i + BS].to(runner.dev)).logits.cpu())
            save(fn, lgs, {"vector": json.dumps(dict(sorted(sched.items())))})
            seen[key] = fn
    status = "completed"
except BaseException:
    tb = traceback.format_exc()
    open(os.path.join(OUT, "FAILURE_RECORD.txt"), "w").write(tb)
    print(tb, flush=True)
    raise
finally:
    meta = {"status": status, "dry_run": args.dry_run, "n": n, "elapsed_seconds": round(time.time() - t_all, 1), "threads": torch.get_num_threads(), "torch": torch.__version__,
            "timm": timm.__version__, "python": platform.python_version(), "platform": platform.platform(), "log": log,
            "frozen_method_configs_sha256": open(os.path.join(P4, "frozen_method_configs.sha256")).read().strip()}
    json.dump(meta, open(os.path.join(P4, "confirmatory_environment.json") if not args.dry_run else os.path.join(OUT, "environment.json"), "w"), indent=1)
    if not args.dry_run:
        json.dump({"status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(MARK, "w"))
print("DONE", status)
