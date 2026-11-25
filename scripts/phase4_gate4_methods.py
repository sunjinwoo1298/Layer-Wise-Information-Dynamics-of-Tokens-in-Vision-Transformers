#!/usr/bin/env python3
"""
Phase 4 Gate 4 (P4-DEV ONLY): evaluate EVERY feasible ToMe / EViT-style candidate and select per target with the frozen deterministic rule.
Targets: primary 40/50/60/70 % and the supplementary native DiffRate executed FLOPs (targets 0.6-0.9). P4-CONFIRM is never loaded.
Outputs (results/phase4_adaptive_method_benchmark/gate4/): methods_candidates.csv, methods_selection.json, methods_candidate_predictions.npz.
Feasibility uses REALIZED token counts from the engine (identical to the official ToMe counts for the whole grid; see parity).
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.phase4.data import load_split  # noqa: E402
from src.phase4.engine import flops_pct, run_compressed  # noqa: E402
from src.phase4.evit_style import EViTStyle  # noqa: E402
from src.phase4.selection import kappa_grid, select, tome_grid  # noqa: E402
from src.phase4.tome import ToMe  # noqa: E402

P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
G4 = os.path.join(P4, "gate4")
os.makedirs(G4, exist_ok=True)
torch.set_num_threads(int(os.environ.get("P4_THREADS", "4")))
TOL = 1.0
native = json.load(open(os.path.join(P4, "diffrate_dev_validation.json")))["targets"]
TARGETS = [("primary", float(b), f"B{b}") for b in (40, 50, 60, 70)] + \
          [("native", native[t]["executed_analytic_flops_pct"], f"DR{t}") for t in ("0.6", "0.7", "0.8", "0.9")]

d = load_split("dev")
X, y = d["images"], d["labels"].numpy()
n = len(X)
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
with torch.no_grad():
    ref = torch.cat([model(X[i:i + 25]) for i in range(0, n, 25)]).argmax(1).numpy()     # unpruned reference (timm native forward)


def make(method, cfg):
    return ToMe(cfg[0] if cfg[1] == 0 else (cfg[0], float(cfg[1]))) if method == "tome" else EViTStyle(cfg)


def realized(method, cfg):
    _, info = run_compressed(model, X[:1], make(method, cfg))
    return flops_pct(info["entering"], info["leaving"]), info["leaving"]


grids = {"tome": [(r, float(i)) for (r, i) in tome_grid()], "evit": kappa_grid()}
cand = {}
for m, g in grids.items():
    for gi, cfg in enumerate(g):
        fp, leaving = realized(m, cfg)
        cand[(m, cfg)] = {"method": m, "config": cfg, "gen_index": gi, "flops_pct": fp, "final_tokens": leaving[-1], "leaving": leaving}

feas = {}
for track, tgt, tag in TARGETS:
    for m in grids:
        feas[(m, tag)] = [k for k, v in cand.items() if k[0] == m and abs(round(v["flops_pct"], 6) - tgt) <= TOL + 1e-9]
need = sorted({k for ks in feas.values() for k in ks}, key=lambda k: (k[0], cand[k]["gen_index"]))
print("targets", [(t, round(v, 4)) for _, v, t in TARGETS])
print("feasible counts", {f"{m}:{t}": len(v) for (m, t), v in feas.items()}, "| unique candidates to evaluate:", len(need), flush=True)

ck = os.path.join(G4, "_methods_progress.pkl")
done = pickle.load(open(ck, "rb")) if os.path.isfile(ck) else {}
t_all = time.time()
for j, k in enumerate(need):
    if k in done:
        continue
    t0 = time.time()
    preds, aux_f, aux_s = [], 0.0, 0.0
    ev = None
    for i in range(0, n, 25):
        comp = make(*k)
        lg, info = run_compressed(model, X[i:i + 25], comp)
        preds.append(lg.argmax(1).numpy())
        aux_f += info["aux_flops"]; aux_s += info["aux_seconds"]
        ev = ev or info["events"]
    pr = np.concatenate(preds)
    done[k] = {"pred": pr.astype(np.int16), "agree_count": int((pr == ref).sum()), "top1_count": int((pr == y).sum()),
               "aux_flops_per_image": aux_f / n, "aux_seconds_total": aux_s, "seconds": time.time() - t0}
    pickle.dump(done, open(ck, "wb"))
    print(f"[{j + 1}/{len(need)}] {k} flops={cand[k]['flops_pct']:.3f} agree={done[k]['agree_count']} top1={done[k]['top1_count']} ({time.time() - t0:.0f}s)", flush=True)

rows, sel_out = [], {}
for track, tgt, tag in TARGETS:
    for m in ("tome", "evit"):
        c_list = []
        for k in feas[(m, tag)]:
            c = cand[k]
            c_list.append({"config": k[1] if m == "tome" else None, "kappa": k[1] if m == "evit" else None, "flops_pct": c["flops_pct"],
                           "agree_count": done[k]["agree_count"], "gen_index": c["gen_index"], "top1_count": done[k]["top1_count"]})
        s = select(c_list, n, tgt, "tome" if m == "tome" else "evit") if c_list else {"chosen": None, "tie_set": [], "n_candidates": 0}
        best = s.get("best_count")
        for c in c_list:
            rows.append({"track": track, "target_tag": tag, "target_flops_pct": round(tgt, 4), "method": m, "config": json.dumps(c["config"] if m == "tome" else c["kappa"]),
                         "realized_flops_pct": round(c["flops_pct"], 6), "abs_flops_gap_pp": round(abs(round(c["flops_pct"], 6) - tgt), 6),
                         "agree_count": c["agree_count"], "agreement_pct": round(100 * c["agree_count"] / n, 3), "top1_count": c["top1_count"],
                         "top1_pct": round(100 * c["top1_count"] / n, 3), "in_tie_set": any(t is c for t in s["tie_set"]),
                         "selected": s["chosen"] is c, "n": n})
        sel_out[f"{tag}|{m}"] = {"track": track, "target_flops_pct": tgt, "method": m, "n_feasible_candidates": len(c_list),
                                 "best_agree_count": best, "n": n,
                                 "tie_set": [{"config": t["config"] if m == "tome" else t["kappa"], "agree_count": t["agree_count"],
                                              "flops_pct": t["flops_pct"], "abs_gap": abs(round(t["flops_pct"], 6) - tgt)} for t in s["tie_set"]],
                                 "chosen": None if s["chosen"] is None else {
                                     "config": s["chosen"]["config"] if m == "tome" else s["chosen"]["kappa"], "agree_count": s["chosen"]["agree_count"],
                                     "top1_count": s["chosen"]["top1_count"], "realized_flops_pct": s["chosen"]["flops_pct"],
                                     "final_tokens": cand[(m, s["chosen"]["config"] if m == "tome" else s["chosen"]["kappa"])]["final_tokens"],
                                     "leaving_per_block": cand[(m, s["chosen"]["config"] if m == "tome" else s["chosen"]["kappa"])]["leaving"]}}
pd.DataFrame(rows).to_csv(os.path.join(G4, "methods_candidates.csv"), index=False)
json.dump(sel_out, open(os.path.join(G4, "methods_selection.json"), "w"), indent=1, default=str)
np.savez_compressed(os.path.join(G4, "methods_candidate_predictions.npz"), ref=ref, labels=y,
                    **{f"{k[0]}__{json.dumps(k[1])}": v["pred"] for k, v in done.items()})
json.dump({"n": n, "elapsed_seconds": round(time.time() - t_all, 1), "unique_candidates_evaluated": len(need),
           "threads": torch.get_num_threads()}, open(os.path.join(G4, "methods_run_meta.json"), "w"), indent=1)
print("DONE", round(time.time() - t_all, 1), "s")
