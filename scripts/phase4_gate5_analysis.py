#!/usr/bin/env python3
"""
Phase 4 Gate 5 ANALYSIS (inference-free): turns the saved per-image arrays of the Gate 5 runner into the protocol-fixed metrics, comparisons and tables.

    python scripts/phase4_gate5_analysis.py --split dev     --input-dir <dir with the runner's .npz files> --out-dir <dir>
    python scripts/phase4_gate5_analysis.py --split confirm [--input-dir ...] [--out-dir ...]   (verifies analysis_freeze.json first)

Everything below is fixed BEFORE any confirmatory output exists (benchmark_protocol.md sections 7-8, protocol_amendments.md); nothing is configurable from data.
Metrics : agreement with the unpruned prediction, top-1, top-5 (counts / n, Wilson 95 % CI), analytic FLOPs from the frozen files.
Paired  : diff = method - comparator (positive favours the method); paired bootstrap 95 % percentile CI (10,000 resamples of images, seed 42, re-seeded
          per comparison); exact two-sided McNemar (binomial on discordant pairs); Holm over each stated family (alpha 0.05 on adjusted p).
Families: PRIMARY agreement (16) and PRIMARY top-1 (16, separate Holm): {ToMe, EViT-style} vs geometric x {40,50,60,70} x {2-6-10, 2-8-10}.
          REPLICATION (own Holm; agreement family of 24 = 4 budgets x 2 patterns x floors {16,24,32}: proposed vs geometric; top-1 reported as its own 24-family).
          SECONDARY (Holm per method x comparator-type x metric): {ToMe, EViT-style} vs proposed(f16, per pattern), vs single-stage, vs constrained (f16, feasible cells).
          EXPLORATORY: vs unconstrained equal-drop (Holm per method x metric); supplementary native-FLOPs track: DiffRate 0.6-0.9 vs matched geometric / constrained(f16) /
          single-stage / proposed(f16) / ToMe / EViT-style (one Holm family per metric, within the track only). Per-cell results always shown; pooled means descriptive only.
A cell exists only if the frozen schedule files define it (e.g. infeasible constrained cells are 'not_defined'); a defined cell with a missing array is an ERROR.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
BUDGETS = (40, 50, 60, 70)
PATTERNS = ("2-6-10", "2-8-10")
FLOORS = (16, 24, 32)
NATIVE = {"0.6": 52, "0.7": 58, "0.8": 64, "0.9": 72}
Z = 1.959963984540054
ALPHA, NBOOT, SEED = 0.05, 10000, 42


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def mcnemar_exact(b, c):
    """two-sided exact McNemar on discordant counts b (method-only correct) and c (comparator-only correct)."""
    m = b + c
    if m == 0:
        return 1.0
    return float(min(1.0, sps.binomtest(min(b, c), m, 0.5, alternative="two-sided").pvalue))


def paired_bootstrap(d, nboot=NBOOT, seed=SEED, chunk=500):
    rng = np.random.default_rng(seed)
    n = len(d)
    out = np.empty(nboot)
    df = d.astype(np.float64)
    for s in range(0, nboot, chunk):
        e = min(s + chunk, nboot)
        out[s:e] = df[rng.integers(0, n, size=(e - s, n))].mean(1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, split, indir):
        self.split, self.indir = split, indir
        man = pd.read_csv(os.path.join(P4, "splits", f"p4_{split}_manifest.csv"))
        if man.image_id.duplicated().any():
            raise SystemExit("duplicate image ids in manifest")
        ref = np.load(self._f("reference_unpruned"), allow_pickle=False)
        self.n = len(man)
        if [str(x) for x in ref["ids"]] != man.image_id.tolist():
            raise SystemExit("reference ids do not match manifest (order/ids)")
        if not np.array_equal(ref["labels"], man.label.to_numpy()):
            raise SystemExit("reference labels do not match manifest labels")
        self.y, self.ref = ref["labels"].astype(np.int64), ref["pred"].astype(np.int64)
        if len(self.y) != self.n or len(self.ref) != self.n:
            raise SystemExit("reference length != manifest")
        self.cache = {}
        self.checked = ["reference_unpruned"]

    def _f(self, name):
        p = os.path.join(self.indir, name + ".npz")
        if not os.path.isfile(p):
            raise SystemExit(f"MISSING required array file: {p}")
        return p

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        z = np.load(self._f(name), allow_pickle=False)
        pred, top5 = z["pred"].astype(np.int64), z["top5"].astype(np.int64)
        if pred.shape != (self.n,) or top5.shape != (self.n, 5):
            raise SystemExit(f"{name}: wrong shape {pred.shape}/{top5.shape} (n={self.n})")
        if pred.min() < 0 or pred.max() > 999 or top5.min() < 0 or top5.max() > 999:
            raise SystemExit(f"{name}: class index outside 0..999")
        if not np.array_equal(top5[:, 0], pred):
            raise SystemExit(f"{name}: top5[:,0] != pred")
        r = {"agree": (pred == self.ref).astype(np.int8), "top1": (pred == self.y).astype(np.int8), "top5": (top5 == self.y[:, None]).any(1).astype(np.int8)}
        self.cache[name] = r
        self.checked.append(name)
        return r


def load_catalogue():
    """Everything the analysis may refer to, from the frozen files only."""
    f3 = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
    fs = json.load(open(os.path.join(P4, "supplementary_phase3_reselection", "frozen_schedules.json")))
    fc = json.load(open(os.path.join(P4, "frozen_method_configs.json")))
    cat = {}
    for prefix, fz in (("p3", f3), ("p3sup", fs)):
        for e in fz["proposed_schedules"] + fz["baseline_schedules"]:
            src = e.get("alias_of") or e["name"]
            cat[f"{prefix}__{e['name']}"] = {"file": f"{prefix}__{src}", "family": e["family"], "flops_pct": round(100 * e["normalized_flops"], 4), "kind": prefix}
    for key, v in fc["selected"].items():
        tag, m = key.split("|")
        cat[f"{m}_{tag}"] = {"file": f"{m}_{tag}", "family": m, "flops_pct": v["realized_flops_pct"], "kind": "phase4", "config": json.dumps(v["config"]),
                             "final_tokens": v["leaving_tokens_per_block"][-1]}
    for t, v in fc["diffrate_released"].items():
        cat[f"diffrate_{t}"] = {"file": f"diffrate_{t}", "family": "diffrate", "flops_pct": v["executed_analytic_flops_pct"], "kind": "phase4", "config": f"released {t}",
                                "final_tokens": v["executed_leaving"][-1]}
    return cat


def geo(b, pat, pfx="p3"): return f"{pfx}__L2_B{b}_geometric_three_stage_{pat}"
def prop(b, pat, f=16, pfx="p3"): return f"{pfx}__L2_B{b}_proposed_f{f}_{pat}"
def cons(b, pat, f=16, pfx="p3"): return f"{pfx}__L2_B{b}_constrained_f{f}_three_stage_{pat}"
def eqd(b, pat, pfx="p3"): return f"{pfx}__L2_B{b}_equal_drop_three_stage_{pat}"
def one(b, pfx="p3"): return f"{pfx}__L2_B{b}_one_stage"


# ----------------------------------------------------------------------------- comparisons
def compare(D, cat, a, b, metric):
    A, B = D.get(cat[a]["file"])[metric], D.get(cat[b]["file"])[metric]
    n = D.n
    ka, kb = int(A.sum()), int(B.sum())
    bb, cc = int(((A == 1) & (B == 0)).sum()), int(((A == 0) & (B == 1)).sum())
    d = A.astype(np.int64) - B.astype(np.int64)
    lo, hi = paired_bootstrap(d)
    return {"method": a, "comparator": b, "metric": metric, "n": n, "method_count": ka, "comparator_count": kb, "method_pct": 100 * ka / n, "comparator_pct": 100 * kb / n,
            "diff_pp": 100 * (ka - kb) / n, "boot_ci_lo_pp": 100 * lo, "boot_ci_hi_pp": 100 * hi, "method_only_correct": bb, "comparator_only_correct": cc,
            "mcnemar_p": mcnemar_exact(bb, cc), "method_flops_pct": cat[a]["flops_pct"], "comparator_flops_pct": cat[b]["flops_pct"]}


def run_family(D, cat, family, pairs, metric, holm_group=None):
    rows = []
    for tag, a, b in pairs:
        if a not in cat or b not in cat:
            rows.append({"family": family, "cell": tag, "metric": metric, "status": "not_defined", "method": a, "comparator": b})
            continue
        r = compare(D, cat, a, b, metric)
        r.update({"family": family, "cell": tag, "status": "ok"})
        rows.append(r)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        adj = holm([r["mcnemar_p"] for r in ok])
        for r, ap in zip(ok, adj):
            r["holm_family_size"] = len(ok)
            r["holm_adj_p"] = float(ap)
            r["significant_holm"] = bool(ap < ALPHA)
            r["direction"] = "method>comparator" if r["diff_pp"] > 0 else ("method<comparator" if r["diff_pp"] < 0 else "equal")
    return rows


def build_all(D, cat):
    fam = {}
    P4M = ("tome", "evit")
    prim = [(f"{m}_B{b}__vs_G_{p}", f"{m}_B{b}", geo(b, p)) for m in P4M for b in BUDGETS for p in PATTERNS]
    for met in ("agree", "top1"):
        fam[f"primary_{met}"] = run_family(D, cat, f"PRIMARY_{met}", prim, met)
    rep = [(f"B{b}_{p}_f{f}", prop(b, p, f), geo(b, p)) for b in BUDGETS for p in PATTERNS for f in FLOORS]
    for met in ("agree", "top1"):
        fam[f"replication_{met}"] = run_family(D, cat, f"REPLICATION_{met}", rep, met)
    for m in P4M:
        for met in ("agree", "top1"):
            fam[f"secondary_{m}_vs_P_{met}"] = run_family(D, cat, f"SECONDARY_{m}_vs_P_{met}", [(f"{m}_B{b}__vs_P16_{p}", f"{m}_B{b}", prop(b, p)) for b in BUDGETS for p in PATTERNS], met)
            fam[f"secondary_{m}_vs_S_{met}"] = run_family(D, cat, f"SECONDARY_{m}_vs_S_{met}", [(f"{m}_B{b}__vs_S", f"{m}_B{b}", one(b)) for b in BUDGETS], met)
            fam[f"secondary_{m}_vs_C_{met}"] = run_family(D, cat, f"SECONDARY_{m}_vs_C_{met}", [(f"{m}_B{b}__vs_C16_{p}", f"{m}_B{b}", cons(b, p)) for b in BUDGETS for p in PATTERNS], met)
            fam[f"exploratory_{m}_vs_U_{met}"] = run_family(D, cat, f"EXPLORATORY_{m}_vs_U_{met}", [(f"{m}_B{b}__vs_U_{p}", f"{m}_B{b}", eqd(b, p)) for b in BUDGETS for p in PATTERNS], met)
    for met in ("agree", "top1"):                                          # supplementary native-FLOPs track: ONE Holm family per metric, within the track only
        pairs = []
        for t, nb in NATIVE.items():
            M3 = f"diffrate_{t}"
            for p in PATTERNS:
                pairs += [(f"DR{t}__vs_G_{p}", M3, geo(nb, p, "p3sup")), (f"DR{t}__vs_C16_{p}", M3, cons(nb, p, 16, "p3sup")), (f"DR{t}__vs_P16_{p}", M3, prop(nb, p, 16, "p3sup"))]
            pairs += [(f"DR{t}__vs_S", M3, one(nb, "p3sup")), (f"DR{t}__vs_ToMe", M3, f"tome_DR{t}"), (f"DR{t}__vs_EViT", M3, f"evit_DR{t}")]
        rows = run_family(D, cat, f"EXPLORATORY_native_track_{met}", pairs, met)
        fam[f"exploratory_native_track_{met}"] = rows
    return fam


def metrics_table(D, cat):
    rows = []
    for name in sorted(cat):
        r = D.get(cat[name]["file"])                                       # strict: every catalogued entity must have its array file
        row = {"entity": name, "family": cat[name]["family"], "flops_pct": cat[name]["flops_pct"], "n": D.n, "config": cat[name].get("config", "")}
        for met, lab in (("agree", "agreement"), ("top1", "top1"), ("top5", "top5")):
            k = int(r[met].sum()); lo, hi = wilson(k, D.n)
            row.update({f"{lab}_count": k, f"{lab}_pct": 100 * k / D.n, f"{lab}_ci_lo": 100 * lo, f"{lab}_ci_hi": 100 * hi})
        rows.append(row)
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--input-dir")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    me = os.path.abspath(__file__)
    if a.split == "confirm":
        fr = json.load(open(os.path.join(P4, "phase4_analysis_freeze.json")))
        if sha(me) != fr["analysis_script_sha256"]:
            sys.exit("analysis script differs from the frozen analysis hash; refusing")
        for rel, h in fr["dependencies_sha256"].items():
            if sha(os.path.join(ROOT, rel)) != h:
                sys.exit("frozen analysis dependency changed: " + rel)
    indir = a.input_dir or (os.path.join(P4, "per_image_results") if a.split == "confirm" else os.path.join(P4, "analysis_dev_validation", "inputs"))
    out = a.out_dir or (os.path.join(P4, "confirmatory_analysis") if a.split == "confirm" else os.path.join(P4, "analysis_dev_validation", "outputs"))
    os.makedirs(out, exist_ok=True)
    cat = load_catalogue()
    D = Data(a.split, indir)
    fam = build_all(D, cat)
    mt = metrics_table(D, cat)
    mt.round(6).to_csv(os.path.join(out, "metrics_all.csv"), index=False, lineterminator="\n")
    summary = []
    for k, rows in fam.items():
        df = pd.DataFrame(rows)
        df.round(6).to_csv(os.path.join(out, f"comparisons_{k}.csv"), index=False, lineterminator="\n")
        ok = df[df.status == "ok"]
        summary.append({"family": k, "cells_defined": len(ok), "cells_not_defined": int((df.status != "ok").sum()), "holm_significant": int(ok.significant_holm.sum()) if len(ok) else 0,
                        "mean_diff_pp_descriptive": float(ok.diff_pp.mean()) if len(ok) else float("nan"), "better_by_point_estimate": int((ok.diff_pp > 0).sum()) if len(ok) else 0,
                        "worse_by_point_estimate": int((ok.diff_pp < 0).sum()) if len(ok) else 0})
    sm = pd.DataFrame(summary)
    sm.round(6).to_csv(os.path.join(out, "family_summary.csv"), index=False, lineterminator="\n")
    L = [f"# Phase 4 analysis — split `{a.split}` (n={D.n})", "",
         "Diff = method − comparator in percentage points (positive favours the method). Holm adjusts the exact-McNemar p within each stated family; no result is called superior from a point estimate alone. "
         "Pooled means are descriptive only." + (" **P4-DEV output: selection/validation data, not confirmatory evidence.**" if a.split == "dev" else ""), "", "## Family summary", "",
         sm.round(4).to_markdown(index=False), "", "## Primary comparisons", ""]
    for k in ("primary_agree", "primary_top1"):
        df = pd.DataFrame(fam[k])
        L += [f"### {k}", "", df[["cell", "method_count", "comparator_count", "diff_pp", "boot_ci_lo_pp", "boot_ci_hi_pp", "mcnemar_p", "holm_adj_p", "significant_holm"]].round(4).to_markdown(index=False), ""]
    for k in ("replication_agree", "replication_top1"):
        df = pd.DataFrame(fam[k])
        L += [f"### {k} (limitation: 995-class, 2,985/1,990-image subset; not an exact replication of the Phase 3 evaluation)", "",
              df[["cell", "method_count", "comparator_count", "diff_pp", "boot_ci_lo_pp", "boot_ci_hi_pp", "mcnemar_p", "holm_adj_p", "significant_holm"]].round(4).to_markdown(index=False), ""]
    L += ["## Metrics (all entities)", "", mt[["entity", "flops_pct", "agreement_count", "agreement_pct", "agreement_ci_lo", "agreement_ci_hi", "top1_count", "top1_pct", "top5_pct"]].round(3).to_markdown(index=False), ""]
    open(os.path.join(out, "analysis_report.md"), "wb").write(("\n".join(L) + "\n").encode())
    import scipy
    outs = {f: sha(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f != "analysis_manifest.json"}
    man = {"split": a.split, "n": D.n, "analysis_script_sha256": sha(me), "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
           "bootstrap": {"n_resamples": NBOOT, "seed": SEED, "per_comparison_reseed": True}, "alpha": ALPHA, "arrays_validated": len(D.checked),
           "input_sha256": {n: sha(os.path.join(indir, n + ".npz")) for n in sorted(set(D.checked))}, "output_sha256": outs}
    json.dump(man, open(os.path.join(out, "analysis_manifest.json"), "w"), indent=1)
    print(sm.round(3).to_string())


if __name__ == "__main__":
    main()
