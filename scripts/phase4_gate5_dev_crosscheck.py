#!/usr/bin/env python3
"""Independent P4-DEV cross-check of the analysis outputs (does NOT import the analysis script): recompute counts from raw arrays and compare with (a) the analysis metrics,
(b) the Gate 4 selection/comparator/DiffRate records produced by different code paths, (c) an independent McNemar / Holm / duplicate-id / class-alignment check."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
A = os.path.join(P4, "analysis_dev_validation")
inp, outd = os.path.join(A, "inputs"), sys.argv[1] if len(sys.argv) > 1 else os.path.join(A, "outputs_run1")
man = pd.read_csv(os.path.join(P4, "splits", "p4_dev_manifest.csv"))
res = {"checks": []}


def chk(name, ok, detail=""):
    res["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)


ref = np.load(os.path.join(inp, "reference_unpruned.npz"))
y = man.label.to_numpy()
chk("no duplicate image ids in manifest / reference", not man.image_id.duplicated().any() and len(set(ref["ids"].tolist())) == len(man))
chk("reference ids == manifest order", [str(i) for i in ref["ids"]] == man.image_id.tolist())
chk("class alignment: labels == manifest labels", np.array_equal(ref["labels"], y))
chk("class alignment: labels in 5..999 (classes 0-4 excluded)", y.min() >= 5 and y.max() <= 999)
files = sorted(f for f in os.listdir(inp) if f.endswith(".npz"))
bad = [f for f in files if np.load(os.path.join(inp, f))["pred"].shape != (len(man),)]
chk("every array file has n=1990 predictions (no missing/extra rows)", not bad, f"{len(files)} files")


def counts(fn):
    z = np.load(os.path.join(inp, fn + ".npz"))
    p = z["pred"].astype(int)
    return int((p == ref["pred"]).sum()), int((p == y).sum()), int((z["top5"] == y[:, None]).any(1).sum())


mt = pd.read_csv(os.path.join(outd, "metrics_all.csv")).set_index("entity")
# (a) analysis metrics == raw recomputation (entity -> file resolution via the frozen files)
f3 = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
fs = json.load(open(os.path.join(P4, "supplementary_phase3_reselection", "frozen_schedules.json")))
mism = 0
fmap = {}
for pre, fz in (("p3", f3), ("p3sup", fs)):
    for e in fz["proposed_schedules"] + fz["baseline_schedules"]:
        fmap[f"{pre}__{e['name']}"] = f"{pre}__{e.get('alias_of') or e['name']}"
for ent in mt.index:
    fn = fmap.get(ent, ent)
    a, t1, t5 = counts(fn)
    if (a, t1, t5) != (mt.loc[ent, "agreement_count"], mt.loc[ent, "top1_count"], round(mt.loc[ent, "top5_pct"] * len(man) / 100)):
        mism += 1
chk("analysis metrics_all counts == independent recomputation (agreement, top-1, top-5)", mism == 0, f"{len(mt)} entities, {mism} mismatches")
# (b1) Gate 4 methods selection (different code path/time) -> selected ToMe/EViT counts
sel = json.load(open(os.path.join(P4, "gate4", "methods_selection.json")))
m = [(k, v["chosen"]["agree_count"], v["chosen"]["top1_count"]) for k, v in sel.items()]
mm = [k for k, ag, t1 in m if counts("%s_%s" % (k.split("|")[1], k.split("|")[0]))[:2] != (ag, t1)]
chk("ToMe/EViT-style selected counts == Gate 4 methods_selection.json (16 cells)", not mm, str(mm))
# (b2) Phase 3 comparators regenerated in Gate 4
p3 = pd.read_csv(os.path.join(P4, "gate4", "phase3_comparators_dev.csv"))
mm = [r.schedule for r in p3.itertuples() if counts("p3__" + (r.alias_of if isinstance(r.alias_of, str) else r.schedule))[:2] != (r.agree_count, r.top1_count)]
chk("Phase 3 schedule counts == Gate 4 phase3_comparators_dev.csv", not mm, f"{len(p3)} rows; mismatches {mm}")
# (b3) DiffRate dev validation
dr = json.load(open(os.path.join(P4, "diffrate_dev_validation.json")))["targets"]
mm = [t for t in ("0.6", "0.7", "0.8", "0.9") if counts("diffrate_" + t)[:2] != (dr[t]["dev_agreement_count"], dr[t]["dev_top1_count"])]
chk("DiffRate 0.6-0.9 counts == diffrate_dev_validation.json", not mm, str(mm))
# (b4) supplementary Phase 3 re-selection: held-out rows must match the full-dev arrays restricted to held-out images
sup = pd.read_csv(os.path.join(P4, "supplementary_phase3_reselection", "heldout_dev_results.csv"))
held = pd.read_csv(os.path.join(P4, "splits", "p4_dev_heldout1490_manifest.csv")).dev_position.to_numpy()
bad = []
for r in sup.itertuples():
    z = np.load(os.path.join(inp, "p3sup__" + (r.alias_of if isinstance(r.alias_of, str) else r.schedule) + ".npz"))["pred"].astype(int)
    if (int((z[held] == ref["pred"][held]).sum()), int((z[held] == y[held]).sum())) != (r.agree_count, r.top1_count):
        bad.append(r.schedule)
chk("supplementary Phase 3 held-out counts == recomputed from full-dev arrays on the 1,490 held-out images", not bad, f"{len(sup)} rows; mismatches {bad[:3]}")
# (c) independent McNemar / Holm / bootstrap on the primary agreement family
prim = pd.read_csv(os.path.join(outd, "comparisons_primary_agree.csv"))


def mcn(b, c):
    mm_ = b + c
    return 1.0 if mm_ == 0 else min(1.0, 2 * sum(math.comb(mm_, i) for i in range(min(b, c) + 1)) / 2 ** mm_)


def fam_file(name):
    return os.path.join(inp, name + ".npz")


ok_p, ok_h, raw = True, True, []
for r in prim.itertuples():
    fa = r.method
    fb = fmap[r.comparator]
    pa, pb = np.load(fam_file(fa))["pred"].astype(int), np.load(fam_file(fb))["pred"].astype(int)
    A_, B_ = (pa == ref["pred"]), (pb == ref["pred"])
    b, c = int((A_ & ~B_).sum()), int((~A_ & B_).sum())
    p = mcn(b, c); raw.append(p)
    ok_p &= abs(p - r.mcnemar_p) < 1e-6 and (A_.sum() - B_.sum()) * 100 / len(man) - r.diff_pp < 1e-5
chk("primary agreement family: McNemar p-values and differences recomputed independently (16 cells)", ok_p)
o = sorted(range(len(raw)), key=lambda i: raw[i]); adj = [0] * len(raw); run = 0
for rank, i in enumerate(o):
    run = max(run, min(1, (len(raw) - rank) * raw[i])); adj[i] = run
chk("primary agreement family: Holm-adjusted p and significance recomputed independently", np.allclose(adj, prim.holm_adj_p.to_numpy(), atol=1e-6) and [a < 0.05 for a in adj] == prim.significant_holm.tolist())
chk("primary comparison direction: diff = method - comparator (sign agrees with counts)", all((r.method_count - r.comparator_count) * r.diff_pp >= 0 for r in prim.itertuples()))
chk("primary family size 16 and Holm size recorded 16", len(prim) == 16 and (prim.holm_family_size == 16).all())
rep = pd.read_csv(os.path.join(outd, "comparisons_replication_agree.csv"))
chk("replication family size 24", len(rep) == 24 and (rep.holm_family_size == 24).all())
b = np.load(fam_file("tome_B50"))["pred"].astype(int)
chk("bootstrap CI brackets point estimate (primary)", ((prim.boot_ci_lo_pp <= prim.diff_pp + 1e-9) & (prim.diff_pp <= prim.boot_ci_hi_pp + 1e-9)).all())
json.dump(res, open(os.path.join(A, "crosscheck_results.json"), "w"), indent=1)
print("ALL PASS" if all(c["ok"] for c in res["checks"]) else "SOME FAILED")
