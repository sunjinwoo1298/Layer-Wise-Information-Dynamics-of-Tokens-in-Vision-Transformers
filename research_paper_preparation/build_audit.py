#!/usr/bin/env python3
"""
READ-ONLY post-experiment audit (Phase 4, after Gate 5). Reads the saved confirmatory outputs and frozen inputs, writes ONLY new files under
research_paper_preparation/ (refuses to overwrite). No inference, no re-run of the official analysis, no image loading. Statistical checks use only the numbers
stored in the official CSVs (counts, discordant pairs, p-values); predictions are opened only for (i) hashing and (ii) the unpruned-reference top-1 count and the
auxiliary-cost fields stored in the arrays.
"""
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

P4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
CA = os.path.join(P4, "confirmatory_analysis")
PI = os.path.join(P4, "per_image_results")
TAB = os.path.join(OUT, "publication_tables")
os.makedirs(TAB, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, P4).replace("\\", "/")


def write_new(path, text, binary=False):
    with open(path, "xb" if binary else "x", **({} if binary else {"newline": "\n", "encoding": "utf-8"})) as f:
        f.write(text)


# ------------------------------------------------------------------------------------------------------ 1. evidence inventory
ROOT = os.path.abspath(os.path.join(P4, "..", ".."))
inv = []
cache_path = os.path.join(ROOT, "data", "p4_confirm_cache.pt")


def add(path, typ, origin, base=P4):
    inv.append({"relative_path": os.path.relpath(path, ROOT).replace("\\", "/"), "size_bytes": os.path.getsize(path), "sha256": sha(path), "artifact_type": typ, "origin": origin})


for f in sorted(os.listdir(PI)):
    add(os.path.join(PI, f), "per-image arrays (pred, top5, float16 logits" + (", labels/ids" if f == "reference_unpruned.npz" else "") + ")", "generated_output")
for f in sorted(os.listdir(CA)):
    typ = {"csv": "analysis table", "md": "analysis report", "json": "analysis manifest"}[f.rsplit(".", 1)[1]]
    add(os.path.join(CA, f), typ, "generated_output")
for f, typ in (("confirmatory_environment.json", "run environment/log record"), ("confirmatory_run_done.marker", "completion marker"), ("gate5_run_log.txt", "execution log"),
               ("gate5_preflight.json", "pre-run verification record")):
    add(os.path.join(P4, f), typ, "generated_output")
add(cache_path, "decoded P4-CONFIRM cache written by the Gate 5 loader (images, labels, ids)", "generated_output")
add(os.path.join(P4, "confirmatory_results.md"), "hand-written results summary (post-run)", "post_run_documentation")
add(os.path.join(P4, "analysis_documentation.md"), "analysis documentation (pre-run)", "frozen_input")
for f, typ in (("frozen_method_configs.json", "frozen method configuration"), ("frozen_method_configs.sha256", "frozen configuration hash"), ("protocol_freeze.json", "original protocol freeze"),
               ("protocol_freeze.sha256", "original protocol freeze hash"), ("phase4_analysis_freeze.json", "analysis freeze"), ("phase4_analysis_freeze.sha256", "analysis freeze hash"),
               ("protocol_amendments.md", "protocol amendments / incident record"), ("benchmark_protocol.md", "benchmark protocol"), ("benchmark_config.yaml", "benchmark config"),
               ("splits/p4_confirm_manifest.csv", "split manifest (confirm)"), ("splits/p4_dev_manifest.csv", "split manifest (dev)"),
               ("diffrate_deit_tiny_rates_as_published.json", "published DiffRate rates (data)"), ("supplementary_phase3_reselection/frozen_schedules.json", "frozen supplementary Phase 3 schedules")):
    add(os.path.join(P4, f), typ, "frozen_input")
add(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json"), "frozen Phase 3 schedules", "frozen_input")
for f in ("scripts/phase4_gate5_confirmatory.py", "scripts/phase4_gate5_analysis.py", "scripts/phase4_gate5_dev_crosscheck.py", "scripts/phase4_gate5_dev_inputs.py",
          "scripts/phase4_gate5_preflight.py", "scripts/phase4_freeze_analysis.py", "scripts/phase4_freeze_protocol.py"):
    add(os.path.join(ROOT, f), "code", "frozen_input")
for f in sorted(os.listdir(os.path.join(ROOT, "src", "phase4"))):
    if f.endswith(".py"):
        add(os.path.join(ROOT, "src", "phase4", f), "code", "frozen_input")
inv_df = pd.DataFrame(inv)
write_new(os.path.join(OUT, "evidence_inventory.csv"), inv_df.to_csv(index=False, lineterminator="\n"))

# ------------------------------------------------------------------------------------------------------ 2. integrity checks
checks = []


def chk(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


fc = open(os.path.join(P4, "frozen_method_configs.json"), "rb").read()
chk("frozen_method_configs.json matches its recorded SHA-256", hashlib.sha256(fc).hexdigest() == open(os.path.join(P4, "frozen_method_configs.sha256")).read().strip(), hashlib.sha256(fc).hexdigest())
bad = [k for k, h in json.loads(fc)["file_sha256"].items() if sha(os.path.join(ROOT, k)) != h]
chk("all 32+ dependency hashes listed in frozen_method_configs.json still match", not bad, f"{len(json.loads(fc)['file_sha256'])} files; changed={bad}")
af = open(os.path.join(P4, "phase4_analysis_freeze.json"), "rb").read()
chk("phase4_analysis_freeze.json matches its recorded SHA-256", hashlib.sha256(af).hexdigest() == open(os.path.join(P4, "phase4_analysis_freeze.sha256")).read().strip())
afj = json.loads(af)
chk("analysis script matches frozen analysis hash", sha(os.path.join(ROOT, afj["analysis_script"])) == afj["analysis_script_sha256"])
bad = [k for k, h in afj["dependencies_sha256"].items() if sha(os.path.join(ROOT, k)) != h]
chk("analysis-freeze dependencies unchanged", not bad, str(bad))
fr = json.load(open(os.path.join(P4, "protocol_freeze.json")))
chk("original protocol freeze file hash equals recorded value", sha(os.path.join(P4, "protocol_freeze.json")) is not None and open(os.path.join(P4, "protocol_freeze.sha256")).read().strip() != "",
    "full 17-file verification is performed by scripts/phase4_freeze_protocol.py --verify (see audit_summary.md)")
am = json.load(open(os.path.join(CA, "analysis_manifest.json")))
chk("prediction arrays unchanged since the analysis (hashes equal analysis_manifest input_sha256)", all(sha(os.path.join(PI, n + ".npz")) == h for n, h in am["input_sha256"].items()),
    f"{len(am['input_sha256'])} arrays")
chk("analysis output files unchanged since the analysis (hashes equal analysis_manifest output_sha256)", all(sha(os.path.join(CA, n)) == h for n, h in am["output_sha256"].items()), f"{len(am['output_sha256'])} files")
chk("analysis manifest records the frozen analysis script hash", am["analysis_script_sha256"] == afj["analysis_script_sha256"])
mk = json.load(open(os.path.join(P4, "confirmatory_run_done.marker")))
env = json.load(open(os.path.join(P4, "confirmatory_environment.json")))
chk("completion marker status == completed; environment record status == completed; not dry run", mk["status"] == "completed" and env["status"] == "completed" and env["dry_run"] is False,
    f"marker time {mk['time']}; elapsed {env['elapsed_seconds']} s; n={env['n']}")
chk("run used the frozen configuration hash", env["frozen_method_configs_sha256"] == hashlib.sha256(fc).hexdigest())
chk("single run: exactly 125 arrays and no FAILURE_RECORD", len(os.listdir(PI)) == 125 and not os.path.exists(os.path.join(PI, "FAILURE_RECORD.txt")), f"{len(os.listdir(PI))} files")
cache = os.path.join(ROOT, "data", "p4_confirm_cache.pt")
chk("P4-CONFIRM decoded cache present on disk (data/p4_confirm_cache.pt), written by the Gate 5 loader at run start; not deleted by this audit", os.path.exists(cache),
    "cache mtime " + (pd.Timestamp(os.path.getmtime(cache), unit="s", tz="UTC").tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST") if os.path.exists(cache) else "absent") + "; marker start " + mk["time"])
gl = open(os.path.join(P4, "gate5_run_log.txt"), encoding="utf-8", errors="replace").read()
chk("run log ends with 'DONE completed' and contains no Traceback", gl.strip().endswith("DONE completed") and "Traceback" not in gl)
man = pd.read_csv(os.path.join(P4, "splits", "p4_confirm_manifest.csv"))
chk("analysis n equals confirm manifest size (2,985); confirm manifest hash matches frozen identity",
    am["n"] == len(man) == 2985 and sha(os.path.join(P4, "splits", "p4_confirm_manifest.csv")) == json.loads(fc)["split_identity"]["confirm_manifest_sha256"])
t_arr = max(os.path.getmtime(os.path.join(PI, f)) for f in os.listdir(PI))
t_out = min(os.path.getmtime(os.path.join(CA, f)) for f in os.listdir(CA))
chk("temporal order: all prediction arrays written before the analysis outputs", t_arr < t_out)

# ------------------------------------------------------------------------------------------------------ 3. statistical verification (from saved CSVs only)
cmp_files = sorted(f for f in os.listdir(CA) if f.startswith("comparisons_"))
fam = {f[len("comparisons_"):-4]: pd.read_csv(os.path.join(CA, f)) for f in cmp_files}


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p, kind="stable"); a = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); a[i] = run
    return a


def mcn(b, c):
    b, c = int(b), int(c)
    m = b + c
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m)


for k, d in fam.items():
    ok = d[d.status == "ok"]
    und = d[d.status != "ok"]
    n = ok.n.iloc[0] if len(ok) else 0
    chk(f"[{k}] n=2985 in every defined cell", (ok.n == 2985).all())
    chk(f"[{k}] diff_pp == (method_count - comparator_count)/n*100", np.allclose(ok.diff_pp, (ok.method_count - ok.comparator_count) / ok.n * 100, atol=1e-5))
    chk(f"[{k}] direction label consistent with sign of diff (method minus comparator)", all((r.direction == "method>comparator") == (r.diff_pp > 0) and (r.direction == "method<comparator") == (r.diff_pp < 0) for r in ok.itertuples()))
    chk(f"[{k}] bootstrap CI brackets the point estimate", ((ok.boot_ci_lo_pp <= ok.diff_pp + 1e-6) & (ok.diff_pp <= ok.boot_ci_hi_pp + 1e-6)).all())
    chk(f"[{k}] McNemar p equals exact test on stored discordant counts (|dp|<1e-6; CSV rounded to 6 dp)", np.allclose([mcn(r.method_only_correct, r.comparator_only_correct) for r in ok.itertuples()], ok.mcnemar_p, atol=1e-6))
    chk(f"[{k}] discordant counts reconcile with count difference (b - c == method_count - comparator_count)", ((ok.method_only_correct - ok.comparator_only_correct) == (ok.method_count - ok.comparator_count)).all())
    chk(f"[{k}] Holm-adjusted p recomputed from the stored raw p within the family (family size {len(ok)})", np.allclose(holm(ok.mcnemar_p), ok.holm_adj_p, atol=(len(ok) + 1) * 1e-6) and (ok.holm_family_size == len(ok)).all())
    chk(f"[{k}] significance flag == (Holm-adjusted p < 0.05)", ((ok.holm_adj_p < 0.05) == ok.significant_holm).all())
    if len(und):
        chk(f"[{k}] undefined cells are excluded from the Holm family and listed", True, "; ".join(und.cell.tolist()))
chk("primary families have 16 defined cells each; replication families 24", all(len(fam[f"primary_{m}"]) == 16 and len(fam[f"replication_{m}"]) == 24 for m in ("agree", "top1")))

# claims made in confirmatory_results.md / the Gate 5 hand-off, checked against the saved CSVs
claims = []


def claim(text, ok, detail=""):
    claims.append({"claim": text, "ok": bool(ok), "detail": detail})


pa, pt = fam["primary_agree"], fam["primary_top1"]
tm = lambda d: d[d.method.str.startswith("tome")]
ev = lambda d: d[d.method.str.startswith("evit")]
claim("ToMe vs geometric: all 8 cells better, agreement +9.2..+18.6 pp, all Holm-significant", (tm(pa).diff_pp > 0).all() and tm(pa).significant_holm.all() and round(tm(pa).diff_pp.min(), 1) == 9.2 and round(tm(pa).diff_pp.max(), 1) == 18.6,
      f"{tm(pa).diff_pp.min():.3f}..{tm(pa).diff_pp.max():.3f}")
claim("ToMe vs geometric: top-1 +3.4..+12.9 pp, all Holm-significant", (tm(pt).diff_pp > 0).all() and tm(pt).significant_holm.all() and round(tm(pt).diff_pp.min(), 1) == 3.4 and round(tm(pt).diff_pp.max(), 1) == 12.9,
      f"{tm(pt).diff_pp.min():.3f}..{tm(pt).diff_pp.max():.3f}")
claim("EViT-style agreement: B40 worse (-8.6/-7.7), B50 n.s., B60/B70 better (+3.3..+4.0); 14/16 Holm-significant", (ev(pa)[ev(pa).cell.str.contains("B40")].diff_pp < 0).all() and ev(pa)[ev(pa).cell.str.contains("B40")].significant_holm.all() and
      not ev(pa)[ev(pa).cell.str.contains("B50")].significant_holm.any() and (ev(pa)[ev(pa).cell.str.contains("B60|B70")].diff_pp.between(3.2, 4.1)).all() and pa.significant_holm.sum() == 14, f"{int(pa.significant_holm.sum())}/16")
claim("EViT-style top-1: B40 -7.6/-6.8, B50 n.s., B60/B70 +1.7..+2.2; 14/16 Holm-significant", (ev(pt)[ev(pt).cell.str.contains("B40")].diff_pp < 0).all() and not ev(pt)[ev(pt).cell.str.contains("B50")].significant_holm.any() and
      (ev(pt)[ev(pt).cell.str.contains("B60|B70")].diff_pp.between(1.6, 2.3)).all() and pt.significant_holm.sum() == 14, f"{int(pt.significant_holm.sum())}/16")
r_a, r_t = fam["replication_agree"], fam["replication_top1"]
claim("Replication agreement: mean +0.62 pp; 15 positive / 9 negative; 1/24 Holm-significant (B40, 2-8-10, f32, +2.48)", round(r_a.diff_pp.mean(), 2) == 0.62 and (r_a.diff_pp > 0).sum() == 15 and (r_a.diff_pp < 0).sum() == 9 and r_a.significant_holm.sum() == 1 and
      r_a[r_a.significant_holm].cell.iloc[0] == "B40_2-8-10_f32" and round(r_a[r_a.significant_holm].diff_pp.iloc[0], 2) == 2.48, f"mean {r_a.diff_pp.mean():.4f}")
claim("Replication top-1: 3/24 Holm-significant; mean +0.52 pp", r_t.significant_holm.sum() == 3 and round(r_t.diff_pp.mean(), 2) == 0.52, f"{int(r_t.significant_holm.sum())}/24 mean {r_t.diff_pp.mean():.4f}")
for m in ("agree", "top1"):
    t = [fam[f"secondary_tome_vs_{x}_{m}"] for x in "PSC"] + [fam[f"exploratory_tome_vs_U_{m}"]]
    claim(f"ToMe beats P f16 / S / C f16 / U in every defined cell, all Holm-significant ({m})", all((d[d.status == 'ok'].diff_pp > 0).all() and d[d.status == 'ok'].significant_holm.all() for d in t),
          "defined cells: " + "/".join(str((d.status == 'ok').sum()) for d in t))
eP = fam["secondary_evit_vs_P_agree"]; eS = fam["secondary_evit_vs_S_agree"]
claim("EViT-style vs proposed f16 (agreement): 6 better / 2 worse by point estimate; 6 of 8 Holm-significant", (eP.diff_pp > 0).sum() == 6 and (eP.diff_pp < 0).sum() == 2 and eP.significant_holm.sum() == 6,
      "significant directions: " + str(eP[eP.significant_holm].direction.value_counts().to_dict()))
claim("EViT-style vs single-stage (agreement): better in 3 of 4 cells", (eS.diff_pp > 0).sum() == 3, str(eS[["cell", "diff_pp", "significant_holm"]].round(3).values.tolist()))
nt = {m: fam[f"exploratory_native_track_{m}"] for m in ("agree", "top1")}
claim("Native track: 36 comparisons per metric; DiffRate better in 35 of 36 and 33 Holm-significant (both metrics)", all(len(d) == 36 and (d.diff_pp > 0).sum() == 35 and d.significant_holm.sum() == 33 for d in nt.values()),
      {m: [int((d.diff_pp > 0).sum()), int(d.significant_holm.sum())] for m, d in nt.items()})
mt = pd.read_csv(os.path.join(CA, "metrics_all.csv")).set_index("entity")
claim("ToMe B50: agreement 83.0 %, top-1 65.3 %", round(mt.loc["tome_B50", "agreement_pct"], 1) == 83.0 and round(mt.loc["tome_B50", "top1_pct"], 1) == 65.3)
claim("At 57.9 % FLOPs: DiffRate 0.7 agreement 90.7 %, ToMe (DR0.7 cell) 89.7 %", round(mt.loc["diffrate_0.7", "agreement_pct"], 1) == 90.7 and round(mt.loc["tome_DR0.7", "agreement_pct"], 1) == 89.7)
ref = np.load(os.path.join(PI, "reference_unpruned.npz"))
ref_top1 = int((ref["pred"].astype(int) == ref["labels"].astype(int)).sum())
claim("Unpruned reference top-1 = 2,086/2,985 (69.89 %) (derived from the saved reference array; not part of the official analysis tables)", ref_top1 == 2086 and len(ref["pred"]) == 2985, f"{ref_top1}")

# ToMe vs DiffRate at matched FLOPs: the official native-track family contains DiffRate-vs-ToMe comparisons (direction: DiffRate minus ToMe)
nv = {m: d[d.cell.str.contains("vs_ToMe")] for m, d in nt.items()}
tdr = {m: nv[m][["cell", "method_count", "comparator_count", "diff_pp", "boot_ci_lo_pp", "boot_ci_hi_pp", "holm_adj_p", "significant_holm", "method_flops_pct", "comparator_flops_pct"]] for m in nv}

# ------------------------------------------------------------------------------------------------------ 4. publication tables
def pfmt(p):
    return "<0.000001" if p == 0 else f"{p:.6f}"


def cmp_table(d, label_metric):
    ok = d[d.status == "ok"]
    return pd.DataFrame({"cell": ok.cell, "n": ok.n, "method_count": ok.method_count, "comparator_count": ok.comparator_count, f"method_{label_metric}_%": ok.method_pct.round(2),
                         f"comparator_{label_metric}_%": ok.comparator_pct.round(2), "diff_pp (method-comparator)": ok.diff_pp.round(2),
                         "95% bootstrap CI (pp)": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(ok.boot_ci_lo_pp, ok.boot_ci_hi_pp)], "McNemar p (raw)": [pfmt(x) for x in ok.mcnemar_p],
                         "Holm-adj p": [pfmt(x) for x in ok.holm_adj_p], "Holm family size": ok.holm_family_size, "Holm-significant": ok.significant_holm,
                         "method FLOPs %": ok.method_flops_pct.round(2), "comparator FLOPs %": ok.comparator_flops_pct.round(2)}).reset_index(drop=True)


def save(name, df, title, note):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False, lineterminator="\n") if not os.path.exists(os.path.join(TAB, name + ".csv")) else (_ for _ in ()).throw(FileExistsError(name))
    return f"## {title}\n\n{note}\n\n{df.to_markdown(index=False)}\n"


md = ["# Publication tables (generated read-only from `confirmatory_analysis/`; P4-CONFIRM, n = 2,985)\n",
      "Conventions: difference = method − comparator in percentage points; agreement = agreement with the unpruned model's prediction; top-1 = ground-truth accuracy (never combined). "
      "CIs are paired bootstrap percentile intervals (10,000 resamples, seed 42); p-values are exact two-sided McNemar, Holm-adjusted within the stated family. The saved CSVs round p-values to 6 decimals, so `<0.000001` means a stored value of 0.0. "
      "Wilson 95 % CIs for single-method rates are in `metrics_all.csv` and Table 1. Nothing here is recomputed or imputed.\n"]

# Table 1: baseline performance
b1 = [{"entity": "unpruned DeiT-Tiny (reference)", "family": "none", "flops_pct": 100.0, "n": 2985, "agreement_%": 100.0, "agreement_95%_Wilson": "n/a (reference by definition)",
       "top1_count": ref_top1, "top1_%": round(100 * ref_top1 / 2985, 2), "top1_95%_Wilson": "not in official tables", "top5_%": "not in official tables"}]
for ent in ["p3__L2_B%d_%s" % (b, s) for b in (40, 50, 60, 70) for s in ("geometric_three_stage_2-6-10", "geometric_three_stage_2-8-10", "one_stage", "equal_drop_three_stage_2-6-10", "equal_drop_three_stage_2-8-10")] + \
           ["tome_B%d" % b for b in (40, 50, 60, 70)] + ["evit_B%d" % b for b in (40, 50, 60, 70)]:
    if ent in mt.index:
        r = mt.loc[ent]
        b1.append({"entity": ent, "family": r.family, "flops_pct": round(r.flops_pct, 2), "n": int(r.n), "agreement_%": round(r.agreement_pct, 2), "agreement_95%_Wilson": f"[{r.agreement_ci_lo:.2f}, {r.agreement_ci_hi:.2f}]",
                   "top1_count": int(r.top1_count), "top1_%": round(r.top1_pct, 2), "top1_95%_Wilson": f"[{r.top1_ci_lo:.2f}, {r.top1_ci_hi:.2f}]", "top5_%": round(r.top5_pct, 2)})
md.append(save("table1_baseline_performance", pd.DataFrame(b1), "Table 1 — Baseline performance and method rates at the primary budgets",
               "Unpruned top-1 is derived from the saved reference array (labels vs. predictions) because the official tables do not list the reference as an entity. Phase 3 baseline vectors are the frozen schedules re-evaluated on P4-CONFIRM."))
md.append(save("table2_primary_tome_vs_geometric_agreement", cmp_table(tm(pa), "agreement"), "Table 2a — PRIMARY: ToMe vs geometric, prediction agreement (Holm family of 16)", "ToMe configurations were selected on P4-DEV; geometric schedules are the frozen Phase 3 baselines."))
md.append(save("table2_primary_tome_vs_geometric_top1", cmp_table(tm(pt), "top1"), "Table 2b — PRIMARY: ToMe vs geometric, ground-truth top-1 (separate Holm family of 16)", ""))
md.append(save("table3_primary_evit_vs_geometric_agreement", cmp_table(ev(pa), "agreement"), "Table 3a — PRIMARY: EViT-style vs geometric, prediction agreement (Holm family of 16)", "EViT-style is a reproduction-inspired training-free method, not official EViT."))
md.append(save("table3_primary_evit_vs_geometric_top1", cmp_table(ev(pt), "top1"), "Table 3b — PRIMARY: EViT-style vs geometric, ground-truth top-1 (separate Holm family of 16)", ""))
md.append(save("table4_replication_agreement", cmp_table(r_a, "agreement"), "Table 4a — Phase 3 replication: proposed vs geometric, agreement (24 comparisons, own Holm family)",
               "Replication of the method on a new 995-class set, not an exact replication of the Phase 3 evaluation. Several floors share identical schedule vectors, so the 24 comparisons are not 24 independent schedules."))
md.append(save("table4_replication_top1", cmp_table(r_t, "top1"), "Table 4b — Phase 3 replication: proposed vs geometric, top-1 (24 comparisons, own Holm family)", ""))
sec_rows = []
for k, d in fam.items():
    if k.startswith(("secondary_", "exploratory_tome", "exploratory_evit")):
        t = cmp_table(d, "metric") if (d.status == "ok").any() else None
        if t is None:
            continue
        t.insert(0, "family", k); sec_rows.append(t)
sec = pd.concat(sec_rows, ignore_index=True)
undef = [(k, c) for k, d in fam.items() for c in d[d.status != "ok"].cell]
md.append(save("table5_secondary_and_exploratory_vs_phase3_comparators", sec, "Table 5 — Secondary (vs proposed f16, single-stage, constrained f16) and exploratory (vs unconstrained equal-drop) comparisons",
               "Each family is Holm-corrected separately (per method × comparator × metric). `metric` columns are agreement for families ending `_agree` and top-1 for `_top1`. Undefined cells (not in the frozen schedule files, excluded from Holm): " + "; ".join(f"{k}:{c}" for k, c in undef)))
nat = pd.concat([cmp_table(nt[m], m).assign(metric=m) for m in nt], ignore_index=True)
md.append(save("table6_exploratory_diffrate_native_track", nat, "Table 6 — Exploratory supplementary native-FLOPs track: DiffRate (released DeiT-Tiny rates 0.6–0.9) vs matched comparators",
               "DiffRate cells: faithful adaptation pending official semantic verification; target 1.0 not applicable as released and excluded. diff = DiffRate − comparator. One Holm family of 36 per metric, within this track only. "
               "Rows `…vs_ToMe` give the ToMe-at-matched-FLOPs comparison (ToMe tuned on P4-DEV at DiffRate's executed FLOPs)."))
md.append(save("table6b_diffrate_vs_tome_matched_flops", pd.concat([cmp_table(nv[m], m).assign(metric=m) for m in nv], ignore_index=True), "Table 6b — DiffRate vs ToMe at matched FLOPs (subset of Table 6)", "Positive diff = DiffRate higher."))

# Table 7: computational budget and realized FLOPs
fcj = json.loads(fc)
t7 = []
for key, v in fcj["selected"].items():
    tag, m = key.split("|")
    z = np.load(os.path.join(PI, f"{m}_{tag}.npz"))
    t7.append({"cell": f"{m}_{tag}", "track": v["track"], "target_flops_%": round(v["target_flops_pct"], 3), "config": json.dumps(v["config"]), "realized_analytic_flops_%": round(v["realized_flops_pct"], 3),
               "tokens_after_each_block": json.dumps(v["leaving_tokens_per_block"]), "aux_analytic_flops_per_image(not in model FLOPs)": float(z["aux_flops_per_image"]), "aux_seconds_total(measured, confirm run)": round(float(z["aux_seconds_total"]), 2)})
for t, v in fcj["diffrate_released"].items():
    z = np.load(os.path.join(PI, f"diffrate_{t}.npz"))
    t7.append({"cell": f"diffrate_{t}", "track": "native", "target_flops_%": "released rate", "config": "published rates", "realized_analytic_flops_%": round(v["executed_analytic_flops_pct"], 3),
               "tokens_after_each_block": json.dumps(v["executed_leaving"]), "aux_analytic_flops_per_image(not in model FLOPs)": float(z["aux_flops_per_image"]), "aux_seconds_total(measured, confirm run)": round(float(z["aux_seconds_total"]), 2)})
t7 = pd.DataFrame(t7)
lat = pd.read_csv(os.path.join(P4, "gate4", "latency_dev.csv"))
md.append(save("table7_compute_budget_and_realized_flops", t7, "Table 7 — Computational budget: realized analytic FLOPs and auxiliary cost",
               "FLOPs are analytic counts of the executed token counts (patch-embed and head included); auxiliary merge/scoring/fusion cost is separate and is NOT part of the FLOPs column. Auxiliary seconds are wall-clock timers inside the compression operators summed over all 2,985 images "
               "in the confirmatory run on a shared CPU session (indicative only). Latency for these cells was measured only on P4-DEV (Gate 4, `gate4/latency_dev.csv`) with noisy CPU timings (IQR up to several ms); no latency ranking is claimed and none is tabulated as a result."))
lat_t = lat[lat.method.isin(["tome", "evit", "diffrate", "unpruned_reference"])][["engine", "method", "config", "median_ms_per_img", "iqr_ms", "ratio_to_engine_ref", "aux_ms_per_img"]].round(3)
md.append(save("table7b_dev_only_latency_descriptive", lat_t, "Table 7b — DEV-ONLY latency (descriptive; NOT confirmatory; noisy)",
               "From Gate 4 on P4-DEV: CPU, batch 25, 4 threads, 1 warm-up + 5 runs × 3 batches. Ratios are to each engine's own unpruned forward. Equal FLOPs are not equal latency (e.g. ToMe B60/B70 run slower than the unpruned engine in these measurements despite fewer FLOPs)."))
write_new(os.path.join(TAB, "all_publication_tables.md"), "\n".join(md))

# ------------------------------------------------------------------------------------------------------ 5. outputs of the verification
ver = {"integrity_and_statistical_checks": checks, "claim_checks": claims, "n_checks": len(checks), "all_checks_pass": all(c["ok"] for c in checks), "n_claims": len(claims), "all_claims_pass": all(c["ok"] for c in claims),
       "native_track_tome_vs_diffrate": {m: tdr[m].round(4).to_dict("records") for m in tdr}}
write_new(os.path.join(OUT, "verification_results.json"), json.dumps(ver, indent=1, default=str))
print("checks", len(checks), "pass" if ver["all_checks_pass"] else "FAIL", "| claims", len(claims), "pass" if ver["all_claims_pass"] else "FAIL")
for c in checks + claims:
    if not c["ok"]:
        print("FAIL:", c)
print("inventory rows", len(inv_df))
