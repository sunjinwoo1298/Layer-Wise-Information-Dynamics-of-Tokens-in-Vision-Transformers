#!/usr/bin/env python3
"""Phase 3 Gate 1: recompute every number used in the evidence ledger from the RAW result files (no narrative summaries).
Writes results/phase3_algorithm_design/audit_numbers.md (+ .json). Read-only with respect to all previous results."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = "results"
OUT = os.path.join(R, "phase3_algorithm_design")
os.makedirs(OUT, exist_ok=True)
lines, J = [], {}


def H(t):
    lines.append(f"\n## {t}\n")


def T(df, floatfmt=3):
    lines.append(df.to_markdown(index=False, floatfmt=f".{floatfmt}f") + "\n")


def P(s):
    lines.append(s + "\n")


def lastk(js):
    d = json.loads(js) if isinstance(js, str) else js
    v = list(d.values())
    return v[-1] if v else 196


# ---------------------------------------------------------------- A. Phase 1
H("A. Phase 1 single-layer characterization (250 images, 5 classes; results/deit_tiny_imagenet_dev250/layer_metrics.csv)")
l1 = pd.read_csv(f"{R}/deit_tiny_imagenet_dev250/layer_metrics.csv")
l1["scorer"] = l1.scorer.replace({"prediction_kl_approximate": "prediction_kl_approx", "attention_cls": "cls_attention"})
for r in (0.5, 0.25, 0.125):
    s = l1[np.isclose(l1.retention_ratio, r)]
    pv = s.pivot(index="layer", columns="scorer", values="prediction_agreement")
    fl = s.groupby("layer").normalized_flops.first()
    t = pv.copy()
    t["norm_FLOPs"] = fl
    t = t.reset_index()
    t.insert(0, "retention", r)
    P(f"Agreement by layer/scorer at retention {r} (single-layer pruning after block `layer`):")
    T(t)
    t1 = s.pivot(index="layer", columns="scorer", values="top1_accuracy").reset_index()
    t1.insert(0, "retention", r)
    P(f"Top-1 at retention {r}:")
    T(t1)
sig = l1[(l1.scorer != "random") & (l1.retention_ratio < 1)]
g = sig.groupby("scorer").agg(n=("mcnemar_pval", "size"), n_sig_p05=("mcnemar_pval", lambda x: int((x < 0.05).sum()))).reset_index()
P("McNemar vs random (uncorrected p<0.05) over layer x retention cells (Phase 1 file; no multiplicity correction in file):")
T(g, 0)
l1ok = l1[l1.layer < 12]
sp = [(sc, float(spearmanr(d.normalized_flops, d.mean_latency_ms)[0])) for sc, d in l1ok.groupby("scorer")]
P("Spearman(norm FLOPs, mean latency ms) per scorer (layers<12): " + ", ".join(f"{a}: {b:.2f}" for a, b in sp))
P("Layer-12 rows: normalized FLOPs = " + str(sorted(l1[l1.layer == 12].normalized_flops.round(4).unique())) + " (pruning at the final block saves nothing)")
J["phase1_spearman_flops_latency"] = dict(sp)

# ---------------------------------------------------------------- B. progressive eval
H("B. Progressive vs single-layer (250 images, 5 classes; results/phase2_progressive_eval/matched_pairs_comparison.csv)")
mp = pd.read_csv(f"{R}/phase2_progressive_eval/matched_pairs_comparison.csv")
g = mp.groupby("scorer").agg(n=("agreement_diff", "size"), mean_agree_diff_pp=("agreement_diff", lambda x: 100 * x.mean()),
                             n_pos=("agreement_diff", lambda x: int((x > 0).sum())), n_neg=("agreement_diff", lambda x: int((x < 0).sum())),
                             n_p05=("agreement_mcnemar_p", lambda x: int((x < 0.05).sum())),
                             mean_top1_diff_pp=("top1_diff", lambda x: 100 * x.mean()),
                             max_abs_flops_diff_pp=("flops_diff_pp", lambda x: x.abs().max())).reset_index()
P("progressive minus single at matched FLOPs, per scorer (uncorrected McNemar counts; file has no Holm):")
T(g, 2)
g2 = mp[mp.scorer == "cls_attention"].groupby("single_layer").agg(n=("agreement_diff", "size"),
                                                                    mean_agree_diff_pp=("agreement_diff", lambda x: 100 * x.mean())).reset_index()
P("cls_attention only, by comparator single-layer:")
T(g2, 2)

# ---------------------------------------------------------------- C. stage ablation
H("C. Controlled stage-count ablation (250 images; results/phase2_stage_ablation/statistical_tests.csv; geometric-style schedules)")
sa = pd.read_csv(f"{R}/phase2_stage_ablation/statistical_tests.csv")
for fam in ("PRIMARY_agreement", "primary_setting_top1"):
    d = sa[sa.family == fam]
    P(f"{fam}: n={len(d)}, mean diff={100*d['diff'].mean():+.2f} pp, positive={int((d['diff']>0).sum())}, Holm-sig={int(d['significant_holm_0.05'].sum())}, "
      f"raw p<.05={int(d['significant_raw_0.05_uncorrected'].sum())}")
    t = d.groupby("stage_count_a").agg(n=("diff", "size"), mean_pp=("diff", lambda x: 100 * x.mean())).reset_index()
    T(t, 2)

# ---------------------------------------------------------------- D. class-diverse
H("D. Class-diverse validation (2,000 images / 1,000 classes; results/phase2_class_diverse_validation)")
cd = pd.read_csv(f"{R}/phase2_class_diverse_validation/statistical_tests.csv")
cdm = pd.read_csv(f"{R}/phase2_class_diverse_validation/metrics.csv")
P("Unpruned row: " + str(cdm[cdm.scorer == "none"][["agreement_count", "top1_count", "n_images"]].to_dict("records")))
P("**Recomputed headline numbers (cls_attention, seed 42, first layer 2):**")
for fam_a in ("geometric", "equal_drop"):
    for metric in ("agreement", "top1"):
        d = cd[(cd.comparison_type == "multi_vs_one_stage") & (cd.family_a == fam_a) & (cd.scorer == "cls_attention") &
               (cd.first_layer == 2) & (cd.metric == metric)]
        P(f"- multi_vs_one_stage {fam_a} {metric}: n={len(d)}, mean={100*d['diff'].mean():+.2f} pp, higher={int((d['diff']>0).sum())}, "
          f"lower={int((d['diff']<0).sum())}, Holm-sig={int(d['significant_holm_0.05'].sum())}; by stages: " +
          ", ".join(f"{k}-stage {100*v['diff'].mean():+.2f}" for k, v in d.groupby("stage_count_a")))
for fam_a in ("geometric", "equal_drop"):
    d = cd[(cd.comparison_type == "three_vs_two_prefix") & (cd.family_a == fam_a) & (cd.scorer == "cls_attention") &
           (cd.first_layer == 2) & (cd.metric == "agreement")]
    P(f"- three_vs_two_prefix {fam_a} agreement: n={len(d)}, mean={100*d['diff'].mean():+.2f} pp, Holm-sig={int(d['significant_holm_0.05'].sum())}")
ge = cd[(cd.comparison_type == "geometric_vs_equal_drop") & (cd.scorer == "cls_attention") & (cd.first_layer == 2) & (cd.metric == "agreement")]
t = ge[["budget_target", "structure_a", "diff", "ci_low", "ci_high", "p_holm", "flops_diff_pp"]].copy()
for c_ in ("diff", "ci_low", "ci_high"):
    t[c_] = 100 * t[c_]
t["final_k_geo"] = [lastk(cdm[cdm.schedule == s].schedule_json.iloc[0]) for s in ge.schedule_a]
t["final_k_eqd"] = [lastk(cdm[cdm.schedule == s].schedule_json.iloc[0]) for s in ge.schedule_b]
P("geometric minus equal_drop (agreement, pp), cls_attention, layer 2:")
T(t.sort_values(["budget_target", "structure_a"]), 2)
P(f"mean over {len(t)} cells: {t['diff'].mean():+.2f} pp; Holm-significant: {int((t.p_holm<0.05).sum())}")
sc = cdm[(cdm.first_layer == 2) & (cdm.stage_count == 1)].groupby(["budget_target", "scorer", "seed"]).agreement.mean().reset_index()
P("One-stage anchor agreement by scorer (2,000 images; only the budgets where run):")
T(sc.pivot_table(index="budget_target", columns=["scorer", "seed"], values="agreement"), 4)
rd = cd[(cd.scorer == "random") & (cd.comparison_type == "multi_vs_one_stage") & (cd.metric == "agreement") & (cd.first_layer == 2)]
t = rd.groupby(["family_a", "budget_target", "structure_a"]).agg(mean_pp=("diff", lambda x: 100 * x.mean()),
                                                                 sd_pp=("diff", lambda x: 100 * x.std(ddof=1)), n_seeds=("diff", "size")).reset_index()
P(f"random scorer multi-vs-one diffs across seeds 42/43/44 (agreement pp): mean over cells of seed-sd = {t.sd_pp.mean():.2f} pp (max {t.sd_pp.max():.2f}); "
  f"mean diff geometric {t[t.family_a=='geometric'].mean_pp.mean():+.2f}, equal_drop {t[t.family_a=='equal_drop'].mean_pp.mean():+.2f}")
rc = json.load(open(f"{R}/phase2_class_diverse_validation/run_config.json"))
P(f"class-diverse not_run entries: {len(rc.get('not_run', []))}")

# ---------------------------------------------------------------- E. minimum token
H("E. Minimum-token ablation (results/phase2_minimum_token_ablation/statistical_tests.csv)")
mt = pd.read_csv(f"{R}/phase2_minimum_token_ablation/statistical_tests.csv")
for fam in ("PRIMARY_cons_vs_uncons_agreement", "PRIMARY_cons_vs_uncons_top1", "SECONDARY_cons_vs_geo_agreement", "SECONDARY_cons_vs_geo_top1"):
    d = mt[mt.family == fam]
    P(f"- {fam}: n={len(d)}, mean={100*d['diff'].mean():+.2f} pp, higher={int((d['diff']>0).sum())}, lower={int((d['diff']<0).sum())}, "
      f"Holm-sig={int(d['significant_holm_0.05'].sum())}, range [{100*d['diff'].min():+.2f}, {100*d['diff'].max():+.2f}]")
d = mt[(mt.family == "SECONDARY_cons_vs_geo_agreement")]
t = d.groupby("floor_a").agg(n=("diff", "size"), mean_pp=("diff", lambda x: 100 * x.mean()), min_pp=("diff", lambda x: 100 * x.min()),
                             max_pp=("diff", lambda x: 100 * x.max())).reset_index()
P("constrained minus geometric by floor (agreement):")
T(t, 2)
mtm = pd.read_csv(f"{R}/phase2_minimum_token_ablation/metrics.csv")
rcm = json.load(open(f"{R}/phase2_minimum_token_ablation/run_config.json"))
mman = json.load(open(f"{R}/phase2_minimum_token_ablation/schedule_manifest.json"))
P(f"minimum-token not_run entries: {len(rcm.get('not_run', []))}; infeasible combos: {len(mman['infeasible_combinations'])}; "
  f"identical-to-unconstrained: {len(mman['identical_to_unconstrained'])}")

# ---------------------------------------------------------------- F. isolation
H("F. Final-token vs trajectory isolation (results/phase2_final_token_trajectory_isolation/statistical_tests.csv)")
iso = pd.read_csv(f"{R}/phase2_final_token_trajectory_isolation/statistical_tests.csv")
c = iso[(iso.comparison == "geo_vs_ctrl") & (iso.metric == "agreement")].copy()


def outc(r):
    if r.p_holm < 0.05:
        return "geometric_better" if r["diff"] > 0 else "control_better"
    return "similar" if (100 * r.ci_low > -2 and 100 * r.ci_high < 2) else "inconclusive"


c["outcome"] = c.apply(outc, axis=1)
P(f"geo_vs_ctrl agreement: n={len(c)}; outcomes {c.outcome.value_counts().to_dict()}; max |FLOPs diff|={c.flops_diff_pp.abs().max():.3f} pp; "
  f"controls using MORE FLOPs than geometric: {int((c.flops_diff_pp<0).sum())}/{len(c)}; mean diff {100*c['diff'].mean():+.2f} pp")
c["bud"] = (100 * c.budget_target).round().astype(int)
T(c.groupby(["bud", "outcome"]).size().unstack(fill_value=0).reset_index(), 0)
t = c.groupby(pd.cut(c.dist_b_to_geometric, [0, 12, 20, 40]), observed=True).agg(n=("diff", "size"), mean_pp=("diff", lambda x: 100 * x.mean())).reset_index()
P("mean (geo - control) agreement by trajectory deviation bin (descriptive):")
T(t, 2)
c["mid_k"] = [json.loads(v)[1] for v in c.vector_b]
c["geo_mid_k"] = [json.loads(v)[1] for v in c.vector_a]
c["mid_ratio_to_geo"] = c.mid_k / c.geo_mid_k
t = c[["bud", "structure", "vector_a", "vector_b", "flops_diff_pp", "diff", "p_holm", "outcome"]].copy()
t["diff"] = 100 * t["diff"]
P("all 32 controls (diff = geometric minus control, pp):")
T(t, 3)
corr = spearmanr(c.mid_ratio_to_geo, c["diff"])
P(f"Spearman(control mid-stage count / geometric mid-stage count, geo-minus-control diff) = {corr[0]:.2f} (p={corr[1]:.3g}); n={len(c)} (descriptive; 32 selected controls)")
J["iso_outcomes"] = {k: int(v) for k, v in c.outcome.value_counts().items()}
J["iso_spearman_mid_ratio"] = [float(corr[0]), float(corr[1])]
for comp in ("geo_vs_cons", "geo_vs_unc", "cons_vs_unc"):
    d = iso[(iso.comparison == comp) & (iso.metric == "agreement") & iso.predefined]
    P(f"- {comp} (predefined): n={len(d)}, mean {100*d['diff'].mean():+.2f} pp, Holm-sig {int(d['significant_holm_0.05'].sum())}, by budget: " +
      ", ".join(f"B{int(100*b)} {100*v['diff'].mean():+.2f}" for b, v in d.groupby("budget_target")))

# ---------------------------------------------------------------- G. FLOPs vs latency
H("G. FLOPs vs latency (CPU)")
lat = pd.read_csv(f"{R}/phase2_final_token_trajectory_isolation/latency.csv")
l2 = lat[lat.allocation_family != "none"]
P(f"isolation latency: Spearman(FLOPs%, end-to-end ms) over {len(l2)} schedules = {spearmanr(l2.flops_pct, l2.end_to_end_ms_median)[0]:.2f}; "
  f"unpruned forward {lat.unpruned_forward_ms_median.iloc[0]:.0f} ms")
for fam, d in l2.groupby("allocation_family"):
    P(f"- {fam}: n={len(d)}, mean FLOPs {d.flops_pct.mean():.1f}%, e2e {d.end_to_end_ms_median.mean():.0f} ms, scorer overhead {d.scorer_overhead_ms_median.mean():.1f} ms")
m3 = l2[l2.allocation_family.isin(["geometric", "final_count_matched_geometric"])].copy()
m3["cell"] = m3.budget_target.round(2).astype(str) + "|" + m3.structure.astype(str)
sp = m3.groupby("cell").end_to_end_ms_median.agg(lambda x: (x.max() - x.min()) / x.min() * 100)
P(f"within-cell relative spread of end-to-end latency among geometric + its 4 controls (FLOPs within ~1 pp): median {sp.median():.0f}%, max {sp.max():.0f}% (includes timing noise)")
for nm, d in (("class-diverse", cdm), ("min-token", mtm)):
    d = d[d.latency_total_ms_median.notna()]
    g = d.groupby("scorer").agg(n=("scorer", "size"), e2e=("latency_total_ms_median", "mean"), scorer_ms=("scorer_overhead_ms_median", "mean")).reset_index()
    P(f"{nm} latency by scorer (ms; mixed schedules):")
    T(g, 1)

# ---------------------------------------------------------------- H. cross-experiment consistency
H("H. Cross-experiment consistency of the same schedules (cls_attention, seed 42, 2,000 images)")
a = cdm[(cdm.scorer == "cls_attention") & (cdm.seed == 42)].set_index("schedule")
b = mtm[(mtm.scorer == "cls_attention") & (mtm.seed == 42) & (~mtm.is_alias.astype(bool))].set_index("schedule")
im = pd.read_csv(f"{R}/phase2_final_token_trajectory_isolation/metrics.csv")
i = im[(im.scorer == "cls_attention") & (~im.is_alias.astype(bool))].set_index("schedule")
ab, bi = a.index.intersection(b.index), b.index.intersection(i.index)
P(f"class-diverse vs min-token: {len(ab)} shared schedules, identical agreement counts: {int((a.loc[ab].agreement_count == b.loc[ab].agreement_count).sum())}")
P(f"min-token vs isolation: {len(bi)} shared schedules, identical agreement counts: {int((b.loc[bi].agreement_count == i.loc[bi].agreement_count).sum())}")

# ---------------------------------------------------------------- I. final token vs agreement
H("I. Agreement vs final patch count (3-stage cls_attention schedules, distinct vectors pooled over the 2,000-image experiments)")
rows = []
for d in (mtm, im.reset_index(drop=True)):
    d = d[(d.stage_count == 3) & (d.scorer == "cls_attention") & (d.seed == 42)]
    for r in d.itertuples():
        rows.append({"vec": json.dumps(list(json.loads(r.schedule_json).values())), "bud": int(round(100 * r.budget_target)),
                     "final": int(lastk(r.schedule_json)), "agree": r.agreement, "fam": r.allocation_family})
x = pd.DataFrame(rows).drop_duplicates("vec")
x["final_bin"] = pd.cut(x.final, [0, 4, 8, 16, 24, 32, 64, 200])
t = x.groupby(["bud", "final_bin"], observed=True).agree.agg(["size", "mean", "min", "max"]).reset_index()
for cc in ("mean", "min", "max"):
    t[cc] = 100 * t[cc]
P("distinct 3-stage schedules: agreement (%) by budget x final-count bin (descriptive):")
T(t, 1)

# ---------------------------------------------------------------- J. scorer differences at 2,000 images + layer 4
H("J. Scorer comparison at 2,000 images (one-stage layer-2 anchors, paired exact McNemar, UNCORRECTED) and first-layer-4 / KL multi-stage effects")
from scipy.stats import binomtest
pi = pd.read_csv(f"{R}/phase2_class_diverse_validation/per_image_metrics.csv")
rows = []
for s in [x for x in pi.schedule.unique() if x.endswith("one_stage") and x.startswith("L2_")]:
    c = pi[(pi.schedule == s) & (pi.scorer == "cls_attention")].sort_values("image_idx")
    for sc_, seed in (("random", 42), ("random", 43), ("random", 44), ("prediction_kl_approx", 42)):
        r = pi[(pi.schedule == s) & (pi.scorer == sc_) & (pi.seed == seed)].sort_values("image_idx")
        if len(r) == 0:
            continue
        a_, b_ = c.agreement.values.astype(bool), r.agreement.values.astype(bool)
        ao, bo = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
        rows.append({"anchor": s, "other_scorer": sc_, "seed": seed, "cls_minus_other_pp": 100 * (a_.mean() - b_.mean()),
                     "cls_only": ao, "other_only": bo, "p_raw": binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else 1.0})
T(pd.DataFrame(rows), 4)
for L_ in (2, 4):
    for sc_ in ("cls_attention", "prediction_kl_approx"):
        for fam in ("geometric", "equal_drop"):
            d = cd[(cd.comparison_type == "multi_vs_one_stage") & (cd.first_layer == L_) & (cd.scorer == sc_) & (cd.family_a == fam) & (cd.metric == "agreement")]
            if len(d):
                P(f"- first layer {L_}, {sc_}, {fam}: multi minus one-stage agreement mean {100*d['diff'].mean():+.2f} pp, n={len(d)}, higher={int((d['diff']>0).sum())}, "
                  f"Holm-sig={int(d['significant_holm_0.05'].sum())}, budgets {sorted(d.budget_target.unique())}")

open(os.path.join(OUT, "audit_numbers.md"), "w", encoding="utf-8").write(
    "# Phase 3 evidence audit: numbers recomputed from raw files (generated by scripts/phase3_evidence_audit.py)\n" + "\n".join(lines))
json.dump(J, open(os.path.join(OUT, "audit_numbers.json"), "w"), indent=1, default=str)
print("ok", len(lines))
