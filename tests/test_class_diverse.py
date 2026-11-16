"""Tests for the class-diverse validation study (random-init DeiT-Tiny, tiny synthetic data, CPU)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.class_diverse import build_manifest, class_coverage_summary, manifest_hash, select_balanced_rows
from src.experiments.class_diverse import (
    DEFAULT_STRUCTURES, FAMILIES, ClassDiverseConfig, build_statistical_tests, family_counts, generate_family_schedules,
    run_class_diverse,
)
from src.experiments.progressive_eval import expected_processed_counts
from src.experiments.stage_ablation import holm_correction
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper

TOL = 0.01


@pytest.fixture(scope="module")
def est():
    return ComputeEstimator(12, 192, 3, 196, 1000, 1, (16, 16))


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ViTWrapper("deit_tiny_patch16_224", pretrained=False, device="cpu")


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(5)
    return torch.randn(8, 3, 224, 224, generator=g), torch.randint(0, 1000, (8,), generator=g)


@pytest.fixture(scope="module")
def result(wrapper, data, tmp_path_factory):
    cfg = ClassDiverseConfig(pretrained=False, batch_size=4, budgets=[0.5, 0.7], extra_seeds_random=[43],
                             scorers_primary_layer=["cls_attention", "random"], scorers_secondary_layer=["cls_attention"],
                             measure_latency=False, make_plots=False, previous_stage_ablation_dir=None, device="cpu",
                             output_dir=str(tmp_path_factory.mktemp("cd") / "out"))
    return run_class_diverse(cfg, wrapper=wrapper, data=data)


# ---- subset construction -------------------------------------------------------------------------------------------

def test_subset_selection_deterministic_balanced_and_seed_dependent():
    src = np.repeat(np.arange(1000), 50)                                  # 50 rows per class, like ImageNet val
    a, b = select_balanced_rows(src, 2, 42), select_balanced_rows(src, 2, 42)
    assert (a == b).all() and len(a) == 2000 and len(set(a.tolist())) == 2000        # no row twice
    assert (np.bincount(src[a], minlength=1000) == 2).all()                           # every class exactly twice
    assert not (select_balanced_rows(src, 2, 43) == a).all()
    assert len(set(src[a[:25]].tolist())) > 15                                       # shuffled order: batches mix classes


def test_selection_handles_small_classes():
    src = np.array([0, 0, 0, 1, 2, 2])
    rows = select_balanced_rows(src, 2, 1, num_classes=3)
    assert np.bincount(src[rows], minlength=3).tolist() == [2, 1, 2]


def test_class_coverage_summary():
    s = class_coverage_summary([0, 0, 1, 2, 2, 2], num_classes=5)
    assert s["num_images"] == 6 and s["num_distinct_classes"] == 3 and s["classes_missing"] == 2
    assert s["per_class_count_over_present_classes"] == {"min": 1, "median": 2.0, "max": 3}
    assert s["per_class_count_over_all_classes"]["min"] == 0 and s["class_histogram"]["2"] == 3


def test_manifest_hash_is_deterministic_and_order_sensitive():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(4, 3, 8, 8, generator=g), torch.tensor([3, 1, 2, 0])
    m = build_manifest(x, y, np.array([10, 11, 12, 13]))
    assert manifest_hash(m) == manifest_hash(build_manifest(x, y, np.array([10, 11, 12, 13])))
    perm = [1, 0, 2, 3]
    m2 = build_manifest(x[perm], y[perm], np.array([10, 11, 12, 13])[perm])
    assert manifest_hash(m2) != manifest_hash(m)
    assert list(m.columns[:4]) == ["position", "image_id", "source_row", "label"]


# ---- schedules ---------------------------------------------------------------------------------------------------

def test_family_counts_and_rounding():
    assert family_counts("geometric", 196, 0.5, 3) == [98, 49, 24]                  # round-half-even on 24.5
    assert family_counts("equal_drop", 196, 49.0, 3) == [147, 98, 49]
    assert family_counts("equal_drop", 196, 49.4, 2) == [147, 97]
    assert family_counts("equal_drop", 10, 4.0, 3) == [6, 2, 1]                     # floor at 1
    with pytest.raises(ValueError):
        family_counts("nope", 196, 1.0, 2)
    # the families distribute tokens differently: drop per stage shrinks (geometric) vs constant (equal_drop)
    g, e = family_counts("geometric", 196, 0.6, 3), family_counts("equal_drop", 196, 40.0, 3)
    dg = np.diff([196] + g); de = np.diff([196] + e)
    assert abs(dg[0]) > abs(dg[-1]) and abs(de[0] - de[-1]) <= 1


def test_schedules_valid_matched_and_families_differ(est):
    g = generate_family_schedules(est, 2, [0.4, 0.5, 0.6, 0.7], DEFAULT_STRUCTURES[2], TOL)
    assert g["entries"]
    fams = {e["family"] for e in g["entries"]}
    assert fams == {"shared", "geometric", "equal_drop"}
    by = {}
    for e in g["entries"]:
        by.setdefault(e["group"], []).append(e)
        assert e["first_layer"] == 2 and min(e["schedule"]) == 2 and max(e["schedule"]) < 12
        ks = [e["schedule"][l] for l in sorted(e["schedule"])]
        assert all(a > b for a, b in zip([196] + ks, ks)) and min(ks) >= 1 and len(ks) == e["stage_count"]
        counts = expected_processed_counts(e["schedule"], 12, 196)
        assert all(a >= b for a, b in zip(counts, counts[1:]))
        assert e["normalized_flops"] == pytest.approx(e["analytic_flops"] / est.baseline_flops)
    for gid, ents in by.items():
        anchor = next(e for e in ents if e["role"] == "anchor")
        for e in ents:
            assert abs(e["normalized_flops"] - anchor["normalized_flops"]) <= TOL + 1e-12
    for grp in g["groups"]:
        assert grp["max_abs_diff_vs_anchor_pp"] <= 100 * TOL + 1e-9
    # same structure & budget: the two families give different counts
    for gid, ents in by.items():
        for geo in [e for e in ents if e["family"] == "geometric"]:
            alt = next((e for e in ents if e["family"] == "equal_drop" and e["structure"] == geo["structure"]), None)
            if alt is None:      # allowed only if reported as unmatched (never silently dropped)
                assert any(u["family"] == "equal_drop" and u["structure"] == geo["structure"] and
                           u["budget"] == geo["budget_target"] for u in g["unmatched"])
            else:
                assert geo["schedule"] != alt["schedule"]
    for u in g["unmatched"]:
        assert u["family"] in FAMILIES and (u["best_diff_pp"] is None or abs(u["best_diff_pp"]) > 100 * TOL)
    # geometric reproduces the previous stage-ablation generator exactly
    from src.experiments.stage_ablation import DEFAULT_STRUCTURES as PREV, generate_schedules
    prev = {e["name"].replace("_geometric", ""): e["schedule"] for e in generate_schedules(est, 2, [0.4, 0.5, 0.6, 0.7], PREV[2], TOL)["entries"]}
    for e in g["entries"]:
        if e["family"] in ("geometric", "shared"):
            assert prev[e["name"].replace("_geometric", "")] == e["schedule"]


def test_unreachable_budget_reported_not_loosened(est):
    g = generate_family_schedules(est, 2, [0.05], DEFAULT_STRUCTURES[2], TOL)
    assert not g["entries"] and g["unmatched"] and g["unmatched"][0]["reason"].startswith("one-stage anchor")


def test_tight_tolerance_reports_unmatched_instead_of_loosening(est):
    g = generate_family_schedules(est, 2, [0.5], {"two": [(2, 4)], "three": []}, 1e-6)
    assert any("within tolerance" in u["reason"] for u in g["unmatched"]) or g["groups"][0]["complete"]
    for e in g["entries"]:
        assert abs(e["normalized_flops"] - g["entries"][0]["normalized_flops"]) <= 1e-6 + 1e-12


# ---- statistics ----------------------------------------------------------------------------------------------------

def test_holm_correction():
    adj = holm_correction([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_statistics_families_primary_and_correction(result):
    st = result["stats"]
    assert not st.empty
    prim = st[st.is_primary]
    assert set(prim.scorer) == {"cls_attention"} and set(prim.first_layer) == {2} and set(prim.seed) == {42}
    assert set(prim.family) <= {"PRIMARY_stage_agreement", "PRIMARY_stage_top1", "PRIMARY_allocation_agreement", "PRIMARY_allocation_top1"}
    for fam, g in prim.groupby("family"):
        assert (g.family_size == len(g)).all() and (g.p_holm >= g.p_raw - 1e-12).all() and (g.p_holm <= 1).all()
    assert {"multi_vs_one_stage", "three_vs_two_prefix", "geometric_vs_equal_drop"} <= set(st.comparison_type)
    assert set(st[st.comparison_type == "geometric_vs_equal_drop"].flops_diff_pp.abs() <= 1.0 + 1e-9) == {True}
    assert st[(st.first_layer == 4) | (st.scorer != "cls_attention")].family.str.startswith("exploratory").all()
    stage = prim[prim.family == "PRIMARY_stage_agreement"]
    assert set(stage.family_a) == {"geometric", "equal_drop"}                    # both allocation families pooled


def test_paired_stat_matches_manual(result):
    st, pi = result["stats"], result["per_image"]
    r = st[st.is_primary & (st.metric == "agreement")].iloc[0]
    a = pi[(pi.schedule == r.schedule_a) & (pi.scorer == r.scorer) & (pi.seed == r.seed)].sort_values("image_idx").agreement.to_numpy()
    b = pi[(pi.schedule == r.schedule_b) & (pi.scorer == r.scorer) & (pi.seed == r.seed)].sort_values("image_idx").agreement.to_numpy()
    assert r["diff"] == pytest.approx(a.mean() - b.mean()) and r.a_only == int((a & ~b.astype(bool)).sum()) and r.count_a == a.sum()


# ---- end-to-end consistency ----------------------------------------------------------------------------------------

def test_manifest_matches_executed_schedules(result):
    man, m = result["manifest"], result["metrics"]
    man_s = {s["name"]: s["schedule"] for s in man["schedules"]}
    run_s = {r.schedule: json.loads(r.schedule_json) for r in m.itertuples()}
    assert {k: v for k, v in man_s.items() if k in run_s} == run_s and set(man_s) == set(run_s) | {"no_pruning"}
    out = result["run_config"]["config"]["output_dir"]
    saved = json.load(open(os.path.join(out, "schedule_manifest.json")))
    assert {s["name"] for s in saved["schedules"]} == set(man_s) and saved["unmatched_schedules"] == []
    for f in ("subset_manifest.csv", "subset_summary.json", "metrics.csv", "per_image_metrics.csv", "token_counts.csv",
              "statistical_tests.csv", "run_config.json", "summary.md"):
        assert os.path.isfile(os.path.join(out, f)), f


def test_token_counts_follow_schedules(result):
    m, tc = result["metrics"], result["token_counts"]
    prim = m[m.analysis == "primary"]
    for (name, sc, seed), g in tc[tc.schedule.isin(prim.schedule)].groupby(["schedule", "scorer", "seed"]):
        sched = {int(k): v for k, v in json.loads(prim[prim.schedule == name].schedule_json.iloc[0]).items()}
        g = g.sort_values("after_block_layer")
        assert g.patch_tokens_in.tolist() == expected_processed_counts(sched, 12, 196)
        assert (g.total_tokens_out == g.patch_tokens_out + 1).all()                   # CLS preserved
        assert g[g.pruned_here].after_block_layer.tolist() == sorted(sched)


def test_per_image_aggregate_to_metrics_and_no_pruning_equivalence(result, wrapper, data):
    m, pi = result["metrics"], result["per_image"]
    g = pi.groupby(["schedule", "scorer", "seed"]).agg(a=("agreement", "sum"), c=("correct", "sum"), t5=("top5_correct", "sum"),
                                                       n=("image_idx", "count")).reset_index()
    j = m.merge(g, on=["schedule", "scorer", "seed"])
    assert len(j) == len(m) == len(g)
    assert (j.agreement_count == j.a).all() and (j.top1_count == j.c).all() and (j.top5_count == j.t5).all() and (j.n_images == j.n).all()
    b = pi[pi.schedule == "no_pruning"]
    with torch.no_grad():
        assert (b.compressed_pred.to_numpy() == wrapper(data[0]).argmax(-1).numpy()).all()
    assert (b.agreement == 1).all() and m[m.schedule == "no_pruning"].normalized_flops.iloc[0] == pytest.approx(1.0)
    assert {"image_id", "label", "baseline_pred", "compressed_pred", "correct", "agreement"} <= set(pi.columns)


def test_random_reproducible_for_fixed_seed(wrapper, data, tmp_path):
    def run(d):
        cfg = ClassDiverseConfig(pretrained=False, batch_size=4, budgets=[0.6], extra_seeds_random=[], secondary_first_layers=[],
                                 scorers_primary_layer=["random"], measure_latency=False, make_plots=False,
                                 previous_stage_ablation_dir=None, device="cpu", resume=False, output_dir=str(tmp_path / d))
        return run_class_diverse(cfg, wrapper=wrapper, data=data)["per_image"]
    pd.testing.assert_frame_equal(run("a").reset_index(drop=True), run("b").reset_index(drop=True))
