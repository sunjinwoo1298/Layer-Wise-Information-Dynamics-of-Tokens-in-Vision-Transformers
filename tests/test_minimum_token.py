"""Tests for the minimum final-token constraint study (random-init DeiT-Tiny, tiny synthetic data, CPU)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.class_diverse import DEFAULT_STRUCTURES, family_counts, generate_family_schedules
from src.experiments.minimum_token import (
    MinTokenConfig, build_statistical_tests, constrained_counts, generate_min_token_schedules, run_min_token,
    verify_reproduction, verify_same_subset,
)
from src.experiments.progressive_eval import expected_processed_counts
from src.experiments.stage_ablation import holm_correction
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper

TOL = 0.01
BUDGETS = [0.4, 0.5, 0.6, 0.7]
FLOORS = [8, 16, 24, 32]


@pytest.fixture(scope="module")
def est():
    return ComputeEstimator(12, 192, 3, 196, 1000, 1, (16, 16))


@pytest.fixture(scope="module")
def gen(est):
    return generate_min_token_schedules(est, 2, BUDGETS, DEFAULT_STRUCTURES[2], TOL, FLOORS)


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
    cfg = MinTokenConfig(pretrained=False, batch_size=4, budgets=[0.4, 0.6], extra_seeds_random=[43], secondary_scorers=["random"],
                         random_budgets=[0.4], kl_budgets=[], measure_latency=False, make_plots=False, previous_dir=None, device="cpu",
                         resume=False, output_dir=str(tmp_path_factory.mktemp("mt") / "out"))
    return run_min_token(cfg, wrapper=wrapper, data=data)


# ---- rules ---------------------------------------------------------------------------------------------------------

def test_prior_families_reproduced_exactly(est, gen):
    prev = generate_family_schedules(est, 2, BUDGETS, DEFAULT_STRUCTURES[2], TOL, ["geometric", "equal_drop"])
    cur = {e["name"]: e["schedule"] for e in gen["entries"] if e["family"] in ("geometric", "equal_drop", "shared")}
    assert cur == {e["name"]: e["schedule"] for e in prev["entries"]}
    assert gen["unmatched"] == prev["unmatched"]
    # original rule definitions untouched
    assert family_counts("geometric", 196, 0.5, 3) == [98, 49, 24] and family_counts("equal_drop", 196, 49.0, 3) == [147, 98, 49]
    # matches the saved class-diverse manifest when present
    d = "results/phase2_class_diverse_validation"
    if os.path.isfile(os.path.join(d, "schedule_manifest.json")):
        assert verify_reproduction(gen["entries"], d)["verified"]


def test_constrained_counts_rounding_and_floor():
    assert constrained_counts(196, 49.0, 16, 3) == [147, 98, 49]                      # floor not binding -> equal_drop counts
    assert constrained_counts(196, 96.0, 16, 2) == [100, 16]                          # last stage truncated at the floor
    assert constrained_counts(196, 49.25, 1, 1) == family_counts("equal_drop", 196, 49.25, 1)
    assert constrained_counts(196, 0.25, 8, 2) == [196, 196]                          # round-half-even: 195.75 -> 196, 195.5 -> 196
    assert constrained_counts(196, 97.5, 8, 2) == [98, 8] and constrained_counts(196, 97.5, 8, 2) == constrained_counts(196, 97.5, 8, 2)
    for F in FLOORS:
        assert all(k >= F for k in constrained_counts(196, 150.0, F, 3))


def test_floor_enforced_valid_and_matched(est, gen):
    assert gen["entries"]
    by = {}
    for e in gen["entries"]:
        by.setdefault(e["group"], []).append(e)
        assert e["first_layer"] == 2 and max(e["schedule"]) < 12
        ks = [e["schedule"][l] for l in sorted(e["schedule"])]
        assert all(a > b for a, b in zip([196] + ks, ks)) and min(ks) >= 1 and len(ks) == e["stage_count"]       # bounds, strictly decreasing
        counts = expected_processed_counts(e["schedule"], 12, 196)
        assert all(a >= b for a, b in zip(counts, counts[1:]))                                                  # non-increasing
        assert e["normalized_flops"] == pytest.approx(e["analytic_flops"] / est.baseline_flops)
        if e["family"] == "constrained_equal_drop":
            assert e["final_patch_tokens"] == ks[-1] >= e["floor"] and e["floor_satisfied"] and e["floor"] in FLOORS
            assert all(k >= e["floor"] for k in ks)
    for gid, ents in by.items():
        anchor = next(e for e in ents if e["role"] == "anchor")
        assert all(abs(e["normalized_flops"] - anchor["normalized_flops"]) <= TOL + 1e-12 for e in ents)


def test_infeasible_reported_and_never_violates_floor_or_tolerance(est, gen):
    assert gen["infeasible"]                                                                         # some combos really are infeasible
    names = {(e["budget_target"], e["structure"], e["floor"]) for e in gen["entries"] if e["family"] == "constrained_equal_drop"}
    for u in gen["infeasible"]:
        assert (u["budget"], u["structure"], u["floor"]) not in names                                # not silently included
        assert u["best_diff_pp"] is None or abs(u["best_diff_pp"]) > 100 * TOL
    # an impossible floor is infeasible rather than relaxed
    g = generate_min_token_schedules(est, 2, [0.4], {"two": [(2, 8)], "three": []}, TOL, [150])
    assert not [e for e in g["entries"] if e["family"] == "constrained_equal_drop"] and g["infeasible"][0]["floor"] == 150
    # unreachable budget reported
    g = generate_min_token_schedules(est, 2, [0.05], DEFAULT_STRUCTURES[2], TOL, FLOORS)
    assert not g["entries"] and g["unmatched"]


def test_identical_to_unconstrained_marked(gen):
    ident = {i["schedule"]: i["identical_to"] for i in gen["identical"]}
    by_name = {e["name"]: e for e in gen["entries"]}
    for s, t in ident.items():
        assert by_name[s]["schedule"] == by_name[t]["schedule"] and by_name[s]["alias_of"] == t
    # floors at or below the unconstrained final count cannot bind
    for e in gen["entries"]:
        if e["family"] == "constrained_equal_drop" and not e["alias_of"]:
            twin = next((x for x in gen["entries"] if x["family"] == "equal_drop" and x["group"] == e["group"] and x["structure"] == e["structure"]), None)
            if twin is not None:
                assert twin["final_patch_tokens"] < e["floor"] or twin["schedule"] != e["schedule"]


# ---- subset verification -------------------------------------------------------------------------------------------

def test_subset_verification(tmp_path):
    man = pd.DataFrame({"position": range(3), "image_id": ["a", "b", "c"], "label": [0, 1, 2], "image_sha1": ["x", "y", "z"]})
    man.to_csv(tmp_path / "subset_manifest.csv", index=False)
    json.dump({"manifest_sha256": "h"}, open(tmp_path / "subset_summary.json", "w"))
    assert verify_same_subset(man, {"manifest_sha256": "h"}, str(tmp_path))["verified"]
    with pytest.raises(RuntimeError, match="does NOT match"):
        verify_same_subset(man, {"manifest_sha256": "other"}, str(tmp_path))
    bad = man.copy(); bad.loc[1, "image_sha1"] = "q"
    with pytest.raises(RuntimeError, match="does NOT match"):
        verify_same_subset(bad, {"manifest_sha256": "h"}, str(tmp_path))
    with pytest.raises(RuntimeError, match="not found"):
        verify_same_subset(man, {"manifest_sha256": "h"}, str(tmp_path / "missing"))


# ---- end-to-end ----------------------------------------------------------------------------------------------------

def test_manifest_matches_executed_schedules(result):
    man, m = result["manifest"], result["metrics"]
    man_s = {s["name"]: s["schedule"] for s in man["schedules"]}
    run_s = {r.schedule: json.loads(r.schedule_json) for r in m.itertuples()}
    assert run_s == {k: v for k, v in man_s.items() if k in run_s} and set(man_s) == set(run_s) | {"no_pruning"}
    out = result["run_config"]["config"]["output_dir"]
    saved = json.load(open(os.path.join(out, "schedule_manifest.json")))
    assert {s["name"] for s in saved["schedules"]} == set(man_s) and "infeasible_combinations" in saved
    for f in ("metrics.csv", "per_image_metrics.csv", "token_counts.csv", "statistical_tests.csv", "run_config.json", "summary.md"):
        assert os.path.isfile(os.path.join(out, f)), f
    assert all(s["floor_satisfied"] for s in man["schedules"] if s["allocation_family"] == "constrained_equal_drop")


def test_alias_rows_copy_executed_results(result):
    m = result["metrics"]
    al = m[m.is_alias]
    assert len(al) > 0
    for r in al.itertuples():
        src = m[(m.schedule == r.executed_as) & (m.scorer == r.scorer) & (m.seed == r.seed) & ~m.is_alias]
        assert len(src) == 1 and src.agreement_count.iloc[0] == r.agreement_count and src.top1_count.iloc[0] == r.top1_count


def test_token_counts_cls_retention_and_floor(result):
    m, tc = result["metrics"], result["token_counts"]
    ex = m[~m.is_alias & (m.scorer == "cls_attention")]
    for (name, sc, seed), g in tc[tc.schedule.isin(ex.schedule) & (tc.scorer == "cls_attention")].groupby(["schedule", "scorer", "seed"]):
        sched = {int(k): v for k, v in json.loads(ex[ex.schedule == name].schedule_json.iloc[0]).items()}
        g = g.sort_values("after_block_layer")
        assert g.patch_tokens_in.tolist() == expected_processed_counts(sched, 12, 196)
        assert (g.total_tokens_out == g.patch_tokens_out + 1).all()                                          # CLS retained
        fl = ex[ex.schedule == name].floor.iloc[0]
        if not pd.isna(fl):
            assert g.patch_tokens_out.iloc[-1] >= fl


def test_per_image_aggregate_and_no_pruning_equivalence(result, wrapper, data):
    m, pi = result["metrics"], result["per_image"]
    mm = m[~m.is_alias]
    g = pi.groupby(["schedule", "scorer", "seed"]).agg(a=("agreement", "sum"), c=("correct", "sum"), t5=("top5_correct", "sum"),
                                                       n=("image_idx", "count")).reset_index()
    j = mm.merge(g, on=["schedule", "scorer", "seed"])
    assert len(j) == len(mm) == len(g)
    assert (j.agreement_count == j.a).all() and (j.top1_count == j.c).all() and (j.top5_count == j.t5).all() and (j.n_images == j.n).all()
    b = pi[pi.schedule == "no_pruning"]
    with torch.no_grad():
        assert (b.compressed_pred.to_numpy() == wrapper(data[0]).argmax(-1).numpy()).all()
    assert (b.agreement == 1).all() and m[m.schedule == "no_pruning"].normalized_flops.iloc[0] == pytest.approx(1.0)


def test_statistics_families_and_correction(result):
    st = result["stats"]
    assert not st.empty
    prim = st[st.is_primary]
    assert set(prim.family) <= {"PRIMARY_cons_vs_uncons_agreement", "PRIMARY_cons_vs_uncons_top1"}
    assert set(prim.comparison_type) == {"constrained_vs_unconstrained"} and set(prim.scorer) == {"cls_attention"} and set(prim.seed) == {42}
    assert set(prim.floor_a) <= {16, 24, 32}
    for fam, g in prim.groupby("family"):
        assert (g.family_size == len(g)).all() and (g.p_holm >= g.p_raw - 1e-12).all() and (g.p_holm <= 1).all()
    sec = st[st.is_predefined_secondary]
    assert set(sec.comparison_type) <= {"constrained_vs_geometric"}
    assert (st[(st.floor_a == 8) | (st.scorer != "cls_attention")].family.str.startswith("exploratory")).all()
    assert (st.flops_diff_pp.abs() <= 100 * TOL + 1e-9).all()
    # identical-to-unconstrained schedules are not tested as constrained-vs-unconstrained pairs
    ali = {e["name"] for e in result["entries"] if e.get("alias_of")}
    assert not set(st[st.comparison_type.isin(["constrained_vs_unconstrained", "constrained_vs_geometric"])].schedule_a) & ali


def test_paired_stat_matches_manual(result):
    st, pi = result["stats"], result["per_image"]
    r = st[st.is_primary & (st.metric == "agreement")].iloc[0]
    get = lambda s: pi[(pi.schedule == s) & (pi.scorer == r.scorer) & (pi.seed == r.seed)].sort_values("image_idx").agreement.to_numpy().astype(bool)
    a, b = get(r.schedule_a), get(r.schedule_b)
    assert r["diff"] == pytest.approx(a.mean() - b.mean()) and r.a_only == int((a & ~b).sum()) and r.b_only == int((~a & b).sum())


def test_holm():
    assert holm_correction([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_random_reproducible_for_fixed_seed(wrapper, data, tmp_path):
    def run(d):
        cfg = MinTokenConfig(pretrained=False, batch_size=4, budgets=[0.6], extra_seeds_random=[], secondary_scorers=["random"], random_budgets=[0.6],
                             secondary_floors=[16], secondary_structures=["2-6"], kl_budgets=[], measure_latency=False, make_plots=False,
                             previous_dir=None, device="cpu", resume=False, output_dir=str(tmp_path / d))
        return run_min_token(cfg, wrapper=wrapper, data=data)["per_image"]
    a, b = run("a"), run("b")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
