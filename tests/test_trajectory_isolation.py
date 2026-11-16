"""Tests for the final-token vs trajectory isolation experiment (random-init DeiT-Tiny, tiny synthetic data, CPU)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.class_diverse import DEFAULT_STRUCTURES, family_counts, generate_family_schedules
from src.experiments.minimum_token import constrained_counts, verify_same_subset
from src.experiments.progressive_eval import schedule_flops
from src.experiments.stage_ablation import holm_correction
from src.experiments.trajectory_isolation import (
    CONTROL_FAMILY, TrajConfig, build_control_entries, build_isolation_tests, enumerate_control_candidates, generate_isolation_schedules,
    prepare_isolation, run_isolation, select_controls, verify_manifest_artifact, verify_prior_schedules,
)
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper

TOL = 0.01
PREV = "./results/phase2_minimum_token_ablation"
have_prev = os.path.isfile(os.path.join(PREV, "schedule_manifest.json"))


@pytest.fixture(scope="module")
def est():
    return ComputeEstimator(12, 192, 3, 196, 1000, 1, (16, 16))


@pytest.fixture(scope="module")
def gen(est):
    return generate_isolation_schedules(est, TrajConfig())


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ViTWrapper("deit_tiny_patch16_224", pretrained=False, device="cpu")


def _prior():
    man = json.load(open(os.path.join(PREV, "schedule_manifest.json")))
    return {s["name"]: s for s in man["schedules"]}


@pytest.mark.skipif(not have_prev, reason="previous minimum-token artifacts absent")
@pytest.mark.parametrize("fam", ["geometric", "equal_drop", "constrained_equal_drop"])
def test_prior_families_reproduce_saved_schedules(gen, fam):          # tests 1-3
    prior = _prior()
    cur = [e for e in gen["entries"] if e["family"] == fam]
    assert cur
    for e in cur:
        assert e["name"] in prior and {int(k): v for k, v in prior[e["name"]]["schedule"].items()} == e["schedule"]
        assert prior[e["name"]]["allocation_family"] == fam
    n_prior = [s for s in prior.values() if s["allocation_family"] == fam and s["structure"] in ("2-6-10", "2-8-10")]
    assert len(n_prior) == len(cur)


def test_prior_families_match_independent_regeneration(est, gen):     # same rules recomputed from the formulas
    base = generate_family_schedules(est, 2, [0.4, 0.5, 0.6, 0.7], {"two": [], "three": [(2, 6, 10), (2, 8, 10)]}, TOL)
    cur = {e["name"]: e["schedule"] for e in gen["entries"] if e["family"] in ("geometric", "equal_drop")}
    assert cur == {e["name"]: e["schedule"] for e in base["entries"] if e["family"] in ("geometric", "equal_drop")}
    for e in gen["entries"]:
        if e["family"] == "constrained_equal_drop":
            assert list(e["schedule"].values()) == constrained_counts(196, e["D"], e["floor"], 3)
        if e["family"] == "equal_drop":
            assert list(e["schedule"].values()) == family_counts("equal_drop", 196, e["param"] if "param" in e else
                                                                 next(D for D in np.arange(0.25, 195.0, 0.25)
                                                                      if family_counts("equal_drop", 196, float(D), 3) == list(e["schedule"].values())), 3)


@pytest.mark.skipif(not have_prev, reason="previous minimum-token artifacts absent")
def test_verify_prior_schedules_passes_and_detects_changes(gen):
    r = verify_prior_schedules(gen["entries"], gen["infeasible"], PREV, ["2-6-10", "2-8-10"])
    assert r["verified"]
    bad = [dict(e) for e in gen["entries"]]
    g = next(e for e in bad if e["family"] == "geometric")
    g["schedule"] = {k: v + 1 for k, v in g["schedule"].items()}
    with pytest.raises(RuntimeError):
        verify_prior_schedules(bad, gen["infeasible"], PREV, ["2-6-10", "2-8-10"])


def test_controls_preserve_final_count_layers_and_initial_count(gen, est):      # tests 4, 5
    geo = {e["name"]: e for e in gen["entries"] if e["family"] == "geometric"}
    ctrl = [e for e in gen["entries"] if e["family"] == CONTROL_FAMILY]
    assert ctrl
    for c in ctrl:
        g = geo[c["control_of"]]
        assert list(c["schedule"].keys()) == list(g["schedule"].keys())          # stage locations
        assert list(c["schedule"].values())[-1] == list(g["schedule"].values())[-1]          # final count exactly
        assert c["final_patch_tokens"] == g["final_patch_tokens"]
        assert est.N > list(c["schedule"].values())[0]                           # initial count N unchanged (strictly pruned)
        assert c["stage_count"] == g["stage_count"] and c["structure"] == g["structure"]


def test_control_trajectories_valid_and_distinct(gen, est):                      # tests 6, 7
    geo = {e["name"]: e for e in gen["entries"] if e["family"] == "geometric"}
    seen = set()
    for c in [e for e in gen["entries"] if e["family"] == CONTROL_FAMILY]:
        v = list(c["schedule"].values())
        assert all(a > b for a, b in zip([196] + v, v)) and min(v) >= 1
        assert v != list(geo[c["control_of"]]["schedule"].values())               # identical-to-geometric never a control
        assert c["trajectory_dist"] >= TrajConfig().control_min_deviation
        assert (c["control_of"], tuple(v)) not in seen
        seen.add((c["control_of"], tuple(v)))
    cands = gen["candidates"]
    ident = [x for x in cands if x["status"] == "identical_to_geometric"]
    assert ident and all(x["vector"] == x["geometric_vector"] and not x["selected"] for x in ident)   # recorded, not counted
    assert all(x["status"] != "identical_to_geometric" for x in cands if x["selected"])


def test_flops_matching_uses_existing_accounting_and_tolerance(gen, est):        # test 8
    geo = {e["name"]: e for e in gen["entries"] if e["family"] == "geometric"}
    for c in [e for e in gen["entries"] if e["family"] == CONTROL_FAMILY]:
        f = schedule_flops(est, c["schedule"])["normalized_flops"]
        assert f == pytest.approx(c["normalized_flops"], abs=1e-12)
        assert abs(f - geo[c["control_of"]]["normalized_flops"]) <= TOL + 1e-12
    for x in gen["candidates"]:
        if x["status"] == "flops_out_of_tolerance":
            assert abs(x["flops_diff_vs_geometric_pp"]) > 100 * TOL
        if x["status"].startswith("feasible"):
            assert abs(x["flops_diff_vs_geometric_pp"]) <= 100 * TOL + 1e-9


def test_infeasible_controls_recorded_not_relaxed(est):                          # test 9
    geo = next(e for e in generate_isolation_schedules(est, TrajConfig(budgets=[0.5], patterns=["2-6-10"]))["entries"] if e["family"] == "geometric")
    # (a) tolerance far too tight: nothing selected, every constraint kept as is
    tight = build_control_entries(est, [geo], 1e-9, 8)
    assert tight["entries"] == [] and len(tight["infeasible"]) == 1 and "no distinct candidate" in tight["infeasible"][0]["reason"]
    # (b) final count so high that no strictly decreasing intermediate vector exists
    fake = {**geo, "name": "fake", "schedule": {2: 195, 6: 195 - 0, 10: 195}, "layers": [2, 6, 10]}
    fake["schedule"] = {2: 195, 6: 194, 10: 193}
    res = build_control_entries(est, [fake], TOL, 8)
    assert res["entries"] == [] and res["infeasible"] and all(c["final_tokens"] == 193 for c in res["candidates"])
    # (c) two-stage / one-stage geometric has no intermediate stage to vary
    one = {**geo, "name": "one", "schedule": {2: 100}, "layers": [2]}
    assert build_control_entries(est, [one], TOL, 8)["infeasible"]


def test_select_controls_is_accuracy_blind_and_deterministic(est):
    geo = next(e for e in generate_isolation_schedules(est, TrajConfig(budgets=[0.5], patterns=["2-6-10"]))["entries"] if e["family"] == "geometric")
    c = enumerate_control_candidates(est, geo, TOL, 8)
    a, b = select_controls(c), select_controls([dict(x) for x in c])
    assert [x["vector"] for x in a] == [x["vector"] for x in b] and 1 <= len(a) <= 4
    assert {x["control_side"] for x in a} <= {"retaining", "steeper"}


@pytest.mark.skipif(not have_prev, reason="previous minimum-token artifacts absent")
def test_exact_prior_manifest_and_image_order_used():                            # test 10
    a = verify_manifest_artifact(PREV)
    assert a["num_images"] == 2000 and len(a["manifest_sha256_full"]) == 64
    assert a["manifest_sha256_full"].startswith("56279dc3") and a["manifest_sha256_full"].endswith("263aa67")
    man = pd.read_csv(os.path.join(PREV, "subset_manifest.csv"))
    summ = json.load(open(os.path.join(PREV, "subset_summary.json")))
    assert verify_same_subset(man, summ, PREV)["verified"]
    shuffled = man.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError):
        verify_same_subset(shuffled, summ, PREV)
    with pytest.raises(RuntimeError):
        verify_manifest_artifact("./results/does_not_exist")


def _tiny_cfg(out, **kw):
    return TrajConfig(parquet_dir="x", subset_cache="x", budgets=[0.5], patterns=["2-6-10"], floors=[16, 32], primary_floors=[16, 32],
                      prev_min_token_dir=None, prev_class_diverse_dir=None, output_dir=str(out), measure_latency=True, latency_runs=1,
                      latency_batches=1, batch_size=4, resume=False, make_plots=True, **kw)


@pytest.fixture(scope="module")
def tiny_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(8, 3, 224, 224, generator=g), torch.randint(0, 1000, (8,), generator=g)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, wrapper, tiny_data):
    out = tmp_path_factory.mktemp("iso")
    return run_isolation(_tiny_cfg(out), wrapper=wrapper, data=tiny_data), out


def test_baseline_equivalence_and_outputs(tiny_run):                             # test 11 + deliverables
    r, out = tiny_run
    assert not r["failures"]
    base = r["metrics"][r["metrics"].scorer == "none"].iloc[0]
    assert base.agreement == 1.0 and base.agreement_count == 8
    for f in ("metrics.csv", "metrics.json", "per_image_predictions.csv.gz", "schedule_table.csv", "control_candidates.csv.gz",
              "statistical_tests.csv", "latency.csv", "run_manifest.json", "result.md", "subset_manifest.csv", "token_counts.csv"):
        assert os.path.isfile(os.path.join(out, f)), f
    for p in ("1_token_count_by_stage", "2_agreement_vs_flops", "3_top1_vs_flops", "4_paired_agreement_differences", "5_final_tokens_vs_agreement"):
        assert os.path.isfile(os.path.join(out, "plots", p + ".png")), p
    rm = json.load(open(os.path.join(out, "run_manifest.json")))
    assert rm["dataset"]["manifest_sha256_full"] and rm["seed"] == 42 and rm["flops_method"]
    tab = pd.read_csv(os.path.join(out, "schedule_table.csv"))
    assert {"executed", "infeasible"} & set(tab.status)
    pi = pd.read_csv(os.path.join(out, "per_image_predictions.csv.gz"))
    assert {"compressed_pred", "baseline_pred", "top5_preds", "agreement", "correct"} <= set(pi.columns)


def test_stats_families_sign_and_holm(tiny_run):
    r, _ = tiny_run
    st = r["stats"]
    assert len(st) and set(st.comparison) >= {"geo_vs_ctrl", "geo_vs_cons"}
    assert (st.flops_diff_pp.abs() <= 100 * TOL + 1e-9).all() and st.flops_tolerance_satisfied.all()
    c = st[st.comparison == "geo_vs_ctrl"]
    assert (c.family_a == "geometric").all() and (c.same_final_count).all()
    for _, g in st.groupby("family"):
        assert np.allclose(g.p_holm.to_numpy(), holm_correction(g.p_raw.tolist()))
    assert (st[st.comparison == "cons_vs_unc"].family.str.contains("cons_vs_unc")).all()


def test_paired_stats_match_per_image_data(tiny_run):
    r, _ = tiny_run
    pi, st = r["per_image"], r["stats"]
    row = st[(st.comparison == "geo_vs_ctrl") & (st.metric == "agreement")].iloc[0]
    a = pi[pi.schedule == row.schedule_a].sort_values("image_idx").agreement.to_numpy()
    b = pi[pi.schedule == row.schedule_b].sort_values("image_idx").agreement.to_numpy()
    assert row["diff"] == pytest.approx(a.mean() - b.mean()) and row.count_a == a.sum() and row.count_b == b.sum()


def test_repeated_runs_reproducible(tiny_run, wrapper, tiny_data, tmp_path):     # test 12
    r1, _ = tiny_run
    r2 = run_isolation(_tiny_cfg(tmp_path), wrapper=wrapper, data=tiny_data)
    a = r1["per_image"].sort_values(["schedule", "image_idx"]).reset_index(drop=True)
    b = r2["per_image"].sort_values(["schedule", "image_idx"]).reset_index(drop=True)
    assert (a.compressed_pred.to_numpy() == b.compressed_pred.to_numpy()).all()
    assert (a.schedule.to_numpy() == b.schedule.to_numpy()).all()


def test_result_md_states_isolation_status(tiny_run):
    _, out = tiny_run
    txt = open(os.path.join(out, "result.md"), encoding="utf-8").read()
    assert "Final-count-matched controls" in txt and "PRE_geo_vs_ctrl" in txt and "Limitations" in txt
