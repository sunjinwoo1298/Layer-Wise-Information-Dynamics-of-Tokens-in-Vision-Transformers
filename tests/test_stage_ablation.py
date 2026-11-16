"""Tests for the controlled pruning-stage ablation (random-init DeiT-Tiny, tiny synthetic data, CPU)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.progressive_eval import expected_processed_counts
from src.experiments.stage_ablation import (
    DEFAULT_STRUCTURES, StageConfig, build_statistical_tests, generate_schedules, holm_correction,
    run_stage_ablation, verify_subset,
)
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
    cfg = StageConfig(pretrained=False, batch_size=4, budgets=[0.5, 0.7], extra_seeds_random=[43],
                      scorers_primary_layer=["cls_attention", "random"], scorers_secondary_layer=["cls_attention"],
                      measure_latency=False, make_plots=False, previous_eval_dir=None, device="cpu",
                      output_dir=str(tmp_path_factory.mktemp("abl") / "out"))
    return run_stage_ablation(cfg, wrapper=wrapper, data=data)


def test_generation_controls_first_layer_stages_and_counts(est):
    for f in (2, 4):
        g = generate_schedules(est, f, [0.4, 0.5, 0.6, 0.7], DEFAULT_STRUCTURES[f], TOL)
        assert g["entries"]
        for e in g["entries"]:
            assert e["first_layer"] == f and min(e["schedule"]) == f                    # same first pruning layer
            ks = [e["schedule"][l] for l in sorted(e["schedule"])]
            assert len(ks) == e["stage_count"] and all(a > b for a, b in zip([196] + ks, ks)) and min(ks) >= 1
            assert max(e["schedule"]) < 12                                              # never the final block
            assert e["effective_layers"] == e["layers"]
            counts = expected_processed_counts(e["schedule"], 12, 196)
            assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_matched_groups_within_tolerance_and_same_baseline(est):
    g = generate_schedules(est, 2, [0.4, 0.5, 0.6, 0.7], DEFAULT_STRUCTURES[2], TOL)
    by = {}
    for e in g["entries"]:
        by.setdefault(e["group"], []).append(e)
        assert e["normalized_flops"] == pytest.approx(e["analytic_flops"] / est.baseline_flops)
    for gid, ents in by.items():
        anchor = next(e for e in ents if e["role"] == "anchor")
        assert anchor["stage_count"] == 1
        for e in ents:
            assert abs(e["normalized_flops"] - anchor["normalized_flops"]) <= TOL + 1e-12
    for grp in g["groups"]:
        assert grp["max_abs_diff_vs_anchor_pp"] <= 100 * TOL + 1e-9


def test_unreachable_budget_is_reported_not_loosened(est):
    g = generate_schedules(est, 2, [0.05], DEFAULT_STRUCTURES[2], TOL)  # below the one-stage floor
    assert not g["entries"] and g["unmatched"] and g["unmatched"][0]["reason"].startswith("one-stage anchor")


def test_holm_correction():
    p = [0.01, 0.04, 0.03, 0.5]
    adj = holm_correction(p)
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])   # sorted: .01*4, .03*3, .04*2 (monotone), .5*1
    assert all(a >= b for a, b in zip(adj, p)) and max(adj) <= 1.0


def test_manifest_matches_executed_schedules(result):
    man, m = result["manifest"], result["metrics"]
    man_s = {s["name"]: s["schedule"] for s in man["schedules"]}
    run_s = {r.schedule: json.loads(r.schedule_json) for r in m.itertuples()}
    assert {k: v for k, v in man_s.items() if k in run_s} == run_s and set(man_s) == set(run_s) | {"no_pruning"}
    saved = json.load(open(os.path.join(result["run_config"]["config"]["output_dir"], "schedule_manifest.json")))
    assert {s["name"] for s in saved["schedules"]} == set(man_s)
    assert saved["unmatched_schedules"] == []


def test_primary_schedules_share_first_layer_and_token_counts(result):
    m, tc = result["metrics"], result["token_counts"]
    prim = m[m.analysis == "primary"]
    assert set(prim.first_layer) == {2}
    for (name, sc, seed), g in tc[tc.schedule.isin(prim.schedule)].groupby(["schedule", "scorer", "seed"]):
        sched = json.loads(prim[prim.schedule == name].schedule_json.iloc[0])
        sched = {int(k): v for k, v in sched.items()}
        g = g.sort_values("after_block_layer")
        assert g.patch_tokens_in.tolist() == expected_processed_counts(sched, 12, 196)
        assert (g.patch_tokens_out.diff().dropna() <= 0).all() and (g.total_tokens_out == g.patch_tokens_out + 1).all()
        assert g[g.pruned_here].after_block_layer.tolist() == sorted(sched)


def test_per_image_aggregate_to_metrics_and_baseline(result, wrapper, data):
    m, pi = result["metrics"], result["per_image"]
    g = pi.groupby(["schedule", "scorer", "seed"]).agg(a=("agreement", "sum"), c=("correct", "sum"),
                                                       t5=("top5_correct", "sum"), n=("image_idx", "count")).reset_index()
    j = m.merge(g, on=["schedule", "scorer", "seed"])
    assert len(j) == len(m) == len(g)
    assert (j.agreement_count == j.a).all() and (j.top1_count == j.c).all() and (j.top5_count == j.t5).all()
    b = pi[pi.schedule == "no_pruning"]
    with torch.no_grad():
        assert (b.compressed_pred.to_numpy() == wrapper(data[0]).argmax(-1).numpy()).all()
    assert m[m.schedule == "no_pruning"].normalized_flops.iloc[0] == pytest.approx(1.0)


def test_statistics_families_and_correction(result):
    st = result["stats"]
    assert not st.empty
    prim = st[st.is_primary]
    assert set(prim.scorer) == {"cls_attention"} and set(prim.first_layer) == {2} and set(prim.seed) == {42}
    assert set(prim.metric) == {"agreement"} and (prim.family_size == len(prim)).all()
    assert (prim.p_holm >= prim.p_raw - 1e-12).all() and (prim.p_holm <= 1).all()
    assert (st[st.first_layer == 4].family.str.startswith("exploratory")).all()   # layer 4 never primary
    assert {"multi_vs_one_stage", "three_vs_two_prefix"} <= set(st.comparison_type)


def test_random_reproducible_for_fixed_seed(wrapper, data, tmp_path):
    def run(d):
        cfg = StageConfig(pretrained=False, batch_size=4, budgets=[0.6], extra_seeds_random=[], secondary_first_layers=[],
                          scorers_primary_layer=["random"], measure_latency=False, make_plots=False,
                          previous_eval_dir=None, device="cpu", output_dir=str(tmp_path / d))
        return run_stage_ablation(cfg, wrapper=wrapper, data=data)["per_image"]
    pd.testing.assert_frame_equal(run("a").reset_index(drop=True), run("b").reset_index(drop=True))


def test_subset_verification(tmp_path):
    man = pd.DataFrame({"image_idx": range(3), "label": [0, 1, 2], "image_sha1": ["a", "b", "c"]})
    man.to_csv(tmp_path / "image_manifest.csv", index=False)
    assert verify_subset(man, str(tmp_path), 1)["verified"]
    bad = man.copy(); bad.loc[1, "image_sha1"] = "x"
    with pytest.raises(RuntimeError, match="does NOT match"):
        verify_subset(bad, str(tmp_path), 1)
    with pytest.raises(RuntimeError, match="not found"):
        verify_subset(man, str(tmp_path / "missing"), 1)
