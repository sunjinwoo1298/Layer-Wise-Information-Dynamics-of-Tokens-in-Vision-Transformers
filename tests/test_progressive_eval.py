"""
Tests for the Phase 2 progressive-vs-single evaluation runner (random-init DeiT-Tiny, tiny synthetic data, CPU).
Properties tested are weight-independent.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.progressive_eval import (
    DEFAULT_SCHEDULES, EvalConfig, build_schedule_set, expected_processed_counts, find_matched_pairs,
    paired_diff_stats, run_evaluation,
)
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper

SCHEDS = {
    "no_pruning": {}, "early_single": {2: 98}, "middle_single": {6: 98},
    "early_progressive": {2: 98, 4: 49}, "delayed_progressive": {2: 196, 4: 98, 8: 49},
}


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ViTWrapper("deit_tiny_patch16_224", pretrained=False, device="cpu")


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(3)
    return torch.randn(8, 3, 224, 224, generator=g), torch.randint(0, 1000, (8,), generator=g)


def _cfg(tmp_path, **kw):
    base = dict(pretrained=False, batch_size=4, scorers=["random", "cls_attention"], schedules=SCHEDS,
                extra_seeds_random=[43], measure_latency=False, make_plots=False,
                output_dir=str(tmp_path / "out"), device="cpu")
    base.update(kw)
    return EvalConfig(**base)


@pytest.fixture(scope="module")
def result(wrapper, data, tmp_path_factory):
    return run_evaluation(_cfg(tmp_path_factory.mktemp("eval")), wrapper=wrapper, data=data)


def test_no_pruning_matches_original(result, wrapper, data):
    m = result["metrics"]
    b = m[m.schedule == "no_pruning"].iloc[0]
    assert b.scorer == "none" and b.agreement_count == len(data[0]) and b.normalized_flops == pytest.approx(1.0)
    pi = result["per_image"]
    nb = pi[pi.schedule == "no_pruning"]
    with torch.no_grad():
        orig = wrapper(data[0]).argmax(-1).numpy()
    assert (nb.compressed_pred.to_numpy() == orig).all()  # (logit equality asserted inside the runner at <=1e-4)


def test_manifest_matches_schedules_run(result, tmp_path_factory):
    out = result["run_config"]["config"]["output_dir"]
    man = json.load(open(os.path.join(out, "schedule_manifest.json")))
    man_s = {s["name"]: s["schedule"] for s in man["schedules"]}
    m = result["metrics"]
    run_s = {r.schedule: json.loads(r.schedule_json) for r in m.itertuples()}
    assert run_s == man_s
    assert set(man_s) == set(m.schedule.unique())
    assert "1-based" in man["layer_numbering_and_semantics"]


def test_token_counts_match_schedules(result):
    tc = result["token_counts"]
    for e in result["entries"]:
        exp_in = expected_processed_counts(e["schedule"], 12, 196)
        for (_, sc, seed), g in tc[tc.schedule == e["name"]].groupby(["schedule", "scorer", "seed"]):
            g = g.sort_values("after_block_layer")
            assert g.patch_tokens_in.tolist() == exp_in
            assert (g.total_tokens_out == g.patch_tokens_out + 1).all()  # CLS kept
            assert (g.patch_tokens_out.diff().dropna() <= 0).all()
    d = tc[(tc.schedule == "delayed_progressive")]
    assert d[d.after_block_layer == 2].patch_tokens_out.iloc[0] == 196  # 196 key is a no-op


def test_flops_normalised_against_same_baseline(result, wrapper):
    est = ComputeEstimator(12, 192, 3, 196, 1000, 1, wrapper.patch_size)
    m = result["metrics"]
    assert (m.analytic_flops / est.baseline_flops - m.normalized_flops).abs().max() < 1e-12
    assert m[m.schedule == "no_pruning"].analytic_flops.iloc[0] == pytest.approx(est.baseline_flops)
    assert ((m.flops_savings_pct - 100 * (1 - m.normalized_flops)).abs() < 1e-9).all()
    assert (m[m.schedule != "no_pruning"].normalized_flops < 1.0).all()


def test_per_image_aggregates_to_metrics(result):
    m, pi = result["metrics"], result["per_image"]
    g = pi.groupby(["schedule", "scorer", "seed"]).agg(a=("agreement", "sum"), c=("correct", "sum"),
                                                       t5=("top5_correct", "sum"), n=("image_idx", "count")).reset_index()
    j = m.merge(g, on=["schedule", "scorer", "seed"])
    assert len(j) == len(m) == len(g)
    assert (j.agreement_count == j.a).all() and (j.top1_count == j.c).all() and (j.top5_count == j.t5).all()
    assert (j.n_images == j.n).all()
    assert np.allclose(j.agreement, j.a / j.n)


def test_matched_pairs_within_tolerance(wrapper):
    cfg = EvalConfig(pretrained=False, schedules=DEFAULT_SCHEDULES)
    est = ComputeEstimator(12, 192, 3, 196, 1000, 1, wrapper.patch_size)
    entries = build_schedule_set(cfg, est)
    pairs = find_matched_pairs(entries, cfg.match_tolerance)
    assert pairs, "expected matched pairs"
    for p in pairs:
        assert abs(p["progressive_norm_flops"] - p["single_norm_flops"]) <= cfg.match_tolerance + 1e-12
        assert abs(p["flops_diff_pp"]) <= p["tolerance_pp"] + 1e-9
    names = {e["name"] for e in entries}
    for req in DEFAULT_SCHEDULES:
        assert req in names
    # auto-added controls are real single-layer schedules; no duplicate schedules
    keys = [tuple(sorted(e["schedule"].items())) for e in entries]
    assert len(keys) == len(set(keys))
    assert all(e["type"] == "single" for e in entries if e["origin"].startswith("auto"))


def test_matched_comparison_uses_same_scorer_and_seed(result):
    mt = result["matched"]
    assert not mt.empty
    assert set(mt.scorer) <= {"random", "cls_attention"}
    assert (mt.n_images == 8).all()


def test_same_seed_and_order_reproduce_random(wrapper, data, tmp_path):
    a = run_evaluation(_cfg(tmp_path / "a", scorers=["random"]), wrapper=wrapper, data=data)["per_image"]
    b = run_evaluation(_cfg(tmp_path / "b", scorers=["random"]), wrapper=wrapper, data=data)["per_image"]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_paired_diff_stats():
    a = np.array([1, 1, 1, 0, 0, 1] * 10); b = np.array([1, 0, 0, 0, 1, 1] * 10)
    s = paired_diff_stats(a, b)
    assert s["a_only"] == 20 and s["b_only"] == 10 and s["diff"] == pytest.approx((40 - 30) / 60)
    assert 0 <= s["mcnemar_p"] <= 1 and s["boot_ci_low"] <= s["diff"] <= s["boot_ci_high"]
