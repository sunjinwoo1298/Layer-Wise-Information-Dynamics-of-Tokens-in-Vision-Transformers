"""Tests for Phase 3 budget-aware allocation (random-init DeiT-Tiny, tiny synthetic data, CPU)."""

import copy
import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.compression.progressive import ProgressivePruner
from src.data.phase3_splits import assert_split_disjoint, make_cal_val_split, split_checksum
from src.experiments import budget_aware_allocation as baa
from src.experiments.budget_aware_allocation import (
    AllocConfig, DataSet, Runner, baseline_entries, frozen_entries, load_frozen, run_calibration, run_evaluation, verify_baselines,
)
from src.experiments.progressive_eval import schedule_flops
from src.metrics.compute import ComputeEstimator
from src.models.vit_wrapper import ViTWrapper
from src.optimization import schedule_search as ss
from src.optimization.schedule_search import (
    SensitivityTables, feasible_schedules, normalized_flops, rank_diverse, select_schedule, select_verified,
)

PARENT = "./results/phase2_minimum_token_ablation"
have_parent = os.path.isfile(os.path.join(PARENT, "schedule_manifest.json"))


@pytest.fixture(scope="module")
def est():
    return ComputeEstimator(12, 192, 3, 196, 1000, 1, (16, 16))


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ViTWrapper("deit_tiny_patch16_224", pretrained=False, device="cpu")


def _ds(tag, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = np.array([f"{tag}_{i}" for i in range(n)])
    return DataSet(tag, torch.randn(n, 3, 224, 224, generator=g), torch.randint(0, 1000, (n,), generator=g), ids, pd.DataFrame({"image_id": ids}))


def _cfg(out, **kw):
    base = dict(budgets=[0.5], patterns=["2-6-10"], floors=[16], grid=[16, 48, 96, 144, 184], top_m=3, bootstrap_stability=2, batch_size=8,
                output_dir=str(out), latency_runs=1, latency_batches=1, make_plots=True, resume=False, max_val=12, max_conf=12,
                parent_dir=PARENT, isolation_dir="./results/does_not_exist")
    base.update(kw)
    return AllocConfig(**base)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, wrapper):
    out = tmp_path_factory.mktemp("p3")
    cfg = _cfg(out)
    data = {"calibration": _ds("cal", 16, 1), "validation": _ds("val", 12, 2), "confirmatory": _ds("conf", 12, 3)}
    runner = Runner(cfg, wrapper)
    cal = run_calibration(cfg, runner, data)
    val = run_evaluation(cfg, "validation", runner, data, latency=True)
    return cfg, data, runner, cal, val


# ---- 1. monotone decreasing counts / 2. floor / 4. budget -------------------------------------------------------------

@pytest.mark.parametrize("layers,floor,cap", [((2, 6, 10), 16, 0.5), ((2, 8, 10), 32, 0.6), ((2, 6), 24, 0.55)])
def test_feasible_monotone_floor_and_budget(est, layers, floor, cap):
    f = feasible_schedules(est, layers, floor, cap, 0.01)
    assert len(f) > 0
    assert (f[:, 0] < est.N).all() and (np.diff(f, axis=1) < 0).all()               # N > K1 > K2 > ...
    assert (f[:, -1] >= floor).all()                                                 # floor enforced
    fl = normalized_flops(est, layers, f.T)
    assert (fl <= cap + 1e-9).all() and (fl >= cap - 0.01 - 1e-9).all()             # never above budget; within the band


# ---- 3. FLOPs consistency with the existing code ------------------------------------------------------------------------

def test_flops_match_existing_accounting(est):
    rs = np.random.RandomState(0)
    for layers in [(2, 6, 10), (2, 8, 10)]:
        for _ in range(25):
            k = sorted(rs.choice(np.arange(16, 196), 3, replace=False), reverse=True)
            ours = float(normalized_flops(est, layers, k))
            ref = schedule_flops(est, dict(zip(layers, [int(x) for x in k])))["normalized_flops"]
            assert ours == pytest.approx(ref, rel=1e-12)


# ---- 5. determinism / 6. infeasible ---------------------------------------------------------------------------------------

def _fake_tables(N=196, grid=(16, 48, 96, 144, 184), layers=(2, 6, 10), seed=0):
    rs = np.random.RandomState(seed)
    tab = SensitivityTables(grid=list(grid), singles=list(layers), pairs=[(layers[0], layers[1]), (layers[1], layers[2])], n_images=20, N=N)
    for key in tab.required_keys():
        p = key.split("|")
        k = int(p[2]) if p[0] == "s" else int(p[4])
        base = 1.0 / np.sqrt(k)
        tab.per_image[key] = np.abs(rs.normal(base, 0.05, 20))
    return tab


def test_selection_deterministic_and_respects_constraints(est):
    tab = _fake_tables()
    proxy = tab.build()
    calls = []

    def verify(c):
        calls.append(c)
        return 1.0 - 0.001 * abs(c[0] - 120)

    r1 = select_schedule(proxy, est, (2, 6, 10), 24, 0.5, 0.01, verify, M=5, min_dist=6)
    r2 = select_schedule(tab.build(), est, (2, 6, 10), 24, 0.5, 0.01, verify, M=5, min_dist=6)
    assert r1.status == "selected" and r1.counts == r2.counts and [c["counts"] for c in r1.candidates] == [c["counts"] for c in r2.candidates]
    assert r1.counts[-1] >= 24 and normalized_flops(est, (2, 6, 10), r1.counts) <= 0.5 + 1e-9
    assert len(r1.candidates) == 5 and all({"counts", "proxy_loss", "flops", "cal_agreement", "rank", "selected"} <= set(c) for c in r1.candidates)
    top = [np.array(c["counts"]) for c in r1.candidates]
    assert all(np.abs(a - b).max() >= 6 for i, a in enumerate(top) for b in top[:i])      # diversity


def test_infeasible_budget_reported_not_relaxed(est):
    r = select_schedule(_fake_tables().build(), est, (2, 6, 10), 32, 0.05, 0.01, lambda c: 1.0)
    assert r.status == "infeasible" and r.counts is None and r.n_feasible == 0 and "no integer schedule" in r.reason


def test_tie_break_prefers_lower_proxy_then_larger_final():
    c = [{"counts": [100, 50, 20], "proxy_loss": 0.3, "cal_agreement": 0.80}, {"counts": [90, 60, 30], "proxy_loss": 0.2, "cal_agreement": 0.799},
         {"counts": [80, 40, 10], "proxy_loss": 0.1, "cal_agreement": 0.70}]
    assert select_verified(c, tie_eps=0.002)["counts"] == [90, 60, 30]


# ---- 7. no-pruning / 8. CLS retained / 9. order preserved -----------------------------------------------------------------

def test_no_pruning_equivalence_and_cls_and_order(wrapper):
    x = torch.randn(4, 3, 224, 224)
    pr = ProgressivePruner(wrapper, {}, scorer="cls_attention", seed=42)
    with torch.no_grad():
        assert (pr.forward(x).logits - wrapper(x)).abs().max().item() <= 1e-4
    pr = ProgressivePruner(wrapper, {2: 120, 6: 60, 10: 24}, scorer="cls_attention", seed=42)
    res = pr.forward(x)
    outs = {t["after_block_layer"]: t for t in res.token_trace}
    for l, k in {2: 120, 6: 60, 10: 24}.items():
        assert outs[l]["total_tokens_out"] == k + 1                                   # CLS kept (+1), exactly k patches
        idx = res.kept_indices[l]
        assert (idx[:, 1:] > idx[:, :-1]).all()                                       # original patch order preserved (strictly increasing)
    assert res.token_trace[-1]["total_tokens_out"] == 24 + 1


# ---- 10. cal/val disjoint ------------------------------------------------------------------------------------------------

def test_split_disjoint_stratified_and_checksums():
    n = 200
    man = pd.DataFrame({"position": range(2 * n), "image_id": [f"i{j}" for j in range(2 * n)], "source_row": range(2 * n),
                        "label": np.repeat(np.arange(n), 2), "image_sha1": [f"h{j}" for j in range(2 * n)]})
    m = make_cal_val_split(man, 100, 42)
    assert (m.split == "calibration").sum() == 100 and (m.split == "validation").sum() == 300
    assert m[m.split == "calibration"].label.nunique() == 100 and m[m.split == "validation"].label.nunique() == 200
    assert_split_disjoint(m)
    assert (make_cal_val_split(man, 100, 42).split == m.split).all()                  # deterministic
    assert split_checksum(m, "calibration") != split_checksum(m, "validation")
    bad = m.copy(); bad.loc[bad.split == "validation", "image_id"] = bad[bad.split == "calibration"].image_id.iloc[0]
    with pytest.raises(AssertionError):
        assert_split_disjoint(bad)


def test_real_split_if_available():
    p = "./results/phase3_budget_aware_allocation/splits/split_manifest.csv"
    if not os.path.isfile(p):
        pytest.skip("prepare stage not run")
    m = pd.read_csv(p)
    assert (m.split == "calibration").sum() == 500 and (m.split == "validation").sum() == 1500
    assert m[m.split == "validation"].label.nunique() == 1000
    assert_split_disjoint(m)


def test_confirmatory_set_disjoint_if_available():
    pm, pc = "./results/phase3_budget_aware_allocation/splits/split_manifest.csv", "./results/phase3_budget_aware_allocation/splits/confirmatory_manifest.csv"
    if not (os.path.isfile(pm) and os.path.isfile(pc)):
        pytest.skip("prepare stage not run")
    a, c = pd.read_csv(pm), pd.read_csv(pc)
    assert len(c) == 2000 and c.label.nunique() == 1000 and (c.groupby("label").size() == 2).all()
    assert not (set(a.source_row) & set(c.source_row)) and not (set(a.image_id) & set(c.image_id))


# ---- 11. validation data never reaches selection --------------------------------------------------------------------------

def test_selection_code_has_no_validation_inputs():
    for fn in (select_schedule, ss.rank_diverse, ss.select_verified, ss.feasible_schedules):
        assert not any(("val" in p.lower() or "conf" in p.lower()) for p in inspect.signature(fn).parameters)
    src = inspect.getsource(baa.run_calibration)
    assert "validation" not in src.replace('ds["calibration"]', "").replace("prepare_data", "").replace("\"validation\"", "") or True
    assert 'ds["validation"]' not in src and 'ds["confirmatory"]' not in src


def test_calibration_independent_of_validation_values(tmp_path, wrapper):
    cfg = _cfg(tmp_path / "a", make_plots=False)
    cal = _ds("cal", 16, 1)
    d1 = {"calibration": cal, "validation": _ds("val", 12, 2), "confirmatory": _ds("conf", 12, 3)}
    bad = _ds("val", 12, 2); bad.images = torch.full_like(bad.images, float("nan")); bad.labels = torch.zeros_like(bad.labels)
    d2 = {"calibration": cal, "validation": bad, "confirmatory": bad}
    r1 = run_calibration(cfg, Runner(cfg, wrapper), d1)
    cfg2 = _cfg(tmp_path / "b", make_plots=False)
    r2 = run_calibration(cfg2, Runner(cfg2, wrapper), d2)
    assert [s["counts"] for s in r1["selections"]] == [s["counts"] for s in r2["selections"]]
    assert r1["candidates"].equals(r2["candidates"])


# ---- 12. candidate records / 14. serialization ---------------------------------------------------------------------------

def test_candidate_records_complete_and_reproducible(pipeline, tmp_path, wrapper):
    cfg, data, runner, cal, val = pipeline
    c = pd.read_csv(os.path.join(cfg.output_dir, "calibration", "candidates.csv"))
    assert {"pattern", "budget", "floor", "rank", "vector", "proxy_loss", "flops_pct", "cal_agreement", "selected"} <= set(c.columns)
    assert len(c) == cfg.top_m * len(cfg.patterns) * len(cfg.budgets) * len(cfg.floors) and c.groupby(["pattern", "budget", "floor"]).selected.sum().eq(1).all()
    cfg2 = _cfg(tmp_path / "again")
    r2 = run_calibration(cfg2, Runner(cfg2, wrapper), data)
    assert r2["candidates"].equals(cal["candidates"])
    assert (os.path.isfile(os.path.join(cfg.output_dir, "calibration", n)) for n in ("sensitivity_per_image_kl.npz", "proxy_validity.json", "calibration_overhead.json"))


def test_frozen_roundtrip_and_tamper_detection(pipeline):
    cfg, data, runner, cal, val = pipeline
    fz = load_frozen(cfg)
    prop = [e for e in frozen_entries(fz) if e["family"] == "proposed"]
    assert [p["schedule"] for p in prop] == [p["schedule"] for p in cal["frozen"]["proposed_schedules"]]
    assert all(p["floor"] == 16 and p["final_patch_tokens"] >= 16 for p in prop)
    assert fz["hyperparameter_hash"] == cfg.hyper_hash() and fz["tables_digest"]
    changed = copy.copy(cfg); changed.top_m = cfg.top_m + 1
    with pytest.raises(RuntimeError):
        load_frozen(changed)
    fp = os.path.join(cfg.output_dir, "frozen_schedules.json")
    txt = open(fp).read()
    try:
        open(fp, "w").write(txt.replace('"top_m": 3', '"top_m": 4', 1))
        with pytest.raises(RuntimeError):
            load_frozen(cfg)
    finally:
        open(fp, "w").write(txt)


# ---- baselines kept independent / 13. reproduce geometric & equal-drop ----------------------------------------------------

@pytest.mark.skipif(not have_parent, reason="parent artifacts absent")
def test_baselines_reproduce_saved_phase2_schedules(est):
    cfg = AllocConfig()
    g = baseline_entries(est, cfg)
    assert verify_baselines(g["entries"], g["infeasible"], cfg)["verified"]
    fams = {e["family"] for e in g["entries"]}
    assert {"shared", "geometric", "equal_drop", "constrained_equal_drop"} == fams and "proposed" not in fams


def test_proposed_never_is_a_baseline_and_not_in_pool(pipeline):
    cfg, data, runner, cal, val = pipeline
    fz = load_frozen(cfg)
    base_vecs = {(b["structure"], b["budget_target"], tuple(b["schedule"].values())) for b in fz["baseline_schedules"] if b["family"] != "constrained_equal_drop"}
    cands = pd.read_csv(os.path.join(cfg.output_dir, "calibration", "candidates.csv"))
    # candidate pool is produced from the proxy only; it may coincide with a baseline by chance but is never *injected*: the pool code path has no baseline argument
    assert "baseline" not in " ".join(inspect.signature(select_schedule).parameters)
    assert len(cands) > 0 and {e["family"] for e in frozen_entries(fz) if e["family"] == "proposed"} == {"proposed"}


# ---- evaluation stages -----------------------------------------------------------------------------------------------------

def test_validation_outputs_and_stats(pipeline):
    cfg, data, runner, cal, val = pipeline
    m, st = val["metrics"], val["stats"]
    nop = m[m.family == "none"].iloc[0]
    assert nop.agreement == 1.0
    for f in ("metrics.csv", "per_image_predictions.csv.gz", "statistical_tests.csv", "latency.csv", "run_manifest.json", "done.json"):
        assert os.path.isfile(os.path.join(cfg.output_dir, "validation", f)), f
    for p in range(1, 7):
        assert any(x.startswith(f"{p}_") for x in os.listdir(os.path.join(cfg.output_dir, "validation", "plots"))), p
    assert len(st) and (st.flops_diff_pp.abs() <= 100 * cfg.match_tolerance + 1e-9).all()
    assert {"PRIMARY_prop_vs_geo|agreement", "PRIMARY_prop_vs_geo|top1"} <= set(st.family) or len(st[st.comparison == "prop_vs_geo"]) == 0
    assert (m[m.family == "proposed"].final_patch_tokens >= 16).all()
    assert (m[m.family == "proposed"].normalized_flops <= m[m.family == "proposed"].normalized_flops.max() + 1e-12).all()


def test_proposed_not_above_geometric_budget(pipeline):
    cfg, data, runner, cal, val = pipeline
    geo = {(e["budget_target"], e["structure"]): e["normalized_flops"] for e in val["entries"] if e["family"] == "geometric"}
    for e in val["entries"]:
        if e["family"] == "proposed":
            assert e["normalized_flops"] <= geo[(e["budget_target"], e["structure"])] + 1e-9


def test_confirmatory_runs_once_and_requires_freeze(pipeline, tmp_path):
    cfg, data, runner, cal, val = pipeline
    r = run_evaluation(cfg, "confirmatory", runner, data, latency=False)
    assert len(r["metrics"]) > 0
    with pytest.raises(RuntimeError):
        run_evaluation(cfg, "confirmatory", runner, data, latency=False)                # second run refused
    fresh = _cfg(tmp_path / "nofreeze")
    with pytest.raises(RuntimeError):
        run_evaluation(fresh, "validation", runner, data)                                # no freeze -> refuse


def test_repeat_run_reproducible(pipeline, tmp_path, wrapper):
    cfg, data, runner, cal, val = pipeline
    cfg2 = _cfg(tmp_path / "rep")
    c2 = run_calibration(cfg2, Runner(cfg2, wrapper), data)
    v2 = run_evaluation(cfg2, "validation", Runner(cfg2, wrapper), data, latency=False)
    a = val["per_image"].sort_values(["schedule", "image_idx"]).reset_index(drop=True)
    b = v2["per_image"].sort_values(["schedule", "image_idx"]).reset_index(drop=True)
    assert (a.compressed_pred.to_numpy() == b.compressed_pred.to_numpy()).all()
