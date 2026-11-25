"""Phase 4 Gate 3 tests: token counts, CLS preservation, ordering, shape consistency, determinism, no-op equivalence, FLOPs consistency,
selection rule determinism, confirmatory guard. Uses random images + real DeiT-Tiny weights (no P4-CONFIRM data is ever touched)."""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.metrics.compute import ComputeEstimator  # noqa: E402
from src.phase4 import data as p4data  # noqa: E402
from src.phase4.diffrate import DiffRateReleased, IllDefinedMerge  # noqa: E402
from src.phase4.engine import Compressor, flops_from_counts, flops_pct, run_compressed  # noqa: E402
from src.phase4.evit_style import EViTStyle  # noqa: E402
from src.phase4.selection import feasible, kappa_grid, select, tome_grid  # noqa: E402
from src.phase4.tome import ToMe, bipartite_soft_matching, parse_r  # noqa: E402

RATES = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "diffrate_deit_tiny_rates_as_published.json")


@pytest.fixture(scope="module")
def model():
    import timm
    m = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
    return m


@pytest.fixture(scope="module")
def imgs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 3, 224, 224, generator=g)


def test_engine_no_compression_matches_timm(model, imgs):
    with torch.no_grad():
        ref = model(imgs)
    out, info = run_compressed(model, imgs, Compressor())
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert info["entering"] == [197] * 12 and info["leaving"] == [197] * 12
    assert (out.argmax(1) == ref.argmax(1)).all()


def test_flops_consistent_with_phase1_3_estimator(model):
    est = ComputeEstimator(num_layers=12, hidden_dim=192, num_heads=3, num_original_tokens=196)
    assert abs(flops_from_counts([197] * 12, [197] * 12) - est.baseline_flops) < 1e-3
    # a fixed progressive schedule (prune after block 2 to 100 patches, after block 6 to 50): entering == leaving within a block
    counts = [197, 197] + [101] * 4 + [51] * 6
    prog = est.compute_progressive_flops([196, 196] + [100] * 4 + [50] * 6)
    assert abs(flops_from_counts(counts, counts) - prog["total_flops"]) / prog["total_flops"] < 1e-9


def test_tome_counts_rule_and_cls_and_size(model, imgs):
    for spec in (4, (6, -1.0), (5, 0.5)):
        comp = ToMe(spec)
        out, info = run_compressed(model, imgs, comp)
        n = 197
        for i, r in enumerate(parse_r(12, spec)):
            assert info["entering"][i] == n
            n -= max(min(r, (n - 1) // 2), 0)
            assert info["leaving"][i] == n
        assert out.shape == (4, 1000) and torch.isfinite(out).all()
        assert torch.allclose(comp.size.sum(dim=1).squeeze(-1), torch.full((4,), float(info["entering"][0])), atol=1e-3)  # sizes conserve 197


def test_tome_cls_never_merged_and_zero_r_is_identity(model, imgs):
    comp = ToMe(0)
    out, info = run_compressed(model, imgs, comp)
    with torch.no_grad():
        ref = model(imgs)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4) and info["leaving"] == [197] * 12
    x = torch.randn(2, 17, 8)
    merge, r = bipartite_soft_matching(x, 5)
    y = merge(x * 1.0, "sum")
    assert r == 5 and y.shape[1] == 12 and torch.allclose(y[:, 0], x[:, 0])      # CLS untouched and still first
    merge2, r2 = bipartite_soft_matching(torch.randn(1, 5, 4), 99)
    assert r2 == 2                                                                # capped at (T-1)//2


def test_parse_r_matches_documented_forms():
    assert parse_r(12, 13) == [13] * 12
    assert parse_r(12, (13, -1.0))[0] == 26 and parse_r(12, (13, -1.0))[-1] == 0
    assert parse_r(12, (13, 0.0)) == [13] * 12
    assert parse_r(12, [1, 2]) == [1, 2] + [0] * 10


def test_evit_style_semantics(model, imgs):
    comp = EViTStyle(0.7)
    out, info = run_compressed(model, imgs, comp)
    n, expected = 197, []
    for i in range(12):
        if i in (3, 6, 9):
            n = math.ceil(0.7 * (n - 1)) + 1 + 1
        expected.append(n)
    assert info["leaving"] == expected and info["entering"][0] == 197
    assert torch.isfinite(out).all()
    # kappa = 1 -> no-op and identical to unpruned
    c1 = EViTStyle(1.0)
    o1, i1 = run_compressed(model, imgs, c1)
    with torch.no_grad():
        ref = model(imgs)
    assert i1["leaving"] == [197] * 12 and torch.allclose(o1, ref, atol=1e-4, rtol=1e-4)


def test_evit_style_unit_topk_fusion_cls_and_order():
    torch.manual_seed(0)
    B, N, C = 2, 11, 5
    x = torch.randn(B, N, C)
    a = torch.rand(B, N - 1)
    c = EViTStyle(0.5, locations_1indexed=(1,))
    c.begin(B, N, x.device)
    y = c.compress(0, x.clone(), a, None)
    k = math.ceil(0.5 * (N - 1))
    assert y.shape == (B, 1 + k + 1, C)
    assert torch.equal(y[:, 0], x[:, 0])                                           # CLS first, unchanged
    for b in range(B):
        top = torch.topk(a[b], k).indices + 1
        assert torch.equal(y[b, 1:1 + k], x[b, top])                               # descending-attention order
        rest = [j for j in range(1, N) if j not in set(top.tolist())]
        fused = (x[b, rest] * a[b, [j - 1 for j in rest]][:, None]).sum(0)
        assert torch.allclose(y[b, -1], fused, atol=1e-6)                           # raw attention-weighted sum appended last


def test_diffrate_released_counts_and_cls(model, imgs):
    rates = json.load(open(RATES))
    for tgt in ("0.6", "0.7", "0.8", "0.9"):
        comp = DiffRateReleased.from_published(RATES, tgt, strict=True)
        out, info = run_compressed(model, imgs, comp)
        n = 197
        for i in range(12):
            p, m = rates[tgt]["prune_kept_num"][i], rates[tgt]["merge_kept_num"][i]
            n = min(n, p)
            if m < p:
                n = min(n, m)
            assert info["leaving"][i] == n, (tgt, i)
        assert torch.isfinite(out).all() and out.shape == (4, 1000)


def test_diffrate_dense_lists_equal_unpruned(model, imgs):
    comp = DiffRateReleased([197] * 12, [197] * 12)
    out, info = run_compressed(model, imgs, comp)
    with torch.no_grad():
        ref = model(imgs)
    assert info["leaving"] == [197] * 12
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)                         # sorting only permutes tokens


def test_diffrate_target_1_0_illdefined_event_is_logged_not_repaired(model, imgs):
    comp = DiffRateReleased.from_published(RATES, "1.0")
    out, info = run_compressed(model, imgs, comp)
    ev = [e for e in info["events"] if "ill_defined" in e]
    assert ev and ev[0]["block"] == 7 and torch.isfinite(out).all()               # recorded, executed literally
    with pytest.raises(IllDefinedMerge):
        run_compressed(model, imgs, DiffRateReleased.from_published(RATES, "1.0", strict=True))
    published = json.load(open(RATES))["1.0"]["merge_kept_num"]
    assert published[6] == 148 and published[7] == 156                            # published list is unchanged (no repair)


def test_determinism(model, imgs):
    for comp_fn in (lambda: ToMe((8, -1.0)), lambda: EViTStyle(0.6), lambda: DiffRateReleased.from_published(RATES, "0.7")):
        o1, i1 = run_compressed(model, imgs, comp_fn())
        o2, i2 = run_compressed(model, imgs, comp_fn())
        assert torch.equal(o1, o2) and i1["leaving"] == i2["leaving"]


def test_flops_pct_matches_frozen_analytic_predictions():
    assert abs(flops_pct([197] * 12, [197] * 12) - 100.0) < 1e-9
    rates = json.load(open(RATES))
    pred = {"0.6": 52.1945, "0.7": 57.9177, "0.8": 64.4732, "0.9": 71.9240}
    for t, v in pred.items():
        p, m = rates[t]["prune_kept_num"], rates[t]["merge_kept_num"]
        n, ent, out = 197, [], []
        for a, b in zip(p, m):
            ent.append(n); n = min(n, a, b); out.append(n)
        assert abs(flops_pct(ent, out) - v) < 5e-4


def test_selection_rule_is_exact_and_deterministic():
    n = 1990
    cands = [dict(config=(r, i), flops_pct=50.0 + 0.1 * r, agree_count=1000 - 4 * r, gen_index=g)
             for g, (r, i) in enumerate(tome_grid()[:20])]
    s = select(cands, n, 50.0, "tome")
    best = max(c["agree_count"] for c in cands)
    assert all((best - c["agree_count"]) * 500 < n for c in s["tie_set"])         # difference <= 3 images
    t3 = dict(config=(1, 0.0), flops_pct=50.2, agree_count=1000, gen_index=0)
    t4 = dict(config=(2, 0.0), flops_pct=50.0, agree_count=997, gen_index=1)        # 3 images (0.15 pp) below: tied
    t5 = dict(config=(3, 0.0), flops_pct=50.0, agree_count=996, gen_index=2)        # 4 images (0.20 pp): NOT tied
    s2 = select([t3, t4, t5], n, 50.0, "tome")
    assert s2["chosen"] is t4 and t5 not in s2["tie_set"]                            # closer FLOPs wins inside the tie
    k1 = dict(kappa=0.3, flops_pct=40.0, agree_count=500, gen_index=0)
    k2 = dict(kappa=0.305, flops_pct=40.0, agree_count=500, gen_index=1)
    assert select([k2, k1], n, 40.0, "evit")["chosen"] is k1                         # final tie-break: smaller kappa
    assert select([k1, k2], n, 40.0, "evit")["chosen"] is k1
    assert len(kappa_grid()) == 161 and kappa_grid()[0] == 0.2 and kappa_grid()[-1] == 1.0
    assert len(tome_grid()) == 200
    assert feasible([dict(flops_pct=40.9)], 40) and not feasible([dict(flops_pct=41.1)], 40)


def test_confirmatory_split_is_locked(monkeypatch, tmp_path):
    monkeypatch.delenv("PHASE4_CONFIRMATORY", raising=False)
    with pytest.raises(p4data.ConfirmatoryGuardError):                              # locked without the explicit Gate 5 env var
        p4data.load_split("confirm")
    monkeypatch.setenv("PHASE4_CONFIRMATORY", "1")
    monkeypatch.setattr(p4data, "P4", str(tmp_path))                                # simulate: no frozen_method_configs.json
    with pytest.raises(p4data.ConfirmatoryGuardError):
        p4data.load_split("confirm")
    (tmp_path / "frozen_method_configs.json").write_text("{}")
    (tmp_path / "confirmatory_run_done.marker").write_text("x")                     # simulate: already run once
    with pytest.raises(p4data.ConfirmatoryGuardError):
        p4data.load_split("confirm")


def test_manifest_disjointness_and_balance():
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "splits", "p4_dev_manifest.csv"))
    c = pd.read_csv(os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "splits", "p4_confirm_manifest.csv"))
    assert len(d) == 1990 and len(c) == 2985 and d.label.nunique() == 995 and c.label.nunique() == 995
    assert not (set(d.source_row) & set(c.source_row))
    assert (d.groupby("label").size() == 2).all() and (c.groupby("label").size() == 3).all()
    for lbl in range(5):
        assert lbl not in set(d.label) and lbl not in set(c.label)
    par = torch.load(os.path.join(ROOT, "data", "imagenet_class_diverse_subset.pt"), weights_only=False)["source_rows"]
    p3 = torch.load(os.path.join(ROOT, "data", "imagenet_confirmatory_2000.pt"), weights_only=False)["source_rows"]
    used = set(np.asarray(par).tolist()) | set(np.asarray(p3).tolist()) | set(range(250))
    assert not (set(d.source_row) | set(c.source_row)) & used


def test_frozen_config_verification_detects_tampering(monkeypatch, tmp_path):
    import hashlib
    import json
    monkeypatch.setattr(p4data, "P4", str(tmp_path))
    dep = os.path.join(ROOT, "scripts", "phase4_gate5_confirmatory.py")
    body = json.dumps({"file_sha256": {"scripts/phase4_gate5_confirmatory.py": hashlib.sha256(open(dep, "rb").read()).hexdigest()}}).encode()
    (tmp_path / "frozen_method_configs.json").write_bytes(body)
    (tmp_path / "frozen_method_configs.sha256").write_text(hashlib.sha256(body).hexdigest())
    p4data._verify_frozen_config()                                                   # consistent: passes
    (tmp_path / "frozen_method_configs.sha256").write_text("0" * 64)
    with pytest.raises(p4data.ConfirmatoryGuardError):                               # file bytes no longer match recorded hash
        p4data._verify_frozen_config()
    bad = json.dumps({"file_sha256": {"scripts/phase4_gate5_confirmatory.py": "0" * 64}}).encode()
    (tmp_path / "frozen_method_configs.json").write_bytes(bad)
    (tmp_path / "frozen_method_configs.sha256").write_text(hashlib.sha256(bad).hexdigest())
    with pytest.raises(p4data.ConfirmatoryGuardError):                               # a dependency changed after freeze
        p4data._verify_frozen_config()
