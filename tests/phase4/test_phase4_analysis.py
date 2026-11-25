"""Independent checks of the Gate 5 analysis script (synthetic data and the P4-DEV manifest CSV only; P4-CONFIRM is never touched)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("p4an", os.path.join(ROOT, "scripts", "phase4_gate5_analysis.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def test_wilson_matches_scipy_and_textbook():
    for k, n in ((0, 50), (1, 50), (25, 50), (1500, 1990), (1990, 1990), (2800, 2985)):
        lo, hi = an.wilson(k, n)
        ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        assert lo == pytest.approx(ci.low, abs=1e-9) and hi == pytest.approx(ci.high, abs=1e-9)
    lo, hi = an.wilson(8, 10)                                               # textbook value ~ (0.4902, 0.9433)
    assert (lo, hi) == (pytest.approx(0.4902, abs=1e-3), pytest.approx(0.9433, abs=1e-3))


def test_mcnemar_exact_against_enumeration():
    for b, c in ((0, 0), (3, 3), (10, 2), (2, 10), (30, 18), (1, 0)):
        m = b + c
        if m == 0:
            expect = 1.0
        else:
            k = min(b, c)
            expect = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
        assert an.mcnemar_exact(b, c) == pytest.approx(expect, abs=1e-12)
    assert an.mcnemar_exact(10, 2) == an.mcnemar_exact(2, 10)               # symmetric


def test_holm_textbook_and_monotone():
    adj = an.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert an.holm([0.2]) == pytest.approx([0.2])
    assert an.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    rng = np.random.default_rng(0)
    p = rng.random(16)
    a = an.holm(p)
    assert (a >= p - 1e-12).all() and (a <= 1).all()
    o = np.argsort(p)
    assert (np.diff(a[o]) >= -1e-12).all()                                    # adjusted p-values are monotone in raw order
    # brute-force step-down definition
    m = len(p); srt = np.sort(p); ref = []
    for i in range(m):
        ref.append(min(1, max((m - j) * srt[j] for j in range(i + 1))))
    assert np.sort(a) == pytest.approx(ref)


def test_bootstrap_reproducible_and_brackets_difference():
    rng = np.random.default_rng(1)
    d = rng.choice([-1, 0, 1], size=500, p=[0.1, 0.6, 0.3])
    a1, a2 = an.paired_bootstrap(d), an.paired_bootstrap(d)
    assert a1 == a2                                                         # same seed -> identical
    assert a1[0] < d.mean() < a1[1]
    assert an.paired_bootstrap(np.zeros(100))[0] == 0.0


def _mk_indir(tmp_path, n_wrong=None):
    man = pd.read_csv(os.path.join(an.P4, "splits", "p4_dev_manifest.csv"))
    n = len(man)
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 1000, n)
    np.savez(tmp_path / "reference_unpruned.npz", ids=np.asarray(man.image_id.tolist()), labels=man.label.to_numpy(), pred=ref, top5=np.stack([ref] * 5, 1))
    return man, n, ref


def test_data_validation_and_metric_definitions(tmp_path):
    man, n, ref = _mk_indir(tmp_path)
    y = man.label.to_numpy()
    pred = ref.copy()
    pred[:100] = y[:100]                                                    # first 100 are correct w.r.t. labels
    pred[100:150] = (ref[100:150] + 1) % 1000                               # 50 disagree with reference (and are wrong)
    top5 = np.stack([pred, (pred + 1) % 1000, (pred + 2) % 1000, (pred + 3) % 1000, y], 1)   # label always in top-5
    np.savez(tmp_path / "m.npz", pred=pred, top5=top5)
    D = an.Data("dev", str(tmp_path))
    r = D.get("m")
    assert r["agree"].sum() == int((pred == ref).sum()) and r["top1"].sum() == int((pred == y).sum())
    assert r["top5"].sum() == n                                              # label always among top-5 by construction
    assert r["agree"].dtype == np.int8 and set(np.unique(r["agree"])) <= {0, 1}


@pytest.mark.parametrize("break_what", ["ids_order", "label", "missing", "range", "top5_pred", "shape"])
def test_data_validation_failures(tmp_path, break_what):
    man, n, ref = _mk_indir(tmp_path)
    if break_what == "ids_order":
        ids = np.asarray(man.image_id.tolist())[::-1].copy()
        np.savez(tmp_path / "reference_unpruned.npz", ids=ids, labels=man.label.to_numpy(), pred=ref, top5=np.stack([ref] * 5, 1))
        with pytest.raises(SystemExit):
            an.Data("dev", str(tmp_path))
        return
    if break_what == "label":
        lab = man.label.to_numpy().copy(); lab[0] = (lab[0] + 1) % 1000
        np.savez(tmp_path / "reference_unpruned.npz", ids=np.asarray(man.image_id.tolist()), labels=lab, pred=ref, top5=np.stack([ref] * 5, 1))
        with pytest.raises(SystemExit):
            an.Data("dev", str(tmp_path))
        return
    D = an.Data("dev", str(tmp_path))
    pred, top5 = ref.copy(), np.stack([ref] * 5, 1)
    if break_what == "missing":
        with pytest.raises(SystemExit):
            D.get("nope")
        return
    if break_what == "range":
        pred[0] = 1000
    if break_what == "top5_pred":
        top5[:, 0] = (top5[:, 0] + 1) % 1000
    if break_what == "shape":
        pred = pred[:-1]
    np.savez(tmp_path / "bad.npz", pred=pred, top5=top5)
    with pytest.raises(SystemExit):
        D.get("bad")


def test_direction_and_family_logic(tmp_path):
    man, n, ref = _mk_indir(tmp_path)
    good, bad = ref.copy(), ref.copy()
    bad[:300] = (bad[:300] + 1) % 1000                                       # 'bad' disagrees on 300 images
    for nm, p in (("good", good), ("bad", bad)):
        np.savez(tmp_path / f"{nm}.npz", pred=p, top5=np.stack([p] * 5, 1))
    D = an.Data("dev", str(tmp_path))
    cat = {"good": {"file": "good", "flops_pct": 50.0}, "bad": {"file": "bad", "flops_pct": 50.0}, "undefined_comp": {"file": "x", "flops_pct": 1.0}}
    rows = an.run_family(D, cat, "T", [("g_vs_b", "good", "bad"), ("b_vs_g", "bad", "good"), ("g_vs_g", "good", "good"), ("g_vs_missing", "good", "absent")], "agree")
    byc = {r["cell"]: r for r in rows}
    assert byc["g_vs_b"]["diff_pp"] == pytest.approx(100 * 300 / n) and byc["g_vs_b"]["direction"] == "method>comparator"
    assert byc["b_vs_g"]["diff_pp"] == pytest.approx(-100 * 300 / n) and byc["b_vs_g"]["direction"] == "method<comparator"
    assert byc["g_vs_g"]["diff_pp"] == 0 and byc["g_vs_g"]["mcnemar_p"] == 1.0 and not byc["g_vs_g"]["significant_holm"]
    assert byc["g_vs_missing"]["status"] == "not_defined"                    # undefined cells are reported, not dropped or errored
    assert byc["g_vs_b"]["significant_holm"] and byc["g_vs_b"]["holm_family_size"] == 3      # Holm family excludes the undefined cell
    assert byc["g_vs_b"]["method_only_correct"] == 300 and byc["g_vs_b"]["comparator_only_correct"] == 0
    assert byc["g_vs_b"]["boot_ci_lo_pp"] > 0


def test_family_sizes_match_protocol_and_catalogue_is_frozen_only():
    cat = an.load_catalogue()
    prim = [(m, b, p) for m in ("tome", "evit") for b in an.BUDGETS for p in an.PATTERNS]
    assert len(prim) == 16
    assert all(an.geo(b, p) in cat and f"{m}_B{b}" in cat for m, b, p in prim)
    rep = [(b, p, f) for b in an.BUDGETS for p in an.PATTERNS for f in an.FLOORS]
    assert len(rep) == 24 and all(an.prop(b, p, f) in cat and an.geo(b, p) in cat for b, p, f in rep)
    assert all(f"diffrate_{t}" in cat and f"tome_DR{t}" in cat and f"evit_DR{t}" in cat for t in an.NATIVE) and "diffrate_1.0" not in cat
    assert an.NBOOT == 10000 and an.SEED == 42 and an.ALPHA == 0.05
