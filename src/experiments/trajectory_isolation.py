"""
Phase 2: isolating the final-token-count effect from the full reduction trajectory (evaluation only; no optimizer, merging,
fine-tuning, learned routing or new scorers).

Research question: at matched compute, is geometric allocation better mainly because it preserves more tokens at the final stage,
or does the shape of the whole layer-wise reduction trajectory matter independently?

Four schedule families per (budget, stage pattern) cell, patterns 2-6-10 and 2-8-10, first layer 2, budgets 40/50/60/70 %:
  A geometric                  : IMPORTED UNCHANGED (class_diverse.generate_family_schedules)   K_j = max(1, round(N rho^j))
  B final-count-matched geometric control (NEW; a control, not a method):
        same layers, same initial count N, EXACTLY the geometric final count, strictly decreasing integer counts, only the
        intermediate counts differ; every candidate is enumerated exhaustively (all strictly decreasing integer vectors), its
        analytic FLOPs compared with the ORIGINAL geometric schedule (+-tolerance), and classified. Controls are then selected by a
        rule fixed before any model result is seen (see `select_controls`). If none exist the cell is reported INFEASIBLE.
  C constrained equal-drop     : IMPORTED UNCHANGED (minimum_token.generate_min_token_schedules)  K_j = max(F, round(N - jD))
  D unconstrained equal-drop   : IMPORTED UNCHANGED (class_diverse.generate_family_schedules)    K_j = max(1, round(N - jD))
Schedules A, C and D are asserted identical to the saved minimum-token manifest; the image subset is asserted identical (ids, labels,
order, per-image hashes, full SHA-256 recomputed from the saved manifest file) to the previous minimum-token and class-diverse runs.
Keys are 1-based layer numbers (see src/compression/progressive.py): key l = after l blocks, keep exactly k PATCH tokens; CLS always kept.
"""

import hashlib
import itertools
import json
import os
import pickle
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..compression.progressive import ProgressivePruner, RetentionSchedule
from ..data.class_diverse import manifest_hash
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from .class_diverse import _time_multi, ClassDiverseConfig, load_subset
from .layerwise_analysis import wilson_score_interval
from .minimum_token import generate_min_token_schedules, verify_same_subset
from .progressive_eval import (
    LAYER_NUMBERING, _environment, _md_table, _sha256_file, effective_prune_layers, expected_processed_counts, paired_diff_stats,
    schedule_flops,
)
from .stage_ablation import STAGE_NAME, _valid_stage_schedule, holm_correction

CONTROL_FAMILY = "final_count_matched_geometric"
PRIMARY_FLOORS = (16, 24, 32)
EQUIVALENCE_MARGIN_PP = 2.0      # descriptive "practically similar" margin used only in the verdict rule (arbitrary, stated)


@dataclass
class TrajConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    parquet_dir: str = "./data/imagenet_val_parquet"
    subset_cache: str = "./data/imagenet_class_diverse_subset.pt"
    images_per_class: int = 2
    subset_seed: int = 42
    batch_size: int = 25
    seed: int = 42
    budgets: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70])
    match_tolerance: float = 0.01
    first_layer: int = 2
    patterns: List[str] = field(default_factory=lambda: ["2-6-10", "2-8-10"])
    floors: List[int] = field(default_factory=lambda: [8, 16, 24, 32])
    primary_floors: List[int] = field(default_factory=lambda: list(PRIMARY_FLOORS))
    scorer: str = "cls_attention"
    # control selection rule (fixed before running; uses only trajectory/FLOPs information, never accuracy)
    control_min_deviation: int = 8          # a control is "distinct" only if max |K_j - G_j| over intermediate stages >= this
    measure_latency: bool = True
    latency_warmup: int = 1
    latency_runs: int = 5
    latency_batches: int = 3
    prev_min_token_dir: Optional[str] = "./results/phase2_minimum_token_ablation"
    prev_class_diverse_dir: Optional[str] = "./results/phase2_class_diverse_validation"
    output_dir: str = "./results/phase2_final_token_trajectory_isolation"
    device: Optional[str] = None
    make_plots: bool = True
    resume: bool = True
    max_images: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _structures(patterns: List[str]) -> Dict[str, List[Tuple[int, ...]]]:
    out: Dict[str, List[Tuple[int, ...]]] = {"two": [], "three": []}
    for p in patterns:
        t = tuple(int(x) for x in p.split("-"))
        out["two" if len(t) == 2 else "three"].append(t)
    return out


# ----------------------------------------------------------------------------- final-count-matched controls

def _traj_dist(vec: List[int], geo: List[int]) -> int:
    """Max absolute difference over INTERMEDIATE stages (the final count is equal by construction)."""
    return int(max((abs(a - b) for a, b in zip(vec[:-1], geo[:-1])), default=0))


def enumerate_control_candidates(est: ComputeEstimator, geo: Dict[str, Any], tol: float, min_dev: int) -> List[Dict[str, Any]]:
    """Every strictly decreasing integer vector with the geometric stage layers and EXACTLY the geometric final count.

    Status: identical_to_geometric | flops_out_of_tolerance | feasible_near_identical (deviation < min_dev) | feasible_distinct |
    invalid. FLOPs are compared with the ORIGINAL geometric schedule using the existing analytic accounting."""
    layers = tuple(geo["layers"])
    G = [geo["schedule"][l] for l in layers]
    N, s, kf = est.N, len(layers), G[-1]
    rows: List[Dict[str, Any]] = []
    if s < 2:
        return rows
    gf = geo["normalized_flops"]
    for inter in itertools.combinations(range(kf + 1, N), s - 1):       # ascending distinct values in (kf, N)
        vec = list(reversed(inter)) + [kf]
        sched = dict(zip(layers, vec))
        r = {"geometric": geo["name"], "budget": geo["budget_target"], "structure": geo["structure"], "vector": json.dumps(vec),
             "geometric_vector": json.dumps(G), "final_tokens": kf, "initial_tokens": N}
        if not _valid_stage_schedule(sched, layers, est):
            r.update({"status": "invalid", "normalized_flops": np.nan, "flops_diff_vs_geometric_pp": np.nan, "trajectory_dist": np.nan})
            rows.append(r); continue
        f = schedule_flops(est, sched)["normalized_flops"]
        d = _traj_dist(vec, G)
        r.update({"normalized_flops": f, "flops_diff_vs_geometric_pp": 100 * (f - gf), "trajectory_dist": d})
        if vec == G:
            r["status"] = "identical_to_geometric"
        elif abs(f - gf) > tol + 1e-12:
            r["status"] = "flops_out_of_tolerance"
        elif d < min_dev:
            r["status"] = "feasible_near_identical"
        else:
            r["status"] = "feasible_distinct"
        rows.append(r)
    return rows


def select_controls(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predefined, accuracy-blind selection of up to four controls per geometric schedule from the `feasible_distinct` candidates.

    Side: 'retaining' if the first intermediate count K_1 is larger than geometric (K_1 == G_1: K_2 decides), else 'steeper'.
    Per side: the candidate with the largest trajectory deviation ('extreme') and the one whose deviation is closest to half of the
    side's maximum ('mid'; must differ from the extreme). Ties -> smaller |FLOPs diff|, then lexicographic vector."""
    good = [c for c in cands if c["status"] == "feasible_distinct"]
    chosen: List[Dict[str, Any]] = []
    for side in ("retaining", "steeper"):
        grp = []
        for c in good:
            v, g = json.loads(c["vector"]), json.loads(c["geometric_vector"])
            sign = next((np.sign(a - b) for a, b in zip(v[:-1], g[:-1]) if a != b), 0)
            if (sign > 0) == (side == "retaining") and sign != 0:
                grp.append(c)
        if not grp:
            continue
        key = lambda c: (abs(c["flops_diff_vs_geometric_pp"]), c["vector"])
        ext = sorted(grp, key=lambda c: (-c["trajectory_dist"],) + key(c))[0]
        half = ext["trajectory_dist"] / 2.0
        rest = [c for c in grp if c is not ext]
        mid = sorted(rest, key=lambda c: (abs(c["trajectory_dist"] - half),) + key(c))[0] if rest else None
        chosen.append({**ext, "control_side": side, "control_kind": "extreme"})
        if mid is not None:
            chosen.append({**mid, "control_side": side, "control_kind": "mid"})
    return chosen


def build_control_entries(est: ComputeEstimator, geo_entries: List[Dict[str, Any]], tol: float, min_dev: int
                          ) -> Dict[str, Any]:
    entries, cand_rows, infeasible = [], [], []
    L, N = est.num_layers, est.N
    for geo in geo_entries:
        cands = enumerate_control_candidates(est, geo, tol, min_dev)
        picked = select_controls(cands)
        pk = {c["vector"] for c in picked}
        for c in cands:
            c["selected"] = c["vector"] in pk
            c["control_name"] = None
        counts = pd.Series([c["status"] for c in cands]).value_counts().to_dict() if cands else {}
        if not picked:
            reason = ("geometric schedule has fewer than two stages" if not cands else
                      "no valid strictly decreasing intermediate vector" if not any(c["status"] != "invalid" for c in cands) else
                      "no distinct candidate with the exact geometric final count within the FLOPs tolerance "
                      f"(statuses: {counts})")
            infeasible.append({"geometric": geo["name"], "budget": geo["budget_target"], "structure": geo["structure"], "reason": reason})
        for p in picked:
            vec = json.loads(p["vector"]); layers = tuple(geo["layers"]); sched = dict(zip(layers, vec))
            RetentionSchedule.from_dict(sched).validate(L, N)
            name = f"{geo['group']}_geoctrl_{p['control_side']}_{p['control_kind']}_{STAGE_NAME[len(layers)]}_{geo['structure']}"
            for c in cands:
                if c["vector"] == p["vector"]:
                    c["control_name"] = name
            entries.append({"name": name, "schedule": sched, "stage_count": len(layers), "first_layer": layers[0],
                            "structure": geo["structure"], "layers": list(layers), "budget_target": geo["budget_target"],
                            "group": geo["group"], "role": "member", "family": CONTROL_FAMILY, "floor": None, "alias_of": None,
                            "control_of": geo["name"], "control_side": p["control_side"], "control_kind": p["control_kind"],
                            "trajectory_dist": p["trajectory_dist"], "D": None, **schedule_flops(est, sched),
                            "effective_layers": effective_prune_layers(sched, L, N), "final_patch_tokens": vec[-1],
                            "floor_satisfied": None})
        for c in cands:
            c["n_candidates_for_geometric"] = len(cands)
        cand_rows.extend(cands)
    return {"entries": entries, "candidates": cand_rows, "infeasible": infeasible}


# ----------------------------------------------------------------------------- schedule set + verification

def generate_isolation_schedules(est: ComputeEstimator, cfg: TrajConfig) -> Dict[str, Any]:
    g = generate_min_token_schedules(est, cfg.first_layer, cfg.budgets, _structures(cfg.patterns), cfg.match_tolerance, cfg.floors)
    geo = [e for e in g["entries"] if e["family"] == "geometric"]
    c = build_control_entries(est, geo, cfg.match_tolerance, cfg.control_min_deviation)
    entries = g["entries"] + c["entries"]
    for e in entries:
        e.setdefault("control_of", None); e.setdefault("control_side", None); e.setdefault("control_kind", None)
        e.setdefault("trajectory_dist", 0 if e["family"] != CONTROL_FAMILY else e.get("trajectory_dist"))
    return {"entries": entries, "groups": g["groups"], "unmatched": g["unmatched"], "infeasible": g["infeasible"],
            "identical": g["identical"], "candidates": c["candidates"], "control_infeasible": c["infeasible"]}


def verify_prior_schedules(entries: List[Dict[str, Any]], infeasible: List[Dict[str, Any]], prev_dir: Optional[str],
                           patterns: List[str]) -> Dict[str, Any]:
    """Geometric / unconstrained equal-drop / constrained equal-drop / anchor schedules must equal the saved minimum-token manifest."""
    if not prev_dir:
        return {"verified": False, "note": "no previous directory supplied"}
    mp = os.path.join(prev_dir, "schedule_manifest.json")
    if not os.path.isfile(mp):
        raise RuntimeError(f"Cannot verify prior schedules: {mp} not found. Stopping.")
    man = json.load(open(mp))
    prev = {s["name"]: {int(k): v for k, v in s["schedule"].items()} for s in man["schedules"]
            if s["role"] != "baseline" and (s["structure"] in patterns or s["role"] == "anchor")}
    cur = {e["name"]: e["schedule"] for e in entries if e["family"] != CONTROL_FAMILY}
    if cur != prev:
        diff = sorted(set(cur) ^ set(prev)) + [k for k in cur if k in prev and cur[k] != prev[k]]
        raise RuntimeError(f"Re-generated A/C/D schedules do NOT match the minimum-token manifest ({diff[:6]}). Stopping.")
    pi = {(round(u["budget"], 6), u["structure"], u["floor"]) for u in man["infeasible_combinations"] if u["structure"] in patterns}
    ci = {(round(u["budget"], 6), u["structure"], u["floor"]) for u in infeasible}
    if pi != ci:
        raise RuntimeError(f"Infeasible constrained combinations differ from the minimum-token manifest: {sorted(pi ^ ci)}")
    return {"verified": True, "schedules_compared": len(cur), "previous_manifest": mp, "infeasible_combinations_identical": len(ci)}


def verify_manifest_artifact(prev_dir: str) -> Dict[str, Any]:
    """Recompute the FULL SHA-256 from the saved manifest file and compare with the checksum stored beside it."""
    mp, sp = os.path.join(prev_dir, "subset_manifest.csv"), os.path.join(prev_dir, "subset_summary.json")
    if not (os.path.isfile(mp) and os.path.isfile(sp)):
        raise RuntimeError(f"Exact 2,000-image manifest not found in {prev_dir}. Stopping instead of substituting another subset.")
    df = pd.read_csv(mp, dtype={"image_sha1": str})
    rec, stored = manifest_hash(df), json.load(open(sp))["manifest_sha256"]
    if rec != stored:
        raise RuntimeError(f"Recomputed manifest checksum {rec} != stored {stored}. Stopping.")
    return {"manifest_path": os.path.abspath(mp), "summary_path": os.path.abspath(sp), "manifest_sha256_full": rec,
            "manifest_file_sha256": _sha256_file(mp), "num_images": int(len(df))}


# ----------------------------------------------------------------------------- statistics

def _label(e):
    return {"geometric": "geometric", "equal_drop": "unconstrained_equal_drop", "constrained_equal_drop": f"constrained_f{e['floor']}",
            CONTROL_FAMILY: "geo_control"}.get(e["family"], e["family"])


def build_isolation_tests(per_img: pd.DataFrame, entries: List[Dict[str, Any]], scorer: str, seed: int, primary_floors: List[int],
                          tol: float) -> pd.DataFrame:
    """Paired tests on the same images. Sign convention: diff = A - B.
    Predefined families (Holm within each; per metric; across budget x pattern cells -- cells are never pooled into an estimate):
      PRE_geo_vs_cons      A=geometric, B=constrained equal-drop (floors 16/24/32)
      PRE_geo_vs_unc       A=geometric, B=unconstrained equal-drop
      PRE_cons_vs_unc      A=constrained (floors 16/24/32, not identical to unconstrained), B=unconstrained
      PRE_geo_vs_ctrl      A=geometric, B=final-count-matched geometric control   (KEY comparison)
    Floor 8 comparisons are exploratory (diagnostic floor)."""
    sel = per_img[(per_img.scorer == scorer) & (per_img.seed == seed)]
    idx = {k: g.sort_values("image_idx") for k, g in sel.groupby("schedule")}
    by_cell: Dict[Tuple, List[Dict[str, Any]]] = {}
    for e in entries:
        if e["role"] == "member":
            by_cell.setdefault((e["group"], e["structure"]), []).append(e)
    rows, skipped = [], []

    def pred(e):
        return idx.get(e["alias_of"] or e["name"])

    def add(ea, eb, kind, geo):
        ga, gb = pred(ea), pred(eb)
        if ga is None or gb is None:
            return
        dfl = ea["normalized_flops"] - eb["normalized_flops"]
        if abs(dfl) > tol + 1e-12:
            skipped.append({"a": ea["name"], "b": eb["name"], "reason": "pairwise |dFLOPs| > tolerance"}); return
        gv = [geo["schedule"][l] for l in geo["layers"]]
        for metric, col in (("agreement", "agreement"), ("top1", "correct")):
            s = paired_diff_stats(ga[col].to_numpy(), gb[col].to_numpy())
            va, vb = [ea["schedule"][l] for l in ea["layers"]], [eb["schedule"][l] for l in eb["layers"]]
            rows.append({
                "comparison": kind, "budget_target": ea["budget_target"], "structure": ea["structure"], "metric": metric,
                "family_a": _label(ea), "family_b": _label(eb), "floor_b": eb["floor"] if eb["family"] == "constrained_equal_drop" else None,
                "floor_a": ea["floor"], "schedule_a": ea["name"], "schedule_b": eb["name"],
                "vector_a": json.dumps(va), "vector_b": json.dumps(vb), "geometric_vector": json.dumps(gv),
                "final_tokens_a": va[-1], "final_tokens_b": vb[-1], "same_final_count": va[-1] == vb[-1],
                "flops_pct_a": 100 * ea["normalized_flops"], "flops_pct_b": 100 * eb["normalized_flops"], "flops_diff_pp": 100 * dfl,
                "flops_tolerance_satisfied": abs(dfl) <= tol + 1e-12,
                "a_differs_from_geometric": va != gv, "b_differs_from_geometric": vb != gv,
                "dist_a_to_geometric": _traj_dist(va, gv), "dist_b_to_geometric": _traj_dist(vb, gv),
                "n_images": s["n"], "count_a": s["a_count"], "count_b": s["b_count"], "diff": s["diff"], "ci_low": s["boot_ci_low"],
                "ci_high": s["boot_ci_high"], "a_only": s["a_only"], "b_only": s["b_only"], "p_raw": s["mcnemar_p"]})

    for cell, ents in by_cell.items():
        geo = next((e for e in ents if e["family"] == "geometric"), None)
        unc = next((e for e in ents if e["family"] == "equal_drop"), None)
        if geo is None:
            continue
        cons = [e for e in ents if e["family"] == "constrained_equal_drop"]
        for e in cons:
            add(geo, e, "geo_vs_cons", geo)
            if unc is not None and not e["alias_of"]:
                add(e, unc, "cons_vs_unc", geo)
        if unc is not None:
            add(geo, unc, "geo_vs_unc", geo)
        for e in [x for x in ents if x["family"] == CONTROL_FAMILY]:
            add(geo, e, "geo_vs_ctrl", geo)
    df = pd.DataFrame(rows)
    df.attrs["skipped_pairs"] = skipped
    if df.empty:
        return df

    def family(r):
        if r.comparison in ("geo_vs_cons", "cons_vs_unc"):
            f = r.floor_b if r.comparison == "geo_vs_cons" else r.floor_a
            if f not in primary_floors:
                return f"exploratory_diagnostic_floor{int(f)}|{r.comparison}|{r.metric}"
        return f"PRE_{r.comparison}|{r.metric}"

    df["family"] = df.apply(family, axis=1)
    df["predefined"] = df.family.str.startswith("PRE_")
    df["family_size"] = df.groupby("family").family.transform("size")
    df["p_holm"] = np.nan
    for _, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm_correction(g.p_raw.tolist())
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["significant_raw_0.05_uncorrected"] = df.p_raw < 0.05
    return df


def classify_control_outcome(r) -> str:
    """Descriptive verdict per geometric-vs-control comparison (agreement metric)."""
    if r.p_holm < 0.05:
        return "geometric_better" if r["diff"] > 0 else "control_better"
    if 100 * r.ci_low > -EQUIVALENCE_MARGIN_PP and 100 * r.ci_high < EQUIVALENCE_MARGIN_PP:
        return "similar"
    return "inconclusive"


# ----------------------------------------------------------------------------- run

def _metric_row(e, seed, n_img, agree, correct, t5, scorer):
    row = {"schedule": e["name"], "group": e["group"], "budget_target": e["budget_target"], "structure": e["structure"],
           "allocation_family": e["family"], "floor": e["floor"], "role": e["role"], "stage_count": e["stage_count"],
           "scorer": scorer, "seed": seed, "final_patch_tokens": e["final_patch_tokens"],
           "schedule_json": json.dumps({str(k): v for k, v in e["schedule"].items()}),
           "vector": json.dumps(list(e["schedule"].values())), "n_images": n_img, "analytic_flops": e["analytic_flops"],
           "normalized_flops": e["normalized_flops"], "flops_pct": 100 * e["normalized_flops"],
           "trajectory_dist_to_geometric": e.get("trajectory_dist"), "control_of": e.get("control_of"),
           "control_side": e.get("control_side"), "control_kind": e.get("control_kind")}
    for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
        w = wilson_score_interval(int(arr.sum()), n_img)
        row.update({f"{name}_count": int(arr.sum()), name: w["mean"], f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
    return row


def prepare_isolation(cfg: TrajConfig, est: ComputeEstimator) -> Dict[str, Any]:
    """Everything checkable without running the model: manifest verification, schedules, controls, prior-rule reproduction."""
    arts = {}
    if cfg.prev_min_token_dir:
        arts["min_token"] = verify_manifest_artifact(cfg.prev_min_token_dir)
    if cfg.prev_class_diverse_dir:
        arts["class_diverse"] = verify_manifest_artifact(cfg.prev_class_diverse_dir)
    if len({a["manifest_sha256_full"] for a in arts.values()}) > 1:
        raise RuntimeError("Previous artifacts disagree on the manifest checksum. Stopping.")
    sched = generate_isolation_schedules(est, cfg)
    repro = verify_prior_schedules(sched["entries"], sched["infeasible"], cfg.prev_min_token_dir, cfg.patterns) \
        if cfg.prev_min_token_dir and cfg.budgets == [0.4, 0.5, 0.6, 0.7] and cfg.floors == [8, 16, 24, 32] and cfg.max_images is None \
        else {"verified": False, "note": "not checked (smoke / custom setup)"}
    return {"artifacts": arts, "schedules": sched, "reproduction": repro}


def run_isolation(cfg: TrajConfig, wrapper: Optional[ViTWrapper] = None, data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  prepared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t_run = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    wrapper = wrapper or ViTWrapper(cfg.model_name, cfg.pretrained, cfg.device)
    L, N = wrapper.num_layers, wrapper.num_original_tokens
    est = ComputeEstimator(L, wrapper.hidden_dim, wrapper.num_heads, N, wrapper.num_classes, wrapper.num_prefix_tokens, wrapper.patch_size)
    dev = wrapper.device
    out_p = lambda n: os.path.join(cfg.output_dir, n)

    sub = load_subset(ClassDiverseConfig(parquet_dir=cfg.parquet_dir, subset_cache=cfg.subset_cache, images_per_class=cfg.images_per_class,
                                         subset_seed=cfg.subset_seed, max_images=cfg.max_images), data)
    images, labels, manifest, summary = sub["images"], sub["labels"], sub["manifest"], sub["summary"]
    n_img = len(images)
    full = cfg.max_images is None and data is None
    subset_checks = {}
    if full:
        for k, d in (("min_token", cfg.prev_min_token_dir), ("class_diverse", cfg.prev_class_diverse_dir)):
            if d:
                subset_checks[k] = verify_same_subset(manifest, summary, d)
    else:
        subset_checks = {"note": "smoke / test run on supplied or truncated data; exact-manifest verification only in the full run"}
    manifest.to_csv(out_p("subset_manifest.csv"), index=False)
    json.dump(summary, open(out_p("subset_summary.json"), "w"), indent=2)
    batches = [(images[i:i + cfg.batch_size], labels[i:i + cfg.batch_size]) for i in range(0, n_img, cfg.batch_size)]
    with torch.no_grad():
        base_logits = [wrapper(x.to(dev)) for x, _ in batches]
    base_pred = torch.cat([l.argmax(-1) for l in base_logits]).cpu().numpy()
    y_all = labels.cpu().numpy()
    image_id = manifest.image_id.to_numpy()

    prep = prepared or prepare_isolation(cfg, est)
    sc = prep["schedules"]
    entries, groups, infeasible, identical = sc["entries"], sc["groups"], sc["infeasible"], sc["identical"]
    for e in entries:
        e["analysis"] = "isolation"
    base_entry = {"name": "no_pruning", "schedule": {}, "stage_count": 0, "first_layer": None, "structure": "", "layers": [],
                  "budget_target": 1.0, "group": "baseline", "role": "baseline", "family": "none", "floor": None, "alias_of": None,
                  "analysis": "baseline", "final_patch_tokens": N, "floor_satisfied": None, "control_of": None, "control_side": None,
                  "control_kind": None, "trajectory_dist": None, **schedule_flops(est, {})}
    exec_entries = [e for e in entries if not e["alias_of"]]
    runs = [base_entry] + exec_entries
    scorer, seed = cfg.scorer, cfg.seed

    ck_path = out_p("_checkpoint.pkl")
    rows, per_image, token_rows, failures, done = [], [], [], [], set()
    sig = hashlib.sha256((json.dumps(cfg.to_dict(), sort_keys=True, default=str) + summary["manifest_sha256"]
                          + json.dumps([e["name"] + str(e["schedule"]) for e in runs])).encode()).hexdigest()
    if cfg.resume and os.path.isfile(ck_path):
        ck = pickle.load(open(ck_path, "rb"))
        if ck["sig"] == sig:
            rows, per_image, token_rows, failures, done = ck["rows"], ck["per_image"], ck["token_rows"], ck["failures"], ck["done"]
            print(f"[resume] {len(done)} runs already complete", flush=True)

    t_start = time.time()
    for ri, e in enumerate(runs):
        if e["name"] in done:
            continue
        sname = "none" if e["role"] == "baseline" else scorer
        try:
            pruner = ProgressivePruner(wrapper, e["schedule"], scorer="random" if sname == "none" else scorer, seed=seed)
            preds, top5, trace = [], [], None
            for (x, _), bl in zip(batches, base_logits):
                res = pruner.forward(x.to(dev), orig_logits=bl)
                trace = res.token_trace
                preds.append(res.logits.argmax(-1).cpu()); top5.append(res.logits.topk(5, dim=-1).indices.cpu())
                if sname == "none":
                    assert (res.logits - bl).abs().max().item() <= 1e-4
            preds, top5 = torch.cat(preds).numpy(), torch.cat(top5).numpy()
        except Exception as ex:
            failures.append({"schedule": e["name"], "error": f"{type(ex).__name__}: {ex}"})
            print(f"[FAILED] {e['name']}: {ex}", flush=True)
            done.add(e["name"]); continue
        correct, agree, t5 = preds == y_all, preds == base_pred, (top5 == y_all[:, None]).any(1)
        per_image.append(pd.DataFrame({
            "schedule": e["name"], "scorer": sname, "seed": seed, "image_idx": np.arange(n_img), "image_id": image_id, "label": y_all,
            "baseline_pred": base_pred, "compressed_pred": preds, "top5_preds": [" ".join(map(str, r)) for r in top5],
            "baseline_correct": (base_pred == y_all).astype(int), "correct": correct.astype(int), "top5_correct": t5.astype(int),
            "agreement": agree.astype(int)}))
        for t in trace:
            token_rows.append({"schedule": e["name"], "family": e["family"], **t})
        rows.append(_metric_row(e, seed, n_img, agree, correct, t5, sname))
        done.add(e["name"])
        if len(done) % 10 == 0 or ri + 1 == len(runs):
            print(f"[{len(done)}/{len(runs)}] {e['name']} ({time.time() - t_start:.0f}s)", flush=True)
            pickle.dump({"sig": sig, "rows": rows, "per_image": per_image, "token_rows": token_rows, "failures": failures, "done": done},
                        open(ck_path, "wb"))

    metrics = pd.DataFrame(rows)
    unpruned_latency, lat_info, lat_rows = None, {}, []
    if cfg.measure_latency and len(metrics):
        nb = len(batches)
        pick = sorted({int(round(i * (nb - 1) / max(cfg.latency_batches - 1, 1))) for i in range(cfg.latency_batches)})
        xs_all = [batches[i][0].to(dev) for i in pick]
        lat_info = {"batch_indices": pick, "batch_size": int(len(xs_all[0]))}
        with torch.no_grad():
            wrapper(xs_all[0])
            ts = []
            for x in xs_all:
                for _ in range(cfg.latency_runs):
                    t0 = time.perf_counter(); wrapper(x); ts.append((time.perf_counter() - t0) * 1e3)
        unpruned_latency = {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts), "n_timings": len(ts)}
        for e in runs:
            if e["name"] not in set(metrics.schedule):
                continue
            p = ProgressivePruner(wrapper, e["schedule"], "random" if e["role"] == "baseline" else scorer, seed=seed)
            r = _time_multi(p, xs_all, cfg.latency_warmup, cfg.latency_runs)
            lat_rows.append({"schedule": e["name"], "allocation_family": e["family"], "budget_target": e["budget_target"],
                             "structure": e["structure"], "flops_pct": 100 * e["normalized_flops"],
                             "unpruned_forward_ms_median": unpruned_latency["median_ms"], "end_to_end_ms_median": r["total_ms_median"],
                             "end_to_end_ms_mean": r["total_ms_mean"], "end_to_end_ms_min_batch": r["total_ms_min_batch"],
                             "end_to_end_ms_max_batch": r["total_ms_max_batch"],
                             "scorer_overhead_ms_median": 0.0 if e["role"] == "baseline" else r["scorer_ms_median"],
                             "device": str(dev), "batch_size": lat_info["batch_size"], "warmup": cfg.latency_warmup,
                             "timed_runs": cfg.latency_runs, "num_batches": len(xs_all)})
        lat_df = pd.DataFrame(lat_rows)
        lat_df.to_csv(out_p("latency.csv"), index=False)
        lm = lat_df.set_index("schedule")
        for c in ("end_to_end_ms_median", "scorer_overhead_ms_median"):
            metrics[c] = [lm[c].get(s, np.nan) for s in metrics.schedule]
    else:
        lat_df = pd.DataFrame()

    # alias rows: constrained schedules identical to the unconstrained one reuse its (identical) results
    if len(metrics):
        metrics["is_alias"] = False; metrics["executed_as"] = metrics.schedule
        extra = []
        for e in entries:
            if e["alias_of"]:
                for r in metrics[metrics.schedule == e["alias_of"]].itertuples(index=False):
                    d = r._asdict()
                    d.update({"schedule": e["name"], "allocation_family": e["family"], "floor": e["floor"], "is_alias": True,
                              "executed_as": e["alias_of"], "vector": json.dumps(list(e["schedule"].values())),
                              "control_of": None, "control_side": None, "control_kind": None})
                    extra.append(d)
        if extra:
            metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
    per_img_df = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    tc_df = pd.DataFrame(token_rows)
    metrics.to_csv(out_p("metrics.csv"), index=False)
    metrics.to_json(out_p("metrics.json"), orient="records", indent=1)
    per_img_df.to_csv(out_p("per_image_predictions.csv.gz"), index=False, compression="gzip")
    tc_df.to_csv(out_p("token_counts.csv"), index=False)
    stats_df = build_isolation_tests(per_img_df, entries, scorer, seed, cfg.primary_floors, cfg.match_tolerance) if len(per_img_df) else pd.DataFrame()
    stats_df.to_csv(out_p("statistical_tests.csv"), index=False)
    cand_df = pd.DataFrame(sc["candidates"])
    cand_df.to_csv(out_p("control_candidates.csv.gz"), index=False, compression="gzip")

    # schedule table (machine readable; includes infeasible rows)
    sched_rows = []
    for e in [base_entry] + entries:
        v = list(e["schedule"].values())
        sched_rows.append({"name": e["name"], "status": "executed" if not e["alias_of"] else "alias_of_unconstrained", "family": e["family"],
                           "budget_target": e["budget_target"], "structure": e["structure"], "floor": e["floor"], "role": e["role"],
                           "layers": json.dumps(e["layers"]), "vector": json.dumps(v), "final_tokens": v[-1] if v else N,
                           "initial_tokens": N, "flops_pct": 100 * e["normalized_flops"], "alias_of": e["alias_of"],
                           "control_of": e.get("control_of"), "control_side": e.get("control_side"),
                           "trajectory_dist_to_geometric": e.get("trajectory_dist"), "reason": ""})
    for u in infeasible:
        sched_rows.append({"name": f"L{cfg.first_layer}_B{int(round(100 * u['budget']))}_constrained_f{u['floor']}_{u['structure']}",
                           "status": "infeasible", "family": "constrained_equal_drop", "budget_target": u["budget"], "structure": u["structure"],
                           "floor": u["floor"], "reason": f"{u['reason']} (best diff {u['best_diff_pp']:.2f} pp)" if u["best_diff_pp"] is not None else u["reason"]})
    for u in sc["unmatched"]:
        sched_rows.append({"name": f"L{cfg.first_layer}_B{int(round(100 * u['budget']))}_{u['family']}_{u['structure']}", "status": "unmatched",
                           "family": u["family"], "budget_target": u["budget"], "structure": u["structure"], "reason": u["reason"]})
    for u in sc["control_infeasible"]:
        sched_rows.append({"name": f"{u['geometric']}_controls", "status": "infeasible", "family": CONTROL_FAMILY,
                           "budget_target": u["budget"], "structure": u["structure"], "reason": u["reason"]})
    pd.DataFrame(sched_rows).to_csv(out_p("schedule_table.csv"), index=False)

    # reproduction of previous per-schedule results (deterministic scorer => identical counts expected)
    prior_repro = {"checked": False}
    if cfg.prev_min_token_dir and full:
        pm = pd.read_csv(os.path.join(cfg.prev_min_token_dir, "metrics.csv"))
        pm = pm[(pm.scorer == scorer) & (pm.seed == seed)].set_index("schedule")
        cur = metrics[~metrics.is_alias] if "is_alias" in metrics else metrics
        cmp_ = [(r.schedule, r.agreement_count == pm.agreement_count.get(r.schedule), r.top1_count == pm.top1_count.get(r.schedule))
                for r in cur.itertuples() if r.schedule in pm.index]
        bad = [c[0] for c in cmp_ if not (c[1] and c[2])]
        prior_repro = {"checked": True, "n_compared": len(cmp_), "n_identical_counts": len(cmp_) - len(bad), "mismatches": bad}

    run_manifest = {
        "layer_numbering_and_semantics": LAYER_NUMBERING,
        "module_doc": __doc__.strip(),
        "model": {"name": cfg.model_name, "pretrained": cfg.pretrained, **_environment(wrapper)["checkpoint"]},
        "dataset": {"num_images": n_img, "manifest_sha256_full": summary["manifest_sha256"], "manifest_file": out_p("subset_manifest.csv"),
                    "manifest_file_sha256": _sha256_file(out_p("subset_manifest.csv")),
                    "image_id_order_sha256": hashlib.sha256(",".join(map(str, image_id)).encode()).hexdigest(),
                    "image_ids_in_order": "see subset_manifest.csv (column image_id, order = position)",
                    "preprocessing": "Phase 1 transforms via the class-diverse subset builder (cached tensors, unchanged)",
                    "verification_vs_previous_artifacts": subset_checks, "previous_artifact_checksums": prep["artifacts"]},
        "seed": seed, "scorer": {"name": scorer, "module": "src/compression/scoring (unchanged Phase 1 CLS-attention scorer)",
                                 "deterministic": True},
        "flops_method": "ComputeEstimator.compute_progressive_flops via progressive_eval.schedule_flops (full-path analytic FLOPs, normalized to the unpruned model)",
        "match_tolerance_pp": 100 * cfg.match_tolerance, "baseline_flops": est.baseline_flops, "budgets": cfg.budgets, "patterns": cfg.patterns,
        "control_selection_rule": {"min_deviation_tokens": cfg.control_min_deviation, "doc": select_controls.__doc__,
                                   "candidate_comparison_reference": "original geometric schedule FLOPs, +-tolerance"},
        "prior_rule_reproduction": prep["reproduction"], "prior_result_reproduction": prior_repro,
        "schedules": [{"name": e["name"], "family": e["family"], "vector": list(e["schedule"].values()), "layers": e["layers"],
                       "schedule": {str(k): v for k, v in e["schedule"].items()}, "final_tokens": e["final_patch_tokens"],
                       "normalized_flops": e["normalized_flops"], "alias_of": e["alias_of"], "floor": e["floor"],
                       "patch_tokens_processed_per_block": expected_processed_counts(e["schedule"], L, N)} for e in [base_entry] + entries],
        "infeasible_constrained": infeasible, "unmatched": sc["unmatched"], "identical_to_unconstrained": identical,
        "control_infeasible": sc["control_infeasible"],
        "control_candidate_status_counts": cand_df.groupby(["geometric", "status"]).size().unstack(fill_value=0).to_dict("index") if len(cand_df) else {},
        "skipped_pairs": stats_df.attrs.get("skipped_pairs", []) if len(stats_df) else [],
        "environment": _environment(wrapper), "hardware": {"device": str(dev), "cpu": platform.processor(), "machine": platform.machine(),
                                                         "gpu_available": torch.cuda.is_available()},
        "timing": {"method": "time.perf_counter around ProgressivePruner.forward (scoring + pruning + blocks + head); median over timed runs "
                             "on evenly spaced batches after warm-up; same procedure as the class-diverse / minimum-token studies. "
                             "Scorer overhead = time inside scorer calls; unpruned forward reported separately. CPU only: no GPU speedup is implied.",
                   "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs, "batches": lat_info,
                   "unpruned_model_forward": unpruned_latency, "enabled": cfg.measure_latency},
        "failed": failures, "config": cfg.to_dict(), "runtime_seconds_total": time.time() - t_run, "runtime_seconds_model_runs": time.time() - t_start,
        "num_executed_runs": len(runs)}
    json.dump(run_manifest, open(out_p("run_manifest.json"), "w"), indent=2, default=str)
    out = {"metrics": metrics, "per_image": per_img_df, "token_counts": tc_df, "stats": stats_df, "entries": [base_entry] + entries,
           "candidates": cand_df, "latency": lat_df, "infeasible": infeasible, "unmatched": sc["unmatched"], "identical": identical,
           "control_infeasible": sc["control_infeasible"], "manifest": run_manifest, "failures": failures, "subset_summary": summary,
           "estimator": est}
    finalize_reports(out, cfg)
    if os.path.isfile(ck_path) and not failures:
        os.remove(ck_path)
    return out


def finalize_reports(out: Dict[str, Any], cfg: TrajConfig) -> None:
    if cfg.make_plots and len(out["metrics"]):
        from ..visualization.trajectory_plots import make_all_plots
        make_all_plots(out["metrics"], out["stats"], out["entries"], cfg.output_dir)
    write_result(out, cfg)


# ----------------------------------------------------------------------------- result.md

def _fmt_vec(v):
    return str(json.loads(v)) if isinstance(v, str) else str(v)


def write_result(out: Dict[str, Any], cfg: TrajConfig) -> None:
    m, st, man, ss = out["metrics"], out["stats"], out["manifest"], out["subset_summary"]
    cand = out["candidates"]
    n = int(m.n_images.iloc[0]) if len(m) else 0
    ds = man["dataset"]
    L = ["# Phase 2: Final token count vs reduction trajectory (matched-FLOPs isolation experiment)\n",
         "*One model, one class-balanced validation sample, CPU. Raw and Holm-adjusted p-values shown separately. Cells (budget x pattern) are reported separately; aggregates are descriptive only.*\n",
         "## Setup and verification\n",
         f"- Subset: {ss['num_images']} images / {ss['num_distinct_classes']} classes. Full manifest SHA-256 `{ss['manifest_sha256']}`. "
         + ("Recomputed from the saved manifest files of the previous minimum-token and class-diverse runs; ids, labels, order and per-image hashes identical."
            if isinstance(ds["verification_vs_previous_artifacts"], dict) and "min_token" in ds["verification_vs_previous_artifacts"]
            else "Exact-manifest verification not performed in this run (smoke/test)."),
         f"- Prior-rule reproduction: " + (f"A / C / D schedules and the infeasible set equal the saved minimum-token manifest ({man['prior_rule_reproduction']['schedules_compared']} schedules)."
                                           if man["prior_rule_reproduction"].get("verified") else str(man["prior_rule_reproduction"].get("note"))),
         "- Prior per-schedule results: " + (f"{man['prior_result_reproduction']['n_identical_counts']}/{man['prior_result_reproduction']['n_compared']} schedules reproduce the previous agreement and top-1 counts exactly."
                                              + (f" Mismatches: {man['prior_result_reproduction']['mismatches']}" if man['prior_result_reproduction']['mismatches'] else "")
                                              if man["prior_result_reproduction"].get("checked") else "not checked"),
         f"- Scorer `{cfg.scorer}` (deterministic), seed {cfg.seed}, first layer {cfg.first_layer}, patterns {cfg.patterns}, budgets {cfg.budgets}, tolerance ±{100 * cfg.match_tolerance:.1f} pp of baseline FLOPs.",
         f"- Layer numbering: {LAYER_NUMBERING}",
         "- Families: A original geometric; B final-count-matched geometric control (same layers, same initial count N, exactly the geometric final count, only intermediate counts differ, FLOPs within tolerance of the *original geometric* schedule); C constrained equal-drop (floors 16/24/32, floor 8 diagnostic); D unconstrained equal-drop (negative control). A, C, D are the unchanged previous implementations.",
         f"- Control selection (fixed before any result): every strictly decreasing integer vector is enumerated and classified; a control is *distinct* only if its largest intermediate-count deviation from geometric is ≥ {cfg.control_min_deviation} tokens; per side (retaining / steeper first intermediate count) the most-deviating ('extreme') and the half-deviation ('mid') candidates are run. Selection uses trajectory and FLOPs only, never accuracy.",
         "- Sign convention: diff = A − B; for geometric comparisons A = geometric (positive = geometric better); for constrained-vs-unconstrained A = constrained.\n"]

    # --- controls
    L.append("## Final-count-matched controls (the key new part)\n")
    cs = cand.groupby(["geometric", "status"]).size().unstack(fill_value=0) if len(cand) else pd.DataFrame()
    rows = []
    for geo in [e for e in out["entries"] if e["family"] == "geometric"]:
        row = cs.loc[geo["name"]].to_dict() if geo["name"] in cs.index else {}
        ctrl = [e for e in out["entries"] if e["family"] == CONTROL_FAMILY and e["control_of"] == geo["name"]]
        rows.append([f"B{int(100 * geo['budget_target'])}", geo["structure"], list(geo["schedule"].values()), f"{100 * geo['normalized_flops']:.2f}%",
                     sum(row.values()), row.get("feasible_distinct", 0), row.get("feasible_near_identical", 0), row.get("flops_out_of_tolerance", 0),
                     len(ctrl)])
    L.append(_md_table(rows, ["budget", "pattern", "geometric vector", "geom. FLOPs", "all candidates (same final count)", "feasible distinct",
                              "feasible near-identical (<min dev)", "out of FLOPs tol.", "controls run"]))
    nctrl = sum(1 for e in out["entries"] if e["family"] == CONTROL_FAMILY)
    L.append(f"\nControl schedules executed: **{nctrl}**. Controls infeasible for: "
             + (", ".join(f"{u['geometric']}" for u in out["control_infeasible"]) or "none") + ".\n")

    cell = st[(st.comparison == "geo_vs_ctrl") & (st.metric == "agreement")] if len(st) else pd.DataFrame()
    cell1 = st[(st.comparison == "geo_vs_ctrl") & (st.metric == "top1")] if len(st) else pd.DataFrame()
    if len(cell):
        L.append("### Geometric (A) vs final-count-matched controls (B): agreement, per cell [PREDEFINED family PRE_geo_vs_ctrl]\n")
        top1 = {r.schedule_b: r for r in cell1.itertuples()}
        rows = []
        for r in cell.sort_values(["budget_target", "structure", "schedule_b"]).itertuples():
            t = top1.get(r.schedule_b)
            rows.append([f"B{int(100 * r.budget_target)}", r.structure, r.vector_a, r.vector_b, f"{r.final_tokens_a} / {r.final_tokens_b}",
                         f"{r.flops_pct_a:.2f} / {r.flops_pct_b:.2f} (Δ{r.flops_diff_pp:+.2f})", r.dist_b_to_geometric,
                         f"{r.count_a} vs {r.count_b}", f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]",
                         f"{r.p_raw:.3g}", f"{r.p_holm:.3g}", f"{100 * t.diff:+.2f} (Holm p {t.p_holm:.3g})" if t is not None else ""])
        L.append(_md_table(rows, ["budget", "pattern", "geometric vector", "control vector", "final A / B", "FLOPs % A / B", "traj. dist",
                                  "agree count A vs B", "Δ pp [95% paired-bootstrap CI]", "raw p", "Holm p", "top-1 Δ pp"]))
        oc = cell.apply(classify_control_outcome, axis=1)
        vc = oc.value_counts().to_dict()
        L.append(f"\nOutcome per control comparison (agreement; 'similar' = CI inside ±{EQUIVALENCE_MARGIN_PP:g} pp and not Holm-significant; descriptive rule): {vc}.\n")
        L.append("Per cell:\n")
        c2 = cell.assign(outcome=oc.values)
        L.append(_md_table([[f"B{int(100 * b)}", s, ", ".join(f"{k}: {v}" for k, v in g.outcome.value_counts().items()),
                             f"{100 * g['diff'].mean():+.2f}", f"{100 * g['diff'].min():+.2f} to {100 * g['diff'].max():+.2f}"]
                            for (b, s), g in c2.groupby(["budget_target", "structure"])],
                           ["budget", "pattern", "outcomes", "mean Δ pp (descriptive)", "range Δ pp"]))
    else:
        L.append("**No valid final-count-matched controls could be run: the final-count explanation remains unresolved by this experiment.**\n")

    # --- other comparisons
    def cmp_table(kind, title, floors=None):
        d = st[(st.comparison == kind) & (st.metric == "agreement")] if len(st) else pd.DataFrame()
        if len(d) == 0:
            return
        L.append(f"### {title}\n")
        rows = []
        for r in d.sort_values(["budget_target", "structure", "family_b" if kind == "geo_vs_cons" else "family_a"]).itertuples():
            rows.append([f"B{int(100 * r.budget_target)}", r.structure, r.family, f"{r.family_a} vs {r.family_b}", r.vector_a, r.vector_b,
                         f"{r.final_tokens_a} / {r.final_tokens_b}", f"{r.flops_pct_a:.2f} / {r.flops_pct_b:.2f} (Δ{r.flops_diff_pp:+.2f})",
                         "yes" if r.b_differs_from_geometric else "no", f"{r.count_a} vs {r.count_b}",
                         f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]", f"{r.p_raw:.3g}", f"{r.p_holm:.3g}"])
        L.append(_md_table(rows, ["budget", "pattern", "family", "A vs B", "A vector", "B vector", "final A / B", "FLOPs % A / B",
                                  "B differs from geom.", "agree count", "Δ pp [95% CI]", "raw p", "Holm p"]))
        L.append("")

    L.append("## Other predefined comparisons (agreement; top-1 in `statistical_tests.csv`)\n")
    cmp_table("geo_vs_cons", "1. Geometric (A) vs constrained equal-drop (B)")
    cmp_table("geo_vs_unc", "2. Geometric (A) vs unconstrained equal-drop (B)")
    cmp_table("cons_vs_unc", "3. Constrained (A) vs unconstrained equal-drop (B)")

    # --- infeasible
    L.append("## Infeasible / unmatched combinations (never relaxed)\n")
    inf = [f"B{int(100 * u['budget'])} {u['structure']} floor {u['floor']} (best Δ {u['best_diff_pp']:.2f} pp)" for u in out["infeasible"]]
    L.append("- Constrained equal-drop: " + ("; ".join(inf) if inf else "none"))
    L.append("- Unmatched geometric / equal-drop: " + ("; ".join(f"B{int(100 * u['budget'])} {u['structure']} {u['family']}" for u in out["unmatched"]) or "none"))
    L.append("- Constrained schedules identical to the unconstrained one (floor not binding; executed once, Δ = 0 by construction, excluded from cons-vs-unc tests): "
             + (", ".join(f"B{int(100 * i['budget'])} {i['structure']} f{i['floor']}" for i in out["identical"]) or "none"))
    L.append("- Final-count-matched controls: " + (", ".join(u["geometric"] for u in out["control_infeasible"]) or "feasible for every geometric schedule") + "\n")

    # --- descriptive aggregate
    if len(st):
        L.append("## Descriptive aggregate over cells (NOT a pooled test; agreement Δ pp, A − B)\n")
        a = st[(st.metric == "agreement") & st.predefined]
        L.append(_md_table([[k, len(g), f"{100 * g['diff'].mean():+.2f}", f"{100 * g['diff'].min():+.2f}", f"{100 * g['diff'].max():+.2f}",
                             int(g['significant_holm_0.05'].sum())] for k, g in a.groupby("comparison")],
                           ["comparison", "n pairs", "mean", "min", "max", "Holm-significant"]))
        L.append("")

    # --- FLOPs / accuracy table
    L.append("## All schedules (cls_attention, seed 42; Wilson 95% CI)\n")
    mm = m[m.role != "baseline"].sort_values(["budget_target", "structure", "allocation_family", "floor"])
    L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.structure, (r.allocation_family if pd.isna(r.floor) else f"{r.allocation_family} f{int(r.floor)}")
                         + ("" if not r.is_alias else " (=unc.)") + (f" [{r.control_side}/{r.control_kind}]" if isinstance(r.control_side, str) else ""),
                         _fmt_vec(r.vector), r.final_patch_tokens, f"{r.flops_pct:.2f}%",
                         f"{r.agreement_count}/{r.n_images} {100 * r.agreement:.1f}% [{100 * r.agreement_ci_low:.1f}, {100 * r.agreement_ci_high:.1f}]",
                         f"{100 * r.top1:.1f}% [{100 * r.top1_ci_low:.1f}, {100 * r.top1_ci_high:.1f}]"] for r in mm.itertuples()],
                       ["budget", "pattern", "family", "vector", "final k", "FLOPs", "agreement", "top-1"]))

    # --- latency
    lat = out["latency"]
    L.append("\n## Latency (CPU; secondary systems metric; separate from FLOPs)\n")
    if len(lat):
        t = man["timing"]
        L.append(f"Device `{man['hardware']['device']}`, batch {lat.batch_size.iloc[0]}, warm-up {t['warmup']}, {t['timed_runs']} timed runs on {len(t['batches'].get('batch_indices', []))} batches; unpruned forward {t['unpruned_model_forward']['median_ms']:.1f} ms (median). "
                 "End-to-end = scoring + pruning + blocks + head; scorer overhead shown separately. CPU numbers do not establish GPU speedups, and latency is never inferred from FLOPs.\n")
        L.append(_md_table([[f, len(g), f"{g.flops_pct.mean():.1f}%", f"{g.end_to_end_ms_median.mean():.0f}",
                             f"{g.scorer_overhead_ms_median.mean():.1f}"] for f, g in lat[lat.allocation_family != "none"].groupby("allocation_family")],
                           ["family", "schedules", "mean FLOPs", "mean end-to-end ms", "mean scorer overhead ms"]))
        L.append("\nPer-schedule values: `latency.csv`.")
    L.append("\n## Limitations\n")
    L += ["- One model, one 2,000-image class-balanced sample, seed 42, CLS-attention only, CPU only; results may not transfer.",
          "- Controls are a finite, accuracy-blind selection from a larger candidate set (`control_candidates.csv.gz`); they do not cover every possible trajectory.",
          "- Controls hold final count and FLOPs fixed but necessarily trade early against middle-stage counts; 'trajectory' here means that intermediate trade-off only.",
          "- Constrained equal-drop differs from geometric in more than the final count (earlier counts are re-searched to match FLOPs).",
          "- Holm families span budget × pattern cells for multiplicity control only; no pooled headline estimate is made.",
          "- Equivalence margin (±2 pp) used in the descriptive verdict is a judgement call, not a pre-registered equivalence test.",
          "- FLOPs are analytic; latency is noisy CPU timing; no speedup claim is made."]
    open(os.path.join(cfg.output_dir, "result.md"), "w", encoding="utf-8").write("\n".join(map(str, L)) + "\n")
