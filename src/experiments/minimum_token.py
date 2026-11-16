"""
Phase 2: minimum final-token constraint and allocation robustness (evaluation only; no optimizer, merging, fine-tuning,
learned routing or new scorers).

Question: is the poor equal-drop performance seen in the class-diverse study mainly caused by near-empty final token counts,
or does geometric allocation help beyond avoiding them?

Layer numbering / schedule semantics (from src/compression/progressive.py, unchanged): keys are 1-based layer numbers; key l =
"after l blocks have run (0-based block l-1), keep exactly k PATCH tokens". CLS is always kept and is not counted in k. The FLOOR
applies to the patch-token count after the final pruning point (CLS excluded).

Allocation families (geometric and equal_drop are IMPORTED UNCHANGED from class_diverse.py; their schedules are re-generated here and
asserted identical to the saved class-diverse schedule manifest):
  geometric   : K_j = max(1, round(N * rho**j))
  equal_drop  : K_j = max(1, round(N - j * D))
  constrained_equal_drop (NEW, separate family, floor F in {8, 16, 24, 32}):
                K_j = max(F, round(N - j * D)),  j = 1..s
    i.e. equal absolute drops of about D patch tokens per stage, truncated so that no stage keeps fewer than F patch tokens (only the
    last stage(s) can be affected because counts decrease). Rounding: Python round() on the float N - j*D (half-to-even; D on a fixed
    0.25 grid), then max with F. D is the single free parameter, searched on the same grid as equal_drop to minimise |FLOPs - F*|.
    A schedule is valid only if counts are strictly decreasing, the final count >= F, and the effective stage count equals s. If no D
    reaches the one-stage FLOPs anchor within the tolerance, the (budget, structure, floor) combination is INFEASIBLE and reported;
    neither the floor nor the tolerance is ever relaxed. If the best constrained schedule is identical to the unconstrained equal_drop
    one (the floor is not binding), it is recorded as `identical_to_unconstrained`, executed once, and excluded from paired tests
    (the difference is exactly zero by construction).
Matching: the one-stage anchor {2: K} nearest to each nominal budget gives the exact FLOPs target; every member must be within
+-tolerance (fraction of baseline FLOPs) of it; paired comparisons additionally require |dFLOPs| <= tolerance directly.
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..compression.progressive import ProgressivePruner, RetentionSchedule
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from .class_diverse import (
    D_GRID, DEFAULT_STRUCTURES, _time_multi, family_counts, generate_family_schedules, load_subset, ClassDiverseConfig,
)
from .layerwise_analysis import wilson_score_interval
from .progressive_eval import (
    LAYER_NUMBERING, _environment, _md_table, _pct, effective_prune_layers, expected_processed_counts, paired_diff_stats,
    schedule_flops,
)
from .stage_ablation import STAGE_NAME, _valid_stage_schedule, holm_correction

PRIMARY_FLOORS = (16, 24, 32)
ALL_FLOORS = (8, 16, 24, 32)       # 8 = diagnostic


@dataclass
class MinTokenConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    parquet_dir: str = "./data/imagenet_val_parquet"
    subset_cache: str = "./data/imagenet_class_diverse_subset.pt"
    images_per_class: int = 2
    subset_seed: int = 42
    batch_size: int = 25
    seed: int = 42
    extra_seeds_random: List[int] = field(default_factory=lambda: [43, 44])
    budgets: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70])
    match_tolerance: float = 0.01
    first_layer: int = 2
    floors: List[int] = field(default_factory=lambda: list(ALL_FLOORS))
    primary_floors: List[int] = field(default_factory=lambda: list(PRIMARY_FLOORS))
    structures: Optional[Dict[str, List[Tuple[int, ...]]]] = None
    primary_scorer: str = "cls_attention"
    secondary_scorers: List[str] = field(default_factory=lambda: ["random", "prediction_kl_approx"])
    # secondary scorers: anchors + these structures/floors/budgets only (compute budget); stated explicitly in outputs
    secondary_structures: List[str] = field(default_factory=lambda: ["2-6", "2-6-10"])
    secondary_floors: List[int] = field(default_factory=lambda: [16, 32])
    random_budgets: List[float] = field(default_factory=lambda: [0.40, 0.50])
    kl_budgets: List[float] = field(default_factory=lambda: [0.50])
    measure_latency: bool = True
    latency_warmup: int = 1
    latency_runs: int = 5
    latency_batches: int = 3
    kl_latency_runs: int = 3
    kl_latency_batches: int = 1
    previous_dir: Optional[str] = "./results/phase2_class_diverse_validation"
    output_dir: str = "./results/phase2_minimum_token_ablation"
    device: Optional[str] = None
    make_plots: bool = True
    resume: bool = True
    max_images: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["structures"] is not None:
            d["structures"] = {s: [list(t) for t in v] for s, v in d["structures"].items()}
        return d


# ----------------------------------------------------------------------------- constrained family

def constrained_counts(N: int, D: float, floor: int, stages: int) -> List[int]:
    """K_j = max(floor, round(N - j*D)), j = 1..stages (see module docstring for the rounding policy)."""
    return [max(int(floor), int(round(N - j * D))) for j in range(1, stages + 1)]


def _ukey(sched: Dict[int, int]) -> Tuple:
    return tuple(sorted(sched.items()))


def generate_min_token_schedules(est: ComputeEstimator, first_layer: int, budgets: List[float],
                                 structures: Dict[str, List[Tuple[int, ...]]], tol: float, floors: List[int]) -> Dict[str, Any]:
    """geometric + equal_drop via the unchanged class-diverse generator, then constrained_equal_drop for every floor."""
    base = generate_family_schedules(est, first_layer, budgets, structures, tol, ["geometric", "equal_drop"])
    L, N = est.num_layers, est.N
    entries = [{**e, "floor": None, "alias_of": None, "final_patch_tokens": (list(e["schedule"].values())[-1]
                                                                             if e["schedule"] else N)} for e in base["entries"]]
    infeasible, identical = [], []
    by_gid = {}
    for e in entries:
        by_gid.setdefault(e["group"], []).append(e)
    for B in budgets:
        gid = f"L{first_layer}_B{int(round(100 * B))}"
        ents = by_gid.get(gid, [])
        anchor = next((e for e in ents if e["role"] == "anchor"), None)
        if anchor is None:
            continue
        for F in floors:
            for kind in ("two", "three"):
                for layers in structures.get(kind, []):
                    s, best = len(layers), None
                    for D in D_GRID:
                        sched = dict(zip(layers, constrained_counts(N, float(D), F, s)))
                        if sched[layers[-1]] < F or not _valid_stage_schedule(sched, layers, est):
                            continue
                        d = schedule_flops(est, sched)["normalized_flops"] - anchor["normalized_flops"]
                        if best is None or abs(d) < abs(best[0]) - 1e-15:
                            best = (d, sched, float(D))
                    struct = "-".join(map(str, layers))
                    if best is None or abs(best[0]) > tol:
                        infeasible.append({"first_layer": first_layer, "budget": B, "floor": F, "structure": struct, "stage_count": s,
                                           "reason": "no valid constrained_equal_drop schedule with final count >= floor within FLOPs tolerance",
                                           "best_diff_pp": None if best is None else 100 * best[0]})
                        continue
                    RetentionSchedule.from_dict(best[1]).validate(L, N)
                    twin = next((e for e in ents if e["family"] == "equal_drop" and e["structure"] == struct), None)
                    alias = twin["name"] if twin is not None and _ukey(twin["schedule"]) == _ukey(best[1]) else None
                    e = {"name": f"{gid}_constrained_f{F}_{STAGE_NAME[s]}_{struct}", "schedule": best[1], "stage_count": s,
                         "first_layer": layers[0], "structure": struct, "layers": list(layers), "budget_target": B, "group": gid,
                         "role": "member", "family": "constrained_equal_drop", "floor": F, "alias_of": alias,
                         "D": best[2], **schedule_flops(est, best[1]), "effective_layers": effective_prune_layers(best[1], L, N),
                         "final_patch_tokens": best[1][layers[-1]]}
                    entries.append(e); ents.append(e)
                    if alias:
                        identical.append({"schedule": e["name"], "identical_to": alias, "floor": F, "structure": struct, "budget": B})
    for e in entries:
        e["floor_satisfied"] = None if e["floor"] is None else bool(e["final_patch_tokens"] >= e["floor"])
    groups = []
    for g in base["groups"]:
        fl = [e["normalized_flops"] for e in entries if e["group"] == g["group"]]
        a = next((e for e in entries if e["group"] == g["group"] and e["role"] == "anchor"), None)
        groups.append({**g, "n_members_total": len(fl) - 1 if a else 0,
                       "max_abs_diff_vs_anchor_pp": 100 * max(abs(f - a["normalized_flops"]) for f in fl) if a else None})
    return {"entries": entries, "groups": groups, "unmatched": base["unmatched"], "infeasible": infeasible, "identical": identical}


def verify_reproduction(entries: List[Dict[str, Any]], prev_dir: Optional[str]) -> Dict[str, Any]:
    """Assert geometric / equal_drop / anchor schedules equal the saved class-diverse manifest (same names and counts)."""
    if prev_dir is None:
        return {"verified": False, "note": "no previous directory supplied"}
    mp = os.path.join(prev_dir, "schedule_manifest.json")
    if not os.path.isfile(mp):
        raise RuntimeError(f"Cannot verify prior allocation rules: {mp} not found. Stopping.")
    prev = {s["name"]: {int(k): v for k, v in s["schedule"].items()} for s in json.load(open(mp))["schedules"] if s["first_layer"] == 2}
    cur = {e["name"]: e["schedule"] for e in entries if e["family"] in ("geometric", "equal_drop", "shared")}
    if cur != {k: v for k, v in prev.items() if k in cur} or set(cur) != {k for k in prev if k.startswith("L2_")}:
        raise RuntimeError("Re-generated geometric/equal_drop/anchor schedules do NOT match the class-diverse manifest. Stopping.")
    return {"verified": True, "schedules_compared": len(cur), "previous_manifest": mp}


def verify_same_subset(manifest: pd.DataFrame, summary: Dict[str, Any], prev_dir: Optional[str]) -> Dict[str, Any]:
    if prev_dir is None:
        return {"verified": False, "note": "no previous directory supplied"}
    mp, sp = os.path.join(prev_dir, "subset_manifest.csv"), os.path.join(prev_dir, "subset_summary.json")
    if not (os.path.isfile(mp) and os.path.isfile(sp)):
        raise RuntimeError(f"Cannot verify subset: {mp} / {sp} not found. Stopping instead of substituting another subset.")
    prev, ps = pd.read_csv(mp), json.load(open(sp))
    cols = ["position", "image_id", "label", "image_sha1"]
    if len(prev) != len(manifest) or not all((prev[c].astype(str).values == manifest[c].astype(str).values).all() for c in cols) \
            or ps["manifest_sha256"] != summary["manifest_sha256"]:
        raise RuntimeError("Image subset does NOT match the class-diverse manifest (ids/labels/order/hash). Stopping.")
    return {"verified": True, "num_images": int(len(manifest)), "manifest_sha256": summary["manifest_sha256"],
            "identical_ids_labels_order_hashes": True}


# ----------------------------------------------------------------------------- statistics

def _fam_label(e):
    return e["family"] if e["floor"] is None else f"constrained_f{e['floor']}"


def build_statistical_tests(per_img: pd.DataFrame, entries: List[Dict[str, Any]], primary_scorer: str, primary_floors: List[int],
                            tol: float, primary_seed: int = 42, first_layer: int = 2) -> pd.DataFrame:
    """
    Paired tests on identical images (same scorer & seed). PREDEFINED families (Holm within each; floors 16/24/32 pooled):
      PRIMARY_cons_vs_uncons_{agreement,top1}  : constrained_equal_drop minus unconstrained equal_drop, same budget/structure
      SECONDARY_cons_vs_geo_{agreement,top1}   : constrained_equal_drop minus geometric, same budget/structure
    (cls_attention, seed 42, first layer 2). Pairs whose constrained schedule is identical to the unconstrained one are excluded
    (difference is zero by construction) and listed in the manifest. Everything else (floor 8, other scorers/seeds, 
    multi-vs-one-stage, equal_drop vs geometric) is exploratory, Holm within its own family.
    """
    idx = {k: g.sort_values("image_idx") for k, g in per_img.groupby(["schedule", "scorer", "seed"])}
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)
    rows, skipped = [], []

    def add(sc, sd, ea, eb, kind, metric, col):
        ga, gb = idx.get((ea["name"] if not ea["alias_of"] else ea["alias_of"], sc, sd)), \
            idx.get((eb["name"] if not eb["alias_of"] else eb["alias_of"], sc, sd))
        if ga is None or gb is None:
            return
        if abs(ea["normalized_flops"] - eb["normalized_flops"]) > tol + 1e-12:
            skipped.append({"a": ea["name"], "b": eb["name"], "reason": "pairwise |dFLOPs| > tolerance"}); return
        s = paired_diff_stats(ga[col].to_numpy(), gb[col].to_numpy())
        rows.append({
            "comparison_type": kind, "budget_target": ea["budget_target"], "scorer": sc, "seed": sd, "metric": metric,
            "family_a": _fam_label(ea), "floor_a": ea["floor"], "family_b": _fam_label(eb), "floor_b": eb["floor"],
            "schedule_a": ea["name"], "schedule_b": eb["name"], "stage_count": ea["stage_count"], "structure": ea["structure"],
            "final_tokens_a": ea["final_patch_tokens"], "final_tokens_b": eb["final_patch_tokens"],
            "flops_a": ea["normalized_flops"], "flops_b": eb["normalized_flops"],
            "flops_diff_pp": 100 * (ea["normalized_flops"] - eb["normalized_flops"]),
            "n_images": s["n"], "count_a": s["a_count"], "count_b": s["b_count"], "diff": s["diff"],
            "ci_low": s["boot_ci_low"], "ci_high": s["boot_ci_high"], "a_only": s["a_only"], "b_only": s["b_only"], "p_raw": s["mcnemar_p"]})

    avail = sorted({(k[1], k[2]) for k in idx if k[1] != "none"})
    for gid, ents in by_group.items():
        anchor = next((e for e in ents if e["role"] == "anchor"), None)
        mem = [e for e in ents if e["role"] == "member"]
        for e in mem:
            for sc, sd in avail:
                for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                    add(sc, sd, e, anchor, "multi_vs_one_stage", metric, col)
        for e in [m for m in mem if m["family"] == "constrained_equal_drop" and not m["alias_of"]]:
            u = next((x for x in mem if x["family"] == "equal_drop" and x["structure"] == e["structure"]), None)
            g = next((x for x in mem if x["family"] == "geometric" and x["structure"] == e["structure"]), None)
            for sc, sd in avail:
                for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                    if u is not None:
                        add(sc, sd, e, u, "constrained_vs_unconstrained", metric, col)
                    if g is not None:
                        add(sc, sd, e, g, "constrained_vs_geometric", metric, col)
        for e in [m for m in mem if m["family"] == "equal_drop"]:
            g = next((x for x in mem if x["family"] == "geometric" and x["structure"] == e["structure"]), None)
            if g is not None:
                for sc, sd in avail:
                    for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                        add(sc, sd, e, g, "equal_drop_vs_geometric", metric, col)
    df = pd.DataFrame(rows)
    df.attrs["skipped_pairs"] = skipped
    if df.empty:
        return df

    def family(r):
        prim = r.scorer == primary_scorer and r.seed == primary_seed and r.floor_a in primary_floors
        if prim and r.comparison_type == "constrained_vs_unconstrained":
            return f"PRIMARY_cons_vs_uncons_{r.metric}"
        if prim and r.comparison_type == "constrained_vs_geometric":
            return f"SECONDARY_cons_vs_geo_{r.metric}"
        return f"exploratory|{r.comparison_type}|{r.family_a}|{r.scorer}|seed{r.seed}|{r.metric}"

    df["family"] = df.apply(family, axis=1)
    df["is_primary"] = df.family.str.startswith("PRIMARY_")
    df["is_predefined_secondary"] = df.family.str.startswith("SECONDARY_")
    df["family_size"] = df.groupby("family").family.transform("size")
    df["p_holm"] = np.nan
    for fam, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm_correction(g.p_raw.tolist())
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["significant_raw_0.05_uncorrected"] = df.p_raw < 0.05
    return df


# ----------------------------------------------------------------------------- run

def run_min_token(cfg: MinTokenConfig, wrapper: Optional[ViTWrapper] = None,
                  data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
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
    subset_check = verify_same_subset(manifest, summary, cfg.previous_dir) if cfg.max_images is None else \
        {"verified": False, "note": "smoke run on a prefix of the subset"}
    manifest.to_csv(out_p("subset_manifest.csv"), index=False)
    json.dump(summary, open(out_p("subset_summary.json"), "w"), indent=2)
    batches = [(images[i:i + cfg.batch_size], labels[i:i + cfg.batch_size]) for i in range(0, n_img, cfg.batch_size)]
    with torch.no_grad():
        base_logits = [wrapper(x.to(dev)) for x, _ in batches]
    base_pred = torch.cat([l.argmax(-1) for l in base_logits]).cpu().numpy()
    y_all = labels.cpu().numpy()
    image_id = manifest.image_id.to_numpy()

    structures = cfg.structures or DEFAULT_STRUCTURES[cfg.first_layer]
    g = generate_min_token_schedules(est, cfg.first_layer, cfg.budgets, structures, cfg.match_tolerance, cfg.floors)
    entries, groups, unmatched, infeasible, identical = g["entries"], g["groups"], g["unmatched"], g["infeasible"], g["identical"]
    repro = verify_reproduction(entries, cfg.previous_dir) if (cfg.previous_dir and cfg.max_images is None
                                                               and cfg.budgets == [0.4, 0.5, 0.6, 0.7]) else {"verified": False, "note": "not checked (smoke/custom budgets)"}
    for e in entries:
        e["analysis"] = "primary"
    base_entry = {"name": "no_pruning", "schedule": {}, "stage_count": 0, "first_layer": None, "structure": "", "layers": [],
                  "budget_target": 1.0, "group": "baseline", "role": "baseline", "family": "none", "floor": None, "alias_of": None,
                  "analysis": "baseline", "effective_layers": [], "final_patch_tokens": N, "floor_satisfied": None, **schedule_flops(est, {})}
    all_entries = [base_entry] + entries
    exec_entries = [e for e in entries if not e["alias_of"]]

    def in_scope(e, sc):
        if sc == cfg.primary_scorer:
            return True
        if e["role"] == "anchor":
            return True if sc != "prediction_kl_approx" else any(abs(e["budget_target"] - b) < 1e-9 for b in cfg.kl_budgets)
        b_ok = any(abs(e["budget_target"] - b) < 1e-9 for b in (cfg.kl_budgets if sc == "prediction_kl_approx" else cfg.random_budgets))
        f_ok = e["floor"] is None or e["floor"] in cfg.secondary_floors
        return b_ok and f_ok and e["structure"] in cfg.secondary_structures

    scorers = [cfg.primary_scorer] + list(cfg.secondary_scorers)
    seeds_for = lambda sc: [cfg.seed] + (list(cfg.extra_seeds_random) if sc == "random" else [])
    runs = [(base_entry, "none", cfg.seed)] + [(e, sc, s) for e in exec_entries for sc in scorers if in_scope(e, sc) for s in seeds_for(sc)]
    not_run = [{"schedule": e["name"], "scorer": sc, "reason": "secondary scorer limited to anchors + reduced structures/floors/budgets (compute budget)"}
               for e in exec_entries for sc in cfg.secondary_scorers if not in_scope(e, sc)]

    ck_path = out_p("_checkpoint.pkl")
    rows, per_image, token_rows, failures, done = [], [], [], [], set()
    sig = hashlib.sha256((json.dumps(cfg.to_dict(), sort_keys=True, default=str) + summary["manifest_sha256"]).encode()).hexdigest()
    if cfg.resume and os.path.isfile(ck_path):
        ck = pickle.load(open(ck_path, "rb"))
        if ck["sig"] == sig:
            rows, per_image, token_rows, failures, done = ck["rows"], ck["per_image"], ck["token_rows"], ck["failures"], ck["done"]
            print(f"[resume] {len(done)} runs already complete", flush=True)

    t_start = time.time()
    for ri, (e, scorer, seed) in enumerate(runs):
        key = (e["name"], scorer, seed)
        if key in done:
            continue
        try:
            pruner = ProgressivePruner(wrapper, e["schedule"], scorer="random" if scorer == "none" else scorer, seed=seed)
            preds, top5, trace = [], [], None
            for (x, _), bl in zip(batches, base_logits):
                res = pruner.forward(x.to(dev), orig_logits=bl)
                trace = res.token_trace
                preds.append(res.logits.argmax(-1).cpu()); top5.append(res.logits.topk(5, dim=-1).indices.cpu())
                if scorer == "none":
                    assert (res.logits - bl).abs().max().item() <= 1e-4
            preds, top5 = torch.cat(preds).numpy(), torch.cat(top5).numpy()
        except Exception as ex:
            failures.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "error": f"{type(ex).__name__}: {ex}"})
            print(f"[FAILED] {e['name']} / {scorer} / {seed}: {ex}", flush=True)
            done.add(key)
            continue
        correct, agree, t5 = preds == y_all, preds == base_pred, (top5 == y_all[:, None]).any(1)
        per_image.append(pd.DataFrame({
            "schedule": e["name"], "scorer": scorer, "seed": seed, "image_idx": np.arange(n_img), "image_id": image_id, "label": y_all,
            "baseline_pred": base_pred, "compressed_pred": preds, "baseline_correct": (base_pred == y_all).astype(int),
            "correct": correct.astype(int), "top5_correct": t5.astype(int), "agreement": agree.astype(int)}))
        for t in trace:
            token_rows.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "family": e["family"], "floor": e["floor"],
                               "stage_count": e["stage_count"], **t})
        rows.append(_metric_row(e, scorer, seed, n_img, agree, correct, t5))
        done.add(key)
        if len(done) % 10 == 0 or ri + 1 == len(runs):
            print(f"[{len(done)}/{len(runs)}] {e['name']} / {scorer} / seed {seed} ({time.time() - t_start:.0f}s)", flush=True)
            pickle.dump({"sig": sig, "rows": rows, "per_image": per_image, "token_rows": token_rows, "failures": failures, "done": done},
                        open(ck_path, "wb"))

    metrics = pd.DataFrame(rows)
    # latency (CPU), several evenly spaced batches
    unpruned_latency, lat_info = None, {}
    if cfg.measure_latency and len(metrics):
        nb = len(batches)
        pick = sorted({int(round(i * (nb - 1) / max(cfg.latency_batches - 1, 1))) for i in range(cfg.latency_batches)})
        xs_all = [batches[i][0].to(dev) for i in pick]
        lat_info = {"batch_indices": pick, "batch_size": int(len(xs_all[0]))}
        import statistics
        with torch.no_grad():
            wrapper(xs_all[0])
            ts = []
            for x in xs_all:
                for _ in range(cfg.latency_runs):
                    t0 = time.perf_counter(); wrapper(x); ts.append((time.perf_counter() - t0) * 1e3)
        unpruned_latency = {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts), "n_timings": len(ts)}
        lat = {}
        for e in [base_entry] + exec_entries:
            for sc in (["none"] if e["role"] == "baseline" else scorers):
                if e["role"] != "baseline" and not in_scope(e, sc):
                    continue
                kl = sc == "prediction_kl_approx"
                p = ProgressivePruner(wrapper, e["schedule"], "random" if sc == "none" else sc, seed=cfg.seed)
                lat[(e["name"], sc)] = _time_multi(p, xs_all[:cfg.kl_latency_batches] if kl else xs_all, cfg.latency_warmup,
                                                   cfg.kl_latency_runs if kl else cfg.latency_runs)
        for col, key in (("latency_total_ms_median", "total_ms_median"), ("latency_total_ms_mean", "total_ms_mean"),
                         ("latency_total_ms_min_batch", "total_ms_min_batch"), ("latency_total_ms_max_batch", "total_ms_max_batch"),
                         ("scorer_overhead_ms_median", "scorer_ms_median")):
            metrics[col] = [lat.get((r.schedule, r.scorer), {}).get(key, np.nan) if r.seed == cfg.seed else np.nan for r in metrics.itertuples()]
        metrics["latency_device"] = str(dev); metrics["latency_batch_size"] = lat_info["batch_size"]
        metrics["latency_warmup"] = cfg.latency_warmup
        metrics["latency_timed_runs"] = [cfg.kl_latency_runs if r.scorer == "prediction_kl_approx" else cfg.latency_runs for r in metrics.itertuples()]
        metrics["latency_num_batches"] = [min(cfg.kl_latency_batches, len(xs_all)) if r.scorer == "prediction_kl_approx" else len(xs_all)
                                          for r in metrics.itertuples()]
    # alias rows: constrained schedules identical to unconstrained equal_drop (same counts -> same results) get their own metric rows
    if len(metrics):
        metrics["is_alias"] = False; metrics["executed_as"] = metrics.schedule
        extra = []
        for e in entries:
            if e["alias_of"]:
                for r in metrics[metrics.schedule == e["alias_of"]].itertuples(index=False):
                    d = r._asdict()
                    d.update({"schedule": e["name"], "allocation_family": e["family"], "floor": e["floor"], "floor_satisfied": e["floor_satisfied"],
                              "is_alias": True, "executed_as": e["alias_of"], "structure": e["structure"]})
                    extra.append(d)
        if extra:
            metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
    per_img_df = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    tc_df = pd.DataFrame(token_rows)
    metrics.to_csv(out_p("metrics.csv"), index=False)
    per_img_df.to_csv(out_p("per_image_metrics.csv"), index=False)
    tc_df.to_csv(out_p("token_counts.csv"), index=False)
    stats_df = build_statistical_tests(per_img_df, entries, cfg.primary_scorer, cfg.primary_floors, cfg.match_tolerance) if len(per_img_df) else pd.DataFrame()
    stats_df.to_csv(out_p("statistical_tests.csv"), index=False)

    manifest_json = {
        "layer_numbering_and_semantics": LAYER_NUMBERING,
        "allocation_rules": __doc__.split("Allocation families")[1].split("Matching:")[0].strip(),
        "floor_definition": "minimum number of PATCH tokens (CLS excluded) kept after the final pruning point",
        "rounding_policy": "Python round() (half to even) on float N - j*D with D on a 0.25 grid; then max(floor, .); counts strictly decreasing",
        "baseline_flops": est.baseline_flops, "match_tolerance_pp_of_baseline_flops": 100 * cfg.match_tolerance,
        "budgets": cfg.budgets, "floors": cfg.floors, "diagnostic_floor": 8, "primary_floors": cfg.primary_floors,
        "schedules": [{"name": e["name"], "schedule": {str(k): v for k, v in e["schedule"].items()}, "stage_count": e["stage_count"],
                       "first_layer": e["first_layer"], "layers": e["layers"], "structure": e["structure"],
                       "allocation_family": e["family"], "floor": e["floor"], "final_patch_tokens": e["final_patch_tokens"],
                       "floor_satisfied": e["floor_satisfied"], "identical_to_unconstrained": e["alias_of"],
                       "budget_target": e["budget_target"], "group": e["group"], "role": e["role"], "analytic_flops": e["analytic_flops"],
                       "normalized_flops": e["normalized_flops"], "patch_tokens_processed_per_block": expected_processed_counts(e["schedule"], L, N)}
                      for e in all_entries],
        "matched_groups": groups, "unmatched_schedules": unmatched, "infeasible_combinations": infeasible,
        "identical_to_unconstrained": identical, "skipped_pairs": stats_df.attrs.get("skipped_pairs", []) if len(stats_df) else [],
        "reproduction_check": repro}
    json.dump(manifest_json, open(out_p("schedule_manifest.json"), "w"), indent=2)
    run_config = {
        "config": cfg.to_dict(), "subset": {"num_images": n_img, "manifest_sha256": summary["manifest_sha256"], "verification": subset_check,
                                            "distinct_classes": summary["num_distinct_classes"]},
        "environment": _environment(wrapper),
        "seeds": {"primary_all_scorers": cfg.seed, "extra_random_only": cfg.extra_seeds_random},
        "seed_notes": ("Random scorer: a fresh generator is seeded per (schedule, scorer, seed) run and advances across batches and pruning "
                       "calls in fixed image order; seeds 42/43/44 are common to all schedules but draws differ between schedules "
                       "(independent streams). cls_attention and prediction_kl_approx are deterministic."),
        "timing": {"method": ("time.perf_counter around ProgressivePruner.forward (scoring + pruning + blocks + head), median over timed runs on "
                              "evenly spaced batches after warm-up; scorer overhead = time inside scorer calls; prediction_kl_approx totals also "
                              "include one unpruned forward pass (not in scorer overhead)."),
                   "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs, "batches": lat_info, "kl_timed_runs": cfg.kl_latency_runs,
                   "kl_batches": cfg.kl_latency_batches, "unpruned_model_forward": unpruned_latency, "enabled": cfg.measure_latency,
                   "gpu": "not available in this environment; no GPU results" if not torch.cuda.is_available() else "see device"},
        "failed_or_unsupported": failures, "not_run": not_run, "num_runs": len(runs)}
    json.dump(run_config, open(out_p("run_config.json"), "w"), indent=2)
    out = {"metrics": metrics, "per_image": per_img_df, "token_counts": tc_df, "stats": stats_df, "entries": all_entries, "groups": groups,
           "unmatched": unmatched, "infeasible": infeasible, "identical": identical, "manifest": manifest_json, "failures": failures,
           "run_config": run_config, "estimator": est, "subset_summary": summary}
    finalize_reports(out, cfg)
    if os.path.isfile(ck_path) and not failures:
        os.remove(ck_path)
    return out


def _metric_row(e, scorer, seed, n_img, agree, correct, t5):
    row = {"schedule": e["name"], "analysis": e["analysis"], "group": e["group"], "first_layer": e["first_layer"],
           "stage_count": e["stage_count"], "structure": e["structure"], "allocation_family": e["family"], "floor": e["floor"],
           "final_patch_tokens": e["final_patch_tokens"], "floor_satisfied": e["floor_satisfied"], "budget_target": e["budget_target"],
           "scorer": scorer, "seed": seed, "schedule_json": json.dumps({str(k): v for k, v in e["schedule"].items()}), "n_images": n_img,
           "analytic_flops": e["analytic_flops"], "normalized_flops": e["normalized_flops"], "flops_savings_pct": 100.0 * (1.0 - e["normalized_flops"])}
    for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
        w = wilson_score_interval(int(arr.sum()), n_img)
        row.update({f"{name}_count": int(arr.sum()), name: w["mean"], f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
    return row


def finalize_reports(out: Dict[str, Any], cfg: MinTokenConfig) -> None:
    m = out["metrics"]
    if len(m):
        anchors = {e["group"]: e["normalized_flops"] for e in out["entries"] if e["role"] == "anchor"}
        m["flops_diff_vs_anchor_pp"] = [100 * (r.normalized_flops - anchors[r.group]) if r.group in anchors else np.nan for r in m.itertuples()]
        m.to_csv(os.path.join(cfg.output_dir, "metrics.csv"), index=False)
    if cfg.make_plots and len(m):
        from ..visualization.min_token_plots import make_all_plots
        make_all_plots(m, out["stats"], out["infeasible"], cfg.output_dir, cfg.seed)
    write_summary(out, cfg)


def regenerate_reports(output_dir: str) -> Dict[str, Any]:
    rd = lambda n: os.path.join(output_dir, n)
    rc = json.load(open(rd("run_config.json"))); man = json.load(open(rd("schedule_manifest.json")))
    c = rc["config"]; c["structures"] = None
    cfg = MinTokenConfig(**{k: v for k, v in c.items() if k in MinTokenConfig.__dataclass_fields__})
    metrics, per_img = pd.read_csv(rd("metrics.csv")), pd.read_csv(rd("per_image_metrics.csv"))
    entries = [{**s, "schedule": {int(k): v for k, v in s["schedule"].items()}, "family": s["allocation_family"], "alias_of": s["identical_to_unconstrained"]}
               for s in man["schedules"]]
    stats_df = build_statistical_tests(per_img, [e for e in entries if e["role"] != "baseline"], cfg.primary_scorer, cfg.primary_floors, cfg.match_tolerance)
    stats_df.to_csv(rd("statistical_tests.csv"), index=False)
    out = {"metrics": metrics, "per_image": per_img, "stats": stats_df, "entries": entries, "groups": man["matched_groups"],
           "unmatched": man["unmatched_schedules"], "infeasible": man["infeasible_combinations"], "identical": man["identical_to_unconstrained"],
           "failures": rc["failed_or_unsupported"], "run_config": rc, "manifest": man, "subset_summary": json.load(open(rd("subset_summary.json")))}
    finalize_reports(out, cfg)
    return out


# ----------------------------------------------------------------------------- summary

def write_summary(out: Dict[str, Any], cfg: MinTokenConfig) -> None:
    m, st, rc, man, ss = out["metrics"], out["stats"], out["run_config"], out["manifest"], out["subset_summary"]
    seed = cfg.seed
    n = int(m.n_images.iloc[0]) if len(m) else 0
    ver = rc["subset"]["verification"]
    L = ["# Phase 2: Minimum Final-Token Constraint and Allocation Robustness\n",
         "*One model, one class-balanced validation sample. Corrected and uncorrected p-values are shown separately.*\n",
         "## Subset and methods\n",
         f"- Subset: {ss['num_images']} images, {ss['num_distinct_classes']} classes, manifest SHA-256 `{ss['manifest_sha256']}`. "
         + ("**Verified identical** (ids, labels, order, hashes) to `results/phase2_class_diverse_validation`." if ver.get("verified")
            else f"Not verified against the class-diverse manifest ({ver.get('note')}).") ,
         f"- Prior-rule reproduction: " + (f"re-generated geometric / equal_drop / anchor schedules equal the saved class-diverse manifest ({man['reproduction_check']['schedules_compared']} schedules)."
                                           if man["reproduction_check"].get("verified") else str(man["reproduction_check"].get("note"))),
         f"- Model `{cfg.model_name}` (pretrained={cfg.pretrained}); device `{rc['environment']['device']}`; primary scorer `{cfg.primary_scorer}`, seed {seed}; first layer {cfg.first_layer}; same later-layer structures as the class-diverse study.",
         f"- Layer numbering: {LAYER_NUMBERING}",
         f"- **Allocation rules:** {man['allocation_rules']}",
         f"- Floor = {man['floor_definition']}. Rounding: {man['rounding_policy']}. Floors tested: {cfg.floors} (8 = diagnostic; primary floors {cfg.primary_floors}).",
         f"- Budgets {cfg.budgets}; tolerance fixed in advance ±{100 * cfg.match_tolerance:.1f} pp of baseline FLOPs vs the one-stage anchor, and paired comparisons require |ΔFLOPs| ≤ {100 * cfg.match_tolerance:.1f} pp directly. Existing full-path FLOPs calculator, common baseline.",
         "- **Predefined families (Holm within each; floors 16/24/32 pooled):** PRIMARY = constrained vs unconstrained equal-drop (agreement; top-1 separately); SECONDARY = constrained equal-drop vs geometric. Floor 8, other scorers/seeds, multi-vs-one-stage and equal-drop-vs-geometric are exploratory. Schedules whose constrained version equals the unconstrained one are not tested (Δ = 0 by construction).",
         "- `prediction_kl_approx` is the Phase 1 gradient x activation saliency, **not exact KL**. Scorer semantics unchanged.",
         f"- Seeds: {cfg.seed} plus {cfg.extra_seeds_random} for `random`. {rc['seed_notes']}",
         f"- Secondary scorers run on anchors and structures {cfg.secondary_structures}, floors {cfg.secondary_floors}; `random` at budgets {cfg.random_budgets}, `prediction_kl_approx` at {cfg.kl_budgets} (compute). Layer 4 not run.\n"]

    L.append("## Floor feasibility (cls_attention schedules; final patch tokens, layer→k)\n")
    ent = [e for e in out["entries"] if e["role"] == "member"]
    rows = []
    for B in cfg.budgets:
        for st_ in sorted({e["structure"] for e in ent}, key=lambda s: (len(s), s)):
            cells = []
            for fam_lab, sel in [("geometric", lambda e: e["family"] == "geometric"), ("equal_drop", lambda e: e["family"] == "equal_drop")] + \
                    [(f"floor {F}", (lambda F: lambda e: e["family"] == "constrained_equal_drop" and e["floor"] == F)(F)) for F in cfg.floors]:
                x = [e for e in ent if abs(e["budget_target"] - B) < 1e-9 and e["structure"] == st_ and sel(e)]
                if x:
                    k = list(x[0]["schedule"].values())
                    cells.append(f"{k}" + (" =unc." if x[0].get("alias_of") else ""))
                else:
                    inf = [u for u in out["infeasible"] if abs(u["budget"] - B) < 1e-9 and u["structure"] == st_ and fam_lab == f"floor {u['floor']}"]
                    um = [u for u in out["unmatched"] if abs(u["budget"] - B) < 1e-9 and u["structure"] == st_ and u["family"] == fam_lab]
                    cells.append("**INFEASIBLE**" if inf else ("**UNMATCHED**" if um else "—"))
            rows.append([f"B{int(100 * B)}", st_] + cells)
    L.append(_md_table(rows, ["budget", "layers", "geometric", "equal_drop"] + [f"floor {F}" for F in cfg.floors]))
    L.append("\n`=unc.` = identical to the unconstrained equal_drop schedule (floor not binding; executed once). INFEASIBLE = floor and FLOPs tolerance cannot both be met. "
             f"Infeasible: {len(out['infeasible'])}; identical-to-unconstrained: {len(out['identical'])}; unmatched geometric/equal_drop: {len(out['unmatched'])}.\n")
    if out["infeasible"]:
        L.append("Infeasible details: " + "; ".join(f"B{int(100 * u['budget'])} {u['structure']} floor {u['floor']} (best Δ {u['best_diff_pp']:.2f} pp)"
                                                    if u["best_diff_pp"] is not None else f"B{int(100 * u['budget'])} {u['structure']} floor {u['floor']}"
                                                    for u in out["infeasible"]) + "\n")
    L.append("## Matched FLOPs budgets\n")
    L.append(_md_table([[g["group"], g.get("anchor") or "—", f"{g.get('max_abs_diff_vs_anchor_pp'):.2f}" if g.get("max_abs_diff_vs_anchor_pp") is not None else "—"]
                        for g in out["groups"]], ["group", "one-stage anchor", "max |ΔFLOPs| vs anchor over all families (pp)"]))
    L.append("")

    def mtable(sel, title, scorer, seed_):
        sub = m[(m.scorer == scorer) & (m.seed == seed_) & sel(m)].sort_values(["group", "stage_count", "structure", "allocation_family", "floor"])
        if sub.empty:
            return
        L.append(f"### {title}\n")
        L.append(_md_table([[r.group, r.stage_count, r.structure, r.allocation_family + ("" if pd.isna(r.floor) else f" f{int(r.floor)}"),
                             json.dumps(json.loads(r.schedule_json)), r.final_patch_tokens, f"{100 * r.normalized_flops:.2f}%",
                             f"{r.agreement_count}/{r.n_images} {_pct(r.agreement)} [{_pct(r.agreement_ci_low)}, {_pct(r.agreement_ci_high)}]",
                             f"{r.top1_count}/{r.n_images} {_pct(r.top1)} [{_pct(r.top1_ci_low)}, {_pct(r.top1_ci_high)}]"] for r in sub.itertuples()],
                           ["group", "stages", "layers", "family", "layer→k", "final k", "norm. FLOPs", "agreement (Wilson 95% CI)", "top-1 (Wilson 95% CI)"]))
        L.append("")

    def ttable(sel, title, note=""):
        if st.empty:
            return
        s = st[sel(st)].sort_values(["budget_target", "stage_count", "structure", "floor_a"])
        if s.empty:
            return
        L.append(f"### {title}\n")
        if note:
            L.append(note + "\n")
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.structure, f"{r.family_a} vs {r.family_b}", f"{r.final_tokens_a} vs {r.final_tokens_b}",
                             f"{r.flops_diff_pp:+.2f}", f"{r.count_a} vs {r.count_b}",
                             f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]", f"{r.p_raw:.4g}", f"{r.p_holm:.4g}"] for r in s.itertuples()],
                           ["budget", "layers", "A vs B", "final k A vs B", "ΔFLOPs pp", "count A vs B", "Δ pp [95% paired-bootstrap CI]", "raw p (exact McNemar)", "Holm p"]))
        pos, neg = int((s["diff"] > 0).sum()), int((s["diff"] < 0).sum())
        L.append(f"\nFamily size {len(s)}: A higher in {pos}, lower in {neg}, tied {len(s) - pos - neg}; mean Δ {100 * s['diff'].mean():+.2f} pp; Holm-significant: "
                 f"{int(s['significant_holm_0.05'].sum())}; uncorrected p<0.05: {int(s['significant_raw_0.05_uncorrected'].sum())}. (Win counts are descriptive.)\n")

    L.append(f"## PRIMARY: `{cfg.primary_scorer}`, first layer {cfg.first_layer}, constrained vs unconstrained equal-drop (seed {seed})\n")
    ttable(lambda d: d.family == "PRIMARY_cons_vs_uncons_agreement", "Primary family: agreement, constrained (A) minus unconstrained equal_drop (B)")
    ttable(lambda d: d.family == "PRIMARY_cons_vs_uncons_top1", "Same comparison, top-1 accuracy (own Holm family)")
    if not st.empty:
        p = st[st.family == "PRIMARY_cons_vs_uncons_agreement"]
        if not p.empty:
            g = p.groupby(["floor_a", "stage_count"])["diff"].agg(["mean", "min", "max", "count"]).reset_index()
            L.append("### Primary comparison by floor and stage count (agreement Δ pp, constrained − unconstrained)\n")
            L.append(_md_table([[int(r.floor_a), int(r.stage_count), f"{100 * r['mean']:+.2f}", f"{100 * r['min']:+.2f}", f"{100 * r['max']:+.2f}", int(r['count'])]
                                for _, r in g.iterrows()], ["floor", "stages", "mean", "min", "max", "pairs"]))
            L.append("")
    L.append(f"## SECONDARY (predefined): constrained equal-drop vs geometric, `{cfg.primary_scorer}`\n")
    ttable(lambda d: d.family == "SECONDARY_cons_vs_geo_agreement", "Agreement, constrained (A) minus geometric (B)")
    ttable(lambda d: d.family == "SECONDARY_cons_vs_geo_top1", "Top-1, constrained (A) minus geometric (B)")
    if not st.empty:
        p = st[st.family == "SECONDARY_cons_vs_geo_agreement"]
        if not p.empty:
            g = p.groupby(["floor_a", "stage_count"])["diff"].agg(["mean", "min", "max", "count"]).reset_index()
            L.append("### By floor and stage count (agreement Δ pp, constrained − geometric)\n")
            L.append(_md_table([[int(r.floor_a), int(r.stage_count), f"{100 * r['mean']:+.2f}", f"{100 * r['min']:+.2f}", f"{100 * r['max']:+.2f}", int(r['count'])]
                                for _, r in g.iterrows()], ["floor", "stages", "mean", "min", "max", "pairs"]))
            L.append("")
    L.append("## Metrics (all families, cls_attention, seed 42; counts out of N, Wilson 95% CI)\n")
    mtable(lambda d: d.first_layer == cfg.first_layer, "All schedules", cfg.primary_scorer, seed)

    L.append("## EXPLORATORY\n")
    ttable(lambda d: (d.comparison_type == "constrained_vs_unconstrained") & (d.floor_a == 8) & (d.scorer == cfg.primary_scorer) & (d.seed == seed)
           & (d.metric == "agreement"), "Diagnostic floor 8: constrained (A) minus unconstrained equal_drop (B), agreement")
    ttable(lambda d: (d.comparison_type == "equal_drop_vs_geometric") & (d.scorer == cfg.primary_scorer) & (d.seed == seed) & (d.metric == "agreement"),
           "Reproduction: unconstrained equal_drop (A) minus geometric (B), agreement")
    ttable(lambda d: (d.comparison_type == "multi_vs_one_stage") & (d.scorer == cfg.primary_scorer) & (d.seed == seed) & (d.metric == "agreement")
           & (d.family_a.str.startswith("constrained")) & (d.floor_a.isin(cfg.primary_floors)),
           "Constrained equal-drop (A) minus matched one-stage (B), agreement")
    for sc in cfg.secondary_scorers:
        mtable(lambda d: d.first_layer == cfg.first_layer, f"`{sc}` metrics (seed {seed}; reduced schedule set)", sc, seed)
        ttable(lambda d, sc=sc: (d.scorer == sc) & (d.seed == seed) & (d.metric == "agreement") & d.comparison_type.isin(["constrained_vs_unconstrained", "constrained_vs_geometric"]),
               f"`{sc}`: constrained comparisons, agreement (exploratory)")
    rs = st[(st.scorer == "random") & (st.comparison_type == "constrained_vs_unconstrained") & (st.metric == "agreement")] if not st.empty else st
    if not rs.empty and rs.seed.nunique() > 1:
        L.append("### Random scorer: seed-to-seed variation (constrained − unconstrained, agreement; separate from image-level uncertainty)\n")
        g = rs.groupby(["budget_target", "structure", "floor_a"])["diff"].agg(["mean", "std", "count"]).reset_index()
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.structure, int(r.floor_a), f"{100 * r['mean']:+.2f}", f"{100 * r['std']:.2f}", int(r['count'])]
                            for _, r in g.iterrows()], ["budget", "layers", "floor", "mean Δ pp over seeds", "sd across seeds pp", "seeds"]))
        L.append("")

    L.append("## FLOPs, latency and scorer overhead (separate quantities; no speedup inferred from FLOPs)\n")
    if cfg.measure_latency and "latency_total_ms_median" in m:
        ul, tm = rc["timing"]["unpruned_model_forward"], rc["timing"]
        L.append(f"Device `{rc['environment']['device']}` only ({tm['gpu']}); batch {int(m.latency_batch_size.dropna().iloc[0])}; warm-up {cfg.latency_warmup}; "
                 f"{cfg.latency_runs} timed runs on each of {len(tm['batches'].get('batch_indices', []))} batches (prediction_kl_approx: {cfg.kl_latency_runs} × {cfg.kl_latency_batches}); "
                 f"median ms; unpruned forward {ul['median_ms']:.1f} ms. {tm['method']}\n")
        sub = m[(m.seed == seed) & m.latency_total_ms_median.notna() & ~m.is_alias]
        g = sub.groupby(["scorer", "stage_count", "allocation_family"]).agg(total=("latency_total_ms_median", "mean"), sc=("scorer_overhead_ms_median", "mean"),
                                                                            flops=("normalized_flops", "mean"), n=("schedule", "count")).reset_index()
        L.append(_md_table([[r.scorer, int(r.stage_count), r.allocation_family, int(r.n), f"{100 * r.flops:.1f}%", f"{r.total:.0f}", f"{r.sc:.1f}"] for r in g.itertuples()],
                           ["scorer", "stages", "family", "schedules", "mean norm. FLOPs", "mean total ms (incl. selection)", "mean scorer overhead ms"]))
        L.append("\nPer-schedule values are in `metrics.csv` (`latency_*`, `scorer_overhead_ms_median`).\n")
    L.append("## Failed / not-run combinations\n")
    L.append(("Failed runs: none." if not out["failures"] else "\n".join(f"- {f}" for f in out["failures"])) + f"\n\nNot run (explicit, compute budget): {len(rc.get('not_run', []))} schedule/scorer combinations (see `run_config.json`); layer-4 analysis not run.\n")
    L.append("## Limitations\n")
    L.append(f"- {n} class-balanced images from {ss['num_distinct_classes']} classes; one model; CPU only; results may not transfer.\n"
             "- The floor changes the allocation (final count) and, to hold FLOPs fixed, the earlier counts; the constrained family therefore differs from equal_drop in more than the final count.\n"
             "- Only one constrained rule (equal drops truncated at the floor) and two other families; later layers differ across structures.\n"
             "- Secondary scorers cover a reduced schedule set; random draws differ between schedules.\n"
             "- FLOPs are analytic; latency is noisy CPU timing; no speedup claim.\n")
    L.append("## Commands\n")
    L.append("```\npython -m pytest tests -q\npython scripts/run_min_token.py --config configs/min_token.yaml --smoke\npython scripts/run_min_token.py --config configs/min_token.yaml\n```\n")
    with open(os.path.join(cfg.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
