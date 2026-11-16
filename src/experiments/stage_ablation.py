"""
Phase 2: controlled pruning-stage ablation (evaluation only; no optimizer, merging, fine-tuning or new scorers).

Question: with the FIRST pruning layer, scorer, images and analytic FLOPs budget controlled, does spreading pruning over
two or three stages preserve predictions better than pruning once?

Layer numbering / schedule semantics (confirmed from src/compression/progressive.py, unchanged): keys are 1-based
layer numbers. Key l = "after l blocks have run (0-based block index l-1), keep exactly k PATCH tokens"; blocks l..L-1
(0-based) receive the shorter sequence. CLS is always kept and not counted in k. Layer L (final block) is rejected.

Schedule-generation procedure (FLOPs only; accuracy is never consulted):
  1. For each first layer f and nominal budget B (fraction of unpruned analytic FLOPs), the ONE-STAGE anchor is {f: K}
     with K chosen to minimise |FLOPs - B|. Its exact FLOPs F* are the matching target for the whole budget group.
  2. For each multi-stage structure (f, a) or (f, a, b) the counts follow a geometric profile
     K_j = round(N * rho^j), j = 1..s, with the single parameter rho searched on a fine grid to minimise
     |FLOPs - F*|. A schedule is accepted only if |FLOPs - F*| <= tolerance (fraction of baseline FLOPs),
     counts are strictly decreasing, >= 1, and the effective number of pruning stages equals s. Otherwise it is reported
     as unmatched with its best difference (the tolerance is never loosened after seeing results).
"""

import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..compression.progressive import ProgressivePruner, RetentionSchedule
from ..data.dataset import DatasetConfig, get_dataloader
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from .layerwise_analysis import wilson_score_interval
from .progressive_eval import (
    LAYER_NUMBERING, _environment, _md_table, _pct, _sha256_file, _time_pruner,
    effective_prune_layers, expected_processed_counts, paired_diff_stats, schedule_flops,
)

STAGE_NAME = {1: "one_stage", 2: "two_stage", 3: "three_stage"}

# structure = tuple of pruning layers (first layer first). Chosen so every 3-stage structure extends a 2-stage one.
DEFAULT_STRUCTURES = {
    2: {"two": [(2, 4), (2, 6), (2, 8)], "three": [(2, 4, 8), (2, 6, 10), (2, 8, 10)]},
    4: {"two": [(4, 6), (4, 8), (4, 10)], "three": [(4, 6, 8), (4, 8, 10), (4, 6, 10)]},
}


@dataclass
class StageConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    dataset_name: str = "imagenet_val"
    data_dir: str = "./data"
    num_samples: int = 250
    batch_size: int = 25
    dataset_seed: int = 42
    seed: int = 42
    extra_seeds_random: List[int] = field(default_factory=lambda: [43, 44])
    budgets: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70])
    match_tolerance: float = 0.01            # +-1 percentage point of baseline FLOPs
    primary_first_layer: int = 2
    secondary_first_layers: List[int] = field(default_factory=lambda: [4])
    structures: Optional[Dict[int, Dict[str, List[Tuple[int, ...]]]]] = None   # None -> DEFAULT_STRUCTURES
    primary_scorer: str = "cls_attention"
    scorers_primary_layer: List[str] = field(default_factory=lambda: ["cls_attention", "random", "prediction_kl_approx"])
    scorers_secondary_layer: List[str] = field(default_factory=lambda: ["cls_attention", "random"])
    measure_latency: bool = True
    latency_warmup: int = 1
    latency_runs: int = 3
    previous_eval_dir: Optional[str] = "./results/phase2_progressive_eval"   # subset manifest to verify against
    output_dir: str = "./results/phase2_stage_ablation"
    device: Optional[str] = None
    make_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["structures"] is not None:
            d["structures"] = {str(k): {s: [list(t) for t in v] for s, v in sv.items()} for k, sv in d["structures"].items()}
        return d


# ----------------------------------------------------------------------------- schedule generation & matching

def _counts_geometric(N: int, rho: float, stages: int) -> List[int]:
    return [max(1, int(round(N * rho ** j))) for j in range(1, stages + 1)]


def _flops(est: ComputeEstimator, sched: Dict[int, int]) -> float:
    return schedule_flops(est, sched)["normalized_flops"]


def _valid_stage_schedule(sched: Dict[int, int], layers: Tuple[int, ...], est: ComputeEstimator) -> bool:
    N, L = est.N, est.num_layers
    ks = [sched[l] for l in layers]
    if not all(a > b for a, b in zip([N] + ks, ks)) or min(ks) < 1:
        return False
    return effective_prune_layers(sched, L, N) == list(layers)


def generate_schedules(est: ComputeEstimator, first_layer: int, budgets: List[float],
                       structures: Dict[str, List[Tuple[int, ...]]], tol: float) -> Dict[str, Any]:
    """Returns {'entries': [...], 'groups': [...], 'unmatched': [...]} for one first layer (see module docstring)."""
    L, N = est.num_layers, est.N
    entries: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    rho_grid = np.arange(0.005, 1.0, 0.0005)

    def make_entry(name, sched, layers, B, gid, role):
        RetentionSchedule.from_dict(sched).validate(L, N)
        fl = schedule_flops(est, sched)
        return {"name": name, "schedule": sched, "stage_count": len(layers), "first_layer": layers[0],
                "structure": "-".join(map(str, layers)), "layers": list(layers), "budget_target": B, "group": gid,
                "role": role, **fl, "effective_layers": effective_prune_layers(sched, L, N)}

    for B in budgets:
        gid = f"L{first_layer}_B{int(round(100 * B))}"
        # 1. one-stage anchor
        kbest = min(range(1, N), key=lambda k: abs(_flops(est, {first_layer: k}) - B))
        a_sched = {first_layer: kbest}
        a = make_entry(f"{gid}_one_stage", a_sched, (first_layer,), B, gid, "anchor")
        if abs(a["normalized_flops"] - B) > tol:
            unmatched.append({"first_layer": first_layer, "budget": B, "structure": str(first_layer), "stage_count": 1,
                              "reason": "one-stage anchor cannot reach the budget within tolerance",
                              "best_diff_pp": 100 * (a["normalized_flops"] - B)})
            groups.append({"group": gid, "first_layer": first_layer, "budget_target": B, "anchor": None,
                           "members": [], "missing_structures": ["all"], "complete": False})
            continue
        entries.append(a)
        members, missing = [], []
        # 2. multi-stage structures matched to the anchor's exact FLOPs
        for kind in ("two", "three"):
            for layers in structures.get(kind, []):
                s = len(layers)
                best = None
                for rho in rho_grid:
                    counts = _counts_geometric(N, float(rho), s)
                    sched = dict(zip(layers, counts))
                    if not _valid_stage_schedule(sched, layers, est):
                        continue
                    d = _flops(est, sched) - a["normalized_flops"]
                    if best is None or abs(d) < abs(best[0]) - 1e-15:
                        best = (d, sched)
                struct = "-".join(map(str, layers))
                if best is None or abs(best[0]) > tol:
                    unmatched.append({
                        "first_layer": first_layer, "budget": B, "structure": struct, "stage_count": s,
                        "reason": "no valid geometric-profile schedule within tolerance of the one-stage anchor",
                        "best_diff_pp": None if best is None else 100 * best[0]})
                    missing.append(struct)
                    continue
                e = make_entry(f"{gid}_{STAGE_NAME[s]}_{struct}", best[1], layers, B, gid, "member")
                entries.append(e)
                members.append(e["name"])
        fl = [e["normalized_flops"] for e in entries if e["group"] == gid]
        groups.append({"group": gid, "first_layer": first_layer, "budget_target": B, "anchor": a["name"],
                       "members": members, "missing_structures": missing, "complete": not missing,
                       "flops_spread_pp": 100 * (max(fl) - min(fl)),
                       "max_abs_diff_vs_anchor_pp": 100 * max(abs(f - a["normalized_flops"]) for f in fl)})
    return {"entries": entries, "groups": groups, "unmatched": unmatched}


# ----------------------------------------------------------------------------- statistics

def holm_correction(p: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1), returned in input order."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def build_statistical_tests(per_img: pd.DataFrame, entries: List[Dict[str, Any]], primary_scorer: str,
                            primary_layer: int) -> pd.DataFrame:
    """
    Paired tests, multi-stage schedule minus its group's one-stage anchor, same images/scorer/seed.
    Primary family (predefined): scorer=cls_attention, first layer=2, seed=42, metric=agreement, every (budget,
    structure) vs one-stage. Everything else is secondary/exploratory. Holm correction is applied within each family;
    raw and corrected p-values are both stored.
    """
    idx = {k: g.sort_values("image_idx") for k, g in per_img.groupby(["schedule", "scorer", "seed"])}
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)
    rows = []

    def add(scorer, seed, e_a, e_b, kind, metric, col):
        ga, gb = idx.get((e_a["name"], scorer, seed)), idx.get((e_b["name"], scorer, seed))
        if ga is None or gb is None:
            return
        s = paired_diff_stats(ga[col].to_numpy(), gb[col].to_numpy())
        rows.append({
            "comparison_type": kind, "first_layer": e_a["first_layer"], "budget_target": e_a["budget_target"],
            "scorer": scorer, "seed": seed, "metric": metric,
            "schedule_a": e_a["name"], "stage_count_a": e_a["stage_count"], "structure_a": e_a["structure"],
            "schedule_b": e_b["name"], "stage_count_b": e_b["stage_count"], "structure_b": e_b["structure"],
            "flops_a": e_a["normalized_flops"], "flops_b": e_b["normalized_flops"],
            "flops_diff_pp": 100 * (e_a["normalized_flops"] - e_b["normalized_flops"]),
            "n_images": s["n"], "count_a": s["a_count"], "count_b": s["b_count"], "diff": s["diff"],
            "ci_low": s["boot_ci_low"], "ci_high": s["boot_ci_high"], "a_only": s["a_only"], "b_only": s["b_only"],
            "p_raw": s["mcnemar_p"]})

    scorers_seeds = sorted({(k[1], k[2]) for k in idx})
    for gid, ents in by_group.items():
        anchor = next(e for e in ents if e["role"] == "anchor")
        for e in ents:
            if e["role"] != "member":
                continue
            for scorer, seed in scorers_seeds:
                for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                    add(scorer, seed, e, anchor, "multi_vs_one_stage", metric, col)
            # exploratory: a 3-stage schedule vs its 2-stage prefix (same first two layers)
            if e["stage_count"] == 3:
                prefix = next((x for x in ents if x["role"] == "member" and x["stage_count"] == 2
                               and x["layers"] == e["layers"][:2]), None)
                if prefix is not None:
                    for scorer, seed in scorers_seeds:
                        for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                            add(scorer, seed, e, prefix, "three_vs_two_prefix", metric, col)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def family(r):
        if r.comparison_type == "multi_vs_one_stage" and r.scorer == primary_scorer and r.first_layer == primary_layer \
                and r.seed == 42:
            return f"PRIMARY_{r.metric}" if r.metric == "agreement" else f"primary_setting_{r.metric}"
        return f"exploratory|{r.comparison_type}|L{r.first_layer}|{r.scorer}|seed{r.seed}|{r.metric}"

    df["family"] = df.apply(family, axis=1)
    df["is_primary"] = df.family == "PRIMARY_agreement"
    df["family_size"] = df.groupby("family").family.transform("size")
    df["p_holm"] = np.nan
    for fam, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm_correction(g.p_raw.tolist())
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["significant_raw_0.05_uncorrected"] = df.p_raw < 0.05
    return df


# ----------------------------------------------------------------------------- subset verification

def verify_subset(manifest: pd.DataFrame, previous_dir: Optional[str], baseline_top1: int) -> Dict[str, Any]:
    """Compare the image manifest (index, label, sha1) with the previous evaluation. Raises on any mismatch."""
    if previous_dir is None:
        return {"verified": False, "note": "no previous evaluation directory supplied"}
    mp = os.path.join(previous_dir, "image_manifest.csv")
    if not os.path.isfile(mp):
        raise RuntimeError(f"Cannot verify subset: {mp} not found. Stopping instead of claiming an identical subset.")
    prev = pd.read_csv(mp)
    cur = manifest.rename(columns={"image_sha1": "image_sha1"})
    if len(prev) != len(cur) or not (prev.image_idx.values == cur.image_idx.values).all() \
            or not (prev.label.values == cur.label.values).all() \
            or not (prev.image_sha1.values == cur.image_sha1.values).all():
        raise RuntimeError("Image subset/order does NOT match the previous evaluation manifest. Stopping.")
    info = {"verified": True, "previous_manifest": mp, "num_images": int(len(cur)), "identical_order_labels_hashes": True}
    mm = os.path.join(previous_dir, "metrics.csv")
    if os.path.isfile(mm):
        pm = pd.read_csv(mm)
        prev_b = int(pm[pm.schedule == "no_pruning"].top1_count.iloc[0])
        info["previous_baseline_top1_count"] = prev_b
        if prev_b != baseline_top1:
            raise RuntimeError(f"Baseline top-1 count {baseline_top1} differs from previous evaluation {prev_b}.")
    return info


# ----------------------------------------------------------------------------- run

def run_stage_ablation(cfg: StageConfig, wrapper: Optional[ViTWrapper] = None,
                       data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    wrapper = wrapper or ViTWrapper(cfg.model_name, cfg.pretrained, cfg.device)
    L, N = wrapper.num_layers, wrapper.num_original_tokens
    est = ComputeEstimator(L, wrapper.hidden_dim, wrapper.num_heads, N, wrapper.num_classes,
                           wrapper.num_prefix_tokens, wrapper.patch_size)
    dev = wrapper.device

    # ---- data (same loader path as Phase 1 / previous evaluation)
    data_info: Dict[str, Any] = {"dataset": cfg.dataset_name, "batch_size": cfg.batch_size}
    if data is None:
        loader, _ = get_dataloader(DatasetConfig(name=cfg.dataset_name, data_dir=cfg.data_dir, num_samples=cfg.num_samples,
                                                 batch_size=cfg.batch_size, seed=cfg.dataset_seed))
        xs, ys = zip(*[(x, y) for x, y in loader])
        images, labels = torch.cat(xs), torch.cat(ys)
        cache = os.path.join(cfg.data_dir, "imagenet_val_250.pt")
        if os.path.isfile(cache):
            data_info.update({"source_file": cache, "source_sha256": _sha256_file(cache),
                              "selection": f"first {len(images)} images of the cached file, stored order, no shuffling"})
    else:
        images, labels = data
        data_info["selection"] = "tensors supplied directly (tests)"
    n_img = len(images)
    data_info["num_images"] = n_img
    data_info["distinct_classes"] = int(len(set(labels.tolist())))
    manifest = pd.DataFrame({"image_idx": range(n_img), "label": labels.tolist(),
                             "image_sha1": [hashlib.sha1(images[i].numpy().tobytes()).hexdigest() for i in range(n_img)]})
    manifest.to_csv(os.path.join(cfg.output_dir, "image_manifest.csv"), index=False)
    batches = [(images[i:i + cfg.batch_size], labels[i:i + cfg.batch_size]) for i in range(0, n_img, cfg.batch_size)]
    with torch.no_grad():
        base_logits = [wrapper(x.to(dev)) for x, _ in batches]
    base_pred = torch.cat([l.argmax(-1) for l in base_logits]).cpu().numpy()
    y_all = labels.cpu().numpy()
    data_info["subset_verification"] = verify_subset(manifest, cfg.previous_eval_dir, int((base_pred == y_all).sum()))

    # ---- schedules
    structures = cfg.structures or DEFAULT_STRUCTURES
    first_layers = [cfg.primary_first_layer] + list(cfg.secondary_first_layers)
    entries: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    for f in first_layers:
        g = generate_schedules(est, f, cfg.budgets, structures[f], cfg.match_tolerance)
        for e in g["entries"]:
            e["analysis"] = "primary" if f == cfg.primary_first_layer else f"secondary_first_layer_{f}"
        entries += g["entries"]; groups += g["groups"]; unmatched += g["unmatched"]
    base_entry = {"name": "no_pruning", "schedule": {}, "stage_count": 0, "first_layer": None, "structure": "",
                  "layers": [], "budget_target": 1.0, "group": "baseline", "role": "baseline", "analysis": "baseline",
                  "effective_layers": [], **schedule_flops(est, {})}
    all_entries = [base_entry] + entries

    def scorers_for(e):
        return cfg.scorers_primary_layer if e["first_layer"] == cfg.primary_first_layer else cfg.scorers_secondary_layer

    def seeds_for(scorer):
        return [cfg.seed] + (list(cfg.extra_seeds_random) if scorer == "random" else [])

    runs = [(base_entry, "none", cfg.seed)] + [(e, sc, s) for e in entries for sc in scorers_for(e) for s in seeds_for(sc)]
    not_run = [{"first_layer": f, "scorer": sc, "reason": "not run for this analysis (compute budget); not an error"}
               for f in cfg.secondary_first_layers for sc in cfg.scorers_primary_layer
               if sc not in cfg.scorers_secondary_layer]

    rows, per_image, token_rows, failures = [], [], [], []
    t_start = time.time()
    for ri, (e, scorer, seed) in enumerate(runs):
        try:
            pruner = ProgressivePruner(wrapper, e["schedule"], scorer="random" if scorer == "none" else scorer, seed=seed)
            preds, top5, trace = [], [], None
            for (x, _), bl in zip(batches, base_logits):
                res = pruner.forward(x.to(dev), orig_logits=bl)
                if trace is not None:
                    assert [t["patch_tokens_out"] for t in trace] == [t["patch_tokens_out"] for t in res.token_trace]
                trace = res.token_trace
                preds.append(res.logits.argmax(-1).cpu()); top5.append(res.logits.topk(5, dim=-1).indices.cpu())
                if scorer == "none":
                    assert (res.logits - bl).abs().max().item() <= 1e-4
            preds, top5 = torch.cat(preds).numpy(), torch.cat(top5).numpy()
        except Exception as ex:
            failures.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "error": f"{type(ex).__name__}: {ex}"})
            print(f"[FAILED] {e['name']} / {scorer} / {seed}: {ex}", flush=True)
            continue
        correct, agree, t5 = preds == y_all, preds == base_pred, (top5 == y_all[:, None]).any(1)
        per_image.append(pd.DataFrame({
            "schedule": e["name"], "scorer": scorer, "seed": seed, "image_idx": np.arange(n_img), "label": y_all,
            "baseline_pred": base_pred, "compressed_pred": preds, "baseline_correct": (base_pred == y_all).astype(int),
            "correct": correct.astype(int), "top5_correct": t5.astype(int), "agreement": agree.astype(int)}))
        for t in trace:
            token_rows.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "first_layer": e["first_layer"],
                               "stage_count": e["stage_count"], **t})
        row = {"schedule": e["name"], "analysis": e["analysis"], "group": e["group"], "first_layer": e["first_layer"],
               "stage_count": e["stage_count"], "structure": e["structure"], "budget_target": e["budget_target"],
               "scorer": scorer, "seed": seed, "schedule_json": json.dumps({str(k): v for k, v in e["schedule"].items()}),
               "n_images": n_img, "analytic_flops": e["analytic_flops"], "normalized_flops": e["normalized_flops"],
               "flops_savings_pct": 100.0 * (1.0 - e["normalized_flops"])}
        for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
            w = wilson_score_interval(int(arr.sum()), n_img)
            row.update({f"{name}_count": int(arr.sum()), name: w["mean"],
                        f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
        rows.append(row)
        if (ri + 1) % 10 == 0 or ri + 1 == len(runs):
            print(f"[{ri + 1}/{len(runs)}] {e['name']} / {scorer} / seed {seed} ({time.time() - t_start:.0f}s)", flush=True)

    metrics = pd.DataFrame(rows)
    unpruned_latency = None
    if cfg.measure_latency and len(metrics):
        xb = batches[0][0].to(dev)
        with torch.no_grad():
            for _ in range(cfg.latency_warmup):
                wrapper(xb)
            ts = []
            for _ in range(cfg.latency_runs):
                t0 = time.perf_counter(); wrapper(xb); ts.append((time.perf_counter() - t0) * 1e3)
        unpruned_latency = {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts)}
        lat = {}
        for e in all_entries:
            for sc in (["none"] if e["role"] == "baseline" else scorers_for(e)):
                p = ProgressivePruner(wrapper, e["schedule"], "random" if sc == "none" else sc, seed=cfg.seed)
                lat[(e["name"], sc)] = _time_pruner(p, xb, cfg.latency_warmup, cfg.latency_runs)
        for col, key in (("latency_total_ms_median", "total_ms_median"), ("latency_total_ms_mean", "total_ms_mean"),
                         ("scorer_overhead_ms_median", "scorer_ms_median")):
            metrics[col] = [lat.get((r.schedule, r.scorer), {}).get(key, np.nan) if r.seed == cfg.seed else np.nan
                            for r in metrics.itertuples()]
        metrics["latency_device"] = str(dev); metrics["latency_batch_size"] = len(xb)
        metrics["latency_warmup"] = cfg.latency_warmup; metrics["latency_timed_runs"] = cfg.latency_runs
    per_img_df = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    tc_df = pd.DataFrame(token_rows)
    metrics.to_csv(os.path.join(cfg.output_dir, "metrics.csv"), index=False)
    per_img_df.to_csv(os.path.join(cfg.output_dir, "per_image_metrics.csv"), index=False)
    tc_df.to_csv(os.path.join(cfg.output_dir, "token_counts.csv"), index=False)

    stats_df = build_statistical_tests(per_img_df, entries, cfg.primary_scorer, cfg.primary_first_layer) \
        if len(per_img_df) else pd.DataFrame()
    stats_df.to_csv(os.path.join(cfg.output_dir, "statistical_tests.csv"), index=False)

    manifest_json = {
        "layer_numbering_and_semantics": LAYER_NUMBERING,
        "generation_procedure": __doc__.split("Schedule-generation procedure")[1].split(":\n", 1)[1].strip(),
        "baseline_flops": est.baseline_flops, "match_tolerance_pp_of_baseline_flops": 100 * cfg.match_tolerance,
        "budgets": cfg.budgets, "first_layers": {"primary": cfg.primary_first_layer,
                                                   "secondary": cfg.secondary_first_layers},
        "schedules": [{"name": e["name"], "schedule": {str(k): v for k, v in e["schedule"].items()},
                       "stage_count": e["stage_count"], "first_layer": e["first_layer"], "structure": e["structure"],
                       "budget_target": e["budget_target"], "group": e["group"], "role": e["role"],
                       "analysis": e["analysis"], "analytic_flops": e["analytic_flops"],
                       "normalized_flops": e["normalized_flops"],
                       "patch_tokens_processed_per_block": expected_processed_counts(e["schedule"], L, N)}
                      for e in all_entries],
        "matched_groups": groups, "unmatched_schedules": unmatched,
    }
    json.dump(manifest_json, open(os.path.join(cfg.output_dir, "schedule_manifest.json"), "w"), indent=2)
    run_config = {
        "config": cfg.to_dict(), "dataset": data_info, "environment": _environment(wrapper),
        "preprocessing": "Phase 1 pipeline: cached tensors from get_transforms (Resize 256 bicubic, CenterCrop 224, ImageNet mean/std)",
        "seeds": {"primary_all_scorers": cfg.seed, "extra_random_only": cfg.extra_seeds_random},
        "seed_notes": ("Random scorer: a fresh generator is seeded per (schedule, scorer, seed) run and advances across batches and "
                       "pruning calls in fixed image order. The same seed values are used for every schedule (common seeds), "
                       "but the draws themselves differ between schedules. cls_attention and prediction_kl_approx are deterministic."),
        "timing": {"method": "time.perf_counter around ProgressivePruner.forward (scoring + pruning + blocks + head), first batch, "
                             "median of timed runs after warm-up. prediction_kl_approx totals also include one unpruned forward "
                             "pass for the target class (not in scorer overhead).",
                   "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs, "unpruned_model_forward": unpruned_latency,
                   "enabled": cfg.measure_latency},
        "failed_or_unsupported": failures, "scorer_analysis_combinations_not_run": not_run, "num_runs": len(runs)}
    json.dump(run_config, open(os.path.join(cfg.output_dir, "run_config.json"), "w"), indent=2)

    out = {"metrics": metrics, "per_image": per_img_df, "token_counts": tc_df, "stats": stats_df, "entries": all_entries,
           "groups": groups, "unmatched": unmatched, "manifest": manifest_json, "failures": failures,
           "run_config": run_config, "estimator": est}
    if cfg.make_plots and len(metrics):
        from ..visualization.stage_ablation_plots import make_all_plots
        make_all_plots(metrics, stats_df, cfg.output_dir, cfg.seed)
    write_summary(out, cfg)
    return out


def regenerate_reports(output_dir: str) -> Dict[str, Any]:
    """Rebuild statistical tests, plots and summary.md from the saved CSV/JSON (no models are re-run)."""
    rd = lambda n: os.path.join(output_dir, n)
    run_config = json.load(open(rd("run_config.json"))); man = json.load(open(rd("schedule_manifest.json")))
    c = run_config["config"]; c["structures"] = None
    cfg = StageConfig(**{k: v for k, v in c.items() if k in StageConfig.__dataclass_fields__})
    metrics, per_img = pd.read_csv(rd("metrics.csv")), pd.read_csv(rd("per_image_metrics.csv"))
    meta = metrics.drop_duplicates("schedule").set_index("schedule")
    entries = []
    for s in man["schedules"]:
        entries.append({**s, "schedule": {int(k): v for k, v in s["schedule"].items()},
                        "layers": [int(x) for x in s["structure"].split("-")] if s["structure"] else [],
                        "normalized_flops": s["normalized_flops"]})
    stats_df = build_statistical_tests(per_img, [e for e in entries if e["role"] != "baseline"], cfg.primary_scorer,
                                       cfg.primary_first_layer)
    stats_df.to_csv(rd("statistical_tests.csv"), index=False)
    out = {"metrics": metrics, "per_image": per_img, "stats": stats_df, "entries": entries, "groups": man["matched_groups"],
           "unmatched": man["unmatched_schedules"], "failures": run_config["failed_or_unsupported"], "run_config": run_config,
           "manifest": man}
    from ..visualization.stage_ablation_plots import make_all_plots
    make_all_plots(metrics, stats_df, output_dir, cfg.seed)
    write_summary(out, cfg)
    return out


# ----------------------------------------------------------------------------- summary

def write_summary(out: Dict[str, Any], cfg: StageConfig) -> None:
    m, st, rc, man = out["metrics"], out["stats"], out["run_config"], out["manifest"]
    seed, n = cfg.seed, int(m.n_images.iloc[0]) if len(m) else 0
    entries = [e for e in out["entries"] if e["role"] != "baseline"]
    L = ["# Phase 2: Controlled Pruning-Stage Ablation\n",
         "*Small-scale study (one model, one image subset). Corrected and uncorrected p-values are shown separately.*\n",
         "## Methods\n",
         f"- Question: with first pruning layer, scorer, images and analytic FLOPs controlled, do 2 or 3 pruning stages preserve "
         f"predictions better than 1?",
         f"- Model `{cfg.model_name}` (pretrained={cfg.pretrained}); eval mode; device `{rc['environment']['device']}`.",
         f"- Layer numbering: {LAYER_NUMBERING}",
         f"- **Schedule generation** (FLOPs only, accuracy never consulted): {man['generation_procedure']}",
         f"- Budgets (nominal fraction of unpruned FLOPs): {cfg.budgets}; tolerance fixed in advance: ±{100 * cfg.match_tolerance:.1f} pp "
         f"of baseline FLOPs relative to the one-stage anchor. FLOPs use the Phase 1/2 calculator and one common baseline.",
         f"- Structures (layers pruned): two-stage {DEFAULT_STRUCTURES[2]['two']}, three-stage {DEFAULT_STRUCTURES[2]['three']} for first layer 2; "
         f"layer-4 structures in `schedule_manifest.json`.",
         f"- **Predefined primary comparison:** scorer `{cfg.primary_scorer}`, first layer {cfg.primary_first_layer}, seed 42, "
         f"metric = agreement with the unpruned model, every matched (budget, structure) multi-stage schedule vs the one-stage schedule "
         f"of the same budget. Family-wise error control: Holm over that family. Everything else (top-1, other scorers, layer 4, "
         f"3-vs-2-stage) is secondary/exploratory with Holm applied within each separate family.",
         "- `prediction_kl_approx` is the Phase 1 gradient x activation saliency, **not exact KL**. `cls_attention` semantics unchanged.",
         f"- Seeds: {cfg.seed} (all scorers) plus {cfg.extra_seeds_random} for `random` only. {rc['seed_notes']}",
         "- Subset: " + (f"{n} images; verified identical (index, label, sha1, order) to `{rc['dataset']['subset_verification'].get('previous_manifest')}`; "
                         f"baseline top-1 {rc['dataset']['subset_verification'].get('previous_baseline_top1_count')}/{n} matches. "
                         f"Distinct classes in subset: {rc['dataset']['distinct_classes']}."
                         if rc["dataset"]["subset_verification"].get("verified") else "not verified against a previous evaluation."),
         ""]

    # matched groups
    L.append("## Matched FLOPs budgets and unmatched schedules\n")
    rows = []
    for g in out["groups"]:
        rows.append([g["group"], g.get("anchor") or "—", len(g["members"]), ", ".join(g["missing_structures"]) or "none",
                     f"{g.get('max_abs_diff_vs_anchor_pp', float('nan')):.2f}" if g.get("anchor") else "—"])
    L.append(_md_table(rows, ["group (first layer, budget %)", "one-stage anchor", "matched multi-stage schedules",
                              "unmatched structures", "max |ΔFLOPs| vs anchor (pp)"]))
    L.append("")
    L.append("Unmatched details: " + ("none." if not out["unmatched"] else "; ".join(
        f"L{u['first_layer']} B{int(100 * u['budget'])} {u['structure']}: {u['reason']} (best Δ {u['best_diff_pp']} pp)"
        for u in out["unmatched"])) + "\n")

    def table_for(scorer, analysis_pred, title, seed_):
        sub = m[(m.scorer == scorer) & (m.seed == seed_) & analysis_pred(m)].sort_values(["group", "stage_count", "structure"])
        if sub.empty:
            return
        L.append(f"### {title}\n")
        L.append(_md_table([[r.group, r.stage_count, r.structure, json.dumps(json.loads(r.schedule_json)), f"{100 * r.normalized_flops:.2f}%",
                             f"{r.agreement_count}/{r.n_images} {_pct(r.agreement)} [{_pct(r.agreement_ci_low)}, {_pct(r.agreement_ci_high)}]",
                             f"{r.top1_count}/{r.n_images} {_pct(r.top1)} [{_pct(r.top1_ci_low)}, {_pct(r.top1_ci_high)}]"]
                            for r in sub.itertuples()],
                           ["group", "stages", "layers", "layer→k", "norm. FLOPs", "agreement", "top-1"]))
        L.append("")

    def test_table(sel, title, note=""):
        if st.empty:
            return
        s = st[sel(st)].sort_values(["budget_target", "stage_count_a", "structure_a"])
        if s.empty:
            return
        L.append(f"### {title}\n")
        if note:
            L.append(note + "\n")
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", f"{r.structure_a} ({r.stage_count_a}-stage) vs {r.structure_b} ({r.stage_count_b}-stage)",
                             f"{r.flops_diff_pp:+.2f}", f"{r.count_a} vs {r.count_b}",
                             f"{100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]", f"{r.p_raw:.4f}", f"{r.p_holm:.4f}"]
                            for r in s.itertuples()],
                           ["budget", "comparison (A vs B)", "ΔFLOPs pp", "count A vs B", "Δ pp [95% paired-bootstrap CI]",
                            "raw p (McNemar exact)", "Holm p"]))
        k = int(s["significant_holm_0.05"].sum()); pos = int((s["diff"] > 0).sum()); neg = int((s["diff"] < 0).sum())
        L.append(f"\nFamily size {len(s)}: A higher in {pos}, lower in {neg}; Holm-significant (p<0.05) in {k}; "
                 f"uncorrected p<0.05 in {int(s['significant_raw_0.05_uncorrected'].sum())}.\n")

    L.append(f"## PRIMARY: `{cfg.primary_scorer}`, first layer {cfg.primary_first_layer}, one vs two vs three stages (seed {seed})\n")
    table_for(cfg.primary_scorer, lambda d: d.first_layer == cfg.primary_first_layer, "Metrics (counts out of N, Wilson 95% CI)", seed)
    test_table(lambda d: d.family == "PRIMARY_agreement", "Primary family: agreement, multi-stage minus one-stage",
               "This is the predefined primary family; the Holm column is the corrected result to use.")
    test_table(lambda d: d.family == "primary_setting_top1", "Same setting, top-1 accuracy (secondary, own Holm family)")

    L.append("## SECONDARY / EXPLORATORY (kept separate from the primary analysis)\n")
    for sc in cfg.scorers_primary_layer:
        if sc == cfg.primary_scorer:
            continue
        table_for(sc, lambda d: d.first_layer == cfg.primary_first_layer, f"First layer {cfg.primary_first_layer}: `{sc}` metrics (seed {seed})", seed)
        test_table(lambda d, sc=sc: (d.scorer == sc) & (d.seed == seed) & (d.first_layer == cfg.primary_first_layer)
                   & (d.comparison_type == "multi_vs_one_stage") & (d.metric == "agreement"),
                   f"First layer {cfg.primary_first_layer}: `{sc}`, agreement, multi-stage minus one-stage (exploratory)")
    rs = st[(st.scorer == "random") & (st.comparison_type == "multi_vs_one_stage") & (st.metric == "agreement")
            & (st.first_layer == cfg.primary_first_layer)] if not st.empty else st
    if not rs.empty and rs.seed.nunique() > 1:
        L.append("### Random scorer, first layer 2: seed-to-seed variation (separate from image-level uncertainty)\n")
        g = rs.groupby(["budget_target", "structure_a"])["diff"].agg(["mean", "std", "count"]).reset_index()
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.structure_a, f"{100 * r['mean']:+.1f}", f"{100 * r['std']:.1f}", int(r['count'])]
                            for _, r in g.iterrows()], ["budget", "structure", "mean Δagree pp over seeds", "sd across seeds pp", "seeds"]))
        L.append("")
    for f in cfg.secondary_first_layers:
        L.append(f"### First layer {f} (secondary analysis)\n")
        for sc in cfg.scorers_secondary_layer:
            table_for(sc, lambda d, f=f: d.first_layer == f, f"First layer {f}: `{sc}` metrics (seed {seed})", seed)
            test_table(lambda d, sc=sc, f=f: (d.scorer == sc) & (d.seed == seed) & (d.first_layer == f)
                       & (d.comparison_type == "multi_vs_one_stage") & (d.metric == "agreement"),
                       f"First layer {f}: `{sc}`, agreement, multi-stage minus one-stage (exploratory)")
    test_table(lambda d: (d.comparison_type == "three_vs_two_prefix") & (d.scorer == cfg.primary_scorer) & (d.seed == seed)
               & (d.first_layer == cfg.primary_first_layer) & (d.metric == "agreement"),
               f"Exploratory: three-stage vs its two-stage prefix, `{cfg.primary_scorer}`, first layer {cfg.primary_first_layer}, agreement")

    L.append("## FLOPs, latency and scorer overhead (separate quantities; no speedup inferred from FLOPs)\n")
    if cfg.measure_latency and "latency_total_ms_median" in m:
        ul = rc["timing"]["unpruned_model_forward"]
        L.append(f"Device `{rc['environment']['device']}` only; batch {int(m.latency_batch_size.dropna().iloc[0])}; warm-up {cfg.latency_warmup}; "
                 f"timed runs {cfg.latency_runs}; median ms; unpruned model forward {ul['median_ms']:.1f} ms. {rc['timing']['method']}\n")
        sub = m[(m.seed == seed) & (m.first_layer == cfg.primary_first_layer)].sort_values(["scorer", "group", "stage_count", "structure"])
        L.append(_md_table([[r.scorer, r.group, r.stage_count, r.structure, f"{100 * r.normalized_flops:.1f}%",
                             f"{r.latency_total_ms_median:.0f}", f"{r.scorer_overhead_ms_median:.1f}"] for r in sub.itertuples()],
                           ["scorer", "group", "stages", "layers", "norm. FLOPs", "total ms (incl. selection)", "scorer overhead ms"]))
        L.append("")
    else:
        L.append("Latency not measured.\n")
    L.append("## Failed / unsupported / not-run combinations\n")
    L.append(("Failed runs: none." if not out["failures"] else "\n".join(f"- {f}" for f in out["failures"])) + "\n")
    nr = rc.get("scorer_analysis_combinations_not_run", [])
    if nr:
        L.append("Not run (explicit, not silently skipped): " + "; ".join(f"first layer {x['first_layer']} / {x['scorer']}" for x in nr)
                 + " — to bound compute; the primary layer-2 analysis runs all three scorers.\n")
    L.append("## Limitations\n")
    L.append(f"- {n} images from {rc['dataset']['distinct_classes']} classes (a contiguous slice of the validation file); one model; results may not transfer.\n"
             "- Counts follow one documented geometric profile; other count allocations at the same FLOPs may behave differently.\n"
             "- Matched schedules differ in the *later* pruning layers; stage count is confounded with which layers are used.\n"
             "- Random draws differ between schedules even with common seeds.\n"
             "- FLOPs are analytic; measured latency is single-batch CPU timing and is noisy.\n")
    L.append("## Commands\n")
    L.append("```\npython scripts/run_stage_ablation.py --config configs/stage_ablation.yaml --smoke   # small smoke test\n"
             "python scripts/run_stage_ablation.py --config configs/stage_ablation.yaml           # full run\n```\n")
    with open(os.path.join(cfg.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
