"""
Phase 2: class-diverse validation and allocation-robustness study (evaluation only; no optimizer, merging, fine-tuning,
learned routing or new scorers).

Questions: (1) does multi-stage CLS-attention pruning starting at layer 2 preserve predictions better than matched-FLOPs
single-stage pruning on a class-diverse ImageNet subset, and (2) does the answer persist under a second token-count
allocation family?

Layer numbering / schedule semantics (confirmed from src/compression/progressive.py, unchanged): keys are 1-based layer
numbers. Key l = "after l blocks have run (0-based block index l-1), keep exactly k PATCH tokens"; blocks l..L-1 (0-based)
receive the shorter sequence. CLS is always kept and is not counted in k. Layer L (final block) is rejected.

Schedule-generation procedure (FLOPs only; accuracy is never consulted; both allocation rules fixed before any evaluation):
  1. For each first layer f and nominal budget B (fraction of unpruned analytic FLOPs) the ONE-STAGE anchor {f: K} uses the K that
     minimises |FLOPs - B|. Its exact FLOPs F* is the matching target for the whole budget group (shared by both families).
  2. For each multi-stage structure (f, a) or (f, a, b) and each allocation family, counts are a one-parameter profile, with the
     parameter searched on a fixed fine grid to minimise |FLOPs - F*|:
       geometric   : K_j = max(1, round(N * rho**j)), j = 1..s, rho in (0, 1)  [the rule used in the previous stage ablation]
       equal_drop  : K_j = max(1, round(N - j * D)), j = 1..s, D > 0           [approximately equal absolute token reduction
                     D per stage; each stage removes about D patch tokens]
     The two rules distribute tokens differently across stages (geometric: equal retention ratio, so shrinking absolute drops;
     equal_drop: equal absolute drops).
  3. A schedule is accepted only if |FLOPs - F*| <= tolerance (fraction of baseline FLOPs), counts are strictly decreasing, >= 1,
     and the effective number of pruning stages equals s. Otherwise it is reported as unmatched with its best difference
     (the tolerance is never loosened after seeing results).
"""

import hashlib
import json
import os
import pickle
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..compression.progressive import ProgressivePruner, RetentionSchedule
from ..data.class_diverse import build_class_diverse_subset, class_coverage_summary, build_manifest, manifest_hash
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from .layerwise_analysis import wilson_score_interval
from .progressive_eval import (
    LAYER_NUMBERING, _environment, _md_table, _pct, _time_pruner, effective_prune_layers, expected_processed_counts,
    paired_diff_stats, schedule_flops,
)
from .stage_ablation import STAGE_NAME, _valid_stage_schedule, holm_correction

FAMILIES = ("geometric", "equal_drop")
D_GRID = np.arange(0.25, 195.0, 0.25)          # equal_drop parameter grid (patch tokens removed per stage)
RHO_GRID = np.arange(0.005, 1.0, 0.0005)       # geometric parameter grid (identical to the previous stage ablation)

DEFAULT_STRUCTURES = {
    2: {"two": [(2, 4), (2, 6), (2, 8)], "three": [(2, 4, 8), (2, 6, 10), (2, 8, 10)]},
    4: {"two": [(4, 8)], "three": [(4, 6, 8)]},
}


@dataclass
class ClassDiverseConfig:
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
    families: List[str] = field(default_factory=lambda: list(FAMILIES))
    primary_first_layer: int = 2
    secondary_first_layers: List[int] = field(default_factory=lambda: [4])
    structures: Optional[Dict[int, Dict[str, List[Tuple[int, ...]]]]] = None
    # Secondary scorers are run only on the anchors + these structures (compute budget); everything else is reported as not run.
    reduced_structures: List[str] = field(default_factory=lambda: ["2-6", "2-6-10"])
    kl_budgets: List[float] = field(default_factory=lambda: [0.50, 0.60])   # prediction_kl_approx is ~100x slower: these budgets only
    primary_scorer: str = "cls_attention"
    scorers_primary_layer: List[str] = field(default_factory=lambda: ["cls_attention", "random", "prediction_kl_approx"])
    scorers_secondary_layer: List[str] = field(default_factory=lambda: ["cls_attention"])
    measure_latency: bool = True
    latency_warmup: int = 1
    latency_runs: int = 5
    latency_batches: int = 3
    kl_latency_runs: int = 3
    kl_latency_batches: int = 1
    previous_stage_ablation_dir: Optional[str] = "./results/phase2_stage_ablation"
    output_dir: str = "./results/phase2_class_diverse_validation"
    device: Optional[str] = None
    make_plots: bool = True
    resume: bool = True
    max_images: Optional[int] = None       # smoke / tests: truncate the subset to its first max_images (order preserved)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["structures"] is not None:
            d["structures"] = {str(k): {s: [list(t) for t in v] for s, v in sv.items()} for k, sv in d["structures"].items()}
        return d


# ----------------------------------------------------------------------------- allocation families

def family_counts(family: str, N: int, param: float, stages: int) -> List[int]:
    """Patch-token counts for stages 1..s under an allocation family (rounding: Python round -> half to even, then max(1, .))."""
    if family == "geometric":
        return [max(1, int(round(N * param ** j))) for j in range(1, stages + 1)]
    if family == "equal_drop":
        return [max(1, int(round(N - j * param))) for j in range(1, stages + 1)]
    raise ValueError(f"unknown allocation family {family!r}")


def _grid(family: str):
    return RHO_GRID if family == "geometric" else D_GRID


def generate_family_schedules(est: ComputeEstimator, first_layer: int, budgets: List[float],
                              structures: Dict[str, List[Tuple[int, ...]]], tol: float,
                              families: List[str] = list(FAMILIES)) -> Dict[str, Any]:
    """Returns {'entries', 'groups', 'unmatched'} for one first layer; see the module docstring for the procedure."""
    L, N = est.num_layers, est.N
    entries, groups, unmatched = [], [], []
    fl = lambda s: schedule_flops(est, s)["normalized_flops"]

    def make_entry(name, sched, layers, B, gid, role, family):
        RetentionSchedule.from_dict(sched).validate(L, N)
        return {"name": name, "schedule": sched, "stage_count": len(layers), "first_layer": layers[0],
                "structure": "-".join(map(str, layers)), "layers": list(layers), "budget_target": B, "group": gid,
                "role": role, "family": family, **schedule_flops(est, sched),
                "effective_layers": effective_prune_layers(sched, L, N)}

    for B in budgets:
        gid = f"L{first_layer}_B{int(round(100 * B))}"
        kbest = min(range(1, N), key=lambda k: abs(fl({first_layer: k}) - B))
        a = make_entry(f"{gid}_one_stage", {first_layer: kbest}, (first_layer,), B, gid, "anchor", "shared")
        if abs(a["normalized_flops"] - B) > tol:
            unmatched.append({"first_layer": first_layer, "budget": B, "family": "shared", "structure": str(first_layer),
                              "stage_count": 1, "reason": "one-stage anchor cannot reach the budget within tolerance",
                              "best_diff_pp": 100 * (a["normalized_flops"] - B)})
            groups.append({"group": gid, "first_layer": first_layer, "budget_target": B, "anchor": None, "members": [],
                           "missing": ["all"], "complete": False})
            continue
        entries.append(a)
        members, missing = [], []
        for fam in families:
            for kind in ("two", "three"):
                for layers in structures.get(kind, []):
                    s, best = len(layers), None
                    for p in _grid(fam):
                        sched = dict(zip(layers, family_counts(fam, N, float(p), s)))
                        if not _valid_stage_schedule(sched, layers, est):
                            continue
                        d = fl(sched) - a["normalized_flops"]
                        if best is None or abs(d) < abs(best[0]) - 1e-15:
                            best = (d, sched)
                    struct = "-".join(map(str, layers))
                    if best is None or abs(best[0]) > tol:
                        unmatched.append({"first_layer": first_layer, "budget": B, "family": fam, "structure": struct,
                                          "stage_count": s, "reason": f"no valid {fam} schedule within tolerance of the one-stage anchor",
                                          "best_diff_pp": None if best is None else 100 * best[0]})
                        missing.append(f"{fam}:{struct}")
                        continue
                    e = make_entry(f"{gid}_{fam}_{STAGE_NAME[s]}_{struct}", best[1], layers, B, gid, "member", fam)
                    entries.append(e); members.append(e["name"])
        fls = [e["normalized_flops"] for e in entries if e["group"] == gid]
        groups.append({"group": gid, "first_layer": first_layer, "budget_target": B, "anchor": a["name"], "members": members,
                       "missing": missing, "complete": not missing, "flops_spread_pp": 100 * (max(fls) - min(fls)),
                       "max_abs_diff_vs_anchor_pp": 100 * max(abs(f - a["normalized_flops"]) for f in fls)})
    return {"entries": entries, "groups": groups, "unmatched": unmatched}


# ----------------------------------------------------------------------------- statistics

def build_statistical_tests(per_img: pd.DataFrame, entries: List[Dict[str, Any]], primary_scorer: str,
                            primary_layer: int, tol: float, primary_seed: int = 42) -> pd.DataFrame:
    """
    Paired tests on identical images (same scorer & seed). Comparison types:
      multi_vs_one_stage   : member minus its budget group's one-stage anchor (per allocation family)
      three_vs_two_prefix  : 3-stage minus the 2-stage schedule with the same first two layers (same family)
      geometric_vs_equal_drop : geometric minus equal_drop for the same structure/budget (only if |dFLOPs| <= tol)
    Predefined PRIMARY families (each Holm-corrected separately; scorer, first layer, seed as given):
      PRIMARY_stage_agreement / PRIMARY_stage_top1   : multi_vs_one_stage, both allocation families pooled
      PRIMARY_allocation_agreement / PRIMARY_allocation_top1 : geometric_vs_equal_drop
    Everything else is exploratory and Holm-corrected within its own (comparison, layer, scorer, seed, metric, family) family.
    """
    idx = {k: g.sort_values("image_idx") for k, g in per_img.groupby(["schedule", "scorer", "seed"])}
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)
    rows = []

    def add(scorer, seed, ea, eb, kind, metric, col):
        ga, gb = idx.get((ea["name"], scorer, seed)), idx.get((eb["name"], scorer, seed))
        if ga is None or gb is None:
            return
        s = paired_diff_stats(ga[col].to_numpy(), gb[col].to_numpy())
        rows.append({
            "comparison_type": kind, "first_layer": ea["first_layer"], "budget_target": ea["budget_target"],
            "scorer": scorer, "seed": seed, "metric": metric, "family_a": ea["family"], "family_b": eb["family"],
            "schedule_a": ea["name"], "stage_count_a": ea["stage_count"], "structure_a": ea["structure"],
            "schedule_b": eb["name"], "stage_count_b": eb["stage_count"], "structure_b": eb["structure"],
            "flops_a": ea["normalized_flops"], "flops_b": eb["normalized_flops"],
            "flops_diff_pp": 100 * (ea["normalized_flops"] - eb["normalized_flops"]),
            "n_images": s["n"], "count_a": s["a_count"], "count_b": s["b_count"], "diff": s["diff"],
            "ci_low": s["boot_ci_low"], "ci_high": s["boot_ci_high"], "a_only": s["a_only"], "b_only": s["b_only"],
            "p_raw": s["mcnemar_p"]})

    scorers_seeds = sorted({(k[1], k[2]) for k in idx if k[1] != "none"})
    for gid, ents in by_group.items():
        anchor = next(e for e in ents if e["role"] == "anchor")
        members = [e for e in ents if e["role"] == "member"]
        for e in members:
            for sc, sd in scorers_seeds:
                for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                    add(sc, sd, e, anchor, "multi_vs_one_stage", metric, col)
            if e["stage_count"] == 3:
                pre = next((x for x in members if x["stage_count"] == 2 and x["family"] == e["family"]
                            and x["layers"] == e["layers"][:2]), None)
                if pre is not None:
                    for sc, sd in scorers_seeds:
                        for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                            add(sc, sd, e, pre, "three_vs_two_prefix", metric, col)
        for e in [m for m in members if m["family"] == "geometric"]:
            o = next((x for x in members if x["family"] == "equal_drop" and x["structure"] == e["structure"]), None)
            if o is not None and abs(e["normalized_flops"] - o["normalized_flops"]) <= tol + 1e-12:
                for sc, sd in scorers_seeds:
                    for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                        add(sc, sd, e, o, "geometric_vs_equal_drop", metric, col)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def family(r):
        if r.scorer == primary_scorer and r.first_layer == primary_layer and r.seed == primary_seed:
            if r.comparison_type == "multi_vs_one_stage":
                return f"PRIMARY_stage_{r.metric}"
            if r.comparison_type == "geometric_vs_equal_drop":
                return f"PRIMARY_allocation_{r.metric}"
        extra = f"|{r.family_a}" if r.comparison_type != "geometric_vs_equal_drop" else ""
        return f"exploratory|{r.comparison_type}{extra}|L{r.first_layer}|{r.scorer}|seed{r.seed}|{r.metric}"

    df["family"] = df.apply(family, axis=1)
    df["is_primary"] = df.family.str.startswith("PRIMARY_")
    df["family_size"] = df.groupby("family").family.transform("size")
    df["p_holm"] = np.nan
    for fam, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm_correction(g.p_raw.tolist())
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["significant_raw_0.05_uncorrected"] = df.p_raw < 0.05
    return df


# ----------------------------------------------------------------------------- helpers

def _time_multi(pruner, xs: List[torch.Tensor], warmup: int, runs: int) -> Dict[str, float]:
    """Median over (batches x runs) of end-to-end latency, plus the per-batch medians' range and scorer overhead median."""
    per = [_time_pruner(pruner, x, warmup, runs) for x in xs]
    tot = [p["total_ms_median"] for p in per]
    return {"total_ms_median": statistics.median(tot), "total_ms_mean": statistics.mean(p["total_ms_mean"] for p in per),
            "total_ms_min_batch": min(tot), "total_ms_max_batch": max(tot),
            "scorer_ms_median": statistics.median(p["scorer_ms_median"] for p in per)}


def load_subset(cfg: ClassDiverseConfig, data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
    if data is not None:           # tests: tensors supplied directly
        images, labels = data
        man = build_manifest(images, labels, np.arange(len(images)))
        summ = {**class_coverage_summary(labels.tolist()), "manifest_sha256": manifest_hash(man),
                "sampling": {"algorithm": "tensors supplied directly (tests)"}}
        return {"images": images, "labels": labels, "manifest": man, "summary": summ}
    sub = build_class_diverse_subset(cfg.parquet_dir, cfg.images_per_class, cfg.subset_seed, cfg.subset_cache)
    if cfg.max_images:
        n = cfg.max_images
        images, labels = sub["images"][:n], sub["labels"][:n]
        man = sub["manifest"].iloc[:n].copy()
        summ = {**class_coverage_summary(labels.tolist()), "manifest_sha256": manifest_hash(man),
                "sampling": {**sub["summary"]["sampling"], "truncated_to_first": n,
                             "note": "SMOKE: prefix of the seeded random order; full-subset manifest hash is "
                                     + sub["summary"]["manifest_sha256"]}}
        return {"images": images, "labels": labels, "manifest": man, "summary": summ}
    return sub


def _flatten(d, prefix=""):
    return d


# ----------------------------------------------------------------------------- run

def run_class_diverse(cfg: ClassDiverseConfig, wrapper: Optional[ViTWrapper] = None,
                      data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    wrapper = wrapper or ViTWrapper(cfg.model_name, cfg.pretrained, cfg.device)
    L, N = wrapper.num_layers, wrapper.num_original_tokens
    est = ComputeEstimator(L, wrapper.hidden_dim, wrapper.num_heads, N, wrapper.num_classes,
                           wrapper.num_prefix_tokens, wrapper.patch_size)
    dev = wrapper.device
    out_p = lambda n: os.path.join(cfg.output_dir, n)

    sub = load_subset(cfg, data)
    images, labels, manifest, summary = sub["images"], sub["labels"], sub["manifest"], sub["summary"]
    n_img = len(images)
    manifest.to_csv(out_p("subset_manifest.csv"), index=False)
    json.dump(summary, open(out_p("subset_summary.json"), "w"), indent=2)
    batches = [(images[i:i + cfg.batch_size], labels[i:i + cfg.batch_size]) for i in range(0, n_img, cfg.batch_size)]
    with torch.no_grad():
        base_logits = [wrapper(x.to(dev)) for x, _ in batches]
    base_pred = torch.cat([l.argmax(-1) for l in base_logits]).cpu().numpy()
    y_all = labels.cpu().numpy()
    image_id = manifest.image_id.to_numpy()

    # ---- schedules
    structures = cfg.structures or DEFAULT_STRUCTURES
    first_layers = [cfg.primary_first_layer] + list(cfg.secondary_first_layers)
    entries, groups, unmatched = [], [], []
    for f in first_layers:
        g = generate_family_schedules(est, f, cfg.budgets, structures[f], cfg.match_tolerance, cfg.families)
        for e in g["entries"]:
            e["analysis"] = "primary" if f == cfg.primary_first_layer else f"secondary_first_layer_{f}"
        entries += g["entries"]; groups += g["groups"]; unmatched += g["unmatched"]
    base_entry = {"name": "no_pruning", "schedule": {}, "stage_count": 0, "first_layer": None, "structure": "", "layers": [],
                  "budget_target": 1.0, "group": "baseline", "role": "baseline", "family": "none", "analysis": "baseline",
                  "effective_layers": [], **schedule_flops(est, {})}
    all_entries = [base_entry] + entries

    def scorers_for(e):
        return cfg.scorers_primary_layer if e["first_layer"] == cfg.primary_first_layer else cfg.scorers_secondary_layer

    def in_scope(e, sc):   # secondary scorers (and layer 4) only on anchors + reduced structures
        if sc == "prediction_kl_approx" and not any(abs(e["budget_target"] - b) < 1e-9 for b in cfg.kl_budgets):
            return False
        if sc == cfg.primary_scorer or e["role"] == "anchor":
            return True
        return e["structure"] in cfg.reduced_structures or e["first_layer"] != cfg.primary_first_layer

    def seeds_for(sc):
        return [cfg.seed] + (list(cfg.extra_seeds_random) if sc == "random" else [])

    runs = [(base_entry, "none", cfg.seed)] + [(e, sc, s) for e in entries for sc in scorers_for(e) if in_scope(e, sc)
                                                for s in seeds_for(sc)]
    not_run = []
    for e in entries:
        for sc in scorers_for(e):
            if not in_scope(e, sc):
                not_run.append({"schedule": e["name"], "scorer": sc, "reason": "secondary scorer limited to anchors + reduced structures (compute budget)"})
    for f in cfg.secondary_first_layers:
        for sc in cfg.scorers_primary_layer:
            if sc not in cfg.scorers_secondary_layer:
                not_run.append({"first_layer": f, "scorer": sc, "reason": "layer-4 secondary analysis runs fewer scorers (compute budget)"})

    # ---- resume support (long run)
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
            done.add(key)
            continue
        correct, agree, t5 = preds == y_all, preds == base_pred, (top5 == y_all[:, None]).any(1)
        per_image.append(pd.DataFrame({
            "schedule": e["name"], "scorer": scorer, "seed": seed, "image_idx": np.arange(n_img), "image_id": image_id,
            "label": y_all, "baseline_pred": base_pred, "compressed_pred": preds,
            "baseline_correct": (base_pred == y_all).astype(int), "correct": correct.astype(int),
            "top5_correct": t5.astype(int), "agreement": agree.astype(int)}))
        for t in trace:
            token_rows.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "first_layer": e["first_layer"],
                               "stage_count": e["stage_count"], "family": e["family"], **t})
        row = {"schedule": e["name"], "analysis": e["analysis"], "group": e["group"], "first_layer": e["first_layer"],
               "stage_count": e["stage_count"], "structure": e["structure"], "allocation_family": e["family"],
               "budget_target": e["budget_target"], "scorer": scorer, "seed": seed,
               "schedule_json": json.dumps({str(k): v for k, v in e["schedule"].items()}), "n_images": n_img,
               "analytic_flops": e["analytic_flops"], "normalized_flops": e["normalized_flops"],
               "flops_savings_pct": 100.0 * (1.0 - e["normalized_flops"]),
               "flops_diff_vs_anchor_pp": None}
        for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
            w = wilson_score_interval(int(arr.sum()), n_img)
            row.update({f"{name}_count": int(arr.sum()), name: w["mean"],
                        f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
        rows.append(row); done.add(key)
        if (len(done)) % 10 == 0 or ri + 1 == len(runs):
            print(f"[{len(done)}/{len(runs)}] {e['name']} / {scorer} / seed {seed} ({time.time() - t_start:.0f}s)", flush=True)
            pickle.dump({"sig": sig, "rows": rows, "per_image": per_image, "token_rows": token_rows, "failures": failures,
                         "done": done}, open(ck_path, "wb"))

    metrics = pd.DataFrame(rows)
    anchors = {e["group"]: e["normalized_flops"] for e in entries if e["role"] == "anchor"}
    if len(metrics):
        metrics["flops_diff_vs_anchor_pp"] = [100 * (r.normalized_flops - anchors[r.group]) if r.group in anchors else np.nan
                                              for r in metrics.itertuples()]

    # ---- latency (CPU/GPU as available; one device only here), several representative batches
    unpruned_latency, lat_info = None, {}
    if cfg.measure_latency and len(metrics):
        nb = len(batches)
        pick = sorted({int(round(i * (nb - 1) / max(cfg.latency_batches - 1, 1))) for i in range(cfg.latency_batches)})
        xs_all = [batches[i][0].to(dev) for i in pick]
        lat_info = {"batch_indices": pick, "batch_size": int(len(xs_all[0]))}
        with torch.no_grad():
            wrapper(xs_all[0])
            ts = [(lambda t0: (wrapper(x), (time.perf_counter() - t0) * 1e3)[1])(time.perf_counter())
                  for x in xs_all for _ in range(cfg.latency_runs)]
        unpruned_latency = {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts), "n_timings": len(ts)}
        lat = {}
        for e in all_entries:
            for sc in (["none"] if e["role"] == "baseline" else scorers_for(e)):
                if e["role"] != "baseline" and not in_scope(e, sc):
                    continue
                p = ProgressivePruner(wrapper, e["schedule"], "random" if sc == "none" else sc, seed=cfg.seed)
                kl = sc == "prediction_kl_approx"
                lat[(e["name"], sc)] = _time_multi(p, xs_all[:cfg.kl_latency_batches] if kl else xs_all, cfg.latency_warmup,
                                                   cfg.kl_latency_runs if kl else cfg.latency_runs)
        for col, key in (("latency_total_ms_median", "total_ms_median"), ("latency_total_ms_mean", "total_ms_mean"),
                         ("latency_total_ms_min_batch", "total_ms_min_batch"), ("latency_total_ms_max_batch", "total_ms_max_batch"),
                         ("scorer_overhead_ms_median", "scorer_ms_median")):
            metrics[col] = [lat.get((r.schedule, r.scorer), {}).get(key, np.nan) if r.seed == cfg.seed else np.nan
                            for r in metrics.itertuples()]
        metrics["latency_device"] = str(dev); metrics["latency_batch_size"] = lat_info["batch_size"]
        metrics["latency_warmup"] = cfg.latency_warmup
        metrics["latency_timed_runs"] = [cfg.kl_latency_runs if r.scorer == "prediction_kl_approx" else cfg.latency_runs
                                         for r in metrics.itertuples()]
        metrics["latency_num_batches"] = [min(cfg.kl_latency_batches, len(xs_all)) if r.scorer == "prediction_kl_approx"
                                          else len(xs_all) for r in metrics.itertuples()]
    per_img_df = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    tc_df = pd.DataFrame(token_rows)
    metrics.to_csv(out_p("metrics.csv"), index=False)
    per_img_df.to_csv(out_p("per_image_metrics.csv"), index=False)
    tc_df.to_csv(out_p("token_counts.csv"), index=False)
    stats_df = build_statistical_tests(per_img_df, [e for e in entries], cfg.primary_scorer, cfg.primary_first_layer,
                                       cfg.match_tolerance) if len(per_img_df) else pd.DataFrame()
    stats_df.to_csv(out_p("statistical_tests.csv"), index=False)

    manifest_json = {
        "layer_numbering_and_semantics": LAYER_NUMBERING,
        "generation_procedure": __doc__.split("Schedule-generation procedure")[1].split(":\n", 1)[1].strip(),
        "allocation_families": {"geometric": "K_j = max(1, round(N*rho^j))", "equal_drop": "K_j = max(1, round(N - j*D))"},
        "baseline_flops": est.baseline_flops, "match_tolerance_pp_of_baseline_flops": 100 * cfg.match_tolerance,
        "budgets": cfg.budgets, "first_layers": {"primary": cfg.primary_first_layer, "secondary": cfg.secondary_first_layers},
        "schedules": [{"name": e["name"], "schedule": {str(k): v for k, v in e["schedule"].items()},
                       "stage_count": e["stage_count"], "first_layer": e["first_layer"], "structure": e["structure"],
                       "allocation_family": e["family"], "budget_target": e["budget_target"], "group": e["group"],
                       "role": e["role"], "analysis": e["analysis"], "analytic_flops": e["analytic_flops"],
                       "normalized_flops": e["normalized_flops"],
                       "patch_tokens_processed_per_block": expected_processed_counts(e["schedule"], L, N)} for e in all_entries],
        "matched_groups": groups, "unmatched_schedules": unmatched}
    json.dump(manifest_json, open(out_p("schedule_manifest.json"), "w"), indent=2)
    run_config = {
        "config": cfg.to_dict(), "subset": {"num_images": n_img, "manifest_sha256": summary["manifest_sha256"],
                                            "distinct_classes": summary["num_distinct_classes"],
                                            "sampling": summary.get("sampling")},
        "environment": _environment(wrapper),
        "preprocessing": "Phase 1 pipeline get_transforms: Resize 256 bicubic, CenterCrop 224, ImageNet mean/std",
        "seeds": {"primary_all_scorers": cfg.seed, "extra_random_only": cfg.extra_seeds_random, "subset_sampling_seed": cfg.subset_seed},
        "seed_notes": ("Random scorer: a fresh generator is seeded per (schedule, scorer, seed) run and advances across batches and "
                       "pruning calls in fixed image order. The same seed values (42, 43, 44) are used for every schedule (common "
                       "seeds), but the random draws themselves differ between schedules (independent streams per schedule, not paired). "
                       "cls_attention and prediction_kl_approx are deterministic."),
        "timing": {"method": ("time.perf_counter around ProgressivePruner.forward (scoring + pruning + blocks + head). Median over "
                              "timed runs on several evenly spaced batches (kl_approx: fewer), after warm-up; scorer overhead = "
                              "time inside the scorer calls. prediction_kl_approx totals also include one unpruned forward pass for the "
                              "target class (not in scorer overhead)."),
                   "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs, "batches": lat_info,
                   "kl_timed_runs": cfg.kl_latency_runs, "kl_batches": cfg.kl_latency_batches,
                   "unpruned_model_forward": unpruned_latency, "enabled": cfg.measure_latency,
                   "gpu": "not available in this environment; no GPU results" if not torch.cuda.is_available() else "see device"},
        "failed_or_unsupported": failures, "scorer_analysis_combinations_not_run": not_run, "num_runs": len(runs)}
    json.dump(run_config, open(out_p("run_config.json"), "w"), indent=2)

    out = {"metrics": metrics, "per_image": per_img_df, "token_counts": tc_df, "stats": stats_df, "entries": all_entries,
           "groups": groups, "unmatched": unmatched, "manifest": manifest_json, "failures": failures, "run_config": run_config,
           "estimator": est, "subset_summary": summary}
    finalize_reports(out, cfg)
    if os.path.isfile(ck_path) and not failures:
        os.remove(ck_path)
    return out


def finalize_reports(out: Dict[str, Any], cfg: ClassDiverseConfig) -> None:
    out["historical"] = historical_comparison(out["stats"], cfg)
    if out["historical"] is not None and len(out["historical"]):
        out["historical"].to_csv(os.path.join(cfg.output_dir, "historical_comparison.csv"), index=False)
    if cfg.make_plots and len(out["metrics"]):
        from ..visualization.class_diverse_plots import make_all_plots
        make_all_plots(out["metrics"], out["stats"], cfg.output_dir, cfg.seed)
    write_summary(out, cfg)


def regenerate_reports(output_dir: str) -> Dict[str, Any]:
    """Rebuild stats, plots and summary.md from saved files (no model re-run)."""
    rd = lambda n: os.path.join(output_dir, n)
    run_config = json.load(open(rd("run_config.json"))); man = json.load(open(rd("schedule_manifest.json")))
    c = run_config["config"]; c["structures"] = None
    cfg = ClassDiverseConfig(**{k: v for k, v in c.items() if k in ClassDiverseConfig.__dataclass_fields__})
    metrics, per_img = pd.read_csv(rd("metrics.csv")), pd.read_csv(rd("per_image_metrics.csv"))
    entries = [{**s, "schedule": {int(k): v for k, v in s["schedule"].items()}, "family": s["allocation_family"],
                "layers": [int(x) for x in s["structure"].split("-")] if s["structure"] else []} for s in man["schedules"]]
    stats_df = build_statistical_tests(per_img, [e for e in entries if e["role"] != "baseline"], cfg.primary_scorer,
                                       cfg.primary_first_layer, cfg.match_tolerance)
    stats_df.to_csv(rd("statistical_tests.csv"), index=False)
    out = {"metrics": metrics, "per_image": per_img, "stats": stats_df, "entries": entries, "groups": man["matched_groups"],
           "unmatched": man["unmatched_schedules"], "failures": run_config["failed_or_unsupported"], "run_config": run_config,
           "manifest": man, "subset_summary": json.load(open(rd("subset_summary.json")))}
    finalize_reports(out, cfg)
    return out


# ----------------------------------------------------------------------------- historical comparison

def historical_comparison(stats: pd.DataFrame, cfg: ClassDiverseConfig) -> Optional[pd.DataFrame]:
    """Previous 5-class/250-image ablation (geometric family) vs this run's geometric family; clearly a different dataset."""
    if cfg.previous_stage_ablation_dir is None or stats is None or stats.empty:
        return None
    p = os.path.join(cfg.previous_stage_ablation_dir, "statistical_tests.csv")
    if not os.path.isfile(p):
        return None
    prev = pd.read_csv(p)
    prev = prev[(prev.scorer == cfg.primary_scorer) & (prev.first_layer == cfg.primary_first_layer) & (prev.seed == 42)
                & (prev.comparison_type == "multi_vs_one_stage") & (prev.metric == "agreement")]
    cur = stats[(stats.scorer == cfg.primary_scorer) & (stats.first_layer == cfg.primary_first_layer) & (stats.seed == 42)
                & (stats.comparison_type == "multi_vs_one_stage") & (stats.metric == "agreement") & (stats.family_a == "geometric")]
    m = prev.merge(cur, on=["budget_target", "structure_a"], suffixes=("_prev", "_new"))
    cols = ["budget_target", "structure_a", "stage_count_a_prev", "n_images_prev", "diff_prev", "p_raw_prev", "p_holm_prev",
            "n_images_new", "diff_new", "ci_low_new", "ci_high_new", "p_raw_new", "p_holm_new"]
    return m[[c for c in cols if c in m.columns]]


# ----------------------------------------------------------------------------- summary

def write_summary(out: Dict[str, Any], cfg: ClassDiverseConfig) -> None:
    m, st, rc, man, ss = out["metrics"], out["stats"], out["run_config"], out["manifest"], out["subset_summary"]
    seed, n = cfg.seed, int(m.n_images.iloc[0]) if len(m) else 0
    cov = ss["per_class_count_over_present_classes"]
    L = ["# Phase 2: Class-Diverse Validation and Allocation-Robustness Study\n",
         "*One model, one class-balanced validation sample. Corrected and uncorrected p-values are shown separately.*\n",
         "## Subset (class coverage)\n",
         f"- Images: **{ss['num_images']}**; distinct classes: **{ss['num_distinct_classes']}/{ss['num_classes_total']}**; "
         f"per-class count min/median/max: **{cov.get('min')}/{cov.get('median')}/{cov.get('max')}**; "
         f"class-count histogram {ss['count_histogram']}.",
         f"- Manifest SHA-256: `{ss['manifest_sha256']}` (`subset_manifest.csv`: ordered image id, source row, label, preprocessed-tensor SHA-1, encoded-image SHA-1).",
         f"- Sampling: {rc['subset']['sampling'].get('algorithm', '')}",
         "- **Limitation:** a class-balanced seeded sample of the ImageNet-1k validation set. It is balanced by class, not a random sample "
         "of the validation distribution, and is not claimed to represent the full distribution.\n",
         "## Methods\n",
         f"- Model `{cfg.model_name}` (pretrained={cfg.pretrained}); eval mode; device `{rc['environment']['device']}`; same preprocessing as Phase 1/2.",
         f"- Layer numbering: {LAYER_NUMBERING}",
         f"- **Schedule generation** (FLOPs only, accuracy never consulted; allocation rules fixed before evaluation): {man['generation_procedure']}",
         f"- Budgets (nominal fraction of unpruned FLOPs): {cfg.budgets}; tolerance fixed in advance: ±{100 * cfg.match_tolerance:.1f} pp of baseline FLOPs vs the one-stage anchor "
         "(so two matched schedules differ by at most 2 pp; the allocation-family comparison additionally requires |ΔFLOPs| ≤ 1 pp directly). FLOPs: existing full-path analytic calculator, one common baseline.",
         f"- **Predefined primary families** (cls_attention, first layer {cfg.primary_first_layer}, seed 42; each Holm-corrected separately): "
         "(1) stage count: every matched 2-/3-stage schedule vs the one-stage schedule of its budget, both allocation families pooled, agreement; same for top-1; "
         "(2) allocation: geometric vs equal_drop for each (budget, structure), agreement; same for top-1. All else (random, prediction_kl_approx, layer 4, 3-vs-2-stage) is exploratory with Holm within its own family.",
         "- `prediction_kl_approx` is the Phase 1 gradient x activation saliency, **not exact KL**; scorer semantics unchanged.",
         f"- Seeds: {cfg.seed} (all scorers) plus {cfg.extra_seeds_random} for `random` only. {rc['seed_notes']}",
         f"- Secondary scorers (random, prediction_kl_approx) run on the anchors and structures {cfg.reduced_structures} only (prediction_kl_approx additionally only at budgets {cfg.kl_budgets} because it is ~100x slower); layer 4 runs {cfg.scorers_secondary_layer} on a reduced structure set (explicit, for compute).",
         ""]
    L.append("## Matched FLOPs budgets and unmatched schedules\n")
    L.append(_md_table([[g["group"], g.get("anchor") or "—", len(g["members"]), ", ".join(g["missing"]) or "none",
                         f"{g.get('max_abs_diff_vs_anchor_pp', float('nan')):.2f}" if g.get("anchor") else "—"] for g in out["groups"]],
                       ["group", "one-stage anchor", "matched multi-stage schedules", "unmatched", "max |ΔFLOPs| vs anchor (pp)"]))
    L.append("\nUnmatched details: " + ("none." if not out["unmatched"] else "; ".join(
        f"L{u['first_layer']} B{int(100 * u['budget'])} {u['family']} {u['structure']}: {u['reason']} (best Δ {u['best_diff_pp']} pp)"
        for u in out["unmatched"])) + "\n")

    def mtable(scorer, pred, title, seed_):
        sub = m[(m.scorer == scorer) & (m.seed == seed_) & pred(m)].sort_values(["group", "stage_count", "allocation_family", "structure"])
        if sub.empty:
            return
        L.append(f"### {title}\n")
        L.append(_md_table([[r.group, r.stage_count, r.allocation_family, r.structure, json.dumps(json.loads(r.schedule_json)),
                             f"{100 * r.normalized_flops:.2f}%",
                             f"{r.agreement_count}/{r.n_images} {_pct(r.agreement)} [{_pct(r.agreement_ci_low)}, {_pct(r.agreement_ci_high)}]",
                             f"{r.top1_count}/{r.n_images} {_pct(r.top1)} [{_pct(r.top1_ci_low)}, {_pct(r.top1_ci_high)}]"]
                            for r in sub.itertuples()],
                           ["group", "stages", "family", "layers", "layer→k", "norm. FLOPs", "agreement (Wilson 95% CI)", "top-1 (Wilson 95% CI)"]))
        L.append("")

    def ttable(sel, title, note=""):
        if st.empty:
            return
        s = st[sel(st)].sort_values(["budget_target", "family_a", "stage_count_a", "structure_a"])
        if s.empty:
            return
        L.append(f"### {title}\n")
        if note:
            L.append(note + "\n")
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.family_a, f"{r.structure_a} vs {r.structure_b}", f"{r.flops_diff_pp:+.2f}",
                             f"{r.count_a} vs {r.count_b}", f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]",
                             f"{r.p_raw:.4g}", f"{r.p_holm:.4g}"] for r in s.itertuples()],
                           ["budget", "family (A)", "A vs B", "ΔFLOPs pp", "count A vs B", "Δ pp [95% paired-bootstrap CI]", "raw p (exact McNemar)", "Holm p"]))
        k, pos, neg = int(s["significant_holm_0.05"].sum()), int((s["diff"] > 0).sum()), int((s["diff"] < 0).sum())
        L.append(f"\nFamily size {len(s)}: A higher in {pos}, lower in {neg}, tied in {len(s) - pos - neg}; mean Δ {100 * s['diff'].mean():+.2f} pp; "
                 f"Holm-significant (p<0.05): {k}; uncorrected p<0.05: {int(s['significant_raw_0.05_uncorrected'].sum())}. "
                 "(Win counts are descriptive; the Holm column is the inferential result.)\n")

    pl = cfg.primary_first_layer
    L.append(f"## PRIMARY: `{cfg.primary_scorer}`, first layer {pl}, one vs two vs three stages, both allocation families (seed {seed})\n")
    mtable(cfg.primary_scorer, lambda d: d.first_layer == pl, "Metrics (counts out of N, Wilson 95% CI)", seed)
    ttable(lambda d: d.family == "PRIMARY_stage_agreement", "Primary family 1: agreement, multi-stage (A) minus one-stage (B)")
    ttable(lambda d: d.family == "PRIMARY_stage_top1", "Primary family 2: top-1 accuracy, multi-stage (A) minus one-stage (B)")
    ttable(lambda d: d.family == "PRIMARY_allocation_agreement", "Primary family 3: allocation, geometric (A) minus equal_drop (B), agreement")
    ttable(lambda d: d.family == "PRIMARY_allocation_top1", "Primary family 4: allocation, geometric (A) minus equal_drop (B), top-1")
    if not st.empty:
        s = st[st.family == "PRIMARY_stage_agreement"]
        if not s.empty:
            g = s.groupby(["family_a", "stage_count_a"])["diff"].agg(["mean", "min", "max", "count"]).reset_index()
            L.append("### Persistence across allocation families (primary family 1, agreement Δ vs one-stage)\n")
            L.append(_md_table([[r.family_a, int(r.stage_count_a), f"{100 * r['mean']:+.2f}", f"{100 * r['min']:+.2f}", f"{100 * r['max']:+.2f}", int(r['count'])]
                                for _, r in g.iterrows()], ["family", "stages", "mean Δ pp", "min Δ pp", "max Δ pp", "comparisons"]))
            L.append("")
    L.append("## SECONDARY / EXPLORATORY (separate from the primary analysis)\n")
    ttable(lambda d: (d.comparison_type == "three_vs_two_prefix") & (d.scorer == cfg.primary_scorer) & (d.seed == seed)
           & (d.first_layer == pl) & (d.metric == "agreement"),
           f"Exploratory: three-stage (A) vs its two-stage prefix (B), `{cfg.primary_scorer}`, agreement (both families; Holm per family)")
    for sc in cfg.scorers_primary_layer:
        if sc == cfg.primary_scorer:
            continue
        mtable(sc, lambda d: d.first_layer == pl, f"First layer {pl}: `{sc}` metrics (seed {seed})", seed)
        ttable(lambda d, sc=sc: (d.scorer == sc) & (d.seed == seed) & (d.first_layer == pl) & (d.comparison_type == "multi_vs_one_stage")
               & (d.metric == "agreement"), f"First layer {pl}: `{sc}`, agreement, multi-stage (A) minus one-stage (B) (exploratory)")
    rs = st[(st.scorer == "random") & (st.comparison_type == "multi_vs_one_stage") & (st.metric == "agreement")
            & (st.first_layer == pl)] if not st.empty else st
    if not rs.empty and rs.seed.nunique() > 1:
        L.append("### Random scorer: seed-to-seed variation (separate from image-level uncertainty)\n")
        g = rs.groupby(["budget_target", "family_a", "structure_a"])["diff"].agg(["mean", "std", "count"]).reset_index()
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.family_a, r.structure_a, f"{100 * r['mean']:+.2f}", f"{100 * r['std']:.2f}", int(r['count'])]
                            for _, r in g.iterrows()], ["budget", "family", "structure", "mean Δ pp over seeds", "sd across seeds pp", "seeds"]))
        L.append("")
    for f in cfg.secondary_first_layers:
        L.append(f"### First layer {f} (secondary, exploratory, kept separate)\n")
        for sc in cfg.scorers_secondary_layer:
            mtable(sc, lambda d, f=f: d.first_layer == f, f"First layer {f}: `{sc}` metrics (seed {seed})", seed)
            ttable(lambda d, sc=sc, f=f: (d.scorer == sc) & (d.seed == seed) & (d.first_layer == f) & (d.comparison_type == "multi_vs_one_stage")
                   & (d.metric == "agreement"), f"First layer {f}: `{sc}`, agreement, multi-stage (A) minus one-stage (B)")

    L.append("## FLOPs, latency and scorer overhead (separate quantities; no speedup inferred from FLOPs)\n")
    if cfg.measure_latency and "latency_total_ms_median" in m:
        ul, tm = rc["timing"]["unpruned_model_forward"], rc["timing"]
        L.append(f"Device `{rc['environment']['device']}` only ({tm['gpu']}); batch {int(m.latency_batch_size.dropna().iloc[0])}; warm-up {cfg.latency_warmup}; "
                 f"{cfg.latency_runs} timed runs on each of {len(tm['batches'].get('batch_indices', []))} evenly spaced batches "
                 f"(prediction_kl_approx: {cfg.kl_latency_runs} runs × {cfg.kl_latency_batches} batch); median ms; unpruned model forward "
                 f"{ul['median_ms']:.1f} ms. {tm['method']}\n")
        sub = m[(m.seed == seed) & (m.first_layer == pl) & m.latency_total_ms_median.notna()].sort_values(["scorer", "group", "stage_count", "allocation_family", "structure"])
        L.append(_md_table([[r.scorer, r.group, r.stage_count, r.allocation_family, r.structure, f"{100 * r.normalized_flops:.1f}%",
                             f"{r.latency_total_ms_median:.0f} ({r.latency_total_ms_min_batch:.0f}–{r.latency_total_ms_max_batch:.0f})",
                             f"{r.scorer_overhead_ms_median:.1f}"] for r in sub.itertuples()],
                           ["scorer", "group", "stages", "family", "layers", "norm. FLOPs", "total ms (incl. selection; range over batches)", "scorer overhead ms"]))
        L.append("")
    else:
        L.append("Latency not measured.\n")

    L.append("## Historical comparison (DIFFERENT DATASET: previous 250-image, 5-class contiguous subset)\n")
    h = out.get("historical")
    if h is not None and len(h):
        L.append("Same scorer (`cls_attention`), first layer 2, geometric family, agreement Δ (multi-stage minus one-stage), same budgets/structures. "
                 "Previous = 5 classes, 250 images. New = class-diverse subset. Different images, so differences are not paired.\n")
        L.append(_md_table([[f"B{int(100 * r.budget_target)}", r.structure_a, f"{100 * r.diff_prev:+.1f}", f"{r.p_holm_prev:.3g}",
                             f"{100 * r.diff_new:+.2f} [{100 * r.ci_low_new:+.2f}, {100 * r.ci_high_new:+.2f}]", f"{r.p_holm_new:.3g}"] for r in h.itertuples()],
                           ["budget", "structure", "previous Δ pp (5-class)", "previous Holm p (its 24-test family)", "new Δ pp [95% CI]", "new Holm p (primary family)"]))
        L.append(f"\nMean Δ: previous {100 * h.diff_prev.mean():+.2f} pp vs new (geometric rows) {100 * h.diff_new.mean():+.2f} pp.\n")
    else:
        L.append("Previous stage-ablation results not available.\n")
    L.append("## Failed / unsupported / not-run combinations\n")
    L.append(("Failed runs: none." if not out["failures"] else "\n".join(f"- {f}" for f in out["failures"])) + "\n")
    nr = rc.get("scorer_analysis_combinations_not_run", [])
    if nr:
        L.append(f"Not run (explicit, to bound compute): {len(nr)} schedule/scorer combinations (listed in `run_config.json`).\n")
    L.append("## Limitations\n")
    L.append(f"- {n} images ({ss['num_distinct_classes']} classes); class-balanced sample, not the full validation set; one model; CPU only.\n"
             "- Two allocation families only; other count allocations at equal FLOPs may behave differently.\n"
             "- Matched schedules differ in the *later* pruning layers; stage count is confounded with which layers are used.\n"
             "- Random draws differ between schedules even with common seeds.\n"
             "- FLOPs are analytic; latency is measured on a few CPU batches and is noisy; no speedup claim is made from FLOPs.\n"
             "- Secondary scorers and layer 4 cover a reduced schedule set.\n")
    L.append("## Commands\n")
    L.append("```\npython -m pytest tests -q\n"
             "python scripts/run_class_diverse.py --config configs/class_diverse.yaml --prepare-only   # build + verify subset\n"
             "python scripts/run_class_diverse.py --config configs/class_diverse.yaml --smoke          # small smoke test\n"
             "python scripts/run_class_diverse.py --config configs/class_diverse.yaml                  # full run\n```\n")
    with open(os.path.join(cfg.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
