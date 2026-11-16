"""
Phase 2 next step: controlled evaluation of progressive vs single-layer token pruning at matched analytic FLOPs.
Evaluation infrastructure only: reuses the Phase 1 model/data/scorers/metrics and the Phase 2 ProgressivePruner.

Layer numbering (confirmed from src/compression/progressive.py, unchanged): schedule keys are 1-based layer numbers.
Key `l` = "after `l` Transformer blocks have run (0-based block index l-1), keep exactly `k` PATCH tokens"; blocks
with 0-based index l..L-1 see the shorter sequence. CLS is always kept and is not counted in `k`. Layer 12 (final
block) is not a valid pruning point. A key whose k equals the current patch count is a legal no-op.
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..compression.progressive import ProgressivePruner, RetentionSchedule
from ..data.dataset import DatasetConfig, get_dataloader
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from .layerwise_analysis import wilson_score_interval

LAYER_NUMBERING = (
    "1-based layer numbers (same as Phase 1). Key l = after l blocks have run (0-based block index l-1), keep exactly "
    "k PATCH tokens; blocks l..L-1 (0-based) receive the shorter sequence. CLS always kept and not counted in k. "
    "Layer L (final block) is rejected. k equal to the current patch count is a no-op. Positional embeddings are "
    "added once at the input and never re-added."
)

# Required candidate schedules (spec section 3) plus a few extra progressive shapes for variety.
DEFAULT_SCHEDULES: Dict[str, Dict[int, int]] = {
    "no_pruning": {},
    "early_single": {2: 98},
    "middle_single": {6: 98},
    "late_single": {10: 98},
    "early_progressive": {2: 98, 4: 49},
    "gradual_progressive": {2: 147, 6: 98, 10: 49},
    "delayed_progressive": {2: 196, 4: 98, 8: 49},
    "mild_progressive": {4: 147, 8: 98},
    "aggressive_progressive": {2: 98, 4: 73, 6: 49, 8: 24},
    "late_progressive": {6: 98, 8: 73, 10: 49},
}


@dataclass
class EvalConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    dataset_name: str = "imagenet_val"
    data_dir: str = "./data"
    num_samples: int = 250
    batch_size: int = 25
    dataset_seed: int = 42
    seed: int = 42                                   # primary seed (all scorers)
    extra_seeds_random: List[int] = field(default_factory=lambda: [43, 44])  # random scorer only
    scorers: List[str] = field(default_factory=lambda: ["random", "cls_attention", "prediction_kl_approx"])
    schedules: Optional[Dict[str, Dict[int, int]]] = None   # None -> DEFAULT_SCHEDULES
    auto_match: bool = True                          # add single-layer controls whose FLOPs match each progressive
    match_tolerance: float = 0.02                    # |normalized FLOPs diff| <= 2 percentage points of baseline
    single_layers: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    measure_latency: bool = True
    latency_warmup: int = 1
    latency_runs: int = 3
    output_dir: str = "./results/phase2_progressive_eval"
    device: Optional[str] = None
    make_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["schedules"] is not None:
            d["schedules"] = {n: {str(k): v for k, v in s.items()} for n, s in d["schedules"].items()}
        return d


# ----------------------------------------------------------------------------- schedules & FLOPs

def expected_processed_counts(schedule: Dict[int, int], num_layers: int, num_patches: int) -> List[int]:
    """Patch tokens each block (0-based index i) PROCESSES under `schedule`, computed analytically."""
    cur, counts = num_patches, []
    for i in range(num_layers):
        counts.append(cur)
        k = schedule.get(i + 1)
        if k is not None and k < cur:
            cur = k
    return counts


def effective_prune_layers(schedule: Dict[int, int], num_layers: int, num_patches: int) -> List[int]:
    """1-based layers at which the patch-token count actually drops (no-op keys are ignored)."""
    counts = expected_processed_counts(schedule, num_layers, num_patches)
    return [i + 1 for i, (a, b) in enumerate(zip(counts, counts[1:])) if b < a]


def effective_prune_points(schedule: Dict[int, int], num_layers: int, num_patches: int) -> int:
    return len(effective_prune_layers(schedule, num_layers, num_patches))


def schedule_type(schedule: Dict[int, int], num_layers: int, num_patches: int) -> str:
    n = effective_prune_points(schedule, num_layers, num_patches)
    return "baseline" if n == 0 else ("single" if n == 1 else "progressive")


def schedule_flops(est: ComputeEstimator, schedule: Dict[int, int]) -> Dict[str, float]:
    counts = expected_processed_counts(schedule, est.num_layers, est.N)
    r = est.compute_progressive_flops(counts)
    return {"analytic_flops": r["total_flops"], "normalized_flops": r["normalized_flops"]}


def build_schedule_set(cfg: EvalConfig, est: ComputeEstimator) -> List[Dict[str, Any]]:
    """Named schedules + FLOPs-matched single-layer controls (chosen from FLOPs only, never from accuracy)."""
    L, N = est.num_layers, est.N
    base = cfg.schedules if cfg.schedules is not None else DEFAULT_SCHEDULES
    entries: List[Dict[str, Any]] = []
    seen: Dict[Tuple, str] = {}

    def add(name: str, sched: Dict[int, int], origin: str) -> str:
        sched = {int(k): int(v) for k, v in sched.items()}
        RetentionSchedule.from_dict(sched).validate(L, N)
        key = tuple(sorted(sched.items()))
        if key in seen:
            return seen[key]
        seen[key] = name
        entries.append({"name": name, "schedule": sched, "type": schedule_type(sched, L, N), "origin": origin,
                        "effective_layers": effective_prune_layers(sched, L, N),
                        **schedule_flops(est, sched)})
        return name

    for name, sched in base.items():
        add(name, sched, "specified")

    if cfg.auto_match:
        for e in [x for x in entries if x["type"] == "progressive"]:
            for layer in cfg.single_layers:
                if not 1 <= layer < L:
                    continue
                best = min(
                    range(1, N),  # k < N so the schedule truly prunes
                    key=lambda k: abs(est.compute_progressive_flops(
                        expected_processed_counts({layer: k}, L, N))["normalized_flops"] - e["normalized_flops"]),
                )
                diff = schedule_flops(est, {layer: best})["normalized_flops"] - e["normalized_flops"]
                if abs(diff) <= cfg.match_tolerance:
                    add(f"single_L{layer}_k{best}", {layer: best}, f"auto-matched to {e['name']}")
    return entries


def find_matched_pairs(entries: List[Dict[str, Any]], tol: float) -> List[Dict[str, Any]]:
    """
    Primary matching rule (fixed before any accuracy is seen): for each progressive schedule and each pruning layer
    used by single-layer schedules, take the single-layer schedule at that layer with the nearest normalized FLOPs,
    provided |diff| <= tol (tol in baseline-FLOPs fraction; 0.02 = 2 percentage points).
    """
    progs = [e for e in entries if e["type"] == "progressive"]
    singles = [e for e in entries if e["type"] == "single"]
    pairs = []
    for p in progs:
        by_layer: Dict[int, Dict[str, Any]] = {}
        for s in singles:
            layer = s["effective_layers"][0]  # the single layer at which tokens are actually pruned
            d = abs(s["normalized_flops"] - p["normalized_flops"])
            if d <= tol and (layer not in by_layer or d < abs(by_layer[layer]["normalized_flops"] - p["normalized_flops"])):
                by_layer[layer] = s
        for layer, s in sorted(by_layer.items()):
            pairs.append({
                "progressive": p["name"], "single": s["name"], "single_layer": layer,
                "progressive_schedule": {str(k): v for k, v in p["schedule"].items()},
                "single_schedule": {str(k): v for k, v in s["schedule"].items()},
                "progressive_norm_flops": p["normalized_flops"], "single_norm_flops": s["normalized_flops"],
                "flops_diff_pp": 100.0 * (p["normalized_flops"] - s["normalized_flops"]),
                "tolerance_pp": 100.0 * tol,
            })
    return pairs


# ----------------------------------------------------------------------------- statistics

def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on discordant pairs)."""
    return 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_diff_stats(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 12345) -> Dict[str, float]:
    """Progressive (a) minus single (b) on the same images: diff, discordant counts, McNemar p, bootstrap CI."""
    a, b = a.astype(bool), b.astype(bool)
    n = len(a)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = a.astype(float)[idx].mean(1) - b.astype(float)[idx].mean(1)
    return {
        "n": n, "a_count": int(a.sum()), "b_count": int(b.sum()),
        "diff": float(a.mean() - b.mean()), "a_only": only_a, "b_only": only_b,
        "mcnemar_p": exact_mcnemar(only_a, only_b),
        "boot_ci_low": float(np.percentile(boots, 2.5)), "boot_ci_high": float(np.percentile(boots, 97.5)),
    }


# ----------------------------------------------------------------------------- helpers

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment(wrapper) -> Dict[str, Any]:
    import timm
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    cfg = getattr(wrapper.model, "pretrained_cfg", {}) or {}
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "timm": timm.__version__, "device": str(wrapper.device), "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(), "cpu": platform.processor(), "git_commit": commit,
        "checkpoint": {"model": wrapper.model_name, "pretrained": wrapper.pretrained,
                       "pretrained_cfg": {k: (v if isinstance(v, (str, int, float, bool, list, tuple, type(None))) else str(v))
                                          for k, v in cfg.items()}},
    }


def _time_pruner(pruner: ProgressivePruner, x: torch.Tensor, warmup: int, runs: int) -> Dict[str, float]:
    dev = x.device if x.device.type != "cpu" else None
    for _ in range(warmup):
        pruner.forward(x)
    tot, sc = [], []
    for _ in range(runs):
        if dev is not None:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = pruner.forward(x)
        if dev is not None:
            torch.cuda.synchronize()
        tot.append((time.perf_counter() - t0) * 1e3)
        sc.append(r.scorer_time_ms)
    return {"total_ms_median": statistics.median(tot), "total_ms_mean": statistics.mean(tot),
            "scorer_ms_median": statistics.median(sc)}


def _md_table(rows: List[List[Any]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- main evaluation

def run_evaluation(
    cfg: EvalConfig,
    wrapper: Optional[ViTWrapper] = None,
    data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, Any]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    wrapper = wrapper or ViTWrapper(cfg.model_name, cfg.pretrained, cfg.device)
    L, N = wrapper.num_layers, wrapper.num_original_tokens
    est = ComputeEstimator(L, wrapper.hidden_dim, wrapper.num_heads, N, wrapper.num_classes,
                           wrapper.num_prefix_tokens, wrapper.patch_size)

    # ---- data: identical loader path to Phase 1 (cached ImageNet-val tensor file, original order, no shuffle)
    data_info: Dict[str, Any] = {"dataset": cfg.dataset_name, "batch_size": cfg.batch_size}
    if data is None:
        loader, n_found = get_dataloader(DatasetConfig(name=cfg.dataset_name, data_dir=cfg.data_dir,
                                                       num_samples=cfg.num_samples, batch_size=cfg.batch_size,
                                                       seed=cfg.dataset_seed))
        xs, ys = zip(*[(x, y) for x, y in loader])
        images, labels = torch.cat(xs), torch.cat(ys)
        cache = os.path.join(cfg.data_dir, "imagenet_val_250.pt")
        if cfg.dataset_name.startswith("imagenet") and os.path.isfile(cache):
            data_info.update({"source_file": cache, "source_sha256": _sha256_file(cache),
                              "selection": f"first {len(images)} images of the cached file in stored order "
                                           f"(no shuffling; identical loader call to Phase 1)"})
    else:
        images, labels = data
        data_info["selection"] = "tensors supplied directly (tests)"
    n_img = len(images)
    data_info["num_images"] = n_img
    manifest = pd.DataFrame({
        "image_idx": range(n_img), "label": labels.tolist(),
        "image_sha1": [hashlib.sha1(images[i].numpy().tobytes()).hexdigest() for i in range(n_img)],
    })
    manifest.to_csv(os.path.join(cfg.output_dir, "image_manifest.csv"), index=False)
    batches = [(images[i:i + cfg.batch_size], labels[i:i + cfg.batch_size]) for i in range(0, n_img, cfg.batch_size)]
    dev = wrapper.device

    # ---- schedules
    entries = build_schedule_set(cfg, est)
    pairs = find_matched_pairs(entries, cfg.match_tolerance)

    # ---- unpruned baseline logits (cached; also the `orig_logits` the Phase 1 saliency scorer targets)
    with torch.no_grad():
        base_logits = [wrapper(x.to(dev)) for x, _ in batches]
    base_pred = torch.cat([l.argmax(-1) for l in base_logits]).cpu().numpy()
    y_all = labels.cpu().numpy()

    rows: List[Dict[str, Any]] = []
    per_image: List[pd.DataFrame] = []
    token_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def seeds_for(scorer: str) -> List[int]:
        return [cfg.seed] + (list(cfg.extra_seeds_random) if scorer == "random" else [])

    runs = [(e, "none", cfg.seed) for e in entries if e["type"] == "baseline"]
    for e in entries:
        if e["type"] != "baseline":
            for sc in cfg.scorers:
                runs += [(e, sc, s) for s in seeds_for(sc)]
    t_start = time.time()
    for ri, (e, scorer, seed) in enumerate(runs):
        sched = e["schedule"]
        try:
            pruner = ProgressivePruner(wrapper, sched, scorer="random" if scorer == "none" else scorer, seed=seed)
            preds, top5 = [], []
            trace = None
            for (x, _), bl in zip(batches, base_logits):
                res = pruner.forward(x.to(dev), orig_logits=bl)
                if trace is not None:
                    assert [t["patch_tokens_out"] for t in trace] == [t["patch_tokens_out"] for t in res.token_trace]
                trace = res.token_trace
                preds.append(res.logits.argmax(-1).cpu())
                top5.append(res.logits.topk(5, dim=-1).indices.cpu())
                if scorer == "none":  # no-pruning path must reproduce the original model
                    assert (res.logits - bl).abs().max().item() <= 1e-4
            preds = torch.cat(preds).numpy()
            top5 = torch.cat(top5).numpy()
        except Exception as ex:  # report, never silently skip
            failures.append({"schedule": e["name"], "scorer": scorer, "seed": seed, "error": f"{type(ex).__name__}: {ex}"})
            print(f"[FAILED] {e['name']} / {scorer} / seed {seed}: {ex}", flush=True)
            continue

        correct = preds == y_all
        agree = preds == base_pred
        t5 = (top5 == y_all[:, None]).any(1)
        rec = {"schedule": e["name"], "schedule_type": e["type"], "scorer": scorer, "seed": seed}
        per_image.append(pd.DataFrame({
            **rec, "image_idx": np.arange(n_img), "label": y_all, "baseline_pred": base_pred,
            "compressed_pred": preds, "baseline_correct": (base_pred == y_all).astype(int),
            "correct": correct.astype(int), "top5_correct": t5.astype(int), "agreement": agree.astype(int)}))
        for t in trace:
            token_rows.append({"schedule": e["name"], "scorer": scorer, "seed": seed, **t})

        row = {**rec, "schedule_json": json.dumps({str(k): v for k, v in sched.items()}),
               "n_images": n_img, "analytic_flops": e["analytic_flops"], "normalized_flops": e["normalized_flops"],
               "flops_savings_pct": 100.0 * (1.0 - e["normalized_flops"])}
        for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
            w = wilson_score_interval(int(arr.sum()), n_img)
            row.update({f"{name}_count": int(arr.sum()), name: w["mean"],
                        f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
        rows.append(row)
        if (ri + 1) % 10 == 0 or ri + 1 == len(runs):
            print(f"[{ri + 1}/{len(runs)}] {e['name']} / {scorer} / seed {seed}  ({time.time() - t_start:.0f}s)", flush=True)

    metrics = pd.DataFrame(rows)

    # ---- latency (primary seed only; CPU/GPU recorded separately via `latency_device`)
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
        for e in entries:
            for sc in (["none"] if e["type"] == "baseline" else cfg.scorers):
                p = ProgressivePruner(wrapper, e["schedule"], "random" if sc == "none" else sc, seed=cfg.seed)
                lat[(e["name"], sc)] = _time_pruner(p, xb, cfg.latency_warmup, cfg.latency_runs)
        for col, key in (("latency_total_ms_median", "total_ms_median"), ("latency_total_ms_mean", "total_ms_mean"),
                         ("scorer_overhead_ms_median", "scorer_ms_median")):
            metrics[col] = [lat.get((r.schedule, r.scorer), {}).get(key, np.nan) if r.seed == cfg.seed else np.nan
                            for r in metrics.itertuples()]
        metrics["latency_device"] = str(dev)
        metrics["latency_batch_size"] = len(xb)
        metrics["latency_warmup"] = cfg.latency_warmup
        metrics["latency_timed_runs"] = cfg.latency_runs
    metrics.to_csv(os.path.join(cfg.output_dir, "metrics.csv"), index=False)
    per_img_df = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    per_img_df.to_csv(os.path.join(cfg.output_dir, "per_image_metrics.csv"), index=False)
    pd.DataFrame(token_rows).to_csv(os.path.join(cfg.output_dir, "token_counts.csv"), index=False)

    # ---- manifest / config
    manifest_json = {
        "layer_numbering_and_semantics": LAYER_NUMBERING,
        "baseline_flops": est.baseline_flops,
        "matching_rule": ("For each progressive schedule and each single-layer pruning layer, the single-layer schedule "
                          "with nearest normalized FLOPs, if within tolerance. Chosen from analytic FLOPs only."),
        "match_tolerance_pp_of_baseline_flops": 100.0 * cfg.match_tolerance,
        "schedules": [{"name": e["name"], "schedule": {str(k): v for k, v in e["schedule"].items()},
                       "type": e["type"], "origin": e["origin"], "analytic_flops": e["analytic_flops"],
                       "normalized_flops": e["normalized_flops"],
                       "patch_tokens_processed_per_block": expected_processed_counts(e["schedule"], L, N)}
                      for e in entries],
        "matched_pairs": pairs,
    }
    with open(os.path.join(cfg.output_dir, "schedule_manifest.json"), "w") as f:
        json.dump(manifest_json, f, indent=2)
    run_config = {
        "config": cfg.to_dict(), "dataset": data_info, "environment": _environment(wrapper),
        "seeds": {"primary_all_scorers": cfg.seed, "extra_random_only": cfg.extra_seeds_random},
        "seed_notes": ("Random scorer: a fresh generator is seeded per (schedule, scorer, seed) run and advances across "
                       "batches and pruning calls in fixed image order, so draws differ between schedules. "
                       "cls_attention and prediction_kl_approx are deterministic, so extra seeds are not run for them."),
        "timing": {"method": "time.perf_counter around ProgressivePruner.forward (scoring + pruning + blocks + head) on the "
                             "first batch; median of timed runs after warm-up. For prediction_kl_approx the total also "
                             "includes one unpruned forward pass to get the target class (not in scorer overhead).",
                   "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs,
                   "unpruned_model_forward": unpruned_latency, "enabled": cfg.measure_latency},
        "failed_or_unsupported": failures,
        "num_runs": len(runs),
    }
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    out = {"metrics": metrics, "per_image": per_img_df, "token_counts": pd.DataFrame(token_rows),
           "manifest": manifest_json, "entries": entries, "pairs": pairs, "failures": failures,
           "run_config": run_config, "estimator": est}
    out["matched"] = matched_comparison(per_img_df, pairs, cfg.seed)
    out["matched"].to_csv(os.path.join(cfg.output_dir, "matched_pairs_comparison.csv"), index=False)
    if cfg.make_plots and len(metrics):
        from ..visualization.progressive_plots import make_all_plots
        make_all_plots(metrics, out["matched"], cfg.output_dir, cfg.seed)
    write_summary(out, cfg)
    return out


# ----------------------------------------------------------------------------- matched-compute analysis

def matched_comparison(per_img: pd.DataFrame, pairs: List[Dict[str, Any]], primary_seed: int) -> pd.DataFrame:
    """Progressive-minus-single differences (agreement and top-1) for every matched pair x scorer x seed."""
    if per_img.empty:
        return pd.DataFrame()
    idx = {k: g.sort_values("image_idx") for k, g in per_img.groupby(["schedule", "scorer", "seed"])}
    out = []
    for p in pairs:
        for (sched, scorer, seed), gp in idx.items():
            if sched != p["progressive"]:
                continue
            gs = idx.get((p["single"], scorer, seed))
            if gs is None:
                continue
            row = {"progressive": p["progressive"], "single": p["single"], "single_layer": p["single_layer"],
                   "scorer": scorer, "seed": seed, "progressive_norm_flops": p["progressive_norm_flops"],
                   "single_norm_flops": p["single_norm_flops"], "flops_diff_pp": p["flops_diff_pp"]}
            for metric, col in (("agreement", "agreement"), ("top1", "correct")):
                s = paired_diff_stats(gp[col].to_numpy(), gs[col].to_numpy())
                row.update({f"{metric}_prog_count": s["a_count"], f"{metric}_single_count": s["b_count"],
                            f"{metric}_diff": s["diff"], f"{metric}_prog_only": s["a_only"],
                            f"{metric}_single_only": s["b_only"], f"{metric}_mcnemar_p": s["mcnemar_p"],
                            f"{metric}_boot_ci_low": s["boot_ci_low"], f"{metric}_boot_ci_high": s["boot_ci_high"]})
            row["n_images"] = len(gp)
            out.append(row)
    return pd.DataFrame(out)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_summary(out: Dict[str, Any], cfg: EvalConfig) -> None:
    m, mt, rc = out["metrics"], out["matched"], out["run_config"]
    seed = cfg.seed
    n = int(m["n_images"].iloc[0]) if len(m) else 0
    L = []
    L.append("# Phase 2: Controlled Progressive-Pruning Evaluation\n")
    L.append("*Exploratory results on a small image set; p-values are nominal, uncorrected for multiple comparisons.*\n")
    L.append("## Setup\n")
    d = rc["dataset"]
    L.append(f"- Model: `{cfg.model_name}` (pretrained={cfg.pretrained}); eval mode; device `{rc['environment']['device']}`.")
    L.append(f"- Images: **{n}** ImageNet-val; {d.get('selection', '')}. Manifest: `image_manifest.csv` (index, label, sha1).")
    if "source_sha256" in d:
        L.append(f"- Source file `{d['source_file']}` sha256 `{d['source_sha256'][:16]}...`.")
    L.append(f"- Primary seed {seed} (all scorers); extra seeds {cfg.extra_seeds_random} for `random` only "
             f"(other scorers are deterministic).")
    L.append(f"- Layer numbering: {LAYER_NUMBERING}")
    L.append(f"- Matching tolerance (set before comparing): |normalized FLOPs difference| <= "
             f"{100 * cfg.match_tolerance:.1f} percentage points of the unpruned baseline FLOPs.")
    L.append("- `prediction_kl_approx` is the Phase 1 gradient x activation saliency, **not exact KL**.")
    L.append(f"- {rc['seed_notes']}\n")

    named = [e for e in out["entries"] if e["origin"] == "specified"]
    for sc in cfg.scorers:
        L.append(f"## Metrics: `{sc}` (seed {seed}; counts out of {n}, Wilson 95% CI)\n")
        rows = []
        for e in [x for x in named]:
            r = m[(m.schedule == e["name"]) & (m.scorer == (sc if e["type"] != "baseline" else "none")) & (m.seed == seed)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([e["name"], json.dumps({str(k): v for k, v in e["schedule"].items()}), e["type"],
                         _pct(r.normalized_flops),
                         f"{r.agreement_count}/{n} {_pct(r.agreement)} [{_pct(r.agreement_ci_low)}, {_pct(r.agreement_ci_high)}]",
                         f"{r.top1_count}/{n} {_pct(r.top1)} [{_pct(r.top1_ci_low)}, {_pct(r.top1_ci_high)}]",
                         f"{r.top5_count}/{n}"])
        L.append(_md_table(rows, ["schedule", "keys: layer→k", "type", "norm. FLOPs", "agreement", "top-1 acc", "top-5"]))
        L.append("")

    L.append("## Matched-compute comparison (progressive minus single-layer)\n")
    L.append("Each pair is one progressive schedule vs the single-layer schedule (one per pruning layer) with the nearest "
             "analytic FLOPs within tolerance. Same scorer, same images, same seed within a pair. Differences are "
             "percentage points (pp); CI = paired bootstrap over images; p = exact McNemar (two-sided).\n")
    if mt.empty:
        L.append("No matched pairs were found within tolerance.\n")
    else:
        for sc in cfg.scorers:
            s = mt[(mt.scorer == sc) & (mt.seed == seed)]
            if s.empty:
                continue
            L.append(f"### `{sc}` (seed {seed})\n")
            rows = [[r.progressive, f"{r.single} (L{r.single_layer})", f"{100 * r.progressive_norm_flops:.1f}",
                     f"{100 * r.single_norm_flops:.1f}", f"{r.flops_diff_pp:+.2f}",
                     f"{r.agreement_prog_count} vs {r.agreement_single_count}",
                     f"{100 * r.agreement_diff:+.1f} [{100 * r.agreement_boot_ci_low:+.1f}, {100 * r.agreement_boot_ci_high:+.1f}]",
                     f"{r.agreement_mcnemar_p:.3f}",
                     f"{r.top1_prog_count} vs {r.top1_single_count}",
                     f"{100 * r.top1_diff:+.1f} [{100 * r.top1_boot_ci_low:+.1f}, {100 * r.top1_boot_ci_high:+.1f}]",
                     f"{r.top1_mcnemar_p:.3f}"] for r in s.itertuples()]
            L.append(_md_table(rows, ["progressive", "single", "FLOPs prog %", "FLOPs single %", "ΔFLOPs pp",
                                      "agree (k) prog vs single", "Δagree pp [95% CI]", "McNemar p",
                                      "top-1 (k) prog vs single", "Δtop-1 pp [95% CI]", "McNemar p"]))
            better = int((s.agreement_diff > 0).sum()); worse = int((s.agreement_diff < 0).sum())
            sig_b = int(((s.agreement_diff > 0) & (s.agreement_mcnemar_p < 0.05)).sum())
            sig_w = int(((s.agreement_diff < 0) & (s.agreement_mcnemar_p < 0.05)).sum())
            L.append(f"\nAgreement: progressive higher in {better}, lower in {worse}, tied in {len(s) - better - worse} of "
                     f"{len(s)} pairs; nominal p<0.05 favouring progressive in {sig_b}, favouring single in {sig_w}.\n")

        rs = mt[mt.scorer == "random"]
        if not rs.empty and rs.seed.nunique() > 1:
            L.append("### Random scorer: seed-to-seed variation of the agreement difference (separate from image-level CIs)\n")
            g = rs.groupby(["progressive", "single"]).agreement_diff.agg(["mean", "std", "count"]).reset_index()
            L.append(_md_table([[r.progressive, r.single, f"{100 * r['mean']:+.1f}", f"{100 * r['std']:.1f}", int(r['count'])]
                                for _, r in g.iterrows()],
                               ["progressive", "single", "mean Δagree pp over seeds", "sd across seeds pp", "seeds"]))
            L.append("")

    L.append("## FLOPs and latency (reported separately; latency is not inferred from FLOPs)\n")
    if cfg.measure_latency and "latency_total_ms_median" in m:
        ul = rc["timing"]["unpruned_model_forward"]
        L.append(f"Device `{rc['environment']['device']}` (CPU/GPU numbers are never mixed), batch {int(m.latency_batch_size.dropna().iloc[0])}, "
                 f"warm-up {cfg.latency_warmup}, timed runs {cfg.latency_runs}, median ms; unpruned model forward "
                 f"{ul['median_ms']:.1f} ms. {rc['timing']['method']}\n")
        rows = []
        for e in named:
            for sc in (["none"] if e["type"] == "baseline" else cfg.scorers):
                r = m[(m.schedule == e["name"]) & (m.scorer == sc) & (m.seed == seed)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([e["name"], sc, _pct(r.normalized_flops), f"{r.latency_total_ms_median:.1f}",
                             f"{r.scorer_overhead_ms_median:.1f}"])
        L.append(_md_table(rows, ["schedule", "scorer", "norm. FLOPs", "total latency ms (incl. selection)", "scorer overhead ms"]))
        L.append("")
    else:
        L.append("Latency not measured in this run.\n")

    L.append("## Failed / unsupported combinations\n")
    L.append("None." if not out["failures"] else "\n".join(f"- {f['schedule']} / {f['scorer']} / seed {f['seed']}: {f['error']}"
                                                          for f in out["failures"]))
    L.append("\n- `no_pruning` is scorer-independent and reported once as scorer `none`.\n")
    L.append("## Limitations\n")
    L.append(f"- Only {n} images and one model; CIs are wide and many pairs are not distinguishable.\n"
             "- Matched pairs differ in pruning *layers* as well as in number of steps, so differences reflect schedule shape "
             "in general, not only 'progressive vs single'.\n"
             "- Many pairs are compared; p-values are not corrected. Treat as exploratory.\n"
             "- Random-scorer draws differ between schedules (generator advances), adding noise to random comparisons.\n"
             "- FLOPs are analytic; measured latency on this device may not follow them.\n")
    L.append("## Commands\n")
    L.append("```\npython scripts/run_progressive_eval.py --config configs/progressive_eval.yaml --smoke   # small smoke test\n"
             "python scripts/run_progressive_eval.py --config configs/progressive_eval.yaml           # full run\n```\n")
    with open(os.path.join(cfg.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def regenerate_reports(output_dir: str) -> Dict[str, Any]:
    """Rebuild matched comparison, plots and summary.md from the CSV/JSON files saved by `run_evaluation`
    (no models are re-run; use if only the reporting step needs to be redone)."""
    rd = lambda n: os.path.join(output_dir, n)
    run_config = json.load(open(rd("run_config.json")))
    man = json.load(open(rd("schedule_manifest.json")))
    c = run_config["config"]
    c["schedules"] = None
    cfg = EvalConfig(**{k: v for k, v in c.items() if k in EvalConfig.__dataclass_fields__})
    metrics = pd.read_csv(rd("metrics.csv"))
    per_img = pd.read_csv(rd("per_image_metrics.csv"))
    pairs = man["matched_pairs"]
    entries = [{"name": s["name"], "schedule": {int(k): v for k, v in s["schedule"].items()}, "type": s["type"],
                "origin": s["origin"], "analytic_flops": s["analytic_flops"], "normalized_flops": s["normalized_flops"]}
               for s in man["schedules"]]
    out = {"metrics": metrics, "per_image": per_img, "entries": entries, "pairs": pairs,
           "failures": run_config["failed_or_unsupported"], "run_config": run_config,
           "matched": matched_comparison(per_img, pairs, cfg.seed)}
    out["matched"].to_csv(rd("matched_pairs_comparison.csv"), index=False)
    from ..visualization.progressive_plots import make_all_plots
    make_all_plots(metrics, out["matched"], output_dir, cfg.seed)
    write_summary(out, cfg)
    return out
