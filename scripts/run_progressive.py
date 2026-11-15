#!/usr/bin/env python3
"""
CLI: Phase 2 progressive multi-layer token pruning smoke test.
Usage: python scripts/run_progressive.py --config configs/progressive_smoke.yaml [--scorer cls_attention]
Results go to experiment.output_dir (separate from Phase 1 results).
"""

import argparse
import json
import os
import statistics
import sys
import time

import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.compression.progressive import ProgressivePruner
from src.data.dataset import DatasetConfig, get_dataloader
from src.metrics.compute import ComputeEstimator
from src.metrics.prediction import compute_sample_level_metrics
from src.models.vit_wrapper import ViTWrapper


def _time_ms(fn, runs, device):
    fn()  # warm-up
    ts = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/progressive_smoke.yaml")
    ap.add_argument("--scorer", default=None, help="Override pruning.scorer")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, d, p, e = cfg.get("model", {}), cfg.get("dataset", {}), cfg.get("pruning", {}), cfg.get("experiment", {})
    scorer = args.scorer or p.get("scorer", "random")
    out_dir = args.output_dir or e.get("output_dir", "./results/phase2_progressive_smoke")
    seed = p.get("seed", 42)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    wrapper = ViTWrapper(m.get("name", "deit_tiny_patch16_224"), m.get("pretrained", True), args.device)
    pruner = ProgressivePruner(wrapper, p.get("schedule") or {}, scorer=scorer, seed=seed)
    loader, n_samples = get_dataloader(DatasetConfig(
        name=d.get("name", "synthetic"), data_dir=d.get("data_dir", "./data"),
        num_samples=d.get("num_samples", 10), batch_size=d.get("batch_size", 5), seed=d.get("seed", 42)))

    all_orig, all_prog, all_base, metrics_acc, last = [], [], [], {}, None
    trace = None
    first_x = None
    for x, y in loader:
        x, y = x.to(wrapper.device), y.to(wrapper.device)
        first_x = x if first_x is None else first_x
        with torch.no_grad():
            orig = wrapper(x)
            nopr = pruner.forward_unpruned(x)
        res = pruner.forward(x, orig_logits=orig)
        trace = res.token_trace
        all_orig.append(orig.cpu()); all_base.append(nopr.cpu()); all_prog.append(res.logits.cpu())
        for k_, v in compute_sample_level_metrics(orig, res.logits, y).items():
            metrics_acc.setdefault(k_, []).extend(v)

    orig_l, base_l, prog_l = torch.cat(all_orig), torch.cat(all_base), torch.cat(all_prog)
    baseline_equiv = (orig_l - base_l).abs().max().item()

    est = ComputeEstimator(wrapper.num_layers, wrapper.hidden_dim, wrapper.num_heads,
                           wrapper.num_original_tokens, wrapper.num_classes,
                           wrapper.num_prefix_tokens, wrapper.patch_size)
    per_block = pruner.patch_tokens_per_block(trace, wrapper.num_original_tokens)
    flops = est.compute_progressive_flops(per_block)

    result = {
        "scorer": pruner.scorer_name,
        "schedule": {str(k_): v for k_, v in pruner.schedule.as_dict().items()},
        "num_samples": n_samples,
        "no_pruning_vs_original_max_abs_logit_diff": baseline_equiv,
        "metrics_vs_unpruned_model": {k_: sum(v) / len(v) for k_, v in metrics_acc.items()},
        "analytic_total_flops": flops["total_flops"],
        "analytic_normalized_flops": flops["normalized_flops"],
        "patch_tokens_processed_per_block": per_block,
        "device": str(wrapper.device),
    }
    if e.get("measure_latency", False):
        runs = e.get("latency_runs", 5)
        with torch.no_grad():
            full_ms = _time_ms(lambda: wrapper(first_x), runs, wrapper.device)
        prog_ms = _time_ms(lambda: pruner.forward(first_x), runs, wrapper.device)
        sc_ms = pruner.forward(first_x).scorer_time_ms
        result["latency_ms_batch"] = {
            "batch_size": int(first_x.shape[0]), "unpruned_model": full_ms,
            "progressive_total_incl_scorer": prog_ms, "of_which_scorer_overhead": sc_ms,
            "note": "Wall-clock on this device; reported separately from analytic FLOPs. Not a speedup claim.",
        }

    torch.save({"original_logits": orig_l, "progressive_logits": prog_l}, os.path.join(out_dir, "logits.pt"))
    pd.DataFrame(trace).to_csv(os.path.join(out_dir, "token_counts.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg, "scorer_used": scorer}, f, indent=2)

    print(json.dumps(result, indent=2))
    print(pd.DataFrame(trace).to_string(index=False))
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
