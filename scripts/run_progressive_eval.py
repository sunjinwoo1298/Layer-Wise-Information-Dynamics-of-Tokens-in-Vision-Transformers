#!/usr/bin/env python3
"""
CLI: Phase 2 controlled progressive-vs-single pruning evaluation.
  Smoke:  python scripts/run_progressive_eval.py --config configs/progressive_eval.yaml --smoke
  Full:   python scripts/run_progressive_eval.py --config configs/progressive_eval.yaml
"""

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.progressive_eval import EvalConfig, run_evaluation


def load_config(path: str) -> EvalConfig:
    with open(path) as f:
        c = yaml.safe_load(f)
    m, d, e = c.get("model", {}), c.get("dataset", {}), c.get("evaluation", {})
    return EvalConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        dataset_name=d.get("name", "imagenet_val"), data_dir=d.get("data_dir", "./data"),
        num_samples=d.get("num_samples", 250), batch_size=d.get("batch_size", 25), dataset_seed=d.get("seed", 42),
        seed=e.get("seed", 42), extra_seeds_random=e.get("extra_seeds_random", [43, 44]),
        scorers=e.get("scorers", ["random", "cls_attention", "prediction_kl_approx"]),
        schedules={n: {int(k): int(v) for k, v in (s or {}).items()} for n, s in c["schedules"].items()}
        if c.get("schedules") else None,
        auto_match=e.get("auto_match", True), match_tolerance=e.get("match_tolerance", 0.02),
        single_layers=e.get("single_layers", [2, 4, 6, 8, 10]), measure_latency=e.get("measure_latency", True),
        latency_warmup=e.get("latency_warmup", 1), latency_runs=e.get("latency_timed_runs", 3),
        output_dir=e.get("output_dir", "./results/phase2_progressive_eval"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/progressive_eval.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="20 images, primary seed only, 3 schedules + matched controls; writes to results/phase2_progressive_eval_smoke")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg.num_samples = 20
        cfg.extra_seeds_random = []
        cfg.latency_runs = 1
        cfg.schedules = {k: v for k, v in cfg.schedules.items()
                         if k in ("no_pruning", "early_single", "middle_single", "early_progressive")}
        cfg.output_dir = "./results/phase2_progressive_eval_smoke"
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device

    out = run_evaluation(cfg)
    print(f"[Saved] {cfg.output_dir}  ({len(out['metrics'])} metric rows, {len(out['failures'])} failures)")


if __name__ == "__main__":
    main()
