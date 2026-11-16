#!/usr/bin/env python3
"""
CLI: Phase 2 minimum final-token constraint study.
  Smoke: python scripts/run_min_token.py --config configs/min_token.yaml --smoke
  Full:  python scripts/run_min_token.py --config configs/min_token.yaml
"""

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.minimum_token import MinTokenConfig, run_min_token


def load_config(path: str) -> MinTokenConfig:
    c = yaml.safe_load(open(path))
    m, d, s = c.get("model", {}), c.get("dataset", {}), c.get("study", {})
    return MinTokenConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        parquet_dir=d.get("parquet_dir", "./data/imagenet_val_parquet"),
        subset_cache=d.get("subset_cache", "./data/imagenet_class_diverse_subset.pt"),
        images_per_class=d.get("images_per_class", 2), subset_seed=d.get("seed", 42), batch_size=d.get("batch_size", 25),
        seed=s.get("seed", 42), extra_seeds_random=s.get("extra_seeds_random", [43, 44]),
        budgets=s.get("budgets", [0.4, 0.5, 0.6, 0.7]), match_tolerance=s.get("match_tolerance", 0.01),
        first_layer=s.get("first_layer", 2), floors=s.get("floors", [8, 16, 24, 32]),
        primary_floors=s.get("primary_floors", [16, 24, 32]), primary_scorer=s.get("primary_scorer", "cls_attention"),
        secondary_scorers=s.get("secondary_scorers", ["random", "prediction_kl_approx"]),
        secondary_structures=s.get("secondary_structures", ["2-6", "2-6-10"]),
        secondary_floors=s.get("secondary_floors", [16, 32]), random_budgets=s.get("random_budgets", [0.4, 0.5]),
        kl_budgets=s.get("kl_budgets", [0.5]), measure_latency=s.get("measure_latency", True),
        latency_warmup=s.get("latency_warmup", 1), latency_runs=s.get("latency_timed_runs", 5),
        latency_batches=s.get("latency_batches", 3), kl_latency_runs=s.get("kl_latency_runs", 3),
        kl_latency_batches=s.get("kl_latency_batches", 1), previous_dir=s.get("previous_dir"),
        output_dir=s.get("output_dir", "./results/phase2_minimum_token_ablation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/min_token.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="50 images (prefix), 2 budgets, seed 42, cls_attention + random; writes to results/phase2_minimum_token_ablation_smoke")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.smoke:
        cfg.max_images, cfg.budgets, cfg.extra_seeds_random = 50, [0.40, 0.60], []
        cfg.secondary_scorers, cfg.random_budgets, cfg.kl_budgets = ["random"], [0.40], []
        cfg.latency_runs, cfg.latency_batches = 1, 1
        cfg.output_dir = "./results/phase2_minimum_token_ablation_smoke"
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.device:
        cfg.device = a.device
    if a.no_resume:
        cfg.resume = False
    out = run_min_token(cfg)
    print(f"[Saved] {cfg.output_dir} ({len(out['metrics'])} metric rows, {len(out['failures'])} failures, "
          f"{len(out['infeasible'])} infeasible, {len(out['identical'])} identical-to-unconstrained)")


if __name__ == "__main__":
    main()
