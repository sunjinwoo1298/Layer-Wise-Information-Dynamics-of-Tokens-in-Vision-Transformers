#!/usr/bin/env python3
"""
CLI: Phase 2 controlled pruning-stage ablation.
  Smoke:  python scripts/run_stage_ablation.py --config configs/stage_ablation.yaml --smoke
  Full:   python scripts/run_stage_ablation.py --config configs/stage_ablation.yaml
"""

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.stage_ablation import StageConfig, run_stage_ablation


def load_config(path: str) -> StageConfig:
    c = yaml.safe_load(open(path))
    m, d, a = c.get("model", {}), c.get("dataset", {}), c.get("ablation", {})
    structures = None
    if c.get("structures"):
        structures = {int(f): {k: [tuple(x) for x in v] for k, v in sv.items()} for f, sv in c["structures"].items()}
    return StageConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        dataset_name=d.get("name", "imagenet_val"), data_dir=d.get("data_dir", "./data"),
        num_samples=d.get("num_samples", 250), batch_size=d.get("batch_size", 25), dataset_seed=d.get("seed", 42),
        seed=a.get("seed", 42), extra_seeds_random=a.get("extra_seeds_random", [43, 44]),
        budgets=a.get("budgets", [0.4, 0.5, 0.6, 0.7]), match_tolerance=a.get("match_tolerance", 0.01),
        primary_first_layer=a.get("primary_first_layer", 2), secondary_first_layers=a.get("secondary_first_layers", [4]),
        structures=structures, primary_scorer=a.get("primary_scorer", "cls_attention"),
        scorers_primary_layer=a.get("scorers_primary_layer", ["cls_attention", "random", "prediction_kl_approx"]),
        scorers_secondary_layer=a.get("scorers_secondary_layer", ["cls_attention", "random"]),
        measure_latency=a.get("measure_latency", True), latency_warmup=a.get("latency_warmup", 1),
        latency_runs=a.get("latency_timed_runs", 3), previous_eval_dir=a.get("previous_eval_dir"),
        output_dir=a.get("output_dir", "./results/phase2_stage_ablation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage_ablation.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="25 images, 2 budgets, seed 42 only, cls_attention+random; writes to results/phase2_stage_ablation_smoke")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.num_samples = 25
        cfg.budgets = [0.50, 0.60]
        cfg.extra_seeds_random = []
        cfg.scorers_primary_layer = ["cls_attention", "random"]
        cfg.scorers_secondary_layer = ["cls_attention"]
        cfg.latency_runs = 1
        cfg.previous_eval_dir = None   # smoke uses a 25-image prefix, so the 250-image manifest cannot match
        cfg.output_dir = "./results/phase2_stage_ablation_smoke"
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    out = run_stage_ablation(cfg)
    print(f"[Saved] {cfg.output_dir} ({len(out['metrics'])} metric rows, {len(out['failures'])} failures, "
          f"{len(out['unmatched'])} unmatched)")


if __name__ == "__main__":
    main()
