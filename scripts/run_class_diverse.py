#!/usr/bin/env python3
"""
CLI: Phase 2 class-diverse validation and allocation-robustness study.
  Prepare subset only: python scripts/run_class_diverse.py --config configs/class_diverse.yaml --prepare-only
  Smoke:               python scripts/run_class_diverse.py --config configs/class_diverse.yaml --smoke
  Full:                python scripts/run_class_diverse.py --config configs/class_diverse.yaml
"""

import argparse
import json
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.class_diverse import ClassDiverseConfig, load_subset, run_class_diverse


def load_config(path: str) -> ClassDiverseConfig:
    c = yaml.safe_load(open(path))
    m, d, v = c.get("model", {}), c.get("dataset", {}), c.get("validation", {})
    structures = None
    if c.get("structures"):
        structures = {int(f): {k: [tuple(x) for x in vv] for k, vv in sv.items()} for f, sv in c["structures"].items()}
    return ClassDiverseConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        parquet_dir=d.get("parquet_dir", "./data/imagenet_val_parquet"), subset_cache=d.get("subset_cache", "./data/imagenet_class_diverse_subset.pt"),
        images_per_class=d.get("images_per_class", 2), subset_seed=d.get("seed", 42), batch_size=d.get("batch_size", 25),
        seed=v.get("seed", 42), extra_seeds_random=v.get("extra_seeds_random", [43, 44]), budgets=v.get("budgets", [0.4, 0.5, 0.6, 0.7]),
        match_tolerance=v.get("match_tolerance", 0.01), families=v.get("families", ["geometric", "equal_drop"]),
        primary_first_layer=v.get("primary_first_layer", 2), secondary_first_layers=v.get("secondary_first_layers", [4]),
        structures=structures, kl_budgets=v.get("kl_budgets", [0.5, 0.6]), reduced_structures=v.get("reduced_structures", ["2-6", "2-6-10"]),
        primary_scorer=v.get("primary_scorer", "cls_attention"),
        scorers_primary_layer=v.get("scorers_primary_layer", ["cls_attention", "random", "prediction_kl_approx"]),
        scorers_secondary_layer=v.get("scorers_secondary_layer", ["cls_attention"]),
        measure_latency=v.get("measure_latency", True), latency_warmup=v.get("latency_warmup", 1),
        latency_runs=v.get("latency_timed_runs", 5), latency_batches=v.get("latency_batches", 3),
        kl_latency_runs=v.get("kl_latency_runs", 3), kl_latency_batches=v.get("kl_latency_batches", 1),
        previous_stage_ablation_dir=v.get("previous_stage_ablation_dir"), output_dir=v.get("output_dir", "./results/phase2_class_diverse_validation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/class_diverse.yaml")
    ap.add_argument("--smoke", action="store_true", help="50 images (prefix of the subset), 2 budgets, seed 42, cls_attention+random; "
                                                           "writes to results/phase2_class_diverse_validation_smoke")
    ap.add_argument("--prepare-only", action="store_true", help="build/verify the subset manifest + class coverage, then stop")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.no_resume:
        cfg.resume = False
    if args.prepare_only:
        os.makedirs(cfg.output_dir, exist_ok=True)
        sub = load_subset(cfg)
        sub["manifest"].to_csv(os.path.join(cfg.output_dir, "subset_manifest.csv"), index=False)
        json.dump(sub["summary"], open(os.path.join(cfg.output_dir, "subset_summary.json"), "w"), indent=2)
        s = sub["summary"]
        print(json.dumps({k: s[k] for k in ("num_images", "num_distinct_classes", "classes_missing", "per_class_count_over_all_classes",
                                           "per_class_count_over_present_classes", "count_histogram", "manifest_sha256")}, indent=2))
        return
    if args.smoke:
        cfg.max_images = 50
        cfg.budgets = [0.50, 0.60]
        cfg.extra_seeds_random = []
        cfg.scorers_primary_layer = ["cls_attention", "random"]
        cfg.latency_runs, cfg.latency_batches = 1, 1
        cfg.output_dir = "./results/phase2_class_diverse_validation_smoke"
        cfg.previous_stage_ablation_dir = cfg.previous_stage_ablation_dir
    out = run_class_diverse(cfg)
    print(f"[Saved] {cfg.output_dir} ({len(out['metrics'])} metric rows, {len(out['failures'])} failures, {len(out['unmatched'])} unmatched)")


if __name__ == "__main__":
    main()
