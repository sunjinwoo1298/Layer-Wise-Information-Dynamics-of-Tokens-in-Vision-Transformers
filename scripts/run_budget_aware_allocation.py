#!/usr/bin/env python3
"""
CLI: Phase 3 budget-aware allocation (LS-Chain + verify). Stages must run in order; later stages refuse to run without the freeze.
  python scripts/run_budget_aware_allocation.py --stage prepare            # splits + fresh confirmatory set + checksums
  python scripts/run_budget_aware_allocation.py --stage calibrate          # tables, selection, calibration baselines, FREEZE
  python scripts/run_budget_aware_allocation.py --stage validate           # frozen schedules + baselines on the 1,500 validation images
  python scripts/run_budget_aware_allocation.py --stage confirm            # fresh confirmatory set, run ONCE
  python scripts/run_budget_aware_allocation.py --smoke --stage all        # tiny end-to-end check (writes to ..._smoke)
"""

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.budget_aware_allocation import AllocConfig, prepare_data, run_calibration, run_evaluation


def load_config(path: str) -> AllocConfig:
    c = yaml.safe_load(open(path))
    m, d, s, p = c.get("model", {}), c.get("data", {}), c.get("study", {}), c.get("paths", {})
    return AllocConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        parquet_dir=d.get("parquet_dir", "./data/imagenet_val_parquet"), subset_cache=d.get("subset_cache", "./data/imagenet_class_diverse_subset.pt"),
        confirm_cache=d.get("confirm_cache", "./data/imagenet_confirmatory_2000.pt"), n_cal=d.get("n_cal", 500), split_seed=d.get("split_seed", 42),
        confirm_seed=d.get("confirm_seed", 4242), batch_size=d.get("batch_size", 25), seed=s.get("seed", 42), scorer=s.get("scorer", "cls_attention"),
        first_layer=s.get("first_layer", 2), patterns=s.get("patterns", ["2-6-10", "2-8-10"]), budgets=s.get("budgets", [0.4, 0.5, 0.6, 0.7]),
        floors=s.get("floors", [16, 24, 32]), match_tolerance=s.get("match_tolerance", 0.01), grid=s.get("grid"), top_m=s.get("top_m", 8),
        min_dist=s.get("min_dist", 6), tie_eps=s.get("tie_eps", 0.002), bootstrap_stability=s.get("bootstrap_stability", 10),
        proxy_validity_min_spearman=s.get("proxy_validity_min_spearman", 0.6), latency_warmup=s.get("latency_warmup", 1),
        latency_runs=s.get("latency_runs", 5), latency_batches=s.get("latency_batches", 3),
        parent_dir=p.get("parent_dir", "./results/phase2_minimum_token_ablation"),
        class_diverse_dir=p.get("class_diverse_dir", "./results/phase2_class_diverse_validation"),
        isolation_dir=p.get("isolation_dir", "./results/phase2_final_token_trajectory_isolation"),
        output_dir=p.get("output_dir", "./results/phase3_budget_aware_allocation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/budget_aware_allocation.yaml")
    ap.add_argument("--stage", choices=["prepare", "calibrate", "validate", "confirm", "all"], required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (40/60/60 images, 1 budget, 1 pattern, 1 floor, coarse grid)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.smoke:
        cfg.max_cal, cfg.max_val, cfg.max_conf = 40, 60, 60
        cfg.budgets, cfg.patterns, cfg.floors = [0.50], ["2-6-10"], [16]
        cfg.grid, cfg.top_m, cfg.bootstrap_stability = [16, 48, 96, 144, 184], 3, 2
        cfg.latency_runs, cfg.latency_batches = 1, 1
        cfg.output_dir = "./results/phase3_budget_aware_allocation_smoke"
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.device:
        cfg.device = a.device
    if a.no_resume:
        cfg.resume = False
    stages = ["prepare", "calibrate", "validate", "confirm"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "prepare":
            d = prepare_data(cfg, need_conf=True)
            print("[prepare] splits:", {k: len(v) for k, v in d.items()})
        elif st == "calibrate":
            out = run_calibration(cfg)
            print("[calibrate] frozen:", len(out["frozen"]["proposed_schedules"]), "proposed schedules;", "H1 pooled Spearman:", out["validity"].get("pooled_spearman"))
        elif st == "validate":
            out = run_evaluation(cfg, "validation")
            print(f"[validate] {len(out['metrics'])} metric rows")
        elif st == "confirm":
            out = run_evaluation(cfg, "confirmatory", latency=False)
            print(f"[confirm] {len(out['metrics'])} metric rows")


if __name__ == "__main__":
    main()
