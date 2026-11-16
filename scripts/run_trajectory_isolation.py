#!/usr/bin/env python3
"""
CLI: Phase 2 final-token vs trajectory isolation experiment.
  Check only:  python scripts/run_trajectory_isolation.py --config configs/trajectory_isolation.yaml --prepare-only
  Smoke:       python scripts/run_trajectory_isolation.py --config configs/trajectory_isolation.yaml --smoke
  Full:        python scripts/run_trajectory_isolation.py --config configs/trajectory_isolation.yaml
"""

import argparse
import json
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.trajectory_isolation import TrajConfig, prepare_isolation, run_isolation, CONTROL_FAMILY
from src.metrics.compute import ComputeEstimator


def load_config(path: str) -> TrajConfig:
    c = yaml.safe_load(open(path))
    m, d, s = c.get("model", {}), c.get("dataset", {}), c.get("study", {})
    return TrajConfig(
        model_name=m.get("name", "deit_tiny_patch16_224"), pretrained=m.get("pretrained", True),
        parquet_dir=d.get("parquet_dir", "./data/imagenet_val_parquet"), subset_cache=d.get("subset_cache", "./data/imagenet_class_diverse_subset.pt"),
        images_per_class=d.get("images_per_class", 2), subset_seed=d.get("seed", 42), batch_size=d.get("batch_size", 25),
        seed=s.get("seed", 42), scorer=s.get("scorer", "cls_attention"), budgets=s.get("budgets", [0.4, 0.5, 0.6, 0.7]),
        match_tolerance=s.get("match_tolerance", 0.01), first_layer=s.get("first_layer", 2), patterns=s.get("patterns", ["2-6-10", "2-8-10"]),
        floors=s.get("floors", [8, 16, 24, 32]), primary_floors=s.get("primary_floors", [16, 24, 32]),
        control_min_deviation=s.get("control_min_deviation", 8), measure_latency=s.get("measure_latency", True),
        latency_warmup=s.get("latency_warmup", 1), latency_runs=s.get("latency_timed_runs", 5), latency_batches=s.get("latency_batches", 3),
        prev_min_token_dir=s.get("prev_min_token_dir"), prev_class_diverse_dir=s.get("prev_class_diverse_dir"),
        output_dir=s.get("output_dir", "./results/phase2_final_token_trajectory_isolation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trajectory_isolation.yaml")
    ap.add_argument("--prepare-only", action="store_true", help="verify manifest + prior schedules, enumerate controls, print counts; no model run")
    ap.add_argument("--smoke", action="store_true", help="50-image prefix, budgets 0.40/0.60, writes to results/phase2_final_token_trajectory_isolation_smoke")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.smoke:
        cfg.max_images, cfg.budgets = 50, [0.40, 0.60]
        cfg.latency_runs, cfg.latency_batches = 1, 1
        cfg.output_dir = "./results/phase2_final_token_trajectory_isolation_smoke"
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.device:
        cfg.device = a.device
    if a.no_resume:
        cfg.resume = False
    if a.prepare_only:
        est = ComputeEstimator(12, 192, 3, 196, 1000, 1, (16, 16))
        p = prepare_isolation(cfg, est)
        for k, v in p["artifacts"].items():
            print(f"[manifest:{k}] {v['manifest_path']}\n    full SHA-256 (recomputed == stored): {v['manifest_sha256_full']}  images={v['num_images']}")
        print("[prior-rule reproduction]", p["reproduction"])
        sc = p["schedules"]
        fams = {}
        for e in sc["entries"]:
            fams[e["family"]] = fams.get(e["family"], 0) + 1
        print("[schedules]", fams, "| infeasible constrained:", len(sc["infeasible"]), "| unmatched:", len(sc["unmatched"]),
              "| constrained identical to unconstrained:", len(sc["identical"]))
        print("[patterns]", cfg.patterns, "[budgets]", cfg.budgets)
        for e in sc["entries"]:
            if e["family"] == "geometric":
                n = sum(1 for x in sc["entries"] if x["family"] == CONTROL_FAMILY and x["control_of"] == e["name"])
                c = [x for x in sc["candidates"] if x["geometric"] == e["name"]]
                nd = sum(1 for x in c if x["status"] == "feasible_distinct")
                print(f"  {e['name']}: geometric {list(e['schedule'].values())}  candidates={len(c)} feasible_distinct={nd} controls_selected={n}")
        print("[control_infeasible]", sc["control_infeasible"])
        return
    out = run_isolation(cfg)
    print(f"[Saved] {cfg.output_dir} ({len(out['metrics'])} metric rows, {len(out['failures'])} failures)")


if __name__ == "__main__":
    main()
