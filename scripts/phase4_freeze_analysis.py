#!/usr/bin/env python3
"""Freeze the Gate 5 analysis: records SHA-256 of the analysis script, its tests/cross-check, dependency versions and the frozen inputs it reads. Refuses to overwrite."""
import hashlib, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
out = os.path.join(P4, "phase4_analysis_freeze.json")
if os.path.exists(out):
    sys.exit("analysis freeze already exists; refusing to overwrite")
sha = lambda r: hashlib.sha256(open(os.path.join(ROOT, r), "rb").read()).hexdigest()
man = json.load(open(os.path.join(P4, "analysis_dev_validation", "outputs_run1", "analysis_manifest.json")))
dev_in = {f: hashlib.sha256(open(os.path.join(P4, "analysis_dev_validation", "inputs", f), "rb").read()).hexdigest() for f in sorted(os.listdir(os.path.join(P4, "analysis_dev_validation", "inputs")))}
rel = "results/phase4_adaptive_method_benchmark/"
deps = ["frozen_method_configs.json", "frozen_method_configs.sha256", "protocol_freeze.json", "benchmark_protocol.md", "splits/p4_confirm_manifest.csv", "splits/p4_dev_manifest.csv",
        "supplementary_phase3_reselection/frozen_schedules.json"]
rec = {"analysis_script": "scripts/phase4_gate5_analysis.py", "analysis_script_sha256": sha("scripts/phase4_gate5_analysis.py"),
       "dev_crosscheck_script_sha256": sha("scripts/phase4_gate5_dev_crosscheck.py"), "dev_inputs_generator_sha256": sha("scripts/phase4_gate5_dev_inputs.py"),
       "analysis_tests_sha256": sha("tests/phase4/test_phase4_analysis.py"),
       "versions": {**man["versions"], "tabulate": "0.9.0", "env": "conda env 'minor'"},
       "dependencies_sha256": {**{rel + d: sha(rel + d) for d in deps}, "results/phase3_budget_aware_allocation/frozen_schedules.json": sha("results/phase3_budget_aware_allocation/frozen_schedules.json")},
       "fixed_procedures": {"bootstrap": "10,000 resamples, seed 42, re-seeded per comparison, percentile 95% CI", "mcnemar": "exact two-sided binomial on discordant pairs", "holm_alpha": 0.05,
                            "wilson_z": 1.959963984540054, "diff_direction": "method - comparator", "families": "see script docstring (PRIMARY 16+16, REPLICATION 24+24, SECONDARY per method/comparator/metric, EXPLORATORY incl. native track)"},
       "dev_validation": {"dev_input_sha256": dev_in, "run1_output_sha256": man["output_sha256"], "reproducible_run1_equals_run2": True, "crosscheck_results": "analysis_dev_validation/crosscheck_results.json (all pass)"},
       "note": "Frozen after P4-DEV validation and before any confirmatory output exists. The confirm-mode analysis refuses to run if this script or any listed dependency changes."}
b = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode()
open(out, "wb").write(b)
open(os.path.join(P4, "phase4_analysis_freeze.sha256"), "wb").write((hashlib.sha256(b).hexdigest() + "\n").encode())
print(hashlib.sha256(b).hexdigest(), rec["analysis_script_sha256"])
