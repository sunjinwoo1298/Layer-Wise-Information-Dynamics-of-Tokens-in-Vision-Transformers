#!/usr/bin/env python3
"""Phase 4 Gate 4: assemble development_results.md and frozen_method_configs.json(+.sha256) from the saved P4-DEV outputs. Reads no P4-CONFIRM data."""
import hashlib
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
G4 = os.path.join(P4, "gate4")
SUP = os.path.join(P4, "supplementary_phase3_reselection")
N = 1990


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


sel = json.load(open(os.path.join(G4, "methods_selection.json")))
cands = pd.read_csv(os.path.join(G4, "methods_candidates.csv"))
dr = json.load(open(os.path.join(P4, "diffrate_dev_validation.json")))
p3 = pd.read_csv(os.path.join(G4, "phase3_comparators_dev.csv"))
lat = pd.read_csv(os.path.join(G4, "latency_dev.csv"))
sup = pd.read_csv(os.path.join(SUP, "heldout_dev_results.csv"))
supf = json.load(open(os.path.join(SUP, "frozen_schedules.json")))
fz3 = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))

# ------------------------------------------------------------------ frozen config
cfg = {"status": "Gate 4 development freeze (P4-DEV selection). P4-CONFIRM has NOT been read.", "n_dev": N,
       "method_labels": {"ToMe": "Faithful adaptation; official parity verified for tested configurations",
                         "EViT-style": "Reproduction-inspired; not official EViT; numerical parity unverified",
                         "DiffRate 0.6-0.9": "Faithful adaptation pending official semantic verification",
                         "DiffRate 1.0": "Not applicable as released; literal no-op diagnostic only"},
       "selection_rule": "frozen protocol: max P4-DEV agreement count among candidates with |realized FLOPs - target| <= 1.0 pp; ties (within <=3 images at n=1990) broken by smaller |FLOPs gap|, then ToMe smaller r then smaller inflection / EViT-style smaller kappa, then generation order",
       "selected": {}, "diffrate_released": {}, "diffrate_target_1.0": {"status": "Not applicable as released; literal no-op diagnostic only", "excluded_from_comparison": True}}
for k, v in sel.items():
    c = v["chosen"]
    cfg["selected"][k] = {"track": v["track"], "target_flops_pct": round(v["target_flops_pct"], 4), "method": v["method"], "config": c["config"],
                          "realized_flops_pct": round(c["realized_flops_pct"], 6), "dev_agree_count": c["agree_count"], "dev_top1_count": c["top1_count"],
                          "leaving_tokens_per_block": c["leaving_per_block"], "n_feasible_candidates": v["n_feasible_candidates"], "n_tied": len(v["tie_set"])}
for t, v in dr["targets"].items():
    if t == "1.0":
        continue
    cfg["diffrate_released"][t] = {"published_prune_kept_num": v["published_prune_kept_num"], "published_merge_kept_num": v["published_merge_kept_num"],
                                   "executed_leaving": v["executed_leaving"], "executed_analytic_flops_pct": v["executed_analytic_flops_pct"],
                                   "dev_agree_count": v["dev_agreement_count"], "dev_top1_count": v["dev_top1_count"]}
cfg["phase3_primary_frozen_schedules"] = {"source": "results/phase3_budget_aware_allocation/frozen_schedules.json", "sha256": sha(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")),
                                          "frozen_sha256_file": open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen.sha256")).read().strip(), "re_selection": False}
cfg["supplementary_phase3_reselection_native_caps"] = {"frozen_schedules_sha256": sha(os.path.join(SUP, "frozen_schedules.json")), "frozen_sha256_file": open(os.path.join(SUP, "frozen.sha256")).read().strip(),
                                                       "calibration_manifest_sha256": "e7681155ca6368726fee7de89521daa8f6179c310661761d19e945bbf0284774",
                                                       "heldout_manifest_sha256": "9f4175c26bdbd3d56eafcc1a7e3f74d2c47b28756e3601c9541875113a996d9b",
                                                       "native_caps_fraction": [0.521945, 0.579177, 0.644732, 0.71924]}
cfg["checkpoint_sha256"] = "21d4764d94f6c3ffdb6da3581115a0a1ee2d505537d96883b540e54766407c9e"
gr = json.load(open(os.path.join(G4, "candidate_grid_record.json")))
cfg["candidate_grids"] = {"tome": gr["tome_grid"], "evit": gr["evit_grid"], "grid_realized_flops_csv_sha256": gr["grid_realized_flops_csv_sha256"], "n_grid_points": gr["n_grid_points"],
                          "gate2_analytic_table_differences": gr["differences_vs_gate2_analytic_table"], "authoritative": "Gate 4 realized-FLOPs grid; selected configs below"}
ss = json.load(open(os.path.join(P4, "splits", "split_summary.json")))
cfg["split_identity"] = {"split_summary_json_sha256": sha(os.path.join(P4, "splits", "split_summary.json")), "split_summary_excerpt_keys": sorted(ss.keys()),
                         "dev_manifest_sha256": sha(os.path.join(P4, "splits", "p4_dev_manifest.csv")), "confirm_manifest_sha256": sha(os.path.join(P4, "splits", "p4_confirm_manifest.csv")),
                         "calibration500": json.load(open(os.path.join(P4, "splits", "p4_dev_cal500_summary.json")))}
cfg["unchanged_statistical_protocol"] = {"benchmark_protocol_sha256": sha(os.path.join(P4, "benchmark_protocol.md")), "benchmark_config_sha256": sha(os.path.join(P4, "benchmark_config.yaml")),
                                         "families": "PRIMARY agreement 16 (ToMe, EViT-style vs geometric x 4 budgets x 2 patterns, Holm); PRIMARY top-1 16 (separate Holm); Phase 3 replication 24 (own Holm); secondary/exploratory as in protocol section 8"}
files = ["src/phase4/engine.py", "src/phase4/tome.py", "src/phase4/evit_style.py", "src/phase4/diffrate.py", "src/phase4/selection.py", "src/phase4/data.py",
         "scripts/phase4_gate4_methods.py", "scripts/phase4_gate4_phase3_comparators.py", "scripts/phase4_gate4_reselection.py", "scripts/phase4_gate4_latency.py",
         "results/phase4_adaptive_method_benchmark/protocol_freeze.json", "results/phase4_adaptive_method_benchmark/gate4/methods_selection.json",
         "results/phase4_adaptive_method_benchmark/gate4/methods_candidates.csv", "results/phase4_adaptive_method_benchmark/gate4/phase3_comparators_dev.csv",
         "results/phase4_adaptive_method_benchmark/splits/p4_dev_manifest.csv", "results/phase4_adaptive_method_benchmark/splits/p4_confirm_manifest.csv",
         "results/phase4_adaptive_method_benchmark/splits/p4_dev_cal500_manifest.csv", "results/phase4_adaptive_method_benchmark/splits/p4_dev_heldout1490_manifest.csv",
         "results/phase4_adaptive_method_benchmark/splits/split_summary.json", "results/phase4_adaptive_method_benchmark/benchmark_protocol.md", "results/phase4_adaptive_method_benchmark/benchmark_config.yaml",
         "results/phase4_adaptive_method_benchmark/diffrate_deit_tiny_rates_as_published.json", "results/phase4_adaptive_method_benchmark/diffrate_dev_validation.json",
         "results/phase4_adaptive_method_benchmark/gate4/grid_realized_flops.csv", "results/phase4_adaptive_method_benchmark/gate4/candidate_grid_record.json",
         "results/phase4_adaptive_method_benchmark/supplementary_phase3_reselection/frozen_schedules.json", "results/phase3_budget_aware_allocation/frozen_schedules.json",
         "scripts/phase4_gate5_confirmatory.py", "scripts/phase4_freeze_protocol.py", "scripts/phase4_gate4_grid_record.py", "src/experiments/budget_aware_allocation.py", "src/compression/progressive.py"]
cfg["file_sha256"] = {f: sha(os.path.join(ROOT, f)) for f in files}
b = (json.dumps(cfg, indent=1, sort_keys=True) + "\n").encode()
open(os.path.join(P4, "frozen_method_configs.json"), "wb").write(b)
open(os.path.join(P4, "frozen_method_configs.sha256"), "wb").write((hashlib.sha256(b).hexdigest() + "\n").encode())

# ------------------------------------------------------------------ report
L = []
w = L.append
w("# Phase 4 — Gate 4 development results (P4-DEV only)\n")
w("**These are selection/validation data, not confirmatory evidence.** n = 1,990 P4-DEV images; no P4-CONFIRM image was loaded. Agreement = prediction agreement with the unpruned model (timm-native forward); top-1 = ground-truth accuracy (unpruned dev top-1 = %d/%d). Analytic FLOPs are realized counts; auxiliary cost and latency are separate.\n" % (dr["unpruned_top1_count"], N))
w("Labels: ToMe — Faithful adaptation; official parity verified for tested configurations. EViT-style — Reproduction-inspired; not official EViT; numerical parity unverified. DiffRate 0.6–0.9 — Faithful adaptation pending official semantic verification. DiffRate 1.0 — Not applicable as released; literal no-op diagnostic only.\n")
w("## 1. Selected ToMe / EViT-style configurations\n")
w("| Target | Method | Selected config | Realized FLOPs % | Gap pp | Agree (n/1990) | Agree % | Top-1 % | Feasible cands | In tie set |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, v in sel.items():
    c = v["chosen"]
    w(f"| {k.split('|')[0]} ({v['target_flops_pct']:.3f}) | {v['method']} | {c['config']} | {c['realized_flops_pct']:.3f} | {abs(c['realized_flops_pct'] - v['target_flops_pct']):.3f} | {c['agree_count']} | {100 * c['agree_count'] / N:.2f} | {100 * c['top1_count'] / N:.2f} | {v['n_feasible_candidates']} | {len(v['tie_set'])} |")
w("\nToMe config = [r, inflection]; EViT-style config = kappa (layers 4/7/10, fused token). Full per-candidate list (every feasible candidate, ties, losers): `gate4/methods_candidates.csv`; per-image predictions: `gate4/methods_candidate_predictions.npz`.\n")
w("### Tie sets and tie-break outcomes\n")
for k, v in sel.items():
    if len(v["tie_set"]) > 1:
        w(f"- {k}: tie set (within {{<=3 images}} of best {v['best_agree_count']}): " + "; ".join(f"{t['config']} (agree {t['agree_count']}, FLOPs {t['flops_pct']:.3f}, gap {t['abs_gap']:.3f})" for t in v["tie_set"]) + f" → chosen {v['chosen']['config']}")
w("\n### Feasibility exceptions\n")
w("- No target had zero feasible candidates; no ±1.0 pp exceptions were needed. Candidate counts per cell are in the table above.")
w("- Candidate-count note (explained, not erased): the Gate 2 analytic table generated kappa with np.arange(0.20, 1.0001, 0.005), whose float values (e.g. 0.7500000000000004) raise ceil(kappa*(N-1)) by one token at kappa = 0.750, 0.760, 0.775 (and 0.250, 0.300, 0.400, 0.500, 0.950, 0.995, 1.000). Gate 4 used the protocol's rounded grid round(0.20+0.005*i, 3) with REALIZED engine counts: e.g. kappa 0.750: 69.719 % (Gate 2 table: 70.084 %), 0.760: 70.787 % (71.020 %), 0.775: 72.355 % (72.461 %). Consequence: EViT-style B70 has 5 feasible values (0.74, 0.745, 0.75, 0.755, 0.76) instead of 4, and DR0.9 has 3 (0.765, 0.77, 0.775) instead of 4. No ToMe grid point differs. Exact grids and realized FLOPs: gate4/grid_realized_flops.csv, gate4/candidate_grid_record.json; the frozen configuration file is authoritative.")
w("- EViT-style B40: nearest feasible FLOPs gap is 0.918 pp (within tolerance, but close to the edge).\n")
w("## 2. DiffRate released DeiT-Tiny rates (supplementary native-FLOPs track)\n")
w("| Target | Executed FLOPs % | Agree | Agree % | Top-1 % |\n|---|---|---|---|---|")
for t, v in dr["targets"].items():
    tag = " (diagnostic only — not applicable as released)" if t == "1.0" else ""
    w(f"| {t}{tag} | {v['executed_analytic_flops_pct']:.4f} | {v['dev_agreement_count']} | {100 * v['dev_agreement_count'] / N:.2f} | {100 * v['dev_top1_count'] / N:.2f} |")
w("\nTarget 1.0 is excluded from any comparison or coverage claim. Published rates not repaired; merge semantics unverified vs official.\n")
w("## 3. Phase 3 comparators regenerated on P4-DEV (frozen schedules, no re-selection, all 1,990 images)\n")
w("Note: Phase 3 schedules were selected on a different image set, so P4-DEV is an independent evaluation for them; 37 unique schedule vectors executed.\n")
w("| Schedule | FLOPs % | Agree | Agree % | Top-1 % |\n|---|---|---|---|---|")
for _, r in p3.iterrows():
    w(f"| {r.schedule} | {r.flops_pct:.3f} | {r.agree_count} | {100 * r.agree_count / N:.2f} | {100 * r.top1_count / N:.2f} |")
w("\n## 4. Supplementary Phase 3 re-selection at DiffRate native FLOPs (500-image calibration subset; evaluated on 1,490 held-out P4-DEV images)\n")
w("Caps 52.1945 / 57.9177 / 64.4732 / 71.9240 %. Unchanged Phase 3 pipeline; calibration 6,811 s on 4 CPU threads (cost reported separately from inference FLOPs). Rows below are held-out dev results (n=1,490); budget tag is the rounded cap (B52/58/64/72).\n")
w("| Schedule | FLOPs % | Agree | Agree % | Top-1 % |\n|---|---|---|---|---|")
for _, r in sup.iterrows():
    w(f"| {r.schedule} | {r.flops_pct:.3f} | {r.agree_count} | {100 * r.agree_count / 1490:.2f} | {100 * r.top1_count / 1490:.2f} |")
w("\n## 5. Latency and auxiliary time (SECONDARY; CPU, batch 25, 4 threads, 1 warm-up + 5 runs × 3 batches)\n")
w("**Caveat:** CPU timing is highly variable here (IQRs up to several ms; e.g. Phase 3 geometric 2-8-10 vs 2-6-10 at B50 differ 18.0 vs 25.4 ms at near-equal FLOPs; the unpruned reference in the Phase 1–3 engine had IQR 13 ms). Ratios are to each engine's own unpruned forward; the Phase 4 explicit-attention engine and the Phase 1–3 fused-attention engine are not comparable in absolute ms. Equal FLOPs ≠ equal latency. Values are indicative only; no latency conclusion should be drawn from a single pass.\n")
w("| Engine | Method | Config | ms/img median | IQR | Ratio to engine ref | Aux ms/img |\n|---|---|---|---|---|---|---|")
for _, r in lat.iterrows():
    a = "" if pd.isna(r.aux_ms_per_img) else f"{r.aux_ms_per_img:.2f}"
    w(f"| {r.engine} | {r.method} | {r.config} | {r.median_ms_per_img:.2f} | {r.iqr_ms:.2f} | {r.ratio_to_engine_ref:.3f} | {a} |")
w("\nAuxiliary time = merge/scoring/fusion timers inside the Phase 4 engine; Phase 3 scorer overhead was not separately re-timed here.\n")
w("## 6. Runtimes\n")
w("ToMe/EViT-style: 84 unique candidates × 1,990 images, 4,802 s total. Phase 3 comparators: 37 schedules, 1,504 s. Supplementary re-selection: calibration 6,811 s + held-out 1,333 s. These were measured while other jobs ran concurrently, so elapsed times are indicative only.\n")
w("## 7. Hashes\n")
w("See `frozen_method_configs.json` (SHA-256 in `frozen_method_configs.sha256`) for code, manifest and output file hashes.")
open(os.path.join(P4, "development_results.md"), "wb").write(("\n".join(L) + "\n").encode())
print("written", hashlib.sha256(b).hexdigest())
