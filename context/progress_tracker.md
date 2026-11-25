# Progress Tracker

_Last updated: 2026-11-15_

**Project:** Layer-Wise Information Dynamics of Tokens in Vision Transformers (DeiT-Tiny, token pruning at layer *l*, measuring performance vs. compute).

## Implemented so far

- **Model wrapper** (`src/models/vit_wrapper.py`): ViT/DeiT wrapper with token extraction at any layer, partial forward passes, and parity checks against the original model.
- **Data** (`src/data/dataset.py`): loaders for ImageNet val, CIFAR-100 and synthetic data.
- **Token importance scorers** (`src/importance/`): abstract base class plus Random, Prediction-KL (exact ablation and approximate gradient saliency) and CLS-Attention scorers.
- **Token pruning** (`src/compression/token_pruning.py`): top-k selection that preserves spatial order and keeps the CLS token.
- **Metrics** (`src/metrics/`):
  - Prediction: agreement, top-1/top-5, confidence retention, KL, JS, cosine distance.
  - Compute: theoretical FLOPs and CUDA latency/memory benchmarking.
  - Information: marginal efficiency ΔP/ΔC.
- **Experiments** (`src/experiments/`):
  - Runner: orchestration across layers, retention ratios and scorers.
  - Analysis: Wilson and Student-t confidence intervals, McNemar test vs. random.
- **Visualization** (`src/visualization/plots.py`): 6 plots (information vs tokens, information vs compute, compute vs tokens, marginal information, accuracy vs compute, FLOPs vs latency).
- **CLI scripts**: `scripts/run_experiment.py` and `scripts/analyze_results.py`.
- **Configs**: smoke test, CIFAR-100 debug, ImageNet dev-250, full ImageNet.
- **Tests** (`tests/`): compute, instrumentation, metrics, pruning, smoke.
- **Docs**: `README.md` and `PROJECT_OVERVIEW_AI_GUIDE.md` (full handbook, not yet committed).

## Phase 2: progressive multi-layer pruning (infrastructure only)

- `src/compression/progressive.py`: `RetentionSchedule` (validated schedule), `prune_patch_tokens` (keeps CLS, per-image top-k, original order) and `ProgressivePruner` (one forward pass with different k at several layers, plus a no-pruning path and a per-block token-count trace).
- Schedule keys are 1-based layer numbers, the same as Phase 1. Key `l` means "after `l` blocks keep `k` patch tokens". The conversion happens in one place, `ProgressivePruner.forward`. Layer 12 is rejected because pruning after the last block saves nothing.
- Scorers are the unchanged Phase 1 classes: `random`, `prediction_kl_approx`, `cls_attention`. The old names `prediction_kl_approximate` and `attention_cls` still work as aliases.
- Added `ComputeEstimator.compute_progressive_flops` (additive; Phase 1 maths unchanged), `configs/progressive_smoke.yaml`, `scripts/run_progressive.py` and `tests/test_progressive.py`.
- Tests: 30 passed in the `minor` conda env (the 14 new Phase 2 tests plus all Phase 1 tests). The empty schedule matches the original logits (max diff 0.0 in the smoke run, ≤1e-4 required). A one-layer schedule matches Phase 1 for random and CLS-attention.
- Smoke test: 10 ImageNet images, schedule `{2: 98, 4: 49}`. Results are in `results/phase2_progressive_smoke/`, separate from Phase 1.

## Phase 2 evaluation: progressive vs single-layer at matched FLOPs

- `src/experiments/progressive_eval.py` is the runner. It evaluates 28 schedules × 3 scorers × seeds (136 runs): the 7 required schedules, 3 extra progressive schedules, and single-layer controls whose analytic FLOPs match each progressive schedule. Controls are chosen from FLOPs only, within ±2 pp of baseline FLOPs, before any accuracy is seen.
- Per run it records agreement, top-1 and top-5 (counts with Wilson 95% CI), analytic FLOPs, and CPU latency with scorer overhead separated. Matched pairs get a paired bootstrap CI and an exact McNemar p-value.
- Also added: `src/visualization/progressive_plots.py` (4 plots), `configs/progressive_eval.yaml`, `scripts/run_progressive_eval.py` (with `--smoke`), `tests/test_progressive_eval.py` and `regenerate_reports()`, which rebuilds the plots and summary from the saved CSVs.
- Tests: 39 passed in the `minor` env (Phase 1, Phase 2 and the 9 new evaluation tests). Smoke test (20 images) and the full 250-image run both completed with 0 failed runs.
- Outputs are in `results/phase2_progressive_eval/` (metrics.csv, per_image_metrics.csv, token_counts.csv, schedule_manifest.json, run_config.json, summary.md, plots/). Phase 1 and the Phase 2 smoke results were not overwritten.
- Subset: the first 250 images of `data/imagenet_val_250.pt` in stored order, the same loader call as Phase 1. Baseline top-1 is 191/250 = 76.4%, the same as Phase 1.

## Phase 2 controlled pruning-stage ablation

- `src/experiments/stage_ablation.py` generates and runs one-, two- and three-stage schedules with the first pruning layer fixed (layer 2 primary, layer 4 as a separate secondary analysis).
- Schedules are matched on analytic FLOPs. Each budget (40/50/60/70%) has a one-stage anchor. Multi-stage counts follow a geometric profile K_j = round(N·ρ^j), with ρ searched to match the anchor within ±1 pp. All 48 schedule slots matched (0 unmatched), with |ΔFLOPs| ≤ 0.13 pp.
- It adds paired McNemar and bootstrap tests, Holm correction within each test family, a subset check against the previous evaluation's manifest (the run stops on a mismatch), and `regenerate_reports()`.
- Also added: `src/visualization/stage_ablation_plots.py`, `configs/stage_ablation.yaml`, `scripts/run_stage_ablation.py` and `tests/test_stage_ablation.py`.
- Tests: 49 passed in the `minor` env (Phase 1, Phase 2, progressive-eval and the 10 new ablation tests). Smoke test and full run both completed (253 metric rows, 0 failed). Outputs are in `results/phase2_stage_ablation/`.

## Results so far (ImageNet dev-250, `deit_tiny_patch16_224`)

- 250 samples, 6 layers (2–12), 8 retention ratios.
- Early layers (L2–L4) are very sensitive to pruning. Deep layers (L8–L10) are robust: about 94% or more agreement at 12.5% of tokens.
- Prediction-KL (approx) is the best scorer and is significant vs. random (p < 0.001) in most settings.
- CLS-Attention is about the same as random at L2 and improves in middle and late layers.
- Marginal gain ΔP/ΔC is high in early layers and near zero in late layers above 50% tokens.
- Theoretical FLOPs savings are larger than measured latency savings.

## Phase 2 – Class-diverse validation & allocation robustness (implemented, run)
- Spec: `Features/phase2_class_diverse_validation_spec.md`. Evaluation only; Phase 1/2 code, scorers and results unchanged.
- New: `src/data/class_diverse.py` (seeded class-balanced subset from the ImageNet-val parquet shards, manifest + SHA-256, coverage summary), `src/experiments/class_diverse.py` (two allocation families `geometric` / `equal_drop`, FLOPs-matched schedule generation within ±1 pp, paired stats with Holm families, multi-batch latency, resume checkpoint, summary), `src/visualization/class_diverse_plots.py`, `configs/class_diverse.yaml`, `scripts/run_class_diverse.py` (`--prepare-only`, `--smoke`), `tests/test_class_diverse.py` (15 tests).
- Subset: 2,000 images, 1,000/1,000 classes, 2 per class, seed 42; data cached in `data/` (parquet shards downloaded from the same HF dataset the loader already used).
- Tests: 64 passed (49 earlier + 15 new). Smoke (50 images) passed. Full run: 141 runs, 0 failures, 2 unmatched schedules (equal_drop L2 B40 2-8-10; L4 B40 4-6-8), ~2.4 h CPU.
- Outputs: `results/phase2_class_diverse_validation/` (smoke in `..._smoke/`).
- Commands: `python -m pytest tests -q`; `python scripts/run_class_diverse.py --config configs/class_diverse.yaml [--prepare-only|--smoke]`.

## Phase 2 – Minimum final-token constraint ablation (implemented, run)
- Spec: `Features/phase2_minimum_token_ablation_spec.md`. Evaluation only; geometric/equal_drop rules imported unchanged and asserted equal to the saved class-diverse manifest; same 2,000-image subset verified by manifest hash.
- New: `src/experiments/minimum_token.py` (new `constrained_equal_drop` family with patch-token floors 8/16/24/32, FLOPs-matched D search, infeasible / identical-to-unconstrained handling, stats with Holm families, summary), `src/visualization/min_token_plots.py`, `configs/min_token.yaml`, `scripts/run_min_token.py` (`--smoke`), `tests/test_minimum_token.py` (14 tests).
- Tests: 78 passed. Smoke (50 images) passed. Full run: 219 metric rows, 0 failures, 6 infeasible, 68 identical-to-unconstrained, ~1.6 h CPU.
- Outputs: `results/phase2_minimum_token_ablation/` (smoke in `..._smoke/`).
- Commands: `python -m pytest tests -q`; `python scripts/run_min_token.py --config configs/min_token.yaml [--smoke]`.

## Phase 2 – Final-token vs trajectory isolation (implemented, run)
- Spec: `Features/next_step_final_token_trajectory_isolation.md`. Evaluation only. Families: geometric, NEW final-count-matched geometric controls (same layers/initial/final count, only intermediate counts vary, FLOPs ±1 pp of geometric), constrained and unconstrained equal-drop (prior code unchanged). Patterns 2-6-10 / 2-8-10, budgets 40–70%, cls_attention, seed 42.
- New: `src/experiments/trajectory_isolation.py` (exhaustive candidate enumeration, accuracy-blind control selection, full-SHA manifest verification, prior-schedule/result reproduction checks, predefined Holm families, result.md), `src/visualization/trajectory_plots.py`, `configs/trajectory_isolation.yaml`, `scripts/run_trajectory_isolation.py` (`--prepare-only`, `--smoke`), `tests/test_trajectory_isolation.py` (16 tests).
- Tests: 94 passed (78 + 16). Smoke passed. Full run: 78 metric rows (32 controls), 0 failures; 30/30 prior schedules reproduce prior counts exactly.
- Outputs: `results/phase2_final_token_trajectory_isolation/` (incl. `result.md`, `control_candidates.csv.gz`, `schedule_table.csv`, `run_manifest.json`).

## Phase 3 – Budget-aware allocation: design stage (Gates 1–3 done; no model run yet)
- Spec: `Features/phase3_budget_aware_algorithm_design_spec.md`. Added `scripts/phase3_evidence_audit.py` (recomputes ledger numbers from raw files) and `results/phase3_algorithm_design/{evidence_ledger.md, algorithm_design.md, audit_numbers.md/json}`.
- Chosen design (for review): measured layer-sensitivity chain model + exact FLOPs-constrained solve with calibration verification of the top-8; 500/1,500 class-stratified split + proposed fresh confirmatory test set. Implementation (Gate 4+) not started pending review.

## Phase 3 — Budget-aware allocation (LS-Chain + verify): implemented and evaluated
- Done: evidence audit + design (`results/phase3_algorithm_design/`), implementation (`src/optimization/`, `src/experiments/budget_aware_allocation.py`, `src/data/phase3_splits.py`, `src/visualization/phase3_plots.py`, config, CLI, 21 tests), calibration (500 img), validation (1,500 img), freeze (SHA `b6d0103d…fe20`), fresh confirmatory (2,000 img, run once).
- Outcome: mean +0.87 pp agreement vs geometric on confirmatory (+0.73 validation); 0 Holm-significant; superiority over geometric NOT established. Clearly better than single-stage and (where feasible) constrained equal-drop. No latency difference.
- Details: `results/phase3_budget_aware_allocation/result.md` (hand-written conclusions at the end). Nothing committed.

## Phase 4 — Benchmark against adaptive token-pruning methods (Gates 0–3 done; stopped before Gate 4)
- Gate 0/1/2: audit, related-methods review, fidelity matrix, protocol frozen (`results/phase4_adaptive_method_benchmark/protocol_freeze.json`, sha256 file verified by `scripts/phase4_freeze_protocol.py --verify`). Splits: P4-DEV 1,990 / P4-CONFIRM 2,985 images, 995 classes (classes 0–4 exhausted by Phase 1 dev250).
- Method set: ToMe (faithful adaptation), EViT-style training-free (reproduction-inspired, not EViT), DiffRate released DeiT-Tiny rates (supplementary native-budget track only). DynamicViT and official EViT: not compared (training/checkpoints).
- Gate 3: new `src/phase4/` (engine, tome, evit_style, diffrate, selection, data with locked P4-CONFIRM), 15 new tests; full suite 130 passed. Official ToMe parity (isolated venv, timm 0.4.12): token counts identical for 201 configs, logits bit-identical, full-dev counts identical. DiffRate released lists executed on P4-DEV; target 1.0 has an ill-defined merge (logged, not repaired).
- Errata found: Phase 3 reports print baseline vectors in string-sorted layer order and one Phase 3 hand-written sentence about geometric profile is wrong (`phase3_documentation_errata.md`; Phase 3 files untouched).
- No P4-CONFIRM data touched. Nothing committed.

## Phase 4 — Gate 4 (development evaluation, P4-DEV only) — complete, awaiting review (2026-11-17)
- Selected ToMe (r, inflection) and EViT-style kappa per primary budget (40/50/60/70) and DiffRate native FLOPs (0.6-0.9) from every feasible candidate (84 unique evaluated on 1,990 dev images); selections, tie sets and all candidates in `results/phase4_adaptive_method_benchmark/gate4/` and `development_results.md`.
- Phase 3 frozen schedules regenerated on P4-DEV (37 unique schedules); supplementary Phase 3 re-selection at DiffRate native caps on the fixed 500-image calibration subset, evaluated on 1,490 held-out dev images.
- Latency/aux time measured as secondary (CPU, noisy). DiffRate target 1.0 excluded (diagnostic only).
- `frozen_method_configs.json` SHA-256 c42717bf05e677e426c3422267f0d7f941a39ea30c07ac27733d1b8c269b706e; protocol freeze verified; 130 tests pass.
- Disclosed deviation: a test unintentionally decoded P4-CONFIRM images into a cache (no model run, no statistics); cache deleted, test fixed (see protocol_amendments.md).
- Development results are selection data, not confirmatory evidence. Gate 5 not started; needs explicit approval.

### Phase 4 — Gate 4 close-out and Gate 5 preflight (2026-11-17)
- Gate 4 accepted. P4-CONFIRM cache incident closed in `protocol_amendments.md` (images + labels were decoded/cached once by a test, never read; cache deleted; test isolated; loader now verifies frozen config bytes and dependency hashes). P4-CONFIRM retained; must be disclosed as a data-access deviation.
- `frozen_method_configs.json` re-frozen with grid identity, split/manifest identities, calibration-subset identity, checkpoint SHA-256 and 32 dependency hashes: SHA-256 171fa7be2cc4bd511a9aa491da8c475c480d198f1a5070287a5a8eebd134e5b6. 131 tests pass; protocol freeze verified (17 files); `gate5_preflight.json` all checks pass. Gate 5 runner written and dry-run on 50 P4-DEV images only; NOT executed on P4-CONFIRM — awaiting explicit authorization.

### Phase 4 — Gate 5 analysis frozen (P4-DEV validated; confirmatory run NOT authorized/started) (2026-11-17)
- `scripts/phase4_gate5_analysis.py` (SHA-256 8b5e51041f297c1f6ad0bd2125c69c59993c1073bc0aabf7b58500019eb5df9d); freeze record `phase4_analysis_freeze.json` (SHA-256 e574c540e646217e1592dcfd94a959e9801666ec789c38a4d43befd7742aa2dc); documentation `analysis_documentation.md`.
- P4-DEV pipeline validation: independent cross-check all pass (counts match Gate 4 records), reproducible byte-for-byte, 144 tests pass, Gate 5 preflight all checks pass; frozen_method_configs (171fa7be…) and original protocol freeze (895ee5fe…) unchanged; no marker/partial outputs.

### Phase 4 — Gate 5 confirmatory run executed once (2026-11-17)
Single P4-CONFIRM run completed (6,247 s, no failure). Frozen analysis run once. ToMe beats geometric in all 16 primary cells (Holm-significant); EViT-style worse at B40, null at B50, better at B60/B70; Phase 3 replication 1/24 Holm-significant (no broad superiority). See `results/phase4_adaptive_method_benchmark/confirmatory_results.md` and `confirmatory_analysis/`. Gate 6 (final report) pending.

### Phase 4 — Gate 6 final report written (2026-11-17)
`results/phase4_adaptive_method_benchmark/phase4_final_report.md`, `paper_claims_matrix.md`, `recommended_next_experiment.md`. Assessment: ToMe > geometric in all 16 primary cells; EViT-style budget-dependent; Phase 3 allocator no consistent advantage over geometric and not competitive with ToMe/DiffRate; no novelty claim; workshop (benchmark/negative-result) framing plausible, main-track not ready; next experiment proposed (sensitivity-guided per-layer merge schedules vs equal-effort ToMe, fresh confirmatory split). No paper drafted; nothing committed.
