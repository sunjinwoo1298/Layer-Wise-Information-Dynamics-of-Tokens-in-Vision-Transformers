# Phase 4 — Benchmark Against Established Adaptive Token-Pruning Methods

## Purpose

Phase 3 is complete. The calibration-based allocator improved over the tested single-stage baseline and, in some settings, constrained equal-drop, but it did **not** establish superiority over geometric allocation on the fresh confirmatory evaluation. The sensitivity-chain proxy also showed weak within-cell ranking despite a strong pooled correlation.

The next step is a controlled benchmark answering:

> How does the current progressive token-pruning approach compare with established adaptive token-pruning methods when model, data, compute budgets, and evaluation protocol are controlled?

This is a benchmark/evidence phase, not another round of tuning against geometric. Do not change the Phase 3 method during this phase.

## 1. Non-negotiable rules

1. Preserve all Phase 1–3 source files, results, manifests, predictions, reports, hashes, and hand-written conclusions. Never overwrite or silently regenerate them.
2. Create an isolated directory: `results/phase4_adaptive_method_benchmark/`.
3. Use new Phase 4 scripts/configurations. Do not modify existing experiment code unless a necessary bug fix is documented and approved before implementation.
4. Freeze the protocol, implementations, hyperparameters, and data splits before confirmatory evaluation.
5. Do not use the Phase 3 confirmatory set for Phase 4 tuning, debugging method quality, or hyperparameter selection.
6. Report prediction agreement, actual top-1 accuracy, analytic FLOPs, auxiliary/scorer overhead, calibration cost, and measured latency separately.
7. Never infer latency from FLOPs. Report hardware, batch size, warm-up, repetitions, timing method, and variability.
8. Use paired comparisons, confidence intervals, and a predefined multiple-comparison correction. Positive means alone do not establish superiority.
9. Keep the weak layer-2 CLS-attention scorer limitation visible: the audit found it worse than random at layer 2 across the tested D2000 budgets.
10. Approximate Pred-KL is gradient-times-activation saliency, not exact KL or a formal information-theoretic measure.
11. Do not claim architecture generalization from DeiT-Tiny alone.
12. Do not claim superiority or novelty over published methods unless implementation fidelity supports that claim.
13. If a published baseline cannot be implemented faithfully, label it as an adapted or reproduction-inspired baseline, and explain the deviation.
14. If compute requires reducing scope, do so before confirmatory evaluation and document the change.
15. A negative result is acceptable. Do not tune after seeing confirmatory results.

## 2. Gates — work in order and stop for review

### Gate 0 — Repository and evidence preservation audit

Before coding:
- Inspect repository structure, git status/current commit, Python/PyTorch/torchvision/timm versions, hardware, available ImageNet shards, and pretrained checkpoints.
- Verify the exact Phase 3 frozen-schedule hash and confirmatory-manifest hash against saved artifacts.
- Confirm whether confirmatory predictions/per-image outputs are present and preserved.
- Inventory ImageNet validation images not used in Phase 1–3.
- Inventory existing token-pruning implementations and identify reusable code without modifying it.
- Check whether any existing code already implements a published adaptive baseline.

Create:
- `environment_audit.md`
- `artifact_inventory.md`
- `data_split_plan.md`

Place these in `results/phase4_adaptive_method_benchmark/`.

**Stop after Gate 0. Present the audit and proposed data split for approval.**

### Gate 1 — Related methods and fidelity plan

Investigate at least these methods:
- **DynamicViT** — dynamic token sparsification/pruning.
- **EViT (Token Reorganization)** — attention-based token reorganization, including inattentive-token handling.
- **DiffRate** — layer-wise token compression rates under a compute budget.

Consider one additional recent/relevant method only if it can be implemented fairly within available compute.

For each method, inspect the original paper and official repository. Record:
- Exact title, authors, venue/year, DOI/arXiv identifier.
- Official repository URL and license.
- Supported architectures and checkpoints.
- Whether training, fine-tuning, distillation, learned policy modules, or only inference-time operations are required.
- Whether the method prunes, merges, reorders, or sparsifies tokens.
- Native compute-budget control and mapping to this benchmark's FLOPs budgets.
- Required pretrained weights and local availability.
- Implementation deviations and their expected effect on fairness.

Use primary sources (papers and official repositories), not only blogs or surveys.

Create:
- `related_methods_review.md`
- `baseline_fidelity_matrix.md`
- `implementation_plan.md`

**Stop after Gate 1. Recommend a feasible method set and request approval before implementation.**

### Gate 2 — Freeze benchmark protocol

#### Model and weights
- Primary model: the same frozen `deit_tiny_patch16_224` used in Phases 1–3.
- Record checkpoint source/hash, preprocessing, input resolution, normalization, and model code/version.
- Do not fine-tune the primary model unless a method fundamentally requires it. If training is required, separate trained-method and inference-only tracks. Do not mix them without explicit labeling.

#### Data
- Use a fresh, disjoint ImageNet validation subset for Phase 4 confirmation, if enough unused images remain.
- Prefer a prespecified class-balanced design; explicitly state that this differs from the natural ImageNet validation distribution.
- If remaining data cannot support the intended sample size, propose a smaller sample before running.
- Save exact image IDs, labels, ordering, preprocessing metadata, and SHA-256 manifest.
- Use a separate development/calibration split for methods requiring calibration or budget selection.
- Never use confirmatory labels/predictions to tune parameters, schedules, thresholds, or method choices.
- Document any differences from an official method's evaluation protocol.

#### Compute budgets
Use common target budgets, preferably 40%, 50%, 60%, and 70% of unpruned analytic FLOPs.
- Map each method's native controls to the closest feasible budget.
- Report achieved FLOPs, not just target budgets.
- Predefine a tolerance (recommend ±1 percentage point).
- Mark cells outside tolerance infeasible; never relax tolerance selectively.
- Compare methods only within matched-budget cells.

#### Metrics
Primary:
- Prediction agreement with the unpruned model.
- Actual ImageNet top-1 accuracy.

Secondary:
- Top-5 accuracy, if consistently available.
- Analytic FLOPs and achieved budget.
- Retained token counts per layer.
- Method-specific auxiliary compute/memory overhead.
- Measured end-to-end latency.

For agreement:
- Compare every method with the same unpruned reference prediction per image.
- Save per-image predictions and correctness indicators.
- Report percentage, numerator/denominator, and Wilson 95% interval.
- Use paired bootstrap confidence intervals and exact McNemar tests where appropriate.

For top-1:
- Report correctness against ground-truth labels separately. Agreement is not accuracy.

Multiple comparisons:
- Predefine primary comparisons and Holm correction family before confirmation.
- Report each cell; do not hide variation in a pooled headline.
- Label exploratory comparisons.

#### Runtime
- Same machine/software environment for all methods.
- Same batch sizes, warm-up, repetitions, and synchronization policy.
- Measure baseline, method forward pass, and end-to-end cost separately.
- Include policy/scorer/auxiliary-network cost.
- Report CPU and GPU separately if available.
- Do not claim speedup from analytic FLOPs alone.

Create:
- `benchmark_protocol.md`
- `benchmark_config.yaml`
- `method_config_matrix.csv`

**Stop after Gate 2. Freeze and hash protocol/config before confirmatory runs.**

### Gate 3 — Implement and validate approved methods

- Prefer official code and weights when compatible.
- Record exact source commit/tag and all local modifications.
- Keep each method in an isolated adapter/module.
- Do not alter the pretrained model differently across methods unless required by the original method.
- Test token counts, CLS preservation, ordering/reorganization semantics, shape consistency, determinism, and no-pruning equivalence where applicable.
- Test FLOPs calculation consistently across methods.
- Validate against paper/official code on development data only.
- If official code cannot run, document the fallback and limitations.
- Do not use confirmatory data for debugging.

Create:
- `implementation_notes.md`
- `implementation_deviations.md`
- Phase 4 method-specific tests.

Run the full existing test suite, new Phase 4 tests, smoke tests, and reproducibility tests.

**Stop after Gate 3. Report test results and method fidelity before full evaluation.**

### Gate 4 — Development/calibration evaluation

On development/calibration data only:
- Map native method controls to target FLOPs.
- Tune only parameters explicitly permitted by the frozen protocol.
- Record achieved FLOPs and feasibility.
- Validate token trajectories, per-image predictions, and overhead.
- Do not use confirmatory data for selection.

Freeze:
- method configurations,
- budget mappings,
- thresholds,
- code commits,
- checkpoint hashes.

Create:
- `development_results.md`
- `frozen_method_configs.json`
- `frozen_method_configs.sha256`

**Stop after Gate 4. Request approval before confirmation.**

### Gate 5 — Confirmatory benchmark

Run frozen methods exactly once on the fresh Phase 4 confirmatory split.

For every method × budget cell save:
- exact configuration,
- per-image logits or predictions,
- agreement and top-1 outcomes,
- token counts per layer,
- analytic FLOPs,
- auxiliary compute,
- latency measurements,
- logs and environment metadata.

Compare:
1. Phase 3 proposed allocator.
2. Geometric progressive allocation.
3. Constrained equal-drop allocation.
4. Single-stage pruning.
5. Approved published adaptive methods.

Keep Phase 3 proposed allocation and geometric allocation independent. Do not add geometric schedules to the proposed method's candidate pool.

Primary comparisons:
- Each approved published method versus geometric at matched FLOPs.
- If multiple published methods are approved, define the Holm family across all prespecified method × budget × pattern comparisons.
- Report Phase 3 proposed allocation as an additional prespecified comparator, not a replacement for published baselines.

Secondary:
- Published methods versus Phase 3 proposed allocation.
- Published methods versus single-stage and constrained equal-drop.

Use paired per-image comparisons, bootstrap CIs, exact McNemar tests where appropriate, and Holm correction.

**Do not tune after viewing confirmatory results.**

Create:
- `confirmatory_results.md`
- `per_image_results/`
- `confirmatory_environment.json`

### Gate 6 — Interpretation and paper-readiness

Produce an evidence-based report answering:
1. Which methods are genuinely comparable and faithfully implemented?
2. Which methods achieve each FLOPs budget?
3. Which method has best agreement and actual top-1 at each budget?
4. Are differences supported after correction?
5. What are the latency and overhead trade-offs?
6. Does the current method offer any advantage over published methods?
7. Which claims are supported, unsupported, or contradicted?
8. What is the strongest defensible contribution?
9. Is the work ready for a workshop/conference submission, or is another experiment necessary?

Separate:
- direct reproductions,
- faithful adaptations,
- reproduction-inspired baselines,
- methods that could not be compared fairly.

Create:
- `phase4_final_report.md`
- `paper_claims_matrix.md`
- `recommended_next_experiment.md`

Do not automatically draft a paper or claim novelty. First provide the evidence-based readiness assessment.

## 3. Fairness notes for candidate methods

These methods have different mechanisms and training assumptions:
- **DynamicViT** commonly uses learned dynamic token sparsification and training/distillation. An inference-only approximation must not be called a faithful DynamicViT reproduction.
- **EViT** reorganizes tokens using attention and can fuse inattentive tokens. It is not simply top-k deletion; preserve and report its semantics.
- **DiffRate** learns layer-wise compression rates under a compute budget and may require training. Do not substitute the Phase 3 calibration algorithm and call it DiffRate.

If a method requires training and cannot be reproduced, put it in a separate training-required track or mark it not fairly comparable. Fidelity matters more than the number of baselines.

## 4. Questions to resolve before implementation

Do not guess. Verify from papers, official repositories, checkpoint metadata, and local environment:

1. Which methods officially support the exact DeiT-Tiny architecture/checkpoint?
2. Which require training/distillation, and what pretrained checkpoints are available?
3. Can native controls reach 40/50/60/70% analytic FLOPs within ±1 pp?
4. Which methods belong in a genuinely inference-only track?
5. How should learned policies and token fusion be handled fairly?
6. Is enough unused ImageNet validation data available for a fresh confirmatory split disjoint from all earlier experiments?
7. What are expected CPU/GPU runtime and storage costs?
8. Which method set provides the strongest credible benchmark within available compute?

## 5. Required terminology

Use these definitions consistently:
- **Prediction agreement:** percentage of examples where the pruned model predicts the same class as the unpruned model.
- **Actual top-1 accuracy:** percentage correct against ImageNet ground-truth labels.
- **Analytic FLOPs:** theoretical operation count under the documented counting convention.
- **Measured latency:** wall-clock inference time under the specified hardware/timing protocol.
- **Scorer/auxiliary overhead:** extra computation needed to select, merge, or reorganize tokens.
- **Calibration cost:** offline cost of selecting/fitting method parameters.

Never conflate these metrics.

## 6. Reporting
- Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
- Results saved separately from Phase 1–3; do not overwrite existing
  outputs.
- Update the `context\result.md` file with a brief summary of the new functionality and test results.

## 7. Desired outcome

At the end of Phase 4, determine whether the current work:
- is competitive with established adaptive token-pruning methods,
- contributes a useful budget-aware progressive allocation strategy,
- primarily provides a careful empirical analysis of token-allocation trajectories,
- or needs a different research direction.

An honest, reproducible benchmark is more valuable than another tuned result without a strong baseline.
