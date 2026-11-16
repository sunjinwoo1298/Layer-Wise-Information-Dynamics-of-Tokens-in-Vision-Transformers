# Next Experiment: Class-Diverse Validation and Allocation-Robustness Study

## Goal

Test whether the observed benefit of multi-stage CLS-attention pruning
beginning at layer 2 generalizes beyond the previous 250-image subset
(which contains only five ImageNet classes), and whether it persists
under a second token-count allocation family.

Primary question:

> Does multi-stage CLS-attention pruning beginning at layer 2 preserve
> predictions better than matched-FLOPs single-stage pruning on a
> class-diverse ImageNet subset, and does the result persist under a
> different token-allocation strategy?

This is a validation/ablation task only. Do not implement a budget
optimizer, token merging, fine-tuning, learned routing, or new scoring
algorithms.

## 1. Inspect and preserve existing code

Before coding: 1. Inspect Phase 1/2 code, the stage-ablation
runner/configs/tests, and `results/phase2_stage_ablation/summary.md`. 2.
Reuse existing model loading, ImageNet preprocessing/data loader,
progressive forward pass, scorers, FLOPs calculator, metrics,
statistical tests, and output conventions. 3. Preserve existing behavior
and results; write to a new output directory. 4. Confirm layer numbering
and schedule semantics from code and document them. 5. Do not silently
change existing scorer implementations.

## 2. Class-diverse ImageNet subset

Use the same pretrained `deit_tiny_patch16_224` checkpoint and
preprocessing as prior experiments.

Construct a reproducible validation subset: - Target at least 1,000
images; use 2,000 if runtime/storage permit. - Cover a broad range of
the 1,000 ImageNet classes. - Aim for approximately balanced class
representation where feasible. - Save ordered image identifiers/paths,
labels, and SHA1 (or existing project hash). - Save the manifest and its
hash. - Use a deterministic sampling seed and document the sampling
algorithm. - Keep image order fixed across all schedules, scorers, and
seeds.

Prefer the full validation set if practical. Otherwise use the
class-diverse subset and describe its sampling limitations. Do not call
it representative of the full ImageNet distribution unless justified by
the sampling method.

Before the full run, report image count, distinct class count,
minimum/median/maximum images per class, manifest hash, and a compact
class-coverage summary.

Do not use the old contiguous 250-image/five-class subset as the primary
dataset. It may appear only as a separate historical comparison.

## 3. Experimental factors

### A. Stage count (primary)

Fix the first pruning layer at layer 2. Compare: - One stage: `{2: K}` -
Two stages: `{2: K1, L2: K2}` - Three stages: `{2: K1, L2: K2, L3: K3}`

Use valid later layers supported by the implementation. Counts are patch
tokens excluding CLS, must be non-increasing, and must preserve CLS.

### B. Allocation family (primary)

Evaluate at least two distinct token-count allocation families:

1.  **Geometric family:** reproduce the allocation principle used in the
    previous stage-ablation experiment.
2.  **Alternative family:** use a clearly specified alternative, such as
    approximately equal absolute token reductions across stages.

Define the alternative rule and rounding before examining evaluation
outcomes. The families must differ in how tokens are distributed across
stages, not merely have different names. Generate counts to satisfy
FLOPs budgets and document the algorithm.

Do not assume either family is better.

### C. Scorer

Primary: `cls_attention`.

Secondary: `random`, `prediction_kl_approx`.

Keep scorer analyses separate. `prediction_kl_approx` remains
gradient-times-activation saliency, not exact KL; report its selection
overhead. Preserve existing scorer semantics.

### D. Secondary layer analysis

If practical, repeat a smaller experiment with first pruning layer fixed
at layer 4. Keep it separate from layer-2 primary results.

## 4. Matched-FLOPs design

Evaluate target budgets around 40%, 50%, 60%, and 70% of baseline
analytic FLOPs, if valid schedules permit.

For every target budget, generate one-, two-, and three-stage schedules
for both allocation families, with first pruning layer fixed at 2.

Requirements: 1. Use the existing full-path FLOPs calculator. 2.
Predefine matching tolerance: ±1 percentage point of baseline FLOPs. 3.
Match schedules across stage counts and allocation families within that
tolerance. 4. Record exact normalized FLOPs and differences. 5. Report
unmatched combinations; do not loosen tolerance after seeing results. 6.
Use identical FLOPs accounting for all schedules and baseline. 7.
Generate schedules by a documented procedure, not by selecting them
after inspecting accuracy.

If a target cannot be matched, report it rather than manufacturing a
comparison. Match by analytic FLOPs, not latency.

## 5. Seeds and scorer controls

-   Fixed evaluation image order for every run.
-   Primary seed 42.
-   For random pruning, run at least seeds 42, 43, and 44.
-   Record how RNG advances across pruning calls and whether schedules
    use independent or paired random streams.
-   Use common random seeds across schedule groups where practical,
    without silently changing established semantics.
-   Run CLS attention as primary; secondary scorers where practical.
-   Report unsupported combinations explicitly.

Do not compare different scorers and attribute the difference to stage
count or allocation family.

## 6. Metrics

For each schedule × allocation family × scorer × seed, collect:

**Prediction** - Agreement with unpruned top-1 predictions. - Actual
top-1 accuracy against ImageNet labels. - Top-5 if supported. - Counts,
image count, and 95% confidence intervals.

**Compute** - Full-path analytic FLOPs. - Normalized FLOPs (%
baseline). - FLOPs saved (%). - Exact matched-budget differences.

**Runtime** - End-to-end latency including scoring/selection. - Scorer
overhead separately where measurable. - Device, batch size, warm-up,
timed-run count, and timing method. - CPU and GPU results separately. -
Use enough repetitions and representative batches to reduce noise; avoid
relying only on one batch and three runs if feasible.

**Metadata** - Checkpoint, subset manifest hash, image count/class
coverage, sampling seed/algorithm, evaluation seed, scorer, schedule,
allocation family, layer numbering, schedule semantics, software/device
details.

Do not infer practical speedup from FLOPs savings.

## 7. Statistical analysis

Predefine the primary family: - CLS attention. - First pruning layer
2. - One vs two vs three stages. - Geometric vs alternative
allocation. - Matched FLOPs within ±1 pp. - Outcomes: prediction
agreement and actual top-1 accuracy.

Use paired evaluation on identical images: - Paired bootstrap over
images for agreement and accuracy differences. - McNemar's test for
paired correctness where appropriate. - Effect sizes and 95% confidence
intervals. - Holm correction for multiple comparisons; report raw and
corrected p-values. - Separate image-level uncertainty from random
seed-to-seed variation.

Random and approximate Prediction-KL are secondary. Layer-4 analysis is
separate and exploratory. Do not claim significance from win counts
alone.

## 8. Output artifacts

Create `results/phase2_class_diverse_validation/`. Do not overwrite
previous results.

Save: - `subset_manifest.csv` or `.json`: ordered image IDs, labels,
hashes. - `subset_summary.json`: image/class counts, class histogram,
per-class count stats, manifest hash, sampling procedure. -
`metrics.csv`: one row per schedule × family × scorer × seed. -
`per_image_metrics.csv`: image ID, label, baseline/compressed
predictions, correctness, agreement. - `token_counts.csv`: counts by
block/pruning point. - `schedule_manifest.json`: schedules, stage
counts, family, first layer, FLOPs, matched groups. - `run_config.json`:
checkpoint, subset, seeds, scorer, timing, environment. -
`statistical_tests.csv`: effect sizes, CIs, raw/corrected p-values. -
`summary.md`: methods, subset coverage, primary/secondary findings,
limitations, commands. - `plots/`: agreement vs FLOPs; top-1 vs FLOPs;
matched-budget stage-count comparisons; allocation-family comparisons;
latency vs FLOPs with scorer overhead separate.

Show measured points and uncertainty. Do not imply unsupported
continuous trends.

## 9. Tests and execution

Add/update tests for: - Deterministic subset construction and manifest
hashing. - Class-coverage summary. - Schedule validity and
non-increasing counts. - Allocation-family generation and rounding. -
FLOPs matching tolerance. - Manifest schedules matching executed
schedules. - Per-image metrics aggregating to summary metrics. - Paired
statistical tests and multiple-comparison correction. - No-pruning
equivalence. - Existing Phase 1/2 tests remaining green.

Execution order: 1. Full test suite. 2. Small smoke test on a few
images/schedules. 3. Verify subset manifest and class coverage. 4. Full
class-diverse evaluation.

Do not launch the long run if tests, smoke test, or subset verification
fail.

## 10. Reporting
-   Update the `context\progress_tracker.md` file with a brief summary of
-    the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
-    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new
-    functionality and test results.

## 11. Out of scope

No budget optimizer, RL, differentiable routing, token merging,
fine-tuning, new scoring algorithms, or real-world speedup claims based
only on FLOPs.

## 12. Final report

Report: 1. Files changed and rationale. 2. Exact test/smoke/full
commands. 3. Subset image count, class count, per-class count stats,
manifest hash, sampling method/limitations. 4. Matched budgets,
tolerance, and unmatched groups. 5. Primary layer-2 CLS-attention
results: one vs two vs three stages. 6. Whether the result persists
across both allocation families. 7. Agreement and actual accuracy with
CIs. 8. Paired tests with Holm correction. 9. Secondary scorer results
and random seed variation. 10. FLOPs, latency, and scorer overhead. 11.
Historical comparison with the previous five-class 250-image result,
clearly labelled. 12. Limitations and recommended next step.

Do not claim that the previous layer-2 CLS-attention advantage
generalizes unless the class-diverse results support that conclusion.
