# Next Experiment: Minimum Final-Token Constraint and Allocation Robustness

## Goal

Test whether poor equal-drop progressive-pruning performance is
primarily caused by extremely small final token counts, or whether
geometric allocation has an advantage beyond avoiding near-empty final
representations.

**Primary question:** Under matched analytic FLOPs, does imposing a
minimum final patch-token count improve equal-drop pruning, and can
constrained equal-drop close the gap to geometric allocation?

This is an evaluation/ablation task only. Do not implement a budget
optimizer, token merging, fine-tuning, learned routing, or new scoring
algorithms.

## 1. Inspect and preserve existing code

Before coding: 1. Inspect Phase 1/2 code, the class-diverse validation
runner/configs/tests, and
`results/phase2_class_diverse_validation/summary.md`. 2. Reuse existing
model/data loading, subset manifest, progressive forward pass, scorers,
FLOPs calculator, metrics, statistical tests, and output conventions. 3.
Preserve prior outputs and behavior. Write to a new results directory.
4. Confirm layer numbering, schedule semantics, and allocation-family
definitions from code. 5. Do not silently modify the existing geometric
or equal-drop rules.

## 2. Dataset and model

Use exactly the same class-diverse 2,000-image ImageNet subset: - Same
image identities, labels, order, and manifest hash. - Same
`deit_tiny_patch16_224` checkpoint. - Same preprocessing and label
mapping. - Primary seed 42. - Primary scorer: `cls_attention`.

Verify the subset manifest before running. If it cannot be reproduced
exactly, stop and report the discrepancy rather than silently
substituting another subset.

Secondary scorers: - `random` (at least seeds 42, 43, 44). -
`prediction_kl_approx` where practical.

Keep secondary analyses separate. `prediction_kl_approx` is
gradient-times-activation saliency, not exact KL; report its scoring
overhead.

## 3. Allocation families

Fix the first pruning layer at layer 2 for the primary experiment.
Compare:

1.  **Geometric:** reproduce the exact rule from the previous
    class-diverse experiment.
2.  **Equal-drop:** reproduce the exact rule from the previous
    experiment.
3.  **Constrained equal-drop:** use equal-drop allocation while
    enforcing a minimum number of patch tokens at the final pruning
    point.

Do not change the original geometric or equal-drop definitions.
Constrained equal-drop is a separate family.

### Final-token floors

Test final patch-token floors of: - 16 - 24 - 32

If feasible, include 8 as a diagnostic floor. The floor applies to patch
tokens after the final pruning point, excluding CLS. Record this
explicitly.

### Constrained allocation rule

Document precisely how intermediate counts are calculated. For
example: - Start from initial patch-token count and target final
count. - Distribute the absolute token reduction as evenly as possible
across pruning intervals. - Apply deterministic integer rounding. -
Enforce non-increasing counts. - Enforce the selected final-token
floor. - Ensure the resulting schedule is evaluated at the intended
FLOPs budget.

If a floor and target FLOPs budget cannot both be satisfied, mark the
combination infeasible. Do not silently violate the floor or FLOPs
tolerance.

## 4. Experimental design

Primary settings: - First pruning layer fixed at 2. - Scorer:
`cls_attention`. - FLOPs budgets: 40%, 50%, 60%, and 70% of baseline, as
in the previous study. - Two-stage and three-stage progressive
schedules. - Geometric, equal-drop, and constrained equal-drop with each
feasible floor. - Matched one-stage schedules as baseline comparators.

Primary comparison: constrained vs unconstrained equal-drop at the same
stage count and matched FLOPs.

Secondary comparison: constrained equal-drop vs geometric at the same
stage count and matched FLOPs.

Use the same later pruning-layer placements as the previous experiment
wherever possible, so allocation rule is the main changing factor. If
placements differ, compare like with like and record them.

Do not choose schedules after inspecting accuracy.

## 5. FLOPs matching

1.  Use the existing full-path analytic FLOPs calculator.
2.  Predefine tolerance: ±1 percentage point of baseline FLOPs.
3.  Match constrained and unconstrained schedules within this tolerance.
4.  Record exact normalized FLOPs and pairwise differences.
5.  Report infeasible/unmatched combinations explicitly; do not relax
    tolerance after seeing results.
6.  Use the same baseline and FLOPs accounting convention for all
    schedules.

A floor may prevent a schedule from reaching a target budget. Report
that trade-off transparently.

## 6. Metrics

For every schedule × family × floor × scorer × seed, collect:

**Prediction** - Agreement with unpruned model top-1 predictions. -
Actual top-1 accuracy against ImageNet labels. - Top-5 if supported. -
Correct/incorrect counts and image count. - 95% confidence intervals.

**Compute** - Full-path analytic FLOPs. - Normalized FLOPs and FLOPs
saved. - Exact matched-budget differences. - Final patch-token count and
floor satisfaction.

**Runtime** - End-to-end latency including scoring/selection. - Scorer
overhead separately where measurable. - Device, batch size, warm-up
count, timed-run count, timing method. - Keep CPU and GPU timing
separate; use repeated representative batches.

**Metadata** - Checkpoint, subset manifest hash, seed, scorer, schedule,
family, floor, stage count, pruning layers, allocation rule, rounding
policy, software/device details.

Do not infer real-world speedup from FLOPs savings.

## 7. Statistical analysis

Primary outcomes: prediction agreement and actual top-1 accuracy.

Use paired comparisons on the same 2,000 images: - Constrained
equal-drop vs unconstrained equal-drop. - Constrained equal-drop vs
geometric. - Match stage count, pruning-layer placements, scorer, and
FLOPs as closely as possible.

Use paired bootstrap over images, McNemar's test for paired correctness
where appropriate, effect sizes, and 95% confidence intervals. Apply
Holm correction within predefined comparison families. Separate random
seed-to-seed variation from image-level uncertainty.

Keep secondary scorers and any layer-4 experiment exploratory. Do not
pool them into the primary CLS-attention layer-2 analysis.

## 8. Output artifacts

Create `results/phase2_minimum_token_ablation/`. Do not overwrite
previous results.

Save: - `metrics.csv`: one row per schedule × family × floor × scorer ×
seed. - `per_image_metrics.csv`: image ID, label, baseline/compressed
predictions, correctness, agreement. - `token_counts.csv`: patch-token
counts at every block/pruning point. - `schedule_manifest.json`:
schedules, family, floor, stage count, layer placements, FLOPs, matched
pairs, infeasible cases. - `run_config.json`: checkpoint, subset
manifest hash, seeds, scorer, timing, environment. -
`statistical_tests.csv`: effect sizes, CIs, raw/corrected p-values. -
`summary.md`: methods, results, floor feasibility, comparisons,
limitations, commands. - `plots/`: agreement vs FLOPs; top-1 vs FLOPs;
performance vs final token count; constrained vs unconstrained
equal-drop; constrained equal-drop vs geometric; latency vs FLOPs with
scorer overhead separated.

Show measured points and uncertainty. Clearly label infeasible
combinations.


## 9. Reporting
-   Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new functionality and test results.


## 10. Tests and validation

Add/update tests for: - Exact reproduction of prior geometric and
equal-drop schedules. - Constrained equal-drop floor enforcement. -
Deterministic integer rounding. - Non-increasing token counts and valid
bounds. - CLS retention. - FLOPs tolerance and infeasible floor/budget
handling. - Manifest schedules matching executed schedules. - Per-image
metrics aggregating to summary metrics. - Paired statistical
calculations and multiple-comparison correction. - No-pruning
equivalence. - Existing Phase 1/2 tests remaining green.

Run the full test suite and a small smoke test before the full
2,000-image run.

## 11. Out of scope

No budget optimizer, RL, differentiable routing, token merging,
fine-tuning, new scoring algorithms, or real-world speedup claims based
only on FLOPs.

## 12. Final report

Report: 1. Files changed and rationale. 2. Exact test, smoke, and
full-run commands. 3. Confirmation of the same 2,000-image subset and
manifest hash. 4. Exact geometric/equal-drop rules reused. 5.
Constrained equal-drop rule and rounding policy. 6. Results for floors
8/16/24/32 and infeasible combinations. 7. Matched FLOPs budgets and
tolerance. 8. Primary constrained-vs-unconstrained equal-drop results.
9. Comparison with geometric at matched FLOPs. 10. Agreement and actual
accuracy with confidence intervals. 11. Paired tests with Holm
correction. 12. FLOPs, latency, and scorer overhead. 13. Limitations and
next research step.

Do not claim the minimum-token constraint explains equal-drop failures
unless controlled matched-FLOPs results support that conclusion.
