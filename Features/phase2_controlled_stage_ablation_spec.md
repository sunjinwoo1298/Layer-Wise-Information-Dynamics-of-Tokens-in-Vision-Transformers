# Phase 2 Next Step: Controlled Pruning-Stage Ablation

## Goal

Test whether distributing token pruning across multiple Transformer
layers improves prediction preservation compared with pruning once,
**when the first pruning layer, scorer, evaluation images, and analytic
FLOPs budget are controlled**.

Primary question: With the first pruning layer fixed and total analytic
FLOPs matched, does using two or three pruning stages preserve
predictions better than using one stage?

This is an evaluation/ablation task only. Do not implement a budget
optimizer, token merging, fine-tuning, learned routing, or new scoring
methods.

## 1. Inspect and preserve existing code

Before coding: 1. Inspect Phase 1 and Phase 2 code, configs, tests, and
`results/phase2_progressive_eval/summary.md`. 2. Reuse existing
model/data loading, progressive forward pass, scorers, FLOPs calculator,
metrics, and output conventions. 3. Preserve working Phase 1/2 behavior;
avoid unnecessary refactoring. 4. Confirm layer numbering and schedule
semantics from code and document them. 5. Do not silently change scorer
behavior.

## 2. Controlled design

Fix the **first pruning layer** for all primary schedules. Use layer 2
if supported.

Compare: - One stage: prune only at the fixed first layer. - Two stages:
prune at the fixed first layer and one later layer. - Three stages:
prune at the fixed first layer and two later layers.

Example structures (counts are patch tokens, excluding CLS):

``` yaml
one_stage:   {2: K}
two_stage:   {2: K1, 6: K2}
three_stage: {2: K1, 4: K2, 8: K3}
```

Choose counts using a documented schedule-generation procedure so
complete schedules have matched analytic FLOPs. These examples are
structures, not prescribed optimal counts.

Token counts must be non-increasing. Do not prune after the final block.

If practical, add a secondary analysis fixing the first pruning layer at
layer 4. Keep it separate from the primary layer-2 analysis.

## 3. Match compute budgets

1.  Calculate complete-schedule FLOPs using the existing calculator.
2.  Predefine a matching tolerance (target: ±1 percentage point of
    baseline FLOPs, if feasible).
3.  Generate enough one-, two-, and three-stage schedules to create
    overlapping FLOPs budgets.
4.  Match one-, two-, and three-stage schedules at each budget.
5.  Record exact normalized FLOPs and pair/group differences.
6.  Report unmatched schedules/budgets rather than loosening tolerance
    after seeing results.
7.  Use the same FLOPs convention and baseline for every schedule.

Match by analytic FLOPs, not measured latency.

## 4. Scorers

Primary scorer: `cls_attention`.

Secondary scorers: - `random` - `prediction_kl_approx`

Keep scorer comparisons separate from stage-count comparisons.

`prediction_kl_approx` is gradient-times-activation saliency, not exact
KL. Report its selection overhead and end-to-end runtime. Preserve
existing CLS-attention semantics. For random pruning, use fixed image
order and multiple seeds; record seed and generator behavior. Use common
seeds across schedule groups where practical.

Report unsupported scorer/schedule combinations explicitly; do not
silently skip or substitute.

## 5. Data and reproducibility

Use the same Phase 1/2 evaluation setup: - Same 250 ImageNet validation
images and same order. - Same `deit_tiny_patch16_224` checkpoint. - Same
preprocessing and label mapping. - Primary seed 42. - Additional random
seeds at least 43 and 44.

Verify the subset manifest against the previous evaluation. If it cannot
be recovered, stop and report this rather than claiming an identical
subset. A larger subset may be a separate extension after the primary
experiment; do not combine results.

## 6. Metrics

For each schedule × scorer × seed, collect:

**Prediction** - Agreement with unpruned top-1 predictions. - Actual
top-1 accuracy against ImageNet labels. - Top-5 if already supported. -
Counts and number of images.

**Compute** - Analytic FLOPs for the full compressed path. - Normalized
FLOPs (% baseline). - FLOPs saved (%).

**Runtime** - End-to-end latency including scoring/selection. - Scorer
overhead separately where measurable. - Device, batch size, warm-up,
timed-run count, and timing method. - Keep CPU and GPU results separate.

**Diagnostics** - Token counts after every block/pruning point. - Exact
schedule, stage count, first pruning layer, scorer, and seed.

Do not infer latency gains from FLOPs savings.

## 7. Statistical analysis

-   Report counts and proportions.
-   Provide 95% confidence intervals for agreement and top-1 accuracy.
-   Use paired comparisons on the same images where appropriate: McNemar
    for correctness and/or paired bootstrap over images.
-   Report random seed-to-seed variation separately from image-level
    uncertainty.
-   Correct multiple comparisons in the planned primary family
    (e.g. Holm correction); distinguish corrected and uncorrected
    p-values.
-   Predefine the primary comparison: CLS attention, first pruning layer
    2, matched FLOPs, one vs two vs three stages.
-   Treat other scorers and layer-4 analysis as secondary/exploratory.

Do not claim significance based only on win counts.

## 8. Output artifacts

Create `results/phase2_stage_ablation/`. Do not overwrite prior results.

Save: - `metrics.csv`: one row per schedule × scorer × seed. -
`per_image_metrics.csv`: image ID, baseline/compressed predictions,
label, correctness, agreement. - `token_counts.csv`: token counts at
every block/pruning point. - `schedule_manifest.json`: schedules, stage
counts, first layer, exact FLOPs, matched groups, unmatched schedules. -
`run_config.json`: checkpoint, subset, preprocessing, seeds, scorer,
timing, environment. - `statistical_tests.csv`: effect sizes, confidence
intervals, raw and corrected p-values. - `summary.md`: methods,
findings, matched comparisons, limitations, commands. - `plots/`:
agreement vs FLOPs; top-1 accuracy vs FLOPs; matched-budget
one/two/three-stage comparison; latency vs FLOPs with scorer overhead
clearly shown.

Show measured points and uncertainty. Do not imply continuous trends
from discrete schedules.


## 9. Reporting
-   Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new functionality and test results.



## 10. Tests and validation

Add tests for schedule generation/matching and aggregation. Verify: -
Manifest schedules exactly match executed schedules. - Primary schedules
share the same first pruning layer. - Stage counts and token counts are
correct and non-increasing. - FLOPs use the same baseline/convention. -
Matched groups satisfy the predefined tolerance. - Per-image metrics
aggregate to `metrics.csv`. - No-pruning remains identical to the
original model. - Random selections reproduce for fixed seed/order,
subject to backend determinism. - Existing Phase 1/2 tests still pass.

Run tests and a small smoke test before the full 250-image evaluation.

## 11. Out of scope

No greedy/learned budget allocation, RL, differentiable routing,
merging, fine-tuning, new scorers, or real-world speedup claims based
only on FLOPs.

## 12. Final report

Report: 1. Files changed and rationale. 2. Exact smoke/full commands and
test results. 3. Confirmation of the same 250-image subset/order. 4.
Schedule-generation method and stage counts. 5. Matched FLOPs budgets,
tolerance, and unmatched groups. 6. Primary CLS-attention result: one vs
two vs three stages, first layer fixed at 2. 7. Secondary scorer/layer-4
results separately. 8. Agreement and actual accuracy with confidence
intervals. 9. Paired tests with multiple-comparison correction. 10.
FLOPs, latency, and scorer overhead. 11. Limitations and recommended
next step.

Do not claim more pruning stages are better unless the controlled,
matched-compute results support that conclusion.
