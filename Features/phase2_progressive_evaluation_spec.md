# Phase 2 Next Step: Controlled Progressive-Pruning Evaluation

## Goal

Build a reproducible experiment runner to evaluate multiple progressive
token-pruning schedules on the same evaluation images.

**Research question:** Does progressive multi-layer pruning preserve
predictions better than single-layer pruning at matched analytic FLOPs
budgets?

This task is evaluation infrastructure only. Do not implement a budget
optimizer, token merging, fine-tuning, learned routing, or a new pruning
method.

## 1. Inspect and preserve the project

Before coding: 1. Inspect Phase 1 and Phase 2 implementations, configs,
tests, and output formats. 2. Reuse existing model loading, ImageNet
preprocessing, data loading, scorers, progressive forward pass, FLOPs
calculation, and metrics. 3. Preserve Phase 1 and Phase 2 behavior;
avoid unnecessary refactoring. 4. Confirm layer numbering and schedule
semantics from the code and document them. 5. Do not silently change the
existing scorer implementations.

## 2. Evaluation setup

Use the Phase 1 setup where possible: - Model: pretrained
`deit_tiny_patch16_224`. - Dataset: ImageNet validation. - Subset: same
250 images as Phase 1, with identical identities and order if
available. - Primary seed: 42. - Scorers: `random`, `cls_attention`,
`prediction_kl_approx`. - Evaluation mode: `model.eval()` and existing
evaluation behavior.

If the exact Phase 1 subset cannot be recovered, do not claim a
different subset is the same 250-image set. Make the new subset
definition explicit and reproducible.

For random pruning, record the seed and image order. Explain that the
generator may advance across pruning calls. If additional seeds are run,
keep image order fixed and record each seed.

## 3. Schedules

Evaluate at least these schedules (confirm layer-key convention first):

``` yaml
no_pruning: {}
early_single: {2: 98}
middle_single: {6: 98}
late_single: {10: 98}
early_progressive: {2: 98, 4: 49}
gradual_progressive: {2: 147, 6: 98, 10: 49}
delayed_progressive: {2: 196, 4: 98, 8: 49}
```

These are candidate schedules, not known-optimal schedules.

Add further single-layer and progressive schedules if necessary to
create overlapping analytic FLOPs budgets. This is essential for fair
comparisons.

For every schedule: - Keep CLS; `k` means patch tokens retained. -
Preserve existing token order and positional-information behavior. -
Record token counts after each block or at every pruning point. - Reject
invalid or increasing schedules. - Do not prune after the final block
for compute savings.

## 4. Scorers and controls

Evaluate the existing scorers where supported: - `random` -
`cls_attention` - `prediction_kl_approx`

Include the unpruned baseline. Explicitly report unsupported
combinations.

`prediction_kl_approx` is the existing gradient-times-activation
saliency approximation, **not exact KL**. Include its selection
overhead; do not relabel it as true KL. Do not silently change the Phase
1 `cls_attention` behavior.

## 5. Metrics

For every schedule × scorer × seed, collect:

**Prediction** - Agreement with unpruned model top-1 predictions. -
Actual top-1 accuracy against ImageNet labels. - Top-5 accuracy if
already supported. - Number of evaluated images.

**Compute** - Analytic FLOPs for the complete compressed forward path. -
Normalized FLOPs (% of unpruned baseline). - FLOPs savings (%).

**Runtime** - End-to-end latency including scoring/selection. - Scorer
overhead separately where measurable. - Batch size, device, warm-up
count, timed-run count, and timing method. - Keep CPU and GPU
measurements separate.

**Reproducibility** - Checkpoint/model identifier. - Dataset/subset
identity and saved image manifest or ordering. - Seed, scorer,
schedule. - Software/device information where available. - Layer
numbering and schedule semantics.

Do not infer latency savings from FLOPs savings. Report measured runtime
and analytic FLOPs separately.

## 6. Matched-compute analysis (central requirement)

1.  Compute normalized FLOPs for every schedule.
2.  Match single-layer and progressive schedules with sufficiently
    similar FLOPs.
3.  Define a matching tolerance before comparison (e.g. within 2
    percentage points of baseline FLOPs, if feasible).
4.  If candidate schedules do not overlap, add schedules to create
    comparable budgets.
5.  Compare prediction agreement and actual top-1 accuracy for each
    matched pair.
6.  Report progressive-minus-single-layer differences for both metrics.
7.  Keep scorer consistent within comparisons; do not attribute scorer
    differences to schedule structure.
8.  Include a table of matched pairs and their compute differences.

Do not assume progressive pruning wins. Neutral or negative results are
valid.

## 7. Statistical reporting

-   Report counts and proportions, not percentages alone.
-   Include 95% confidence intervals for agreement and actual accuracy
    where practical.
-   Use paired comparisons on the same images where practical
    (e.g. McNemar's test for correctness or paired bootstrap over
    images).
-   Label exploratory results and avoid overstating significance with
    250 images.
-   If multiple seeds are run, report seed-to-seed variation separately
    from image-level uncertainty.

## 8. Output files

Create a new directory such as `results/phase2_progressive_eval/`. Do
not overwrite smoke-test or Phase 1 results.

Save: - `metrics.csv`: one row per schedule × scorer × seed with
aggregate metrics. - `per_image_metrics.csv`: image identifier, baseline
prediction, compressed prediction, ground-truth label, correctness, and
agreement. - `token_counts.csv`: token counts at each block/pruning
point. - `schedule_manifest.json`: exact schedules, layer numbering,
semantics, and matched-budget pairs. - `run_config.json`: model, subset,
seed, scorer, timing, and environment settings. - `summary.md`:
findings, matched-budget comparisons, limitations, and commands. -
`plots/` containing: - Prediction agreement vs normalized FLOPs. -
Actual top-1 accuracy vs normalized FLOPs. - Latency vs normalized
FLOPs, with scorer overhead shown clearly or separately. -
Matched-budget progressive vs single-layer comparisons.

Use clear labels and units. Show discrete measured points; do not imply
unsupported smooth trends.

## 9. Reporting
-   Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new functionality and test results.

## 10. Tests and validation

Add suitable tests. Verify: - No-pruning baseline matches the original
model. - Manifest schedules exactly match schedules actually run. -
Recorded token counts match each schedule. - FLOPs are normalized
against the same baseline. - Per-image metrics aggregate to
`metrics.csv`. - Matched pairs satisfy the documented tolerance. - Same
seed and image order reproduce random selections, subject to backend
determinism. - Existing Phase 1 and Phase 2 tests still pass.

Run a small smoke test first. Run the full 250-image evaluation only
after smoke tests and tests pass.

## 11. Out of scope

Do not implement: - Greedy or learned budget allocation. - RL or
differentiable routing. - Token merging. - Fine-tuning. - New scoring
algorithms. - Real-world speedup claims based only on FLOPs.

## 12. Final report from Claude Code

Report: 1. Files changed and rationale. 2. Exact smoke/full commands. 3.
Test results. 4. Subset identity and image count. 5. Compact metrics
table for all schedules. 6. Matched-compute progressive vs single-layer
comparisons. 7. Agreement and actual accuracy with uncertainty. 8. FLOPs
and latency, with scorer overhead separated. 9. Failed, skipped, or
unsupported combinations. 10. Limitations and recommended next
experiment.

Do not claim progressive pruning is superior unless matched-compute
results support that conclusion.
