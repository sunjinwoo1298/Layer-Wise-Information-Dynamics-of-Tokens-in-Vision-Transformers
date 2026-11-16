# Next Experiment: Isolate the Effect of the Final Token Floor

## Objective

Run one final controlled allocation-characterization experiment before implementing a budget-aware token-allocation method.

The previous minimum-token ablation found that imposing a final-token floor substantially repaired equal-drop schedules, but constrained equal-drop still underperformed geometric allocation overall. Because the constrained family re-searched its drop amount to match FLOPs, earlier token counts also changed. The previous experiment therefore did not isolate final token count from the full reduction trajectory.

Research question:

> At matched compute, is geometric allocation better mainly because it preserves more tokens at the final stage, or does the shape of the entire layer-wise reduction trajectory matter independently?

Do not claim a causal effect unless the schedule construction isolates the intended variable.

## 1. Fixed setup

Keep fixed:

- Model: `deit_tiny_patch16_224`
- Dataset: exact same class-diverse 2,000-image subset used in the previous minimum-token experiment
- Manifest and image order: reuse prior manifest; verify its full SHA-256 from the project artifact. The previously reported checksum began `56279dc3` and ended `263aa67`; do not rely on this partial checksum.
- Seed: `42`
- Primary scorer: CLS attention
- First pruning layer: layer 2
- Stage patterns: reuse the prior 2-6-10 and 2-8-10 patterns
- FLOPs budgets: 40%, 50%, 60%, 70% of baseline
- FLOPs matching tolerance: ±1 percentage point
- Always retain CLS; preserve original token order
- No fine-tuning, token merging, or optimizer
- Primary metric: agreement with unpruned-model predictions
- Secondary metric: actual top-1 accuracy
- Report analytic FLOPs and measured CPU latency separately

Inspect the prior config, schedule generator, manifest, and results first. Reuse existing conventions. Do not silently change data, preprocessing, weights, image order, scoring, or FLOPs accounting.

If the exact 2,000-image manifest cannot be found and verified, stop and report that. Do not substitute another subset.

## 2. Schedule families

For every feasible matched-FLOPs comparison, evaluate these four families.

### A. Original geometric

Reuse the exact previous implementation:

\[
K_j = \max(1,\operatorname{round}(N\rho^j))
\]

Reuse prior schedule definitions and rounding behavior.

### B. Final-count-matched geometric control

This is a control, not a proposed method.

For each original geometric schedule:
1. Record its complete per-stage token-count vector and final token count.
2. Construct alternative schedules with the same stage locations, same initial count, and exactly the same final count.
3. Vary only intermediate token counts, with strictly decreasing valid integer counts.
4. Search for alternatives whose analytic FLOPs match the original geometric schedule within ±1 percentage point.
5. Do not count a vector identical to geometric as a distinct control.
6. Record every candidate, its full vector, FLOPs, and feasibility.

It may be impossible to change intermediate counts while holding both final count and FLOPs fixed. If no valid alternative exists, mark it infeasible. Do not relax final-count equality or FLOPs tolerance.

This comparison asks whether a different trajectory can achieve similar FLOPs while preserving the same final representation size.

### C. Constrained equal-drop

Reuse the exact previous implementation:

\[
K_j = \max(F,\operatorname{round}(N-jD))
\]

Re-search `D` using the previous grid, rounding, and FLOPs-matching behavior. Primary floors: 16, 24, and 32. Floor 8 may be included as a diagnostic if compute permits.

Do not modify the previous implementation. Record infeasible schedules as infeasible.

### D. Original unconstrained equal-drop

Reuse the exact previous implementation:

\[
K_j = \max(1,\operatorname{round}(N-jD))
\]

This is the negative-control family expected to expose failures when the final count becomes too small.

## 3. Comparison design

For each budget and stage pattern:

1. Original geometric vs constrained equal-drop.
2. Original geometric vs unconstrained equal-drop.
3. Constrained equal-drop vs unconstrained equal-drop.
4. Original geometric vs final-count-matched geometric controls, where feasible.

Use paired predictions on the exact same 2,000 images. Match schedule pairs within ±1 percentage point of baseline FLOPs. Record exact schedule vectors and actual FLOPs differences.

Do not pool budgets or stage patterns into one headline estimate. Report each budget × stage-pattern comparison separately, followed by a clearly labelled descriptive aggregate.

### Critical isolation check

For every comparison, show:
- Full token-count vector at each stage
- Final token count
- FLOPs percentage
- Whether the trajectory differs from geometric
- Whether FLOPs tolerance is satisfied

The final-count-matched geometric control is the key new part. If no valid controls can be constructed, explicitly state that the final-count explanation remains unresolved. Do not infer the explanation from the other comparisons alone.

## 4. Statistics

Follow prior project conventions:

- Paired bootstrap confidence intervals for agreement and top-1 differences
- Exact McNemar tests for paired outcomes where appropriate
- Holm correction within each predefined comparison family
- Wilson intervals for individual proportions
- Report effect sizes in percentage points, raw p-values, and adjusted p-values
- Preserve per-image predictions/logits needed to reproduce all comparisons

Predefine comparison families before inspecting results. Label any post-hoc comparisons exploratory. Report infeasible comparisons and reasons; do not replace them with easier comparisons after seeing outcomes.

## 5. Runtime and implementation

- Reuse the existing evaluation runner and metric utilities where possible.
- Add a separate module/config/script; do not overwrite previous experiment outputs.
- Save a machine-readable schedule table, including infeasible candidates and reasons.
- Save per-image prediction artifacts sufficient to reproduce paired statistics.
- Save a run manifest with model/checkpoint provenance, dataset manifest and full checksum, image IDs/order, transforms, seed, scorer version, schedule vectors, FLOPs method, software versions, hardware/device, and runtime.

### Latency

Measure only as a secondary systems metric:
- CPU only, matching the prior device and batch size
- Same warm-up and repeated-batch procedure
- Report baseline, scorer overhead, and end-to-end latency separately
- Never infer latency from FLOPs
- State that CPU results do not establish GPU speedups

If the prior timing procedure cannot be recovered, document the deviation before running.

## 6. Tests and validation

Add tests verifying:

1. Original geometric vectors exactly reproduce prior schedules.
2. Original unconstrained equal-drop vectors exactly reproduce prior schedules.
3. Constrained equal-drop vectors exactly reproduce prior schedules.
4. Final-count-matched controls preserve geometric final count exactly.
5. Controls preserve stage locations and initial token count.
6. Control trajectories are strictly decreasing and valid.
7. Identical-to-geometric controls are not counted as distinct.
8. FLOPs matching uses the existing accounting and ±1 pp tolerance.
9. Infeasible controls are recorded without relaxing constraints.
10. The exact prior manifest and image order are used.
11. No-pruning baseline equivalence remains exact.
12. Repeated runs with same seed and input order are reproducible.

Run the full existing test suite plus new tests. Do not start the full run if any test fails.

## 7. Deliverables

Create a new experiment output directory containing:

- Config, runner script, experiment module, and tests
- Complete schedule/candidate table
- Per-image prediction artifacts
- Metrics CSV/JSON
- Statistical comparison table with raw and Holm-adjusted p-values
- Latency report
- Run manifest
- Plots:
  1. Token count by stage for all feasible schedules
  2. Agreement vs FLOPs by family
  3. Top-1 accuracy vs FLOPs by family
  4. Paired agreement differences with confidence intervals
  5. Final token count vs agreement
- Concise `result.md` covering findings, infeasibility, limitations, and whether the isolation question was answered

## 8. Interpretation rules

- If final-count-matched alternative trajectories perform similarly to geometric at matched FLOPs, this is evidence that final representation size may explain a substantial part of geometric's advantage.
- If geometric consistently outperforms genuinely different final-count-matched controls, this suggests intermediate trajectory matters too.
- If no valid final-count-matched controls are feasible, say the experiment could not isolate final count from trajectory.
- If constrained equal-drop improves over unconstrained equal-drop but remains below geometric, conclude only that a minimum-token constraint repairs part of the failure.
- If results vary by budget or stage pattern, report the dependence rather than averaging it away.

Do not implement the budget-aware optimizer in this experiment. The goal is to complete controlled characterization and establish a defensible design target.

## 9. Reporting
-   Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new functionality and test results.

## 10. Claude Code execution instructions

First inspect the repository and previous minimum-token artifacts. Before the full run, report:

1. Exact manifest path and verified full checksum
2. Schedule patterns and budgets found in prior artifacts
3. Number of feasible final-count-matched control schedules
4. Test results
5. Deviations from this specification

Proceed to the full run only if the manifest is verified and all tests pass.

At completion, explicitly answer:

> Does the evidence indicate that geometric allocation wins mainly because of its final token count, because of its intermediate trajectory, or is the distinction still not identifiable under matched-FLOPs constraints?

Do not implement an optimizer yet. That is the following step, after this experiment is reviewed.
