# Phase 3 — Evidence-Grounded Design of a Compute-Budget-Aware Token Allocation Algorithm

## Purpose

Analyze the completed Phase 1 and Phase 2 experiments, then design and implement a candidate algorithm that allocates visual tokens across Transformer layers under a target compute budget.

The goal is to turn empirical findings into a simple, efficient, reproducible allocation policy and test whether it improves prediction preservation over existing schedules at matched FLOPs. Begin with an audit of the raw evidence. Do not assume this specification's summaries are more authoritative than repository artifacts.

## 1. Research question

> Given a target compute budget, how should a Vision Transformer distribute its remaining patch tokens across selected pruning stages to preserve the unpruned model's predictions?

The candidate should be budget-aware, layer-aware, trajectory-aware, enforce a defensible minimum final-token count, and have low allocation overhead. Do not claim novelty until relevant published baselines and literature have been checked.

## 2. First: audit all existing results

Locate and inspect raw configs, logs, manifests, metrics, per-image predictions, code, and tests for:
- Phase 1 single-layer characterization
- Progressive multi-layer evaluation
- Controlled stage-count ablation
- Class-diverse validation and allocation robustness
- Minimum-token ablation
- Final-token vs trajectory isolation

Inspect `context/progress_tracker.md`, `context/result.md`, and experiment-specific result files. Raw metric files and reproducible calculations are the source of truth. If narrative summaries conflict with raw results, document the discrepancy; do not silently reconcile or overwrite it.

Create `results/phase3_algorithm_design/evidence_ledger.md` with columns:

| Finding | Supporting experiment | Exact metric/effect | Conditions | Confidence/limitations | Algorithm implication |
|---|---|---|---|---|---|

Cover layer sensitivity, scorer differences, progressive vs single-stage schedules, stage-count effects, geometric vs equal-drop, final-token floors, same-final-count/different-trajectory controls, FLOPs vs latency, and effects that vary by layer/budget/scorer.

Keep prediction agreement and actual top-1 accuracy separate. Agreement is not accuracy.

Classify statements as:
- **Observed:** directly in a metric file or reproducible from saved predictions.
- **Derived:** calculated from raw data with a documented method.
- **Hypothesis:** plausible explanation, not established.
- **Design choice:** proposed decision requiring evaluation.

Verify these caveats against artifacts:
- The 2,000-image class-diverse subset has two images per ImageNet class; it is not the natural validation distribution.
- The primary model so far is `deit_tiny_patch16_224`.
- Recent primary scorer is CLS attention.
- Prediction-KL is approximate gradient×activation saliency, not exact KL.
- FLOPs are analytic; CPU latency is measured separately and includes scorer overhead.
- Some secondary scorer/layer/budget combinations were omitted or reduced.
- Random scoring varies by seed.
- Final-token/trajectory isolation used 32 selected accuracy-blind controls, not exhaustive trajectory enumeration.
- Its 9/32 significant, 10 similar, and 13 inconclusive comparisons do not establish universal geometric superiority.
- Same-final-count controls were within ±0.93 pp of geometric FLOPs; inspect exact vectors and compute.
- A hand-written conclusion may exist in `result.md`; do not overwrite it by regenerating reports.

If raw evidence is missing, mark the claim unverified.

## 3. Formal optimization problem

For `m` pruning stages, let:
- `l_j`: Transformer block after which stage `j` prunes
- `K_j`: retained patch tokens after stage `j`
- `N`: initial patch-token count
- `B`: target normalized analytic FLOPs budget
- `F_min`: minimum final patch-token count

Constraints:

\[
N > K_1 > K_2 > \cdots > K_m \geq F_{\min}
\]

\[
\widehat{\operatorname{FLOPs}}(K_1,\ldots,K_m;l_1,\ldots,l_m)
\leq B\cdot\operatorname{FLOPs}_{\text{baseline}}.
\]

Optimize a calibration-set prediction-preservation proxy:

\[
\max_{K_1,\ldots,K_m}\widehat{P}_{\text{agree}}(K_1,\ldots,K_m)
\]

subject to the compute and monotonicity constraints. Report actual top-1 accuracy separately.

Do not optimize only final token count: the isolation experiment suggests intermediate counts can matter at tight budgets. Do not assume geometric allocation is optimal; it is a strong empirical baseline. FLOPs and latency are separate objectives.

## 4. Design at least two candidate algorithms

Before choosing one, compare at least two approaches in `algorithm_design.md`.

### Candidate A — Calibration-based discrete schedule search
- Generate monotone candidate schedules at selected pruning layers.
- Evaluate on a small fixed calibration subset using the frozen model and existing scorer.
- Estimate agreement relative to the unpruned model.
- Select the best candidate satisfying FLOPs and final-token constraints.
- Use held-out validation data only for final evaluation.
- Control search cost with coarse-to-fine candidate generation, limited candidate pools, or a calibration-only surrogate.
- Report search/calibration cost separately. This may be too expensive for per-request inference-time optimization.

### Candidate B — Lightweight layer-sensitivity allocation
Develop a low-overhead rule using calibration statistics, such as:
- layer-wise prediction/feature changes under token removal;
- logit sensitivity to token removal at candidate layers;
- scorer-derived importance summaries;
- measured degradation for a small set of candidate token counts;
- a minimum final-token constraint and penalty for overly aggressive intermediate reductions.

Explain exactly how signals are measured and converted into a schedule. Do not assume CLS attention scores alone measure layer sensitivity. Validate the proposed proxy against measured schedule degradation.

### Candidate C — Discrete optimization / dynamic programming
Consider this only if the objective can be represented by a reliable additive or pairwise transition cost. Do not force an additive approximation without validating it.

Compare approaches on prediction preservation, overhead, complexity, data requirements, calibration-overfitting risk, compatibility, and whether allocation is per-image or model/dataset/budget-level.

Choose one primary method and justify it using the evidence ledger.

## 5. Recommended initial scope

Unless the audit reveals a strong reason otherwise, begin with a **model-level, calibration-derived schedule**, not a per-image optimizer. This keeps inference allocation overhead low and makes comparison with fixed schedules cleaner.

Inputs:
- frozen model
- calibration subset
- selected pruning stages
- target FLOPs budget
- minimum final-token count
- maximum search/evaluation budget

Outputs:
- deterministic token-count vector
- calibration agreement estimate and measured agreement
- analytic FLOPs
- calibration runtime and memory
- scorer and configuration metadata

Freeze the schedule before held-out evaluation. If choosing another design, explain why.

## 6. Baselines and evaluation

Compare:
1. Unpruned baseline
2. Single-stage pruning at matched FLOPs
3. Original geometric
4. Constrained equal-drop
5. Unconstrained equal-drop (diagnostic)
6. Proposed budget-aware allocation

Keep model checkpoint, preprocessing, image IDs/order, pruning stages, scorer, and FLOPs accounting consistent.

### Data separation

Use the existing 2,000-image class-diverse subset to create a deterministic class-stratified split if no prior split exists; for example, 500 calibration and 1,500 validation images. Record IDs and hashes. Calibration may select schedules; validation must not influence schedule selection or tuning. If the subset is too small for a defensible split, state this and propose a separate calibration set.

### Initial budgets and settings
- Budgets: 40%, 50%, 60%, 70%
- Stage patterns: 2-6-10 and 2-8-10
- Primary scorer: CLS attention
- Floors: 16, 24, 32
- Record infeasible cases; do not silently drop them.

### Metrics
Report separately:
- prediction agreement with unpruned predictions
- actual top-1 accuracy
- top-5 if consistently supported
- analytic and normalized FLOPs
- final and intermediate token counts
- calibration/search overhead
- scorer overhead
- end-to-end CPU latency
- memory if available

Use paired confidence intervals and paired tests; Holm-correct within predefined comparison families. Primary comparison is proposed method vs geometric at matched FLOPs on held-out images.

## 7. Leakage prevention

- All schedule selection and tuning use calibration data only.
- Validation labels and predictions must not enter selection code.
- Save every candidate and its calibration score.
- Freeze schedules before validation.
- If validation is reused for iterative design, label later results exploratory and reserve a new test set for final claims.
- Report calibration-to-validation generalization gap.
- Do not claim architecture generalization until another architecture is tested.
- Do not claim superiority over published methods without faithful comparable baselines.

## 8. Implementation requirements

Inspect existing code and reuse it. Do not rewrite working Phase 1/2 modules. Add a separate Phase 3 implementation, adapting paths to repository conventions. Suggested structure:

- `src/experiments/budget_aware_allocation.py`
- `src/optimization/` (if appropriate)
- `configs/budget_aware_allocation.yaml`
- `scripts/run_budget_aware_allocation.py`
- `tests/test_budget_aware_allocation.py`
- `results/phase3_algorithm_design/evidence_ledger.md`
- `results/phase3_algorithm_design/algorithm_design.md`
- `results/phase3_budget_aware_allocation/`

Tests must cover:
1. Monotone decreasing token counts.
2. Final-token floor enforcement.
3. FLOPs calculation consistency with existing code.
4. Selected schedule does not exceed target budget, apart from documented numerical tolerance.
5. Deterministic selection for fixed inputs and seed.
6. Infeasible budget handling.
7. Exact no-pruning baseline equivalence.
8. CLS always retained.
9. Original token order preserved.
10. Calibration/validation IDs disjoint.
11. Validation data never passed to selection code.
12. Candidate records complete and reproducible.
13. Ability to reproduce geometric/equal-drop schedules when requested.
14. Serialization/reload preserves selected schedule and metadata.

Run the full test suite and a smoke test before full evaluation.

## 9. Required artifacts

### `evidence_ledger.md`
Source-grounded inventory of findings, exact effects, and limitations.

### `algorithm_design.md`
Include:
- formal problem definition
- at least two candidate algorithms
- comparison table
- chosen method and evidence-based rationale
- pseudocode
- calibration requirements
- complexity and overhead estimates
- failure modes and fallback
- explicit testable hypotheses

### Code/config
Separate implementation, config, tests, and reproducible run instructions.

### `result.md`
Include:
- exact data split and hashes
- selected schedules for each budget/stage pattern
- calibration and held-out results
- comparisons against every baseline
- paired CIs and adjusted p-values
- FLOPs, scorer/allocation overhead, latency
- infeasible cases
- calibration-to-validation gap
- limitations and next steps

### Plots
1. Token allocation by stage and budget
2. Agreement vs FLOPs
3. Top-1 accuracy vs FLOPs
4. Proposed vs geometric paired differences with CIs
5. Calibration vs validation performance
6. Allocation/scorer overhead vs end-to-end latency

## 10. Execution gates — do not skip

**Gate 1 — Evidence audit:** inspect raw results and create `evidence_ledger.md`. Report discrepancies and missing artifacts.

**Gate 2 — Algorithm proposal:** create `algorithm_design.md` with at least two candidates. Do not begin the full implementation until the design is consistent with the evidence.

**Gate 3 — Reviewable plan:** summarize selected algorithm, rationale, split, search space, candidate count, expected overhead, risks, and files to add.

**Gate 4 — Implementation/tests:** implement separately; run full tests and smoke test.

**Gate 5 — Calibration selection:** select schedules using calibration only. Save all candidates and freeze selected schedules.

**Gate 6 — Held-out evaluation:** evaluate frozen schedules only. Do not tune on validation outcomes.

**Gate 7 — Report:** produce results and plots. Preserve all previous outputs and hand-written conclusions.

## 11. Research discipline

The intended contribution is a budget-aware, layer-wise allocation algorithm, not simply a renamed schedule heuristic.

A useful result may be:
- better agreement than geometric at matched FLOPs;
- similar agreement with materially lower allocation/scorer overhead;
- identifying budgets where geometric is already near-optimal;
- or showing that current sensitivity signals are insufficient.

A negative result is acceptable if measured correctly. Do not fabricate novelty, information-theoretic guarantees, or latency speedups. Distinguish observed results, hypotheses, and design choices.

**First deliverable:** complete `evidence_ledger.md` and `algorithm_design.md`. Do not start a long full-dataset run until these are complete and the proposed algorithm can be reviewed.
