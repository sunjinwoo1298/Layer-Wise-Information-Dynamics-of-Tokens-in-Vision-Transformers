# Phase 2 --- Progressive Multi-Layer Token Pruning

## Goal

Extend the existing DeiT-Tiny Phase 1 code so a single forward pass can
prune a **different number of patch tokens at multiple selected
Transformer layers**. This milestone is infrastructure only: do not
implement the budget optimizer yet.

## Preserve the Phase 1 setup

-   Reuse the existing pretrained `deit_tiny_patch16_224` model,
    preprocessing, dataset loader, metrics, and result conventions.
-   Keep the original model/checkpoint unchanged.
-   Preserve the CLS token at every layer. Token counts below refer to
    **patch tokens only** (initially 196).
-   Keep retained patch tokens in their original sequence order.
-   Do not silently change the Phase 1 scoring implementations or metric
    definitions.

## Required functionality

### 1. Retention schedule

Add a configurable schedule, for example:

``` yaml
pruning:
  schedule:
    2: 196
    4: 147
    6: 98
    8: 73
    10: 49
  scorer: random  # random | prediction_kl_approx | cls_attention
```

The values are illustrative. Support arbitrary valid schedules,
including pruning at only one layer.

-   Layer keys must map unambiguously to the existing model's block
    indexing. Document whether the configuration uses 1-based
    human-readable layer numbers or 0-based block indices; convert in
    one place only.
-   A schedule means: after the specified block/layer, retain exactly
    `k` patch tokens. The reduced sequence is passed to all subsequent
    blocks.
-   Validate that layer keys are valid, token counts are integers,
    `1 <= k <= current_token_count`, and the schedule never increases
    the token count.
-   If a layer is omitted, do not prune there.
-   Do not prune after the final block as a compute-saving operation; it
    has no downstream Transformer blocks.

### 2. Pruning and token bookkeeping

-   Implement a reusable pruning function that takes the token tensor,
    patch-token scores, and `k`.
-   Keep CLS separate and always retain it.
-   Select the top-k patch tokens, then restore their original sequence
    order before continuing the forward pass.
-   Keep positional information aligned with the retained tokens. For
    DeiT's standard absolute positional embeddings, embeddings are
    already incorporated into token representations; do not add
    positional embeddings a second time.
-   Ensure batch dimensions and per-image selections are handled
    correctly.
-   Make scorer randomness reproducible using the configured seed.
-   Avoid padding or masking tokens while claiming computational
    savings: subsequent blocks must receive a genuinely shorter
    sequence.

### 3. Scorers

Reuse the existing Phase 1 scorer code where possible: - `random` -
`prediction_kl_approx` (retain the existing implementation and label; do
not call it exact KL) - `cls_attention`

If a scorer cannot operate correctly at a particular point in the
forward pass, fail clearly with an explanatory error rather than
silently substituting another scorer.

### 4. Forward-pass behavior

Implement a progressive forward path that: 1. Runs the model normally
until a scheduled pruning point. 2. Computes scores for the current
patch tokens. 3. Retains CLS plus the requested top-k patch tokens. 4.
Passes the shortened sequence through the remaining Transformer blocks.
5. Produces logits through the original model head.

Also provide a no-pruning path. With an empty schedule (or all counts
equal to the current count), logits must match the original model within
numerical tolerance (target max absolute difference `<= 1e-4` in
evaluation mode).

### 5. Tests and validation

Add tests for: - Empty schedule reproduces original logits (max absolute
difference `<= 1e-4`). - CLS token is always retained. - Exactly the
requested number of patch tokens remains after each pruning point. -
Retained tokens preserve original relative order. - Multiple pruning
points work in one forward pass and token counts never increase. -
Invalid layers, invalid counts, and increasing schedules raise clear
errors. - Same seed produces reproducible random selections. - A
one-layer schedule matches the existing Phase 1 single-layer pruning
behavior, where applicable.

Add a small smoke-test command using a 10 image  and a schedule such as
`{2: 98, 4: 49}`. Save logits/metrics and a record of the token count
after each block or pruning point.

## Outputs

-   A modular implementation integrated into the existing repository.
-   A configuration file for progressive pruning.
-   Unit tests and a runnable smoke test.
-   Update the `context\progress_tracker.md` file with a brief summary of the new functionality and test results.
-   Results saved separately from Phase 1; do not overwrite existing
    Phase 1 outputs.
-   Update the `context\result.md` file with a brief summary of the new functionality and test results.

## Scope limits

Do **not** implement the Phase 2 allocation optimizer, greedy search,
reinforcement learning, differentiable routing, token merging, or model
fine-tuning in this task.

Do not claim speedups from FLOPs estimates alone. For this milestone,
correctness of progressive pruning is the priority; report analytic
FLOPs separately from measured latency and include scorer overhead if
latency is measured.

## Final report from the coding agent

When finished, report: 1. Files created or modified. 2. Exact command to
run the smoke test. 3. Test results, including baseline-logit
equivalence. 4. Example schedule and observed token counts. 5. Any
limitations or scorer compatibility issues.

Do not fabricate results. If any requirement cannot be implemented
reliably, explain why and leave it unimplemented rather than silently
approximating it.
