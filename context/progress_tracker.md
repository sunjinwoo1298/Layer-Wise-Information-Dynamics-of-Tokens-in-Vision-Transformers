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

## Results so far (ImageNet dev-250, `deit_tiny_patch16_224`)

- 250 samples, 6 layers (2–12), 8 retention ratios.
- Early layers (L2–L4) are very sensitive to pruning. Deep layers (L8–L10) are robust: about 94% or more agreement at 12.5% of tokens.
- Prediction-KL (approx) is the best scorer and is significant vs. random (p < 0.001) in most settings.
- CLS-Attention is about the same as random at L2 and improves in middle and late layers.
- Marginal gain ΔP/ΔC is high in early layers and near zero in late layers above 50% tokens.
- Theoretical FLOPs savings are larger than measured latency savings.
