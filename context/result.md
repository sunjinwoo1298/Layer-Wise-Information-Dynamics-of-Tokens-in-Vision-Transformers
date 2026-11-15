# Phase 1 Results

**Setup:** pretrained `deit_tiny_patch16_224` (12 layers, 196 patch tokens), 250 ImageNet-val images, seed 42. Baseline top-1 is 76.4% and top-5 is 94.8%.
Layers `l ∈ {2,4,6,8,10,12}` × 8 retention ratios (100% → 12.5%) × 3 scorers.
Raw outputs are in `results/deit_tiny_imagenet_dev250/` (`layer_metrics.csv`, `marginal_efficiency.csv`, `summary.txt`, `plots/`).

## How it works

1. Run blocks `0..l-1` on the full input and capture `[CLS]` plus the 196 patch tokens.
2. A scorer ranks the patch tokens: **Random**, **Prediction-KL (approximate)**, or **CLS-Attention**.
3. Keep the top-k patch tokens, in their original order, plus `[CLS]`.
4. Run blocks `l..11` and the head on those k+1 tokens.
5. **Information** `P_l(k)` is prediction agreement with the unpruned model (Wilson 95% CI). **Compute** `C_l(k)` is analytic FLOPs, normalised to the full model. **Marginal gain** is `ΔP/ΔC` between consecutive k.
6. A McNemar test compares each scorer with Random. At k=N the intercepted pass matches the baseline (sanity check, max diff < 1e-4).

## Prediction agreement (%)

| Layer | k | Norm. FLOPs | Random | Attention-CLS | Pred-KL (approx) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 98 | 56.5% | 78.4 | 78.0 | **91.6** |
| 2 | 24 | 27.6% | 21.6 | 35.6 | **49.6** |
| 4 | 98 | 65.2% | 85.6 | 89.2 | **92.4** |
| 4 | 24 | 42.1% | 45.2 | 56.0 | **65.2** |
| 6 | 98 | 73.9% | 85.2 | 89.2 | **92.8** |
| 6 | 24 | 56.6% | 61.6 | 71.2 | **79.2** |
| 8 | 98 | 82.6% | 91.2 | 96.0 | **97.6** |
| 8 | 24 | 71.0% | 76.4 | 80.8 | **88.4** |
| 10 | 98 | 91.3% | 94.8 | 98.4 | **99.2** |
| 10 | 24 | 85.5% | 82.8 | 91.2 | **94.4** |
| 12 | any | 100% | 100 | 100 | 100 |

## Findings

1. **Information is not uniform across depth.** The same pruning hurts far less when applied later. At k=24, agreement with Pred-KL rises from 49.6% (L2) to 94.4% (L10). Under Random it rises from 21.6% to 82.8%.
2. **Marginal gain falls with depth.** For Pred-KL, `ΔP/ΔC` for k=24→49 is 2.73 at L2, 2.24 at L4, 1.42 at L6, 0.43 at L8 and 1.28 at L10. The last step (k=172→196) is ≤0.18 at L2 and ≈0 at L10. Early layers gain a lot from extra tokens. Late layers gain almost nothing, so late tokens are redundant.
3. **Scorer ranking is Pred-KL > Attention-CLS > Random.**
   - Pred-KL beats Random with p<0.01 in most mid-range settings (k≈50–75%) at L2–L8. It is least clear at L8 (k=25–37%, p=0.06–0.13) and at L10 (k≥62%, p>0.1).
   - Attention-CLS is indistinguishable from Random at L2 for k ≥ 25% (p>0.4), and significantly better only at the lowest k (35.6% vs 21.6%, p=0.0001). It improves in mid and late layers.
4. **Layer 12 is a trivial control.** No blocks remain after pruning, so `[CLS]` is unchanged and agreement is 100%.
5. **FLOPs savings are an upper bound on real savings.** Pruning at L2 to 25% of tokens cuts FLOPs to 37%, but measured latency is only ~20–25 ms vs ~35–40 ms for the full model. Latency is noisy, and the Pred-KL timings also include its backward pass (60–500 ms), so only the Random and Attention latencies compare cleanly with FLOPs.

## Caveats

- Only 250 images and one seed, so CIs are ±3–6 points and some scorer differences are not significant.
- The "Prediction-KL (approximate)" scorer is a gradient×activation saliency of the top-class logit, not an actual KL. The exact leave-one-out KL scorer is implemented but was not run (it takes N forward passes per batch).
- The marginal-gain figure in `PROJECT_OVERVIEW_AI_GUIDE.md` (≈3.66 at L2/L4) does not match the CSV (2.73 and 2.24). Use the CSV values.
- Everything uses only DeiT-Tiny and pretrained ImageNet. Pruning is done once at a single layer, not progressively.

---

# Phase 2 Results: progressive multi-layer pruning (smoke test)

**Command (from the project root, in the `minor` conda env):**
`python scripts/run_progressive.py --config configs/progressive_smoke.yaml`
Outputs: `results/phase2_progressive_smoke/` (`metrics.json`, `token_counts.csv`, `logits.pt`, `config.json`).

**Correctness**
- `pytest tests`: 30 passed (14 new Phase 2 tests plus the Phase 1 suite).
- The no-pruning path matches the original model: max |Δlogit| = 0.0 in the smoke run (target ≤1e-4).
- Tests also check these properties:
  - CLS is always kept.
  - Exactly k patch tokens remain at each pruning point.
  - Retained tokens keep their original order, and the sets are nested across pruning points.
  - The same seed gives the same random selections.
  - Invalid layers, counts and increasing schedules raise `ScheduleError`.
  - A one-layer schedule matches Phase 1 for the random and CLS-attention scorers.
  - A forward hook confirms that later blocks receive a genuinely shorter sequence (197, 197, 99, 99, then 50 for the remaining 8 blocks).

**Example: schedule `{2: 98, 4: 49}`** gives patch tokens per block of 196, 196, 98, 98, then 49 for blocks 5–12. The total token count after the last block is 50.

**Smoke metrics (10 images, versus the unpruned model; too few images for conclusions)**

| Scorer | Agreement | Analytic norm. FLOPs |
|:-:|:-:|:-:|
| random | 0.7 | 40.9% |
| cls_attention | 0.8 | 40.9% |
| prediction_kl_approx | 0.8 | 40.9% |

**Latency (CPU, batch of 5, 5 runs, includes scorer overhead; not a speedup claim, reported separately from FLOPs):**

| Scorer | Unpruned model (ms) | Progressive total (ms) | Scorer overhead (ms) |
|:-:|:-:|:-:|:-:|
| random | ≈130 | ≈68 | ≈0.2 |
| cls_attention | ≈121 | ≈69 | ≈4.9 |
| prediction_kl_approx | ≈117 | ≈682 | ≈452 |

**Limitations / scorer compatibility**
- `prediction_kl_approx` is the Phase 1 gradient×activation saliency, not an exact KL. At each point it targets the original unpruned model's top class, as in Phase 1, so the forward pass computes the original logits once. It scores as if no later pruning happens. Its backward pass makes it much slower than the other scorers.
- `cls_attention` keeps the Phase 1 behaviour: it uses block `l-1`'s attention weights on that block's output tokens. I did not change this.
- Layer 12 (final block) is rejected as a pruning point. Random selection reproduces only for the same seed and the same batch order, because the generator advances across calls.
- The budget optimizer, merging and fine-tuning are not implemented, as specified.
