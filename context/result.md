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

---

# Phase 2 Evaluation: progressive vs single-layer pruning at matched FLOPs

**Commands:** `python scripts/run_progressive_eval.py --config configs/progressive_eval.yaml --smoke` (20 images) and the same command without `--smoke` (full run, about 22 min on CPU). Output is in `results/phase2_progressive_eval/`; the full write-up is `summary.md`.

**Setup:** 250 ImageNet-val images (the Phase 1 subset, same order; baseline top-1 191/250 = 76.4%), 28 schedules, 3 scorers, seed 42. Extra seeds 43 and 44 were run for `random` only. Matching tolerance is ±2 pp of baseline FLOPs; there are 18 matched pairs per scorer, each with |ΔFLOPs| ≤ 0.14 pp. 136 runs, 0 failed. Tests: 39 passed.

**Agreement with the unpruned model, seed 42 (k/250):**

| Schedule | norm. FLOPs | random | cls_attention | pred_kl_approx |
|:-:|:-:|:-:|:-:|:-:|
| early_single {2:98} | 56.5% | 191 | 195 | 229 |
| middle_single {6:98} | 73.9% | 212 | 223 | 232 |
| late_single {10:98} | 91.3% | 238 | 246 | 248 |
| early_progressive {2:98,4:49} | 40.9% | 158 | 174 | 211 |
| gradual_progressive {2:147,6:98,10:49} | 61.0% | 201 | 220 | 230 |
| delayed_progressive {2:196,4:98,8:49} | 57.4% | 200 | 216 | 221 |
| mild_progressive {4:147,8:98} | 73.6% | 225 | 238 | 241 |
| aggressive_progressive {2:98,4:73,6:49,8:24} | 39.0% | 145 | 173 | 215 |
| late_progressive {6:98,8:73,10:49} | 68.0% | 211 | 221 | 233 |

Top-1 accuracy, Wilson CIs and top-5 are in `summary.md`. All Wilson 95% CIs are about ±5 pp.

**Matched-compute result (progressive minus single-layer, same scorer):**
- **Against a single-layer prune at layer 2 (the fairest comparison):** the result depends on the scorer.
  - `random`: mean Δagreement is −0.8 pp, and progressive is higher in only 2 of 6 pairs.
  - `prediction_kl_approx`: mean Δ is +1.1 pp, and progressive is higher in 3 of 6 pairs.
  - `cls_attention`: mean Δ is +6.2 pp, and progressive is higher in 6 of 6 pairs. It is significant (nominal p≤0.001) in 3 of them: gradual, delayed and mild. Its top-1 gain is +4.7 pp.
- **Against single-layer prunes at layers 4–8:** progressive is often much better. For example, at about 40% FLOPs a layer-4 single prune must cut to 14–20 tokens and loses 20–47 pp. This mostly shows that very aggressive late-start single pruning is bad, not that progressive pruning is better in general. It is not a clean "progressive wins" result.
- **Across all 18 pairs per scorer:** progressive has higher agreement in 14 (random), 17 (cls_attention) and 14 (prediction_kl_approx) of 18 pairs. None favour single-layer at nominal p<0.05. These p-values are uncorrected.
- **Seed variation (random scorer):** the seed-to-seed sd of the agreement difference is 0.4–6.4 pp. This is comparable to the image-level CIs, so the random-scorer differences against L2 singles are within noise.

**Compute and latency (CPU, batch 25, 3 timed runs after 1 warm-up, reported separately from FLOPs):**
- Latency is noisy and measured on one batch only.
- Unpruned forward is about 642 ms. Random and cls_attention progressive passes at 40–60% FLOPs took about 300–430 ms.
- Scorer overhead: random ≈0.2–0.5 ms, cls_attention ≈18–57 ms, `prediction_kl_approx` ≈350–2800 ms.
- `prediction_kl_approx` makes the pass 3–6× slower than unpruned (1.7–3.9 s), so its accuracy advantage does not come with a speedup.

**Limitations:**
- 250 images and one model; the many comparisons are uncorrected.
- Pair differences also reflect which layers are pruned, not only the number of steps.
- Random draws differ between schedules, since the generator advances.
- Latency is single-batch CPU timing.

**Suggested next experiment:** more images (and seeds) on a finer grid of schedules at the same budget. Hold the first pruning layer fixed to separate "progressive" from "which layer" effects. Then try a larger model.

---

# Phase 2 Controlled Pruning-Stage Ablation (first layer fixed, FLOPs matched)

**Commands:** `python scripts/run_stage_ablation.py --config configs/stage_ablation.yaml --smoke`, then the same command without `--smoke` (about 38 min on CPU). Output is in `results/phase2_stage_ablation/`; the full tables are in `summary.md`.

**Setup:** the same 250 images as the previous evaluation. The run verified the manifest (index, label, sha1, order) and the baseline top-1 of 191/250 against it. The subset contains only **5 ImageNet classes** (a contiguous slice of the file). Budgets are 40/50/60/70% of FLOPs, each with one-stage, three two-stage and three three-stage schedules. The tolerance was ±1 pp, fixed in advance. All 8 budget groups (first layers 2 and 4) matched, with 0 unmatched and a maximum |ΔFLOPs| of 0.13 pp. Tests: 49 passed.

**Primary result (predefined: `cls_attention`, first layer 2, seed 42, agreement, multi-stage minus one-stage, 24 comparisons, Holm-corrected):**
- Multi-stage had higher agreement than one-stage in 24 of 24 comparisons, with a mean of +4.0 pp for two-stage and +5.9 pp for three-stage.
- Only **4 of 24 are significant after Holm correction** (the 60% budget: 2-4, 2-4-8, 2-6-10 and 2-8-10). Uncorrected, 11 of 24 reach p<0.05.
- Three-stage is higher than its two-stage prefix in 11 of 12 pairs (mean +1.9 pp), but none is Holm-significant. That is exploratory.
- Top-1 accuracy shows the same direction but weaker: 23 of 24 higher, 1 Holm-significant.

Examples of agreement (counts of 250):

| Budget | One-stage | Best two-stage | Best three-stage |
|:-:|:-:|:-:|:-:|
| 40% | 164 | 179 | 183 |
| 50% | 187 | 199 | 206 |
| 60% | 198 | 217 | 218 |
| 70% | 215 | 223 | 228 |

**Secondary and exploratory results (each its own Holm family):**
- `prediction_kl_approx`, layer 2: mean +1.7 pp (two-stage) and +2.8 pp (three-stage), with 1 Holm-significant comparison each.
- `random`, layer 2: mean +2.5 pp (two-stage) and −0.8 pp (three-stage), with 0 Holm-significant comparisons. This is within seed-to-seed noise: the sign of the mean difference changes across seeds 42, 43 and 44.
- **Layer 4 (first layer fixed at 4) does not reproduce the advantage.**
  - `cls_attention`: mean −0.6 pp (two-stage) and −3.0 pp (three-stage). 2 of the three-stage comparisons are Holm-significant, and they favour the single stage.
  - `random`: mean −2.7 to −3.4 pp (two-stage) and −6.4 to −7.1 pp (three-stage). Several of these are Holm-significant and favour the single stage.
  - These layer-4 schedules often retain very few tokens (e.g. 3–12 tokens after the later stages).
- So "more stages is better" holds only for the CLS-attention scorer with the first layer at 2. It is not general.

**FLOPs and latency (CPU, batch 25, 3 runs, separate from FLOPs):** latency is noisy in this run. The unpruned forward pass took about 1 s and pruned passes about 0.65–0.73 s for random and CLS-attention. Multi-stage scorer overhead grows with stage count: `cls_attention` about 36, 57 and 71 ms for 1, 2 and 3 stages; `prediction_kl_approx` about 2.7, 3.8 and 4.1 s, which makes it 4–6× slower than the other scorers. Since extra stages cost extra scorer calls, equal analytic FLOPs do not mean equal runtime.

**Limitations:** only 5 classes and 250 images; one model; one count-allocation family (geometric); stage count is confounded with which later layers are used; random draws differ between schedules; latency is single-batch CPU timing. The Holm correction removes most of the nominally significant results.

**Next step:** a class-diverse subset (more than 250 images covering all 1000 classes), with a second count-allocation family, to check whether the layer-2 CLS-attention advantage survives.

## Phase 2 – Class-diverse validation (2,000 images, all 1,000 classes, 2/class; seed 42)
Unpruned: top-1 71.2% (1424/2000). Matched analytic FLOPs (±1 pp), budgets 40/50/60/70%, first layer 2, `cls_attention`, seed 42; paired tests, Holm per predefined family.
- **Geometric allocation (rule from the 5-class ablation): the multi-stage advantage generalizes.** Agreement vs one stage: higher in 24/24, mean +5.5 pp (2-stage ≈ +5.2, 3-stage ≈ +5.9), 24/24 Holm-significant; top-1 higher in 24/24, mean +3.7 pp, 17/24 Holm-significant. Three-stage vs its two-stage prefix: mean +0.7 pp (exploratory; across both families 7 of 23 comparisons Holm-significant).
- **Equal-drop allocation: not robust.** Mean −4.1 pp vs one stage (16 higher / 7 lower, all 7 lower are Holm-significant). Losses come from schedules whose last stage keeps only 1–4 tokens (e.g. {2:92, 8:1}: 8% agreement); with final counts ≥ ~24 it is similar to geometric (+2–10 pp over one stage at 60–70% FLOPs).
- Geometric vs equal_drop (same structure): geometric better by up to 47 pp at 40–50% FLOPs, ≈ equal at 60–70%. So the result depends on token allocation, not stage count alone.
- Secondary (exploratory): random layer 2 geometric +1 to +3 pp (seed-dependent), equal_drop 3-stage ≈ −21 pp; prediction_kl_approx (budgets 50/60% only) geometric +2.0/+2.6 pp, equal_drop 3-stage −28 pp. Layer 4 (cls_attention): geometric −1.6/−4.6 pp, equal_drop −18 pp, i.e. no advantage.
- Historical 5-class/250-image geometric deltas (+2 to +8 pp, mostly Holm-nonsignificant) are consistent in sign and size with the new ones (+2 to +12 pp, all significant); the difference in significance is sample size.
- Latency (CPU, batch 25, 3 batches × 5 runs): unpruned ≈ 908 ms; cls_attention pruned 519/580/635 ms (1/2/3 stages), scorer 28/49/64 ms; prediction_kl_approx 3.6–4.9 s (scorer ≈ 2.2–3.4 s). Equal FLOPs ≠ equal latency; no speedup claim from FLOPs.
- Limitations: class-balanced sample (not the validation distribution), one model, two allocation families, CPU only, secondary scorers on reduced schedule sets.

## Phase 2 – Minimum final-token constraint (2,000 images, same subset; `cls_attention`, seed 42, layer 2, matched FLOPs ±1 pp)
Floor F = minimum patch tokens kept after the last pruning point; constrained equal-drop = equal drops truncated at F, D re-searched to match FLOPs.
- **Floor binds only at 40–50% FLOPs** (and 2-8-10 at 60%); elsewhere the constrained schedule equals unconstrained equal-drop (68 cases, untested). 6 combos infeasible (B40 2-8-10 all floors; 2-6-10 floors 24/32).
- **Primary (constrained vs unconstrained equal-drop, 17 testable pairs):** agreement higher in 17/17, mean +28 pp, 16/17 Holm-significant (the exception: B50 2-8 f32 vs 31 tokens, +0.65 pp, n.s.). Gains are largest where equal-drop ended with 1–4 tokens (+25 to +47 pp); small (+2 to +8 pp) when unconstrained already kept 21–24. Top-1 shows the same pattern.
- **Secondary (constrained vs geometric):** constrained is lower in 17/17 (mean −4.3 pp agreement, 14 Holm-significant). Floor 32 closes most of the gap (−0.1 to −3.3 pp; B40 2-8 f32 ≈ geometric); floor 16 stays −5 to −11 pp.
- **Interpretation:** a very small final token count explains most of the equal-drop failure (matched FLOPs, paired), but it is not the whole story: even with floor ≥ 24 equal-drop-style allocation is mostly still below geometric, which keeps a larger final count and a gentler early drop. Floor 8 (diagnostic) is still clearly worse than floors ≥ 16 where it binds (agreement 36–41% at B40, 56% at B50 2-6-10 / 2-8-10 vs 64–68% with floors 16–32).
- Secondary scorers (reduced set): `random` constrained gain +9 to +42 pp mean over seeds 42–44 (seed sd 0.6–1.9 pp); `prediction_kl_approx` B50 2-6-10 f16/f32 vs equal-drop +53/+57 pp, vs geometric −7/−3 pp (approx. KL, not exact).
- Latency (CPU): equal FLOPs ≠ equal latency; prediction_kl_approx scorer overhead ≈ 2.2–2.5 s vs ≈ 25–45 ms for cls_attention. No speedup claim.
- Limitations: constrained family also changes earlier counts to hold FLOPs; one rule; layer 4 not run; many secondary combos not run.

## Phase 2 – Final-token vs trajectory isolation (same 2,000 images, cls_attention, seed 42; patterns 2-6-10 / 2-8-10; matched FLOPs ±1 pp)
- 32 final-count-matched geometric controls (same final count, different intermediate counts) were feasible for all 8 budget×pattern cells; 9/32 are Holm-significantly worse than geometric (+1.5 to +12.7 pp agreement; 8 at 40–50% FLOPs), 10 similar, 13 inconclusive. At 60–70% FLOPs differences ≤ ~1.5 pp.
- Unconstrained equal-drop (final 1–2 tokens) −36 to −44 pp vs geometric at 40–50%; floors repair +22 to +41 pp but constrained stays +2.3 to +11 pp below geometric at 40–50%.
- Conclusion: final count explains the catastrophic failure but not all of geometric's advantage at tight budgets; intermediate trajectory (esp. not collapsing the middle stage) matters independently there; at 60–70% not distinguishable. Details: `results/phase2_final_token_trajectory_isolation/result.md`.
- Tests: 94 passed. Limitations: 32 non-exhaustive controls, one model/scorer/seed, CPU only.

## Phase 3 – Evidence audit highlights (design stage; no new model results)
- Raw-file audit found 7 documented discrepancies/unverified items (ledger §3), e.g. the "+22 to +41 pp" floor-repair wording includes diagnostic floor 8, and Phase 1's 250 images come from only 5 classes.
- New derived finding: at layer 2 on the 2,000-image subset the CLS-attention scorer is worse than random at all four budgets (−1.45 to −4.90 pp over 12 comparisons) and far worse than approximate Pred-KL (−12 to −18 pp); allocation gains so far are relative to a weak scorer.
- No schedule has yet been shown to beat geometric at matched FLOPs (0/32 controls significantly better).

## Phase 3 — Budget-aware allocation results (summary)
Proposed schedules chosen from calibration-only sensitivity tables (proxy pooled Spearman 0.98; within-cell mean 0.53). Matched-FLOPs agreement vs geometric: validation +0.73 pp, confirmatory +0.87 pp mean (18/24 positive, 0 Holm-significant; best B40 2-8-10 +2.25 pp, Holm p 0.06). vs single-stage +4.4 pp (all Holm-significant); vs constrained equal-drop +2.33 pp (7 Holm-significant). Latency equal to geometric. Calibration cost 87 min CPU offline. Full report: `results/phase3_budget_aware_allocation/result.md`.

## Phase 4 (Gates 0–3) summary
Frozen protocol and implementations for a controlled benchmark of ToMe, an EViT-style training-free baseline and DiffRate's released DeiT-Tiny rates against the Phase 3 allocator, geometric, equal-drop and single-stage baselines on a fresh 995-class split. Verified so far (P4-DEV only, diagnostics): our ToMe reproduces official ToMe exactly; DiffRate's released lists execute (target 1.0 partly ill-defined). Preliminary dev-only observation (not a finding): merge-based methods keep much higher agreement at similar analytic FLOPs than the Phase 3 prune-only schedules. Files: `results/phase4_adaptive_method_benchmark/`.

## Phase 4 — Gate 4 (development evaluation, P4-DEV only) — complete, awaiting review (2026-11-17)
- Selected ToMe (r, inflection) and EViT-style kappa per primary budget (40/50/60/70) and DiffRate native FLOPs (0.6-0.9) from every feasible candidate (84 unique evaluated on 1,990 dev images); selections, tie sets and all candidates in `results/phase4_adaptive_method_benchmark/gate4/` and `development_results.md`.
- Phase 3 frozen schedules regenerated on P4-DEV (37 unique schedules); supplementary Phase 3 re-selection at DiffRate native caps on the fixed 500-image calibration subset, evaluated on 1,490 held-out dev images.
- Latency/aux time measured as secondary (CPU, noisy). DiffRate target 1.0 excluded (diagnostic only).
- `frozen_method_configs.json` SHA-256 c42717bf05e677e426c3422267f0d7f941a39ea30c07ac27733d1b8c269b706e; protocol freeze verified; 130 tests pass.
- Disclosed deviation: a test unintentionally decoded P4-CONFIRM images into a cache (no model run, no statistics); cache deleted, test fixed (see protocol_amendments.md).
- Development results are selection data, not confirmatory evidence. Gate 5 not started; needs explicit approval.

### Phase 4 — Gate 4 close-out and Gate 5 preflight (2026-11-17)
- Gate 4 accepted. P4-CONFIRM cache incident closed in `protocol_amendments.md` (images + labels were decoded/cached once by a test, never read; cache deleted; test isolated; loader now verifies frozen config bytes and dependency hashes). P4-CONFIRM retained; must be disclosed as a data-access deviation.
- `frozen_method_configs.json` re-frozen with grid identity, split/manifest identities, calibration-subset identity, checkpoint SHA-256 and 32 dependency hashes: SHA-256 171fa7be2cc4bd511a9aa491da8c475c480d198f1a5070287a5a8eebd134e5b6. 131 tests pass; protocol freeze verified (17 files); `gate5_preflight.json` all checks pass. Gate 5 runner written and dry-run on 50 P4-DEV images only; NOT executed on P4-CONFIRM — awaiting explicit authorization.

### Phase 4 — Gate 5 analysis frozen (P4-DEV validated; confirmatory run NOT authorized/started) (2026-11-17)
- `scripts/phase4_gate5_analysis.py` (SHA-256 8b5e51041f297c1f6ad0bd2125c69c59993c1073bc0aabf7b58500019eb5df9d); freeze record `phase4_analysis_freeze.json` (SHA-256 e574c540e646217e1592dcfd94a959e9801666ec789c38a4d43befd7742aa2dc); documentation `analysis_documentation.md`.
- P4-DEV pipeline validation: independent cross-check all pass (counts match Gate 4 records), reproducible byte-for-byte, 144 tests pass, Gate 5 preflight all checks pass; frozen_method_configs (171fa7be…) and original protocol freeze (895ee5fe…) unchanged; no marker/partial outputs.

### Phase 4 — Gate 5 confirmatory run executed once (2026-11-17)
Single P4-CONFIRM run completed (6,247 s, no failure). Frozen analysis run once. ToMe beats geometric in all 16 primary cells (Holm-significant); EViT-style worse at B40, null at B50, better at B60/B70; Phase 3 replication 1/24 Holm-significant (no broad superiority). See `results/phase4_adaptive_method_benchmark/confirmatory_results.md` and `confirmatory_analysis/`. Gate 6 (final report) pending.

### Phase 4 — Gate 6 final report written (2026-11-17)
`results/phase4_adaptive_method_benchmark/phase4_final_report.md`, `paper_claims_matrix.md`, `recommended_next_experiment.md`. Assessment: ToMe > geometric in all 16 primary cells; EViT-style budget-dependent; Phase 3 allocator no consistent advantage over geometric and not competitive with ToMe/DiffRate; no novelty claim; workshop (benchmark/negative-result) framing plausible, main-track not ready; next experiment proposed (sensitivity-guided per-layer merge schedules vs equal-effort ToMe, fresh confirmatory split). No paper drafted; nothing committed.
