# Layer-Wise Information Dynamics of Tokens in Vision Transformers — Full Project Explanation (Phases 1–4)

*A plain-language, end-to-end account of what this project asked, how it was tested, what was found, and what the findings do and do not support. Written so that someone new to the project can follow it. Last updated 2026-11-17.*

> **Where the details live.** This document summarises. The authoritative numbers are in the `results/` folders named in each section, and the project log is in `context/progress_tracker.md` and `context/result.md`. Nothing in this project has been committed to Git.

---

## 0. The one-minute version

- **Subject:** a small pretrained image classifier (DeiT-Tiny, a Vision Transformer) that looks at an image as 196 small patches ("tokens"). Running it costs compute. Could we **drop or merge tokens inside the network** to save compute while still getting the same answers?
- **Phase 1** asked *where* in the network tokens can be removed with least damage. Answer: **late layers tolerate removal far better than early layers.**
- **Phase 2** asked whether removing tokens **gradually in several stages** beats removing them **all at once**, at equal compute. Answer: **yes, if the schedule is shaped sensibly** (a "geometric" shape works; "equal drops that end with 1–2 tokens" fail badly).
- **Phase 3** built a **data-driven method** to pick the per-stage token counts under a compute budget. Answer: it gave **about +0.6 to +0.9 percentage points** over the simple geometric schedule, which was **not statistically reliable**. We could not claim it is better.
- **Phase 4** compared our approach against published training-free methods (**ToMe**, an **EViT-style** method, and **DiffRate**'s released settings) on fresh images, with the analysis frozen in advance. Answer: **ToMe (merging tokens) was clearly better than our pruning-only schedules at every budget**; our method showed no consistent advantage; no novelty is claimed.
- **Honest bottom line:** the project produced a careful, controlled **comparison / negative-result** study. It did **not** produce a method that beats published ones.

---

## 1. Background: what is being studied

### 1.1 The model
- **DeiT-Tiny** (`deit_tiny_patch16_224`, from the `timm` library), pretrained on ImageNet-1k. It is **frozen** throughout: we never train or fine-tune it.
- Architecture: input image 224×224 is cut into 16×16 patches → **196 patch tokens** + 1 special **[CLS] token** = 197 tokens. Each token is a 192-dimensional vector. There are **12 transformer blocks** (each: multi-head self-attention with 3 heads, then an MLP). The final prediction is read from the [CLS] token.
- Cost of a block grows with the number of tokens *T*: attention costs about *T²·d* and the MLP / projections about *T·d²*. **Fewer tokens in later blocks ⇒ less compute.**

### 1.2 The idea: token reduction
Many patches (for example, plain background) matter little to the final answer. If we remove or combine some of them partway through the network, later blocks process fewer tokens and cost less. Two families:
- **Pruning:** throw tokens away (keep the top-*k* by some score). Phases 1–3.
- **Merging:** combine similar tokens into one instead of discarding them (ToMe, DiffRate). Phase 4.

### 1.3 Terms used everywhere
| Term | Meaning |
|---|---|
| **Layer *l*** | "after *l* blocks". A prune "at layer 2" happens after block 2. |
| **Schedule** | Which layers prune and how many patch tokens remain after each, e.g. `{2: 90, 6: 41, 10: 19}`. Written compactly as a pattern like **2-6-10**. |
| **Budget (B40/B50/B60/B70)** | Target compute = about 40/50/60/70 % of the unpruned model's **analytic FLOPs**. |
| **Analytic FLOPs** | Compute counted by formula from the token counts per block (not measured time). |
| **Prediction agreement** | % of images where the reduced model gives the **same top-1 class as the unpruned model**. Measures "did we preserve the model's behaviour". Does not use ground-truth labels. |
| **Top-1 accuracy** | % of images where the predicted class equals the **true label**. Reported **separately** from agreement. |
| **Latency** | Measured wall-clock time (CPU). **Never conflated with FLOPs**; fewer FLOPs ≠ faster in practice. |
| **Geometric schedule** | Token counts shrink by a constant ratio per stage: *K_j = round(N·ρ^j)*, with ρ tuned to hit the budget. |
| **Equal-drop schedule** | Removes the same number of tokens at each stage (can end with only 1–2 tokens). |
| **Single-stage** | One prune at one layer, tuned to the same FLOPs. |
| **Floor (F)** | A minimum number of patch tokens that must survive the last stage (8/16/24/32). |
| **Holm correction** | A standard fix for running many tests at once, so "significant" results are not just luck. |

---

## 2. Datasets and images used (all phases)

**Source:** the ImageNet-1k **validation set** (50,000 images, 1,000 classes, 50 per class), read from locally cached parquet shards of the same Hugging Face dataset the loader uses. **No training data was used, and nothing was trained.** The images are always resized/normalised the standard DeiT way.

| Phase | Subset | Size | Classes | How chosen | Used for |
|---|---|---|---|---|---|
| 1 and early 2 | **dev250** (`imagenet_val_250.pt`) | 250 | **only 5** (classes 0–4, all 50 images each) | first 250 rows in stored order | Exploration. *Limitation:* not class-diverse. |
| 2 (validation) and 3 (parent) | **class-diverse subset** | 2,000 | 1,000 | 2 per class, seed 42 | Validating Phase 2 findings; basis for Phase 3 splits |
| 3 | **Calibration** | 500 | 500 classes drawn | class-stratified from the parent, seed 42 | Choose the schedule (no labels needed for most of it) |
| 3 | **Validation** | 1,500 | 1,000 | the rest of the parent | Check the method before the final test |
| 3 | **Fresh confirmatory** | 2,000 | 1,000 | 2 per class, seed 4242, disjoint from the parent (verified) | Final test, run **once** |
| 4 | **P4-DEV** | 1,990 | 995 | 2 per class, seed 2026, never used before | Selecting each method's settings |
| 4 | **P4-CONFIRM** | 2,985 | 995 | 3 per class, seed 7331, never used before | Final test, run **once** |

Notes:
- Phase 4 splits exclude the 5 classes (0–4) because Phase 1's dev250 already used **all** their images; so Phase 4 covers **995 classes**, and the sample is class-**balanced**, not the natural ImageNet distribution. All Phase 4 images are disjoint from the **4,230 images** used in Phases 1–3.
- Every subset has a manifest (indices, labels, image hashes) and a SHA-256 so it can be re-verified.
- Baseline (unpruned) DeiT-Tiny top-1 differs by subset (that is a property of the images, not an inconsistency): 76.4 % on dev250, 71.2 % on the 2,000-image set, **69.89 % on P4-CONFIRM**.

---

## 3. Phase 1 — Single-layer pruning: *where* does information live?

**Research questions** (`README.md`):
1. Does token task-sensitivity change across depth?
2. How fast does performance degrade as we keep fewer tokens at layer *l*?
3. What is the marginal gain per unit of compute at each depth?

**Method.** Run blocks 0..*l*−1 on all tokens, then **score** the 196 patch tokens, **keep the top-*k*** (plus [CLS], original order preserved), and run the remaining blocks on the shorter sequence. Layers *l* ∈ {2,4,6,8,10,12} × 8 retention ratios (100 % → 12.5 %). Three scorers were compared so that the result isn't an artefact of one choice:
- **Random** (baseline, uniform noise).
- **CLS-attention:** how much attention the [CLS] token pays to each patch, averaged over heads: *S(t_i) = (1/H) Σ_h A_{h,cls,i}*.
- **Prediction-KL (approximate):** gradient × activation saliency of the top-class logit (named "KL" for its role; it is *not* an exact leave-one-out KL, which was implemented but too slow to run).

**Metrics (kept deliberately separate to avoid circular evaluation):** prediction agreement, top-1/top-5 accuracy, confidence retention, KL/JS divergence; analytic FLOPs; CPU latency. Agreement uses Wilson 95 % intervals; scorer comparisons use McNemar's test. Marginal efficiency: ΔP/ΔC = [P(k_{j+1}) − P(k_j)] / [Ĉ(k_{j+1}) − Ĉ(k_j)], where Ĉ is FLOPs relative to the unpruned model.

**FLOPs model.** Total cost = embedding + Σ over blocks of block-cost(tokens in that block) + head, divided by the unpruned total. Tokens are counted per block, so a prune at layer *l* lowers cost only for blocks after *l*.

**Findings** (250 images, one seed; `results/deit_tiny_imagenet_dev250/`):
1. **Depth matters.** At *k* = 24 tokens, agreement with Pred-KL rises from **49.6 % (layer 2)** to **94.4 % (layer 10)**. Early layers are fragile; late layers are robust.
2. **Marginal gain falls with depth:** extra tokens help a lot early (ΔP/ΔC ≈ 2.7 at layer 2) and almost nothing late (≈0 at layer 10).
3. **Scorer order: Pred-KL > CLS-attention > Random**, but CLS-attention ≈ Random at layer 2.
4. Layer 12 is a trivial control (pruning after the last block changes nothing).
5. **Theoretical FLOPs savings overstate real speedups** (measured CPU latency improves less).

**Caveats:** only 250 images from **5 classes**; ±3–6 pt intervals; one model.

---

## 4. Phase 2 — Progressive (multi-stage) pruning and schedule shape

**Question:** at the **same FLOPs**, is pruning gradually in several stages better than one big prune? And does *how* tokens are allocated across stages matter?

**Infrastructure.** A `ProgressivePruner` applies different *k* at several layers in one forward pass (verified: with no pruning it reproduces the original logits, max diff ≤ 1e-4). Layer keys are "after *l* blocks".

**Studies, in order** (each used a fixed analysis plan and Holm correction within predefined test families):
1. **Progressive vs single-layer at matched FLOPs** (250 images, 28 schedules × 3 scorers). Controls were chosen from FLOPs only, before seeing accuracy. Result: depends on the scorer; mostly positive for CLS-attention (+6.2 pp vs a layer-2 single prune), not a clean "progressive always wins".
2. **Controlled stage ablation** (first layer fixed at 2; one/two/three stages; budgets 40–70 %; geometric counts). On the 5-class set multi-stage was higher in 24/24 comparisons (+4.0 pp two-stage, +5.9 pp three-stage) but only 4/24 survived Holm; with first layer 4 the advantage did **not** reproduce.
3. **Class-diverse validation** (2,000 images, all 1,000 classes): the geometric multi-stage advantage **generalised** (higher in 24/24, mean +5.5 pp, 24/24 Holm-significant). The **equal-drop** allocation did **not**: mean −4.1 pp, with catastrophic failures (as low as 8 % agreement) when the last stage kept only 1–4 tokens.
4. **Minimum final-token constraint** ("floors" 8/16/24/32): enforcing a floor repaired most equal-drop failures (+28 pp mean, 16/17 significant), but constrained equal-drop still stayed below geometric (mean −4.3 pp).
5. **Final-token vs trajectory isolation:** controls with the *same final count* but different intermediate counts showed the **trajectory also matters** at tight budgets (40–50 %), but not at 60–70 %.

**Phase 2 conclusions.**
- Multi-stage beats single-stage **only if** the allocation is sensible (geometric-like) **and** the first prune is early with a scorer that works there.
- **A very small final token count is the main cause of failure**, but not the whole story at tight budgets.
- At 60–70 % FLOPs, allocation barely matters.
- **Side finding (from the Phase 3 audit):** on the 2,000-image set at layer 2, CLS-attention is *worse than random* (−1.5 to −4.9 pp) and far worse than approximate Pred-KL (−12 to −18 pp). Allocation gains so far were measured relative to a weak scorer.

Outputs: `results/phase2_*`.

---

## 5. Phase 3 — A budget-aware allocation method ("LS-Chain + verify")

**Goal.** Given a FLOPs budget, a set of pruning layers (patterns 2-6-10 and 2-8-10) and a floor, automatically choose the token counts *K₁ > K₂ > K₃ ≥ F* that best preserve predictions — better than the hand-designed geometric rule.

**Why this design (from an evidence audit of Phase 1–2).** Sensitivity differs by layer; trajectories matter at tight budgets; final count matters; allocation matters little at 60–70 %; a model-level (fixed) schedule adds **zero inference overhead**.

**The algorithm (in words, then maths).**
1. **Measure sensitivity once, on the 500-image calibration set (no labels).** For layer *a* and count *k*: *D(a,k)* = average KL divergence between the unpruned output distribution and the output after pruning only at layer *a* to *k* tokens. For consecutive layers *(a,b)*, measure the two-stage version *D2*. (312 pruned runs.)
2. **Chain model of total damage:**
   *L̂(K) = D(l₁,K₁) + Σ_{j≥2} [ D2(l_{j−1},K_{j−1}; l_j,K_j) − D(l_{j−1},K_{j−1}) ]*
   i.e. each stage's cost depends on the count it starts from, so collapsing the middle stage is penalised using measured data.
3. **Exact constrained solve:** enumerate all integer schedules (~10⁶) whose analytic FLOPs lie in the band [cap − 1 pp, cap] (cap = FLOPs of the matching geometric schedule, so the new schedule never spends more) and with *K_m ≥ F*; rank by *L̂*.
4. **Verify:** take the top 8 diverse candidates and measure their true agreement on the calibration set (75 runs); pick the best (ties → lower *L̂*, then larger final count).
5. **Freeze** the schedule (SHA-256 recorded) before any validation data is touched. Geometric/equal-drop are **baselines only**, never candidates.

**Protocol.** 500 calibration / 1,500 validation / 2,000 fresh confirmatory (run once). 24 cells = 2 patterns × 4 budgets × 3 floors (only **10 distinct schedules**, because floors are often non-binding, so cells are not independent).

**Results** (`results/phase3_budget_aware_allocation/result.md`):
- Proposed vs **geometric**: validation **+0.73 pp**, confirmatory **+0.87 pp** mean agreement (18 of 24 cells positive); **0 comparisons significant after Holm** (best: B40 2-8-10, +2.25 pp, Holm p = 0.06). **Superiority over geometric was not established.**
- vs **single-stage:** better in all 24 (+4.4 pp, all significant). vs **constrained equal-drop:** +2.33 pp, 7 significant (where feasible).
- No latency difference vs geometric; offline calibration cost ≈ 87 min on CPU.
- The sensitivity proxy ranks coarse differences well (pooled Spearman 0.98) but is weak *within* a cell (mean 0.53).

**Phase 3 conclusion:** a leakage-safe, zero-overhead method that repeatedly gives about +1 pp over geometric but cannot be shown to beat it.
*(An additive 2026-11-16 erratum corrects how some baseline vectors were printed; no result changed.)*

---

## 6. Phase 4 — Benchmark against published adaptive token-reduction methods

**Research question.** Is our Phase 3 approach competitive with, or better than, established **training-free** token-reduction methods on a frozen DeiT-Tiny, at matched analytic FLOPs, when everything is pre-specified and tested once on fresh images?

### 6.1 Methods compared, and honesty labels
| Method | What it does | Label used |
|---|---|---|
| **ToMe** (Bolya et al., 2023) | **Merges** similar tokens: splits tokens into two alternating sets A and B, computes cosine similarity of the attention **keys**, links each A-token to its most similar B-token, merges the top-*r* links by **size-weighted averaging**, and applies **proportional attention** (adds log of token size to attention logits). [CLS] is never merged. | **Faithful adaptation**; our re-implementation **matched the official code exactly** (identical token counts; bit-identical logits) for the tested configurations. |
| **EViT-style** | At layers 4/7/10 keeps the top fraction κ of tokens by [CLS] attention and **fuses the rest into one token**. | **Reproduction-inspired, not official EViT** (EViT is normally trained; our fused token is a raw attention-weighted sum; numerical parity unverified). |
| **DiffRate** (released rates 0.6–0.9) | Published learned prune+merge rate lists for DeiT-Tiny, executed in our engine. | **Supplementary track**, merge semantics not verified against official execution; its rates can't hit our 40–70 % grid, so it is compared only at its own native FLOPs. |
| Phase 3 baselines | geometric, equal-drop, single-stage, constrained, and the proposed "LS-Chain + verify" | Our own pruning-only schedules |
| **Not compared** | DynamicViT, official EViT (need training); DiffRate target 1.0 (not applicable as released) | Excluded |

### 6.2 Design (the "gates")
The work was staged in gates 0–6, with a stop for review at each:
- **Gates 0–2:** audit, related-work review, fidelity matrix, split plan, and a **protocol freeze** (17 files hashed).
- **Gate 3:** implementation and tests (official-ToMe parity checked in an isolated environment).
- **Gate 4 (P4-DEV only):** pick each method's setting **per budget**. ToMe grid: 200 points (*r* = 1..40 × inflection ∈ {−1, −0.5, 0, 0.5, 1}); EViT-style grid: 161 κ values (0.20–1.00). A setting is **feasible** if its analytic FLOPs are within ±1 pp of the budget. Selection rule (fixed in advance): highest dev agreement; near-ties (≤ 3 images of 1,990) broken by smaller FLOPs gap, then simpler parameter, then generation order. Phase 3 schedules were regenerated on dev as comparators.
- **Gate 5:** freeze all configurations (SHA-256 `171fa7be…`), freeze and validate the **analysis script** on dev data, then run **one** confirmatory inference on P4-CONFIRM and the frozen analysis **once**.
- **Gate 6:** final report, claims matrix, next-experiment proposal.

### 6.3 Statistics (pre-specified)
Wilson intervals; **paired bootstrap** (10,000 resamples, seed 42) for differences; **exact McNemar** test; **Holm correction** within each pre-declared family (primary agreement: 16 comparisons, primary top-1: 16, replication: 24 each, plus secondary and exploratory families). Difference = method − comparator, in percentage points. Agreement, top-1, FLOPs, auxiliary overhead, calibration cost and latency are **always reported separately**.

### 6.4 Results (confirmatory, P4-CONFIRM, n = 2,985; unpruned top-1 = 69.89 %)

**Best settings per budget** (agreement with unpruned / top-1):
| Budget | ToMe | EViT-style | Geometric (2-6-10) |
|---|---|---|---|
| ≈40 % FLOPs | **72.63 / 59.26** | 46.30 / 39.56 | 54.87 / 47.14 |
| ≈50 % | **83.02 / 65.29** | 70.99 / 58.86 | 70.95 / 58.89 |
| ≈60 % | **90.62 / 68.31** | 82.75 / 65.49 | 79.46 / 63.25 |
| ≈70 % | **94.71 / 69.65** | 88.78 / 67.91 | 85.46 / 65.76 |

**Key comparisons (Holm-corrected):**
- **ToMe vs geometric:** better in **all 16 primary cells** on both metrics, all significant (agreement +9.2 to +18.6 pp, top-1 +3.4 to +12.9 pp). ToMe also beats every Phase 3 variant tested (proposed floor-16, single-stage, constrained, equal-drop).
- **EViT-style vs geometric:** **worse** at ≈40 %, **no difference** at ≈50 %, **better** at ≈60–70 % (+3.3 to +4.0 pp). So "EViT-style is better" is *not* a safe blanket claim.
- **Phase 3 replication (proposed vs geometric, 24 comparisons):** mean +0.62 pp, 15 positive / 9 negative, **only 1 of 24 significant** after Holm. This matches Phase 3: **no consistent advantage.**
- **DiffRate (supplementary):** better than our pruning baselines in 35 of 36 matched comparisons (33 significant), but versus ToMe at matched FLOPs it was significantly better **only at its 0.6 setting** (+2.28 pp agreement). DiffRate does *not* clearly beat ToMe.
- **Latency (development set only, CPU, noisy):** ToMe ran *slower* than the unpruned model at the 60 % and 70 % budgets in our explicit-attention engine despite fewer FLOPs. **No latency ranking is supported.**

### 6.5 What this means
- **Merging (ToMe) preserves far more of the model's behaviour than pruning-only schedules** at equal analytic FLOPs. This is an empirical regularity here; the *mechanism* was not isolated.
- **Our sensitivity-based allocator is not competitive** with ToMe or DiffRate, and has **no consistent advantage over the simple geometric schedule.**
- **No novelty is claimed.** ToMe, EViT and DiffRate are published; our allocator was not shown to be better than its own baseline.

### 6.6 Disclosed deviation (data-access incident)
During Gate 4, a unit test unintentionally **decoded and cached the P4-CONFIRM images, labels and IDs** (no model inference, no label inspection, no statistics). The cache was deleted, the test fixed, and a guard added that verifies the frozen configuration before the confirmatory split can be loaded. There are **no OS audit logs**, so we cannot prove nothing was accessed; we therefore **never claim the split was "untouched"**. The full record is in `results/phase4_adaptive_method_benchmark/protocol_amendments.md`, and it must be disclosed in any paper.

---

## 7. The whole story in one chain

| Phase | Question | Answer |
|---|---|---|
| 1 | Where can tokens be removed safely? | Late layers; early layers are fragile. |
| 2 | Does multi-stage beat single-stage at equal FLOPs? | Yes, with a geometric-like shape and a sufficient final token count; equal-drop ending in 1–2 tokens fails. |
| 3 | Can a measured-sensitivity method beat the geometric shape? | About +0.6 to +0.9 pp, not statistically reliable; not established. |
| 4 | Does it compete with published training-free methods? | No. ToMe (merging) is clearly better at every budget; no consistent advantage over geometric. |

**Overall research finding.** At matched analytic FLOPs on frozen DeiT-Tiny, **how tokens are reduced (merge vs prune) matters much more than fine-tuning the per-layer allocation of a pruning schedule.** Allocation tuning gives, at best, about a one-point gain that we could not confirm.

## 8. Limitations (read before citing any number)
- **One model** (DeiT-Tiny) and one dataset (ImageNet-val subsets), training-free, CPU only. Nothing is shown for larger ViTs, other datasets, or fine-tuned methods.
- Phase 4 samples are **class-balanced over 995 classes**, not the natural ImageNet distribution; Phase 1 used only 5 classes.
- **FLOPs are analytic.** Latency is development-only, noisy and CPU-only; **FLOPs savings do not equal speedups.**
- Methods were tuned with different histories (ToMe/EViT-style on P4-DEV; Phase 3 schedules frozen earlier on other data), so the comparison is controlled but not perfectly symmetric.
- Only 10 distinct Phase 3 schedules underlie 24 replication cells.
- EViT-style is not official EViT; DiffRate semantics unverified.
- The P4-CONFIRM incident described in §6.6.

## 9. Suggested next step (proposal only; not authorised or started)
Test whether the Phase 3 allocation procedure helps when the reduction operator is a good one: **sensitivity-guided per-layer merge schedules vs equal-effort-tuned ToMe**, on DeiT-Tiny then DeiT-Small, with a **fresh** confirmatory split (P4-CONFIRM is spent). Secondary: a controlled latency study (ideally with a GPU). See `results/phase4_adaptive_method_benchmark/recommended_next_experiment.md`.

## 10. Where to find things
| What | Where |
|---|---|
| Phase 1 results | `results/deit_tiny_imagenet_dev250/` |
| Phase 2 results | `results/phase2_progressive_eval/`, `phase2_stage_ablation/`, `phase2_class_diverse_validation/`, `phase2_minimum_token_ablation/`, `phase2_final_token_trajectory_isolation/` |
| Phase 3 design and results | `results/phase3_algorithm_design/`, `results/phase3_budget_aware_allocation/result.md` |
| Phase 4 protocol, amendments, incident record | `results/phase4_adaptive_method_benchmark/` (`benchmark_protocol.md`, `protocol_amendments.md`) |
| Phase 4 confirmatory outputs | `.../confirmatory_analysis/`, `.../confirmatory_results.md`, `.../per_image_results/` |
| Phase 4 final report, claims, next experiment | `.../phase4_final_report.md`, `paper_claims_matrix.md`, `recommended_next_experiment.md` |
| Audit and publication tables | `research_paper_preparation/` |
| Specs for each phase | `Features/` |
| Code | `src/` (models, importance, compression, metrics, experiments, optimization, phase4), `scripts/`, `configs/`, `tests/` (144 tests at last run) |
| Running log | `context/progress_tracker.md`, `context/result.md` |

*Environment:* Python in the `minor` conda env; run tests with `python -m pytest tests -q`.
