# Audit summary — post-Gate-5 read-only verification (2026-11-17)

Scope: read-only audit of the frozen inputs and the single confirmatory run's outputs. Everything was produced by `research_paper_preparation/build_audit.py` (new file; creates files with exclusive-create, so it cannot overwrite). Machine-readable results: `verification_results.json` (201 integrity/statistical checks, 14 claim checks).

## What was verified (all against saved files; no inference, no re-run of the official analysis)
**Evidence integrity**
- `frozen_method_configs.json` hash equals its recorded value (`171fa7be…e5b6`) and all 32 dependency hashes it lists still match; the original protocol freeze verifies (17 files, `phase4_freeze_protocol.py --verify`); `phase4_analysis_freeze.json` matches its recorded hash (`e574c540…a2dc`); the analysis script matches its frozen hash (`8b5e5104…df9d`) and the analysis-freeze dependencies are unchanged; `protocol_amendments.md` is unchanged (`d4b5fbd8…ec7`).
- The 125 prediction arrays have the same SHA-256 values that the official analysis manifest recorded as its inputs, and all 25 analysis output files match the analysis manifest's output hashes (so neither predictions nor results changed after the analysis); every array was written before the first analysis output.
- Completion marker `completed`; run environment record `completed`, not a dry run, n = 2,985, 6,246.7 s, frozen configuration hash matches; log ends `DONE completed` with no traceback; 125 arrays, no failure record; confirm manifest hash equals the frozen identity.

**Statistics (from the saved CSVs; 22 comparison files, per-file checks within the 201 total)** — for every family: n = 2,985 in each defined cell; diff = (method count − comparator count)/n; direction label consistent with sign (method minus comparator); bootstrap CI brackets the point estimate; McNemar p equals the exact test recomputed from the stored discordant counts (tolerance 1e-6, files round to 6 dp); discordant counts reconcile with count differences; Holm-adjusted p recomputed from stored raw p's matches (tolerance scaled for 6-dp rounding); significance flag equals adjusted p < 0.05; primary families have 16 defined cells, replication 24; undefined cells (constrained-schedule cells not defined by the frozen file) are excluded from their Holm family and listed.

**Reported findings vs saved outputs** (14 claims checked): ToMe vs geometric (16 cells, both metrics); EViT-style pattern by budget; 14/16 significant for each metric; replication mean +0.617 pp, 15/9 signs, 1/24 Holm-significant (B40 2-8-10 f32, +2.48 pp), top-1 3/24, mean +0.517; ToMe better in all defined secondary/exploratory cells; EViT-style vs proposed f16 and vs single-stage; native track 35/36 and 33 significant for each metric; ToMe B50 83.0 % / 65.3 %; unpruned top-1 2,086/2,985 (derived from the saved reference array).

## Discrepancies and corrections (originals not modified)
1. **Rounding error in my earlier text.** `confirmatory_results.md` and my Gate 5 hand-off say ToMe at the DR0.7 cell had 89.7 % agreement. The saved value is 2,676/2,985 = **89.648 %, i.e. 89.6 %** (DiffRate 0.7: 90.687 %, 90.7 % is correct). `confirmatory_results.md` was left unchanged (it is hashed in the inventory); use 89.6 %. This is the one failed claim check.
2. **Omitted nuance:** I wrote that EViT-style beat single-stage in 3 of 4 cells; the exception is B40, where it was significantly *worse* (−3.65 pp agreement, −3.55 pp top-1). Against proposed f16 the significant results split 4 better / 2 worse (the two worse are B40).
3. **DiffRate vs ToMe:** my hand-off said DiffRate was "better in 35 of 36" matched comparisons; against ToMe specifically only DR0.6 was significant (agreement +2.28 pp, top-1 +1.81 pp); DR0.7–DR0.9 were not (Table 6b). Both statements are true of the saved outputs but the second is the relevant one for ToMe.
4. **Cache file:** `data/p4_confirm_cache.pt` (1.8 GB) exists again because the Gate 5 runner's loader writes it by design; the pre-run preflight (which verified its absence) preceded that. It was not touched by this audit beyond hashing.
5. Saved p-values are rounded to 6 dp; "<0.000001" in the tables stands for a stored 0.0.

## What could not be verified
- **No OS audit logs**: access to P4-CONFIRM by any process other than those recorded cannot be excluded by logs; the incident account rests on the session record and code review.
- The confirmatory predictions were not independently regenerated (explicitly out of scope), so correctness of the inference itself rests on the frozen code, the dev-time exact-reproduction checks (Gate 4 and the 50-image dry run) and the matching hashes — not on a second confirmatory computation.
- Per-cell latency on P4-CONFIRM does not exist (latency was measured only on P4-DEV); top-5 for the unpruned reference and its Wilson CI are not in the official tables.
- DiffRate semantic equivalence to official DiffRate; ToMe parity beyond the tested configurations; EViT-style numerical parity.

## Conclusion
The reported findings **match the saved confirmatory outputs**, with the corrections in items 1–3 above (one rounding slip, two nuances). No frozen file, confirmatory prediction, marker or result was modified; no inference, analysis re-run or Git commit occurred in this audit.
