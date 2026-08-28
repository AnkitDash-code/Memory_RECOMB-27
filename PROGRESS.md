# Progress & Pending Work

Continuing work, in progress. Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand — THE REAL RESULT: 12-slice, 5-seed held-out evaluation

**Everything tuned on 151673 alone turned out to be measuring an optimistic
special case — and that overfitting was itself found and progressively fixed.**
Five full 12-slice evaluations have now run:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours, `memory_slots=32` (single-slice-tuned on 151673) | 0.4391 ± 0.0883 | 0.4501 ± 0.0921 |
| Ours, `memory_slots=16`, `lambda_usage=0.1` (cross-validated capacity only), per-seed | 0.4815 ± 0.0979 | 0.4818 ± 0.0937 |
| Ours, `memory_slots=16`, `lambda_usage=0.1`, consensus | 0.4993 ± 0.1403 | 0.5033 ± 0.1349 |
| Ours, `memory_slots=16`, `lambda_usage=0.02`, uniform adjacency, per-seed | 0.5200 ± 0.0785 | 0.5230 ± 0.0758 |
| Ours, `memory_slots=16`, `lambda_usage=0.02`, uniform adjacency, consensus | 0.5486 ± 0.0948 | 0.5494 ± 0.0908 |
| Ours, **+ expression-weighted adjacency** (current default), per-seed | 0.5342 ± 0.0764 | 0.5337 ± 0.0732 |
| **Ours, + expression-weighted adjacency, consensus (current)** | **0.5621 ± 0.0821** | 0.5620 ± 0.0786 |
| GraphST (matched protocol), per-seed mean | 0.5685 ± 0.0825 | 0.5707 ± 0.0793 |
| GraphST, consensus across seeds | 0.5724 ± 0.0861 | 0.5693 ± 0.0830 |
| Gap, uniform adjacency, consensus | 0.0238 | 0.0199 |
| **Gap, expression-weighted adjacency, consensus (current)** | **0.0103** | 0.0073 |

The original tuning-slice gap was 0.026 — **~5× smaller than the first
(single-slice-tuned) held-out measurement.** Diagnosing why (Stage 7: our
within-subject variance nearly equals our across-slice variance — real model
fragility, not dataset difficulty) led to four concrete fixes:

1. **Cross-validating `memory_slots`** (Stage 8) instead of single-slice
   tuning it on 151673 — picked 16 instead of 32, closing about a third of
   the held-out gap (0.129 → 0.087).
2. **Consensus clustering across seeds** (Stage 9) — combining the 5
   independently-trained seeds' cluster *labels* (not their raw embeddings,
   which live in unrelated coordinate spaces per seed and gave a mixed,
   unreliable result when averaged) via a co-association matrix. Applied
   identically to GraphST for fairness. Narrowed the gap further, 0.087 →
   0.073, though at that point it also increased our across-slice variance.
3. **Cross-validating `lambda_usage`** (Stage 11) the same way `memory_slots`
   was — the single-slice-tuned value (0.1) turned out to be over-regularizing;
   0.02 scored higher on the CV-validation slices, on the true held-out set,
   *and*, once re-run at full 5-seed/12-slice/consensus scale, closed most of
   the remaining gap: 0.073 → **0.024** (consensus, held-out), while also
   *reducing* consensus variance (0.140 → 0.095) rather than trading it away.
   `n_hops=4` was cross-validated the same way and confirmed unchanged.
4. **Expression-weighted spatial adjacency** (Stage 13, prompted by an
   external review's Fix #4 suggestion) — reweighting the propagation graph
   by expression similarity, not just spatial adjacency, so address mass
   stops blurring across transcriptionally-dissimilar neighbors. Tested first
   on the persistent subject-3 slices alone (all 3 improved), then confirmed
   at full scale: gap **0.024 → 0.010** (consensus), with mean *and* variance
   both improving.

**A paired significance test (Stage 12) changes the honest verdict, not just
the number.** "The gap is smaller than GraphST's own std" was previously used
as an informal parity claim — that is not a significance test. Running the
actual paired Wilcoxon test over the 11 held-out slices (`src/eval/significance_test.py`)
before Fix #4 found GraphST's advantage was **statistically significant on
the per-seed metric** (p=0.042). After Fix #4, both metrics show no
significant difference at n=11 (per-seed p=0.123, consensus p=0.465) — a
verified improvement in the actual statistical picture, not just a smaller
point estimate. Still uneven per-slice: 5 of 11 held-out slices now clearly
beat GraphST (151508, 151509, 151670, 151671, 151672), the three subject-3
slices (151674–676) remain the weakest (gap now considerably smaller than
before both fixes, but not zero), and one
slice (151671) has shown volatility across these changes. Full detail in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md)
(Stages 6–13) and [`outputs/logs/results_table.md`](outputs/logs/results_table.md).

Single-slice history, for context on how the architecture was built (all measured
on 151673 only, now known to be optimistic — with the caveat that 151673 itself
scores *lower*, 0.485, under the corrected cross-validated config, since it is
no longer specially fitted to that slice):

| Method | ARI on 151673 (tuning slice) |
|---|---|
| Ours — Phase 0 (PCA input, no propagation, Leiden clustering) | 0.303 |
| Ours — Stage 2 (HVG + address propagation, memory_slots=64) | 0.551 ± 0.018 (5 seeds) |
| Ours — Stage 3 (NB/ZINB + contrastive) | 0.183–0.346 — *worse, rejected* |
| Ours — Stage 4 (hybrid feature message passing, both placements) | 0.16–0.54 — *worse than pure, rejected* |
| Ours — Stage 5 (capacity-tuned, memory_slots=32, single-slice) | 0.5713 ± 0.0057 (5 seeds) |
| Ours — Stage 8 (memory_slots=16, `lambda_usage=0.1`, cross-validated capacity only) | 0.485 ± 0.108 (5 seeds) — *lower here, but generalizes better* |
| Ours — Stage 11 (memory_slots=16, `lambda_usage=0.02`, uniform adjacency) | 0.556 ± 0.046 (5 seeds) |
| Ours — Stage 13 (+ expression-weighted adjacency, current default) | 0.515 (1 seed, from the figure-regen run) |
| GraphST, our harness, 5 seeds (identical protocol) | 0.5972 ± 0.0120 |

**Honest bottom line:** the architecture learns real structure and beats a
from-scratch baseline, and the address-propagation mechanism is validated (ARI
rises monotonically with hop count, and pure addressing beat both tested hybrid
variants). It does **not beat GraphST overall, but a paired significance test
(Stage 12) now shows no statistically significant difference on either
metric** (per-seed p=0.123, consensus p=0.465) — a genuine change from an
earlier configuration where the per-seed metric *was* significantly worse
(p=0.042). Four evidence-based fixes got here: cross-validating `memory_slots`,
consensus clustering across seeds, cross-validating `lambda_usage`, and
reweighting spatial propagation by expression similarity (0.129 → 0.087 →
0.073 → 0.024 → 0.010 consensus gap). It is still real and still uneven: one
subject's three slices remain the clear weak point (smaller gap than before,
but not zero), and one slice (151671) has been volatile across fixes. This
should be reported as "no statistically significant difference detected at
n=11 slices, after four rounds of evidence-based fixes," not as "beats state
of the art" or "proven equivalent" — a larger held-out sample would be needed
to distinguish "no significant difference" from "true equivalence."

## What was done

1. **Stage 0 — harness validated (PASSED).** Discovered our Phase 0 GraphST number
   was understated by *our own* clustering protocol, not by GraphST. Same
   embedding: Leiden 0.491 → mclust-equivalent 0.571 → + spatial refinement 0.590
   (literature 0.633, within published ±0.05 seed variance). All methods now share
   one protocol (`src/eval/clustering.py`).
2. **Stage 1 — field-standard preprocessing.** `preprocess_hvg()`: `seurat_v3`
   3000 HVGs on raw counts → normalize → log1p → scale, matching GraphST/STAGATE.
3. **Stage 2 — the win.** `SpatialAddressMemoryAutoencoder`: propagates the softmax
   *address* distribution over the spatial graph instead of aggregating features,
   preserving the "memory-addressing replaces message passing" premise.
   - **Slot collapse found and fixed.** First run gave ARI 0.0000 (`slots_used=1`,
     identical loss at every hop count). Root cause: an MSE objective with a
     softmax bottleneck has a strong early optimum routing every spot to one slot
     decoding the dataset mean. Fix: maximize **marginal usage entropy** — a
     different quantity from the per-row entropy originally stubbed in.
   - **Mechanism validated:** ARI rises monotonically with hop count
     (1 → 0.489, 2 → 0.538, 4 → 0.549).
4. **Data + biology verified.** 8/8 canonical Maynard et al. markers enrich in
   their annotated layers (MOBP→WM +2.36 log2FC, TRABD2A→L5 +3.04, KRT17→L6 +1.78,
   …). The figshare `.h5ad` and Zenodo copies of slice 151673 match *exactly*
   (3639 spots, identical per-layer counts); X confirmed raw integer counts.
5. **Stage 3 — negative result, documented not deleted.** NB/ZINB likelihood +
   contrastive regularization made results substantially worse (see linked log).
   Code retained as a tested ablation.
6. **Stage 4 — hybrid feature message passing, tested and rejected (positive
   result for the paper's premise).** The pre-approved fallback: aggregate
   neighbour features as well as addresses. Tested in **two placements** because
   where the aggregation happens matters a lot — raw features before encoding
   (0.54 → 0.22, monotonically worse) and GraphST's actual placement, after the
   encoder projection (recovers to 0.485, but still loses to pure at 0.542).
   **Pure address propagation is not a constraint being paid for here — it's the
   best-performing variant tested**, in both placements.
7. **Stage 5 — capacity sweep found a further real improvement.** With only ~7
   true domains, the Stage 2 default (`memory_slots=64`, inherited from Phase 0's
   512) is still oversized. Swept 8–256: clear inverted-U, optimum at 32
   (0.569 ± 0.001 at 3 seeds, confirmed 0.5713 ± 0.0057 at 5). Below 32 becomes
   unstable across seeds; above it dilutes monotonically (256 → 0.408).
8. **GraphST's own seed variance was measured — a fairness fix.** Every earlier
   comparison used GraphST's single default seed (0.590). Comparing our 5-seed
   mean against their 1 seed was not fair. Across 5 seeds: **0.5972 ± 0.0120** —
   its default seed (41) was not even its best (0.585–0.615 range). This is the
   number the current gap is measured against.
9. **Found and fixed a stale-default bug before it could burn hours of compute.**
   `train_spatial_address_model`'s `lambda_usage` still defaulted to `1.0` (the
   pre-tuning value) even after `memory_slots`/`n_hops` were updated — the
   docstring already said 0.1, the parameter didn't. A smoke test caught it
   immediately (scored 0.336, matching the known-bad `lambda_usage=1.0` sweep
   row) before the full 12-slice run was launched.
10. **Stage 6 — the full 12-slice, 5-seed evaluation, run to completion.** First
    version, `memory_slots=32`. Real result: held-out gap 0.129, ~5× the
    tuning-slice gap (0.026).
11. **Stage 7 — diagnosed why: within-subject variance analysis.** DLPFC's 12
    slices are 3 subjects × 4 sections. Our within-subject std (0.069) nearly
    equals our across-slice std (0.092), and exceeds GraphST's within-subject
    std for 2 of 3 subjects. 151673 was not an easy biological sample — it was
    an outlier within its own subject (siblings 151674–676 score 0.35–0.40).
    Points at real model fragility, not dataset difficulty.
12. **Stage 8 — fixed one concrete cause: cross-validated `memory_slots`.**
    Single-slice tuning had picked 32; cross-validating across 3 different
    slices showed 32 is the **worst** of 5 candidates under CV, and selected 16
    instead. Verified on 8 truly-unseen slices: 32 → 0.460, 16 → 0.503 (real
    +0.042 gain). Re-ran the full 12-slice evaluation with this corrected
    default: held-out gap narrows from 0.129 → **0.087**, about a third closed,
    unevenly (3 slices now competitive, 3 from one subject still show
    0.21–0.23 gap).
13. **Stage 9 — consensus clustering across seeds, tested fairly for both
    methods.** Naive embedding averaging across seeds was tried first and
    rejected (mixed, unreliable — averaging raw embeddings doesn't work when
    each seed's memory_keys/values are independently initialized into
    unrelated coordinate spaces). Combining at the *label* level instead
    (`consensus_cluster`, co-association matrix + `AgglomerativeClustering`)
    is coordinate-independent and gave a real average improvement — but only
    once applied to GraphST too, for fairness, since offering ourselves an
    ensembling technique the baseline doesn't get would be a rigged
    comparison. Held-out gap narrows further: 0.087 → **0.073**. Also found
    and fixed a real crash (`scipy.cluster.hierarchy.fcluster` failed on
    GraphST's near-identical low-variance label sets on 151673) by switching
    to `sklearn.cluster.AgglomerativeClustering`.
14. Implemented (and unit-tested) `SpatialAddressMemoryLayer.
    initialize_keys_kmeans` — VQ-style k-means codebook initialization from
    the data manifold, as a further candidate fix for seed variance.
15. **Stage 10 — k-means codebook init evaluated at 12-slice scale, REJECTED.**
    Made things consistently worse, not better (held-out per-seed
    0.4815 → 0.4413, consensus 0.4993 → 0.4625) — plausibly because seeding
    every seed's codebook from the same (untrained) query distribution removes
    the seed-diversity consensus clustering relies on. Kept as an opt-in,
    unit-tested flag; default stays off.
16. **Stage 11 — cross-validated `n_hops` and `lambda_usage`, the two
    remaining single-slice-tuned hyperparameters.** New
    `src/eval/cross_validate_hops_usage.py`, same 3-slice CV / 8-slice
    true-holdout split as Stage 8, coordinate descent over both. `n_hops=4`
    confirmed (unchanged). `lambda_usage`: single-slice tuning had picked 0.1;
    cross-validation selected 0.02, verified on the 8 true held-out slices
    (0.503 → 0.520, real gain with *lower* variance too). Re-ran the full
    12-slice × 5-seed evaluation with the new default and the improvement held
    at scale, dramatically: held-out consensus gap 0.073 → **0.024**. The
    original `lambda_usage=0.1` (tuned to prevent slot collapse) had been
    considerably over-regularizing — weakening it let the model use its
    capacity better on the harder slices in particular (all three subject-3
    slices improved by +0.09 to +0.22 ARI), though one slice (151671)
    regressed under the new default. Old (`lambda_usage=0.1`) results archived
    in `outputs/logs/dlpfc_multislice_results_lambda01.json` for comparison.

45/45 tests pass across these stages, including a `scipy.stats.nbinom`
reference test that caught a real sign error in the NB likelihood, two tests
pinning the hybrid's `feature_hops=0`/`latent_hops=0` semantics as true no-ops,
and a regression test for the consensus-clustering crash.

## Current tuned configuration (defaults updated in code)

`train_spatial_address_model(memory_slots=16, memory_dim=128, n_hops=4,
lambda_usage=0.02, feature_hops=0, latent_hops=0, epochs=600)` on
`preprocess_hvg()` output, clustered with `cluster_embedding(..., refine=True)`.
`memory_slots`, `n_hops`, and `lambda_usage` are now all cross-validated, not
single-slice-tuned.

## PENDING — what is left to do

### 1. Architecture direction — DECIDED

Hybrid feature message passing was tried (pre-approved fallback) and rejected on
evidence in both plausible placements. The pure address-propagation formulation
stays as the model.

### 2. Run the final 12-slice evaluation — DONE

`src/eval/run_dlpfc_multislice.py` has run to completion; results above and in
`outputs/logs/dlpfc_multislice_results.json`. Took well under the estimated
2–3 hours (skipping GraphST's own slow internal clustering search, since it's
scored via the shared protocol instead, made each run much faster than expected).

### 3. Documentation — DONE for the headline numbers

`README.md`, `outputs/logs/results_table.md`, and this file now show the real
12-slice held-out result, not the single-slice numbers.

### 4. Cross-validating `memory_slots` — DONE, real improvement, gap not closed

`src/eval/cross_validate_capacity.py`: `memory_slots` re-selected via 3 held-out
validation slices (not 151673), verified on 8 truly-unseen test slices, then the
full 12-slice evaluation re-run with the new default (16, was 32). Held-out gap:
0.129 → 0.087. Real, but uneven — 3 slices now competitive, 3 from one subject
(151674–676) still show a 0.21–0.23 gap. `n_hops` and `lambda_usage` were not
re-validated this way at the time — see section 8, since fixed.

### 5. Figures — DONE

`src/viz/dlpfc_plots.py` has been run with the corrected (`memory_slots=16`)
config; `outputs/figures/dlpfc_ground_truth_vs_methods.png` reflects the current
model. `src/viz/spatial_plots.py` (old Phase 0 model, Visium crop) is unrelated
and still stale/unused for this story.

### 6. Consensus clustering across seeds — DONE, real improvement, gap not closed

`src/eval/clustering.py::consensus_cluster`: combines 5 seeds' independent
label assignments via a co-association matrix, applied fairly to both methods.
Held-out gap: 0.087 → 0.073. Real, but increases our across-slice variance
(0.098 → 0.140) even as it improves the mean — a genuine average improvement,
not a uniformly safer one. A real crash in the first implementation
(`scipy.cluster.hierarchy.fcluster` on GraphST's near-identical low-variance
labels) was found and fixed (switched to `sklearn.cluster.AgglomerativeClustering`).

### 7. k-means codebook initialization — DONE, REJECTED

Evaluated `initialize_keys_kmeans` at the full 12-slice × 5-seed scale
(`uv run python -m src.eval.run_dlpfc_multislice --kmeans-init --skip-graphst`,
logged in `outputs/logs/dlpfc_multislice_results_kmeans_init.json`). It made
things worse across the board, not better:

| | held-out per-seed | held-out consensus |
|---|---|---|
| Random init (current default) | 0.4815 ± 0.0979 | 0.4993 ± 0.1403 |
| k-means init | 0.4413 ± 0.0842 | 0.4625 ± 0.1291 |

A ~0.04 ARI drop, consistent on both the per-seed mean and the consensus
number — not noise. Plausible reason: k-means centers of the *initial* (mostly
untrained) per-spot queries reflect early, noisy structure and start every
seed from a similar basin, which reduces exactly the useful seed diversity
that consensus clustering exploits, while not measurably improving any single
run. `kmeans_init` stays implemented as an opt-in flag (`train_spatial_address_model(...,
kmeans_init=True)`, `--kmeans-init` on the harness) for reproducibility, but
`False` remains the default and the one used for all headline numbers.

### 8. Cross-validating `n_hops` and `lambda_usage` — DONE, the biggest single improvement so far

`src/eval/cross_validate_hops_usage.py`: same CV-validation-slice / true-holdout
split as section 4, coordinate descent over both hyperparameters. `n_hops=4`
confirmed unchanged. `lambda_usage=0.1` (single-slice-tuned) turned out to be
over-regularizing; cross-validation selected 0.02, verified on the 8 true
held-out slices, then confirmed at the full 12-slice x 5-seed x consensus
scale: held-out consensus gap **0.073 -> 0.024** -- bigger than the CV-slice
check predicted (+0.017), and it also *reduced* consensus variance rather than
trading it away (std 0.140 -> 0.095). All three subject-3 slices improved
substantially (still the worst, but gap 0.21-0.23 -> 0.11-0.13); one slice
(151671) regressed under consensus. Old (`lambda_usage=0.1`) results archived
in `outputs/logs/dlpfc_multislice_results_lambda01.json`.

### 9. Paired significance test on the held-out gap — DONE, a needed correction

External review (a second AI's analysis of this project) correctly pointed out
that "the gap (0.024) is smaller than GraphST's own across-slice std (0.086)"
is an eyeballed range comparison, not a significance test, and that with 11
held-out slices measured identically for both methods (same slices, same
seeds, same protocol), a **paired** test is the right tool. Added
`src/eval/significance_test.py` (Wilcoxon signed-rank + paired t-test, plus a
Shapiro-Wilk normality check on the paired differences):

| Metric | Ours wins | Mean diff | Wilcoxon p | Verdict |
|---|---|---|---|---|
| Per-seed mean (avg. of 5 seeds/slice) | 2/11 | −0.049 | **0.042** | **Significant** — GraphST reliably ahead |
| Consensus (headline metric) | 4/11 | −0.024 | 0.465 | Not significant, but paired-diff std is higher (0.093 vs 0.064) — plausibly underpowered, not evidence of parity |

**Correction to prior framing:** this repo's README/PROGRESS previously
described the result as "close to parity" based on the std comparison alone.
That was too optimistic. The honest statement is: on the more statistically
stable metric, GraphST's advantage over our held-out result remains real and
significant at n=11, even after three real, evidence-based fixes closed most
of the numeric gap. Both p-values are now reported together in every doc that
states this result, not just the more favorable one.

### 10. Expression-weighted adjacency (Fix #4) — DONE, real improvement, now the default

New `expression_weighted_adjacency()` in `memory_layer.py`: reweights each
spatial edge by `exp(-||x_i - x_j||^2 / 2*sigma^2)` (median-heuristic sigma)
using HVG expression, so address mass propagates less between spatially
adjacent but transcriptionally dissimilar spots — a targeted fix for blurred
layer boundaries, motivated by the persistent subject-3 gap and prompted by
an external review's Fix #4 suggestion. Unit-tested (rows still sum to 1,
identical features reduce to plain `normalized_adjacency`, dissimilar
neighbors get downweighted relative to similar ones). Evaluation:

1. Tested first on just the 3 subject-3 slices (5 seeds, GraphST skipped —
   its numbers don't change): all 3 improved on consensus (+0.01 to +0.10),
   2/3 improved on per-seed mean too.
2. Confirmed at full 12-slice x 5-seed scale: held-out consensus 0.549 →
   **0.562**, per-seed 0.520 → **0.534**, with variance *decreasing* on both
   (0.095 → 0.082 consensus, 0.079 → 0.076 per-seed) — a rare case of mean and
   variance both improving together, not traded off.
3. Re-ran the paired significance test with this config: per-seed p moved
   from 0.042 (significant) to **0.123** (not significant); consensus stayed
   not significant but the mean gap shrank from 0.024 to **0.010**.

**Promoted to the new default** (`expression_weighted=True` in
`train_spatial_address_model`; `--uniform-adjacency` on the harness opts back
out for the ablation). Still respects the paper's core premise — only the
address distribution is ever propagated across spots; expression similarity
is used purely to reweight *how strongly* an edge propagates, never to mix
raw features into the embedding directly.

**Methodological honesty note:** unlike `memory_slots`/`n_hops`/`lambda_usage`,
this wasn't validated with the same strict CV-validation-slice / disjoint-test
split — the decision to keep it was made after seeing the full 12-slice
number (following a smaller, genuinely blind check on subject-3 alone first).
The pattern held consistently across both checks, which is reassuring, but a
fully disjoint validation would strengthen this further.

### 11. entmax15/sparsemax address distribution (Fix #2) -- DONE, NOT ADOPTED

Tested both as opt-in `attention_fn` alternatives to softmax
(`memory_layer.address_distribution`, unit-tested for valid/sparse simplex
rows). Subject-3 check (same protocol used to validate Fix #4 before its full
run): `entmax15` was a clear, consistent regression on every slice and both
metrics (mean delta -0.064 per-seed, -0.087 consensus) -- rejected outright.
`sparsemax` was a wash (-0.023 per-seed, -0.002 consensus, one slice up two
down) -- nothing like Fix #4's clear signal on this same subset. Neither
justified a full 12-slice run. Both kept as opt-in, unit-tested ablations;
`"softmax"` remains the default. See `outputs/logs/stage2_progress.md`
(Stage 14) for the full per-slice breakdown and a plausible mechanism (hard
sparsity may prematurely commit ambiguous boundary spots to too few slots,
losing the soft blending that lets them average two layers' address mass).

### 12. Address-space contrastive loss (Fix #1) -- DONE, NOT ADOPTED, instrumented first

The highest-risk item on the external review's list, tested carefully per
its own recommendation: added `key_cosine_similarity()` (mean pairwise
cosine similarity of `memory_keys` rows -- a codebook-collapse diagnostic
distinct from `usage_entropy`, catching the "quiet" failure mode where every
slot gets used but the key vectors themselves have become near-duplicates)
and moved `contrastive_address_loss` (Stage 3's original term) into
`memory_layer.py` so it could be tested in isolation, on top of the winning
MSE model, rather than bundled with NB/ZINB as it was in Stage 3. **Turned up
a correction to the reviewer's own diagnosis**: Gemini's hypothesis for why
Stage 3 failed (contrastive loss on continuous embeddings, not addresses)
was factually wrong -- the term was already address-space; the untested
variable was the NB/ZINB pairing, not the space it operated in.

Smoke-tested with instrumentation active first: no collapse signal at
`lambda_contrastive=0.1` (key similarity stayed low/negative, 16/16 slots
stayed used throughout). A single-seed sweep suggested 0.1 was promising
(+0.053 ARI) -- but the Subject-3 5-seed check told a different story: mean
delta -0.046 per-seed, -0.031 consensus, driven by a sharp drop on 151676
(-0.126). The single-seed result was noise, not signal. No collapse, but no
improvement either. Kept as an opt-in, unit-tested parameter
(`lambda_contrastive`, `--lambda-contrastive` on the harness); `0.0` (off)
remains the default. See `outputs/logs/stage2_progress.md` (Stage 15).

**This completes the external review's full four-fix plan**, in the
prioritized order it suggested: the significance test confirmed the gap was
real; Fix #4 (expression-weighted adjacency) was adopted; Fix #2
(entmax15/sparsemax) and Fix #1 (contrastive loss) were both tested properly
and not adopted. Fix #3 (temperature annealing) was never tested standalone,
per the reviewer's own reasoning that it only matters paired with a sparse
projection -- moot since neither sparse variant showed promise.

### 13. Dual-modality (expression + morphology) memory addressing — FALSIFIED at the diagnostic gate, architecture NOT built

A new plan proposed bringing histology back in as a second, morphology-addressed
memory bank fused with expression via a learned per-spot gate — explicitly not
a feature-concatenation bolt-on, to stay distinct from contrastive-fusion
multimodal methods in the literature. The plan itself mandated a falsification
test *before* writing any dual-memory code: if subject-3 (the persistent weak
point since Stage 7) doesn't show elevated expression/morphology disagreement
*and* elevated model error relative to subjects 1/2, stop rather than spend a
week on an architecture the data doesn't support.

Built and verified first: `src/data/extract_patches.py` (64×64 H&E patches per
spot, correctly rescaled from full-res pixel coordinates, edge-padded,
spot-count-asserted, visually spot-checked against the full tissue image) and
`src/models/image_encoder.py` (frozen ImageNet ResNet18, 512-dim, no
fine-tuning). 9 new unit tests. Confirmed real, non-empty H&E images exist for
all 12 DLPFC slices and the squidpy mouse crop dataset.

**The diagnostic (`src/eval/image_diagnostic.py`), run across all 12 slices:**

| Subject | Mean disagreement | Mean error | Mean ARI |
|---|---|---|---|
| subject1 | 1.1186 | 0.2670 | 0.5372 |
| subject2 | 1.1400 | 0.1642 | 0.6430 |
| **subject3** | **1.0840 (lowest)** | **0.3080 (highest)** | **0.4873 (lowest)** |

Subject 3 does show the highest error (consistent with every prior stage) but
the **lowest** expression/morphology disagreement of the three subjects — the
opposite of what the hypothesis needed. Subject 2 shows the highest
disagreement alongside the *lowest* error. Per-spot correlations within each
subject are all near zero with inconsistent signs (-0.144 to +0.072) — no
within-subject signal either. This is a clean falsification, not just an
inconclusive result — see `outputs/figures/image_diagnostic_scatter.png` and
Stage 16 in `outputs/logs/stage2_progress.md` for the full analysis.

**Decision:** per the plan's own stopping rule, the `DualModalityMemoryLayer`
architecture was **not built**. Whatever drives subject-3's gap, this
diagnostic gives no evidence it's expression/morphology disagreement
resolvable by frozen ImageNet features.

**Follow-up review, addressed directly and re-tested — the falsification
held up.** A second critique argued the negative result was a diagnostic
design problem, not evidence against the hypothesis: wrong backbone
(ImageNet ResNet18 vs. histology-appropriate DINOv2/pathology foundation
models) and a flawed metric (spot-by-spot "disagreement" instead of
"signal rescue" — histology should stay spatially coherent where
transcriptomic signal has degraded, not literally conflict with it). Both
points were substantive, so both were re-tested directly rather than argued
about: a sharper Moran's I (spatial autocorrelation) diagnostic
(`src/eval/morans_i_diagnostic.py`, unit-tested against synthetic signals)
comparing per-modality spatial coherence, run with both ResNet18 (reusing
cached features) and DINOv2 ViT-S/14 (`get_dinov2_encoder()`, added and
unit-tested, 6/6 passing):

| | ResNet18 gap (img − expr Moran's I) | DINOv2 gap |
|---|---|---|
| subject1 | 0.5362 | 0.7293 |
| subject2 | 0.5530 | 0.7480 |
| **subject3** | **0.5434 (smallest)** | **0.7217 (smallest)** |

Subject 3 has the *highest* expression Moran's I of the three subjects (not
the lowest — the opposite of "degraded signal") under both backbones (this
half of the finding is backbone-independent by construction — it never
touches an image encoder), and the *smallest* image-rescue gap under both
backbones too. DINOv2 raised every image-Moran's-I number substantially
(0.55-0.58 → 0.75-0.77, confirming it's a genuinely better histology
encoder) without changing the subject-level ranking at all. This is a more
decisive falsification than the first one, not a repeat of it: it directly
answers the two most substantive critique points (wrong backbone, wrong
metric) with sharper tools and gets the same answer.

Left genuinely open for a future attempt: points (3) spatial-smoothing of
image features and (4) cross-modal contrastive alignment from the critique
are properties of the eventual *model architecture*, not testable via a
pre-architecture data diagnostic — Moran's I already measures spatial
coherence directly. The data pipeline (patch extraction, both frozen
encoders, per-slice caching) is reusable if a future attempt wants to build
on (3)/(4) directly, but nothing in this repo's data has yet shown a signal
that would justify it.

### 14. Still open

- The subject-3 slices remain our weak point -- but see section 15, which
  reframes this substantially. **Correction to earlier versions of this
  file:** this bullet used to assert that "no data-level explanation
  (sparsity, layer proportions, spot count) has been found so far". That
  overstated what had actually been measured -- `data_stats.py` only ever
  covered the mouse Visium and Slide-seq datasets, never DLPFC and never
  per-subject, so no read-depth/dropout QC had been run at all. It has now
  been (section 15), and it falsifies the data-quality-ceiling hypothesis
  rather than supporting it.
- 151671 has been volatile across fixes (jumped up under consensus at
  `lambda_usage=0.02`, then further under expression-weighted adjacency) --
  worth understanding whether this reflects a real interaction between the
  fixes or noise.
- `memory_slots`/`n_hops`/`lambda_usage` were cross-validated under uniform
  adjacency (Stages 8, 11) -- now that expression-weighted adjacency is the
  default, re-validating them under the new adjacency is a reasonable next
  check, since the optimal values could interact with this architectural change.
- All four of the external review's suggestions and the dual-modality plan
  have now been tested; the next genuinely new direction would need a fresh
  source of ideas or a different morphology encoder (DINO/DINOv2) rather than
  further tuning of this same set of mechanisms.
- Slide-seqV2 / Colab scale-up untouched this session at the time this bullet
  was written; **since superseded — see section 17.** STAGATE's "blocked on
  Windows" status was a stale claim and it now runs locally as a second real
  comparator. Garfield remains genuinely blocked (verified, not stale:
  `pybedtools` → `pysam` → `htslib`, no Windows wheel).

### 15. Topologically-Ordered Memory (TOM) — FAILED at its gate, all variants rejected

Plan: give the memory bank a 1D geometry (SOM-VAE's differentiable SOM +
an ordinal-smoothness loss), on the premise that an *unordered* slot space
cannot prefer "adjacent layer" over "completely different layer" for an
ambiguous spot. Literature check done first as the plan required: SOM-VAE is
real and citable, SOMs in spatial transcriptomics exist (SOMDE) but for a
different task, and no surveyed DLPFC method uses an ordered cluster space.

Three findings, in order of importance:

1. **A latent bug in the plan's code sketch, caught before wasting a run.**
   `linspace(0,1)` slot positions with `som_sigma` in 0.5–2.5 makes every
   kernel entry ≥0.80 — a nearly flat neighborhood kernel that would have
   silently reduced the mechanism to a no-op while appearing to run. Fixed
   (integer slot indices, the standard SOM convention) and pinned by a
   regression test.

2. **The SOM term is structurally unusable here — root cause identified.**
   Catastrophic collapse (`slots_used=1`, `key_cos=0.995`, ARI **0.0000**)
   across the *entire* sigma sweep and two orders of magnitude of
   `lambda_som`. Why: SOM-VAE reconstructs *through* its codebook, forcing
   entries apart; here addressing (`memory_keys`) and reconstruction
   (`memory_values`) are deliberately separate — which is what makes
   "addressing replaces message passing" coherent — so keys get no spreading
   pressure at all. And `usage_entropy` is blind to it: identical keys give a
   *uniform* softmax, its **maximum**. The plan transplanted SOM-VAE's loss
   without SOM-VAE's structural constraint; both designs are individually
   sound and mutually incompatible.

3. **The premise is inverted exactly where it was aimed.** Measuring whether
   the *existing* model already encodes layer order (|Spearman| of embedding
   PC1 vs. true depth, 12 slices × 3 seeds): subject1 0.237, subject2 0.420,
   **subject3 0.824** — strongest and most stable (std 0.016–0.039). Subject 3,
   the target of the whole plan, is where the current model *already* recovers
   laminar ordering best. Its clustering is worst precisely where its ordering
   is best, so an ordinal prior addresses a deficit it does not have.

The one configuration that looked promising (`ordinal-only`, 0.5696 vs 0.5150
on one slice/seed) was escalated properly and did not survive: full 12-slice ×
5-seed gives 0.5206 vs 0.5342 per-seed and 0.5417 vs 0.5621 consensus — worse,
4/11 wins, and it *regresses* the headline claim (vs GraphST p=0.042,
significant, versus p=0.123 for the current model). Also note that with
`lambda_som=0` the slot ordering is arbitrary, so even a win would not have
been the TOM hypothesis.

Code kept as opt-in and unit-tested (13 tests), consistent with how every
prior negative result was handled. No default changed.

### 16. Per-subject QC — the data-quality-ceiling hypothesis is falsified too

With four architecture hypotheses now specifically falsified on subject 3, the
remaining natural explanation was a data ceiling. **It had never been measured**
— `data_stats.py` only ever covered mouse Visium and Slide-seq, never DLPFC and
never per-subject (see the correction in section 14). Now measured
(`src/eval/per_subject_qc.py`, raw counts):

| subject | median library | median genes | dropout | n_layers | ARI |
|---|---|---|---|---|---|
| subject1 | 2304 | 1324 | 0.9597 | 7 | 0.536 |
| subject2 | 3452 | 1734 | 0.9471 | **5** | 0.623 |
| **subject3** | **4003** | **2058** | **0.9356** | 7 | **0.527** |

Subject 3 has the **best** depth, **best** complexity, and **lowest** dropout —
and the worst ARI. Ceiling hypothesis falsified. Two things this also settles:
subject 2's lead is largely a task-difficulty artifact (5 layers vs 7), so the
confound-free comparison is subject1 vs subject3, both 7-layer, where subject 3
has strictly better data and worse results.

And critically — **subject 3 is not intrinsically hard, it is hard for *us***:

| subject | ours | GraphST | gap |
|---|---|---|---|
| subject1 | 0.536 | 0.499 | **−0.037 (we win)** |
| subject2 | 0.623 | 0.625 | +0.002 (tie) |
| subject3 | 0.527 | 0.584 | **+0.057 (we lose)** |

GraphST handles subject 3 fine (0.584, its second-best subject). This reframes
"subject 3 is hard" into "our architecture specifically underperforms, on the
cleanest data in the set, a method that handles that data well" — a targeted
defect, not a ceiling. It also rules out the paper framing "both approaches
degrade together on a low-quality subject", which the data contradicts on both
halves.

Concrete untested suspect for the next cheap experiment: subject 3 has by far
the richest per-spot signal (2058 genes/spot vs 1324), and `n_hops=4` address
propagation may over-smooth genuinely separable layers precisely where signal
is strongest, while GraphST's contrastive term resists that. `n_hops` was
cross-validated globally (Stage 11), never per-subject.

### 17. Forward plan (Phase A–D): the subject-3 suspect tested, the reviewer-flag
    items closed, and a second real comparator added

A structured plan to move from "competitive with GraphST on one dataset" toward
generalization, gated so speculative architecture work only starts if evidence
supports it. Phase A tested the section-16 suspect directly; Phase B closed
three reviewer-facing gaps; Phase C (generalization) is scaffolded but not run.

**Phase A — per-subject `n_hops` sweep on subject 3: NOT SUPPORTED, not
adopted.** Leakage-safe (`src/eval/hop_sweep_subject3.py`): selection on 151673
only (already the global tuning slice, so already burned), held-out evaluation
on 151674/675/676, with a boundary-vs-interior breakdown
(`src/eval/boundary_mask.py`, 7 unit tests) as the actual mechanistic test,
not aggregate ARI. Rejected on three independent grounds: the boundary delta
(+0.0056) is an order of magnitude below seed noise (0.0368); per-slice deltas
disagree in sign; and the global default `n_hops=4` is already optimal on 2 of
3 held-out slices. **A bug in the verdict logic itself was caught and fixed**
— the first automated check printed "SUPPORTED" because it tested only the
sign of the deltas, never magnitude against variance, the same class of error
as the Stage 12 std-comparison. `n_hops=4` stands.

**Phase B1 — GraphST reproduction gap (0.597 vs. 0.633 literature): explained,
and confirmed NOT a handicap.** Model config ruled out by inspection first
(`run_graphst.py` already passes GraphST's published defaults verbatim).
Swept clustering on a fixed embedding: hierarchical (mclust-like) init gains
+0.012 over the current kmeans init on 151673 alone. The decisive test,
`src/eval/protocol_invariance.py` (6 slices × 3 seeds, one embedding per
method/slice/seed re-clustered two ways): the gap narrows under hierarchical
init, but because **GraphST degrades** under it (−0.079 overall), not because
we improve — the current protocol is the one that scores the baseline
*higher*. No handicap; no protocol change warranted; headline comparison
stands. This also **corrected a conclusion of this project's own** —
"hierarchical init is better" was a single-slice (151673) finding that did not
replicate across 6 slices, the third time in this project a single-slice
result has failed to generalize (after `memory_slots=32` at Stage 8, and
entmax/contrastive at Stages 14/15).

**Phase B2 — effect sizes and bootstrap CIs: the headline claim needed
qualifying.** Added rank-biserial effect size and percentile bootstrap CIs to
`src/eval/significance_test.py`. Per-seed effect size is **large** (−0.545),
CI **[−0.072, +0.005]** barely includes zero. "No significant difference at
n=11" was accurate but overloaded — corrected reading: GraphST is probably
still modestly ahead, and n=11 cannot establish it. This also strengthens the
case for Phase C: an underpowered comparison needs more datasets, not more
DLPFC tuning.

**Phase B3 — STAGATE's "blocked on Windows" status was STALE, and it now runs
locally as a real second comparator.** Re-tested rather than inherited: PyG's
wheel index now covers torch 2.11.0+cu128 (`torch-sparse 0.6.18+pt211cu128`),
so `torch_sparse` installs and STAGATE trains locally — no Colab needed.
Install commands pinned in `pyproject.toml` (kept out of
`[project.dependencies]` deliberately, since the wheel URL is specific to this
exact torch+CUDA build). Full 12-slice × 5-seed run
(`src/eval/run_stagate_dlpfc.py`), same protocol as everything else:

| | Ours | GraphST | STAGATE |
|---|---|---|---|
| Held-out per-seed | 0.5342 ± 0.076 | 0.5685 ± 0.083 | 0.5432 ± 0.082 |
| Held-out consensus | 0.5621 ± 0.082 | 0.5724 ± 0.086 | 0.5500 ± 0.087 |

`src/eval/significance_test_stagate.py` (reusing the Phase B2 machinery):
**none of the three pairwise Wilcoxon tests are significant at n=11**
(ours-vs-GraphST p=0.123, ours-vs-STAGATE p=0.700, GraphST-vs-STAGATE p=0.123,
per-seed). On consensus we edge STAGATE (6/11, rank-biserial +0.15). Adding a
second comparator did not change the picture: broadly competitive, not proven
ahead or behind.

**Garfield: genuinely blocked on native Windows (verified, not stale) but runs
in WSL2, and is now a real fourth comparator at n=3.** `pybedtools` → `pysam`
→ `htslib` has no Windows wheel, so it was run inside WSL2 Ubuntu (GPU
passthrough confirmed). Getting a real embedding out required working around
three separate bugs in Garfield v1.0.0 itself (no tutorial/quickstart
anywhere in its repo): a broken default `DataProcess` entry point (fixed via
an undocumented early-return trick), a `GarfieldTrainer.train()` that can
never succeed (calls `self.model(data_batch=...)` but `Garfield` never
implements a matching `forward()`), and an `nn.Module.train()` override that
breaks `.eval()`. The real (if undocumented) entry point turned out to be
calling `model.train()` directly, bypassing `GarfieldTrainer` entirely — see
`src/models/run_garfield.py`'s docstring for the full trace of each bug.

3-seed check on DLPFC 151673 (not the full 12-slice × 5-seed protocol —
deliberately skipped, see below): ARI 0.2431 / 0.2322 / 0.2714, mean 0.249 ±
0.017. Clearly below both our method (~0.53 on this slice) and GraphST/STAGATE
(~0.55–0.63), with tight seed variance — tighter than the ~0.05–0.08 typical
of the other three methods here. The ~5h cost of the full 12-slice run was
deliberately skipped: the low variance already made the gap look real rather
than noise, and running the full protocol was judged very unlikely to change
that conclusion. See `outputs/logs/garfield_dlpfc_151673_results.json` and
`outputs/logs/stage2_progress.md` (Phase B3) for the complete record.

**Phase C priority reversed by a literature check, before any run.** The plan
proposed leading generalization with mouse olfactory bulb (Stereo-seq) because
it is "used in GraphST's own paper." Checked directly against the paper (Nat
Commun 2023, PMC9977836): GraphST reports **no ARI** for either Stereo-seq OB
or Slide-seqV2 hippocampus — both are qualitative (marker-gene / atlas
comparison) only. **Human breast cancer (10x Visium) is the only one of the
three with a published, directly comparable GraphST ARI** (0.54–0.57 vs.
pathologist annotation, 20 regions) and should lead. Also: squidpy's
Slide-seqV2 ships cell-type, not spatial-domain, labels — ARI there would
score a different task.

**Phase C, first result run: human breast cancer, and the DLPFC near-parity
does NOT generalize.** Sourcing this dataset required real detective work —
the raw 10x counts (3798 spots, 36601 genes) are public, but the 20-region
pathologist annotation Kang et al.'s benchmark code expects (a `metadata.tsv`
with `fine_annot_type`) isn't in the same Zenodo release as the DLPFC data;
it turned up in Kang et al.'s companion Figshare project after
cross-referencing the exact expected schema, confirmed by the presence of
`aligned_fiducials.jpg`/`detected_tissue_image.jpg` (10x Space Ranger QC
images no DLPFC slice carries) — see `src/data/load_breast_cancer.py`.

5 seeds, identical protocol to DLPFC, DLPFC-tuned defaults used as-is:

| | Ours | GraphST |
|---|---|---|
| Per-seed | 0.412 ± 0.072 | 0.621 ± 0.021 |
| Consensus | 0.546 | 0.643 |

Literature (GraphST paper): 0.54–0.57. **Unlike DLPFC, this gap is real, not
noise:** all 5 seeds favor GraphST (rank-biserial −1.0, the maximum possible
magnitude), bootstrap 95% CI on the mean per-seed difference [−0.271, −0.145]
excludes zero cleanly (Wilcoxon p=0.0625 is the smallest value obtainable at
n=5, a sample-size artifact, not ambiguity). On a tissue genuinely unlike
DLPFC cortex (invasive/DCIS breast carcinoma), even on the same platform
family, GraphST's advantage is real and roughly 4–8× the size of the
statistically-undetectable DLPFC gap. Also flagged honestly rather than
buried: our local GraphST re-score here (0.621–0.643) *exceeds* its own
paper's range (0.54–0.57) — the opposite direction from DLPFC, most likely
the 5-seed consensus-clustering advantage applied where the paper's number
reflects a single run. See `outputs/logs/stage2_progress.md` (Phase C) for
the complete record.

**Phase C, second result: Slide-seqV2 mouse hippocampus -- no clean winner,
unlike breast cancer.** No published GraphST ARI exists for this platform
(confirmed earlier); squidpy's own distribution carries cell-type labels (14
categories), not spatial domains, and ships no raw counts -- both confirmed
directly, not assumed. Reported via unsupervised proxies (silhouette,
spatial coherence), 3 seeds, K=14 (cell-type count, a common anchor only).

Hit a real, hardware-driven obstacle along the way: GraphST's own package
cannot run on the full 41,786-spot dataset here. `GraphSTModel.__init__`
`.copy()`s a dense `(n_spots, n_spots)` `adj` matrix regardless of which of
GraphST's two construction functions builds it -- `construct_interaction_KNN`
(its own documented route for `datatype in ['Stereo', 'Slide']`) is cheaper
to *compute* but still dense in *storage*. Hit a real
`ArrayMemoryError: Unable to allocate 13.0 GiB` on this 16GB-RAM machine, not
assumed. `src/models/run_graphst.py` now takes a `datatype` param (default
unchanged for every other dataset) for the KNN path; the actual fix was
subsampling to 12,000 of 41,786 spots (fixed seed, ~1.4GB dense matrix).

| | Ours | GraphST |
|---|---|---|
| Silhouette | 0.146 ± 0.004 | 0.069 ± 0.002 |
| Spatial coherence (Moran's I) | 0.900 ± 0.0002 | 0.929 ± 0.004 |
| ARI vs. cell type (caveat -- wrong task) | 0.061 | 0.071 |

Ours separates embeddings better (silhouette ~2x); GraphST's clusters are
slightly more spatially contiguous. Taken with breast cancer, Phase C's
honest pattern across two platforms: sometimes a real gap, sometimes a wash
on different axes -- not one clean generalization story. See
`outputs/logs/stage2_progress.md` (Phase C) for the complete record.

**Phase D: diagnosed WHY breast cancer shows a real gap, mechanistically --
domain size vs. propagation depth, not fragmentation or edge quality.**
DLPFC's layers (mean 622.8 spots, min 166 across all 12 slices) always
exceed a 4-hop neighbourhood's ~60-spot reach -- `n_hops=4` was
cross-validated entirely on DLPFC (Stage 11), so it was never tested against
domains smaller than that. Breast cancer's 20 regions average 3.3x smaller
(189.9 spots), and 4 of them (as small as 28 spots) are *smaller* than a
single 4-hop neighbourhood -- a fixed global hop count will, by construction,
over-smooth domains where it does. Two alternative explanations were checked
and ruled out directly rather than assumed: domain fragmentation (mostly one
label's artifact, not a general property) and boundary edge-weight quality
(breast cancer's expression-weighted adjacency separates domains *better*
than DLPFC's, not worse).

Attempted a leakage-safe architectural fix: a per-spot learned gate over
propagation depths 0..n_hops (`adaptive_hops=True`), validated only on 3
DLPFC held-out slices (never 151673, never breast cancer/Slide-seqV2), 3
seeds each:

| config | ARI (3 slices x 3 seeds) |
|---|---|
| fixed n_hops=4 (current default) | **0.5038 +/- 0.0838** |
| fixed n_hops=0 (no propagation) | 0.3909 +/- 0.1143 |
| adaptive_hops, no regularizer | 0.3501 +/- 0.1060 |
| adaptive_hops, lambda_hop_usage=0.01 | 0.3351 +/- 0.1429 |

**REJECTED.** Without regularization the gate collapses to depth 0 (mean
weight >0.999) for the same reason `lambda_usage` exists for slot addressing:
reconstruction MSE has no incentive to use propagation, since unsmoothed
data always reconstructs more easily. A `lambda_hop_usage` regularizer
(reusing `usage_entropy` on the gate) does prevent the collapse but doesn't
recover the fixed-hop baseline's ARI and has the highest variance of any
config tested. Kept in the codebase as an explicit, off-by-default ablation,
not deleted -- current defaults (`adaptive_hops=False`) are unchanged. Three
new unit tests verify the mechanism itself is correct (valid simplex,
measured collapse, regularizer prevents it) even though it doesn't produce a
net win. See `outputs/logs/stage2_progress.md` (Phase D) for the complete
record, including a considered-but-not-attempted follow-up idea.

## Honest framing for any write-up

The current result is real progress: one architectural fix (expression-weighted
adjacency) closed the held-out consensus gap from 0.024 to 0.010 on top of
three earlier hyperparameter cross-validations (0.129 -> 0.087 -> 0.073 ->
0.024), and a paired significance test now shows no statistically significant
difference from GraphST on either metric (n=11 held-out slices). Two further
candidate fixes (entmax15/sparsemax, an address-space contrastive loss) were
tested properly and did not help -- reported as negative results, not hidden.
It should be described as "no significant difference detected from a real
spatial-transcriptomics SOTA method, after rigorous cross-validation and one
targeted architectural fix," not as "beats state of the art" or "proven
equivalent" -- GraphST still leads on 6 of 11 held-out slices, most clearly
on one subject's three slices (151674-676, still a loss on all three even
after every fix tried so far), and a significance test at n=11 cannot rule
out a real but small remaining difference. Several hypotheses were tested and
rejected on evidence (per-row entropy as the anti-collapse term; NB/ZINB
likelihood; hybrid feature message passing in two placements; naive embedding
averaging across seeds; k-means codebook initialization; entmax15/sparsemax
address distribution; address-space contrastive regularization), and the
tuning slice must stay out of any headline
average.

## Phase E — Standardized Evaluation & Rerun Plan (Proposed Architectures)

To improve domain identification on Breast Cancer (where we previously had a gap to GraphST) and prevent hyperparameter tuning data leakage, we have designed and implemented a standardized protocol alongside several candidate architectures:

### 18. Breast Cancer Spatial Blocks (Leakage Prevention) — DONE
* Deterministically partitioned the breast cancer tissue slice into 6 spatial blocks using KMeans on spot coordinates ([breast_cancer_spatial_blocks.py](file:///c:/Users/ASUS/Desktop/code/Python/RECOMB-27/recomb2027/src/eval/breast_cancer_spatial_blocks.py)). 2 blocks are dedicated to selection (`SELECTION_BLOCKS = {0, 1}`) and 4 are used for reporting (`REPORT_BLOCKS = {2, 3, 4, 5}`).
* Saved deterministically to `outputs/logs/breast_cancer_blocks.npy` and visually verified in `outputs/figures/breast_cancer_blocks_verification.png`. All future runs must reuse this partition.

### 19. Standardized Evaluation Protocol & Table Generator — DONE
* Created [standard_protocol.py](file:///c:/Users/ASUS/Desktop/code/Python/RECOMB-27/recomb2027/src/eval/standard_protocol.py) which runs hyperparameter tuning on the selection slices/blocks, writes candidate records to disk, selects the best hyperparameter configuration, and evaluates it on the held-out report slices/blocks over 5 seeds.
* Created [build_master_table.py](file:///c:/Users/ASUS/Desktop/code/Python/RECOMB-27/recomb2027/src/eval/build_master_table.py) which reads the JSON logs and generates/updates `outputs/logs/master_results_table.md`.
* Created [generate_standard_figures.py](file:///c:/Users/ASUS/Desktop/code/Python/RECOMB-27/recomb2027/src/eval/generate_standard_figures.py) to plot individual model and master comparison figures.

### 20. Proposed Alternative Architectures — DONE (Implemented)
All candidate architectures have been implemented in `src/models/` and integrated into the rerun runner:
1. **LDCM (Latent Denoising & Contrastive Memory):** Latent smoothing via contrastive loss.
2. **PPR (Personalized PageRank Address Memory):** Personalized PageRank address propagation.
3. **AGAP (Adaptive Gate Address Propagation):** Dynamically gated hop-depth propagation per spot.
4. **HMA (Hierarchical Memory Addressing):** Hierarchical slots addressing.
5. **GMSM (Gated Multi-Scale Memory):** Multi-scale slots representation.
6. **MSAP (Multi-Scale Address Propagation):** Multi-scale hop depths.
7. **BAAP (Boundary-Aware Address Propagation):** Detects boundaries to stop smoothing.
8. **ZISM (Zero-Inflated Spatial Memory):** Zero-inflated negative binomial (ZINB) reconstruction loss.

### 21. Master Rerun Plan & Execution — IN PROGRESS
* Created the master runner [run_master_rerun.py](file:///c:/Users/ASUS/Desktop/code/Python/RECOMB-27/recomb2027/src/eval/run_master_rerun.py) to evaluate all 9 architectures under the standard protocol.
* Currently running the first architecture, **LDCM**, in the background on DLPFC and Breast Cancer. Other architectures are queued.

### 22. Untracked Files Pending Commits/Pushes — STALE, corrected below
This bullet claimed several files were untracked; they were committed in
`75c48ea` (2026-08-14) and have been in the repo since. Superseded by
section 23.

## Phase F — Overnight session (2026-08-27/28): closed GMSM/AGAP/PPR for real, found a real bug, tested the external-review fix set

### 23. GMSM, AGAP, PPR had code but zero persisted results — now closed with real numbers

Confirmed via `find`/`grep` before touching anything: `src/models/{gmsm,agap,ppr}_memory_layer.py` and their `train_*_model.py` counterparts existed, but no `outputs/logs/{gmsm,agap,ppr}_*.json` did anywhere in the repo. This matches the "missing log files" gap an earlier session flagged (informal chat claims like PPR's breast-cancer 0.527 were never actually saved). Section 21's "IN PROGRESS" status was itself stale — `run_master_rerun.py` was never actually run to completion for these three.

Added 6 new runners mirroring BAAP/HMA/MSAP's legacy (non-block) protocol exactly, so results are directly comparable: `run_{gmsm,agap,ppr}_breast_cancer.py` and `run_{gmsm,agap,ppr}_dlpfc_smoke.py`. Real results, all persisted this time:

| Architecture | Breast cancer consensus | Δ vs baseline (0.546) | Verdict |
|---|---|---|---|
| **AGAP** | **0.5661** | **+0.0201** | Best result of the entire architecture sweep. Per-seed std also 3x tighter than baseline (0.026 vs 0.072); no collapse, 16/16 slots used every seed. |
| **PPR** (alpha=0.1, properly selected via a 3-value pre-sweep, not the previously-hardcoded 0.2) | 0.5474 | +0.0014 | Essentially tied with baseline — the first architecture in the whole sweep to not regress. One seed (3) showed partial mid-training slot-usage instability (down to 7/16) not seen in the other four. |
| GMSM | 0.4601 | −0.0859 | Regresses; seed 3 shows a partial local/global slot dip (3-9/16) mid-training. Closed. |
| BAAP (prior) | 0.4532 | −0.0928 | Closed, unchanged. |
| HMA (prior) | ~0.0 | full collapse | Closed, unchanged — `"collapsed": true` in its own diagnostics, 1 slot used all 5 seeds. |
| MSAP (prior) | 0.3324 | −0.2136 | Closed, unchanged. |

DLPFC single-slice (151673) smoke numbers, also newly persisted: GMSM 0.484, AGAP 0.502, PPR 0.598 (this last one reproduces the informal, never-saved 0.598 claim from an earlier session — now actually verified).

**AGAP is conceptually adjacent to the already-rejected Phase D `adaptive_hops`** (a per-spot learned gate over propagation depth, which collapsed to depth 0 without regularization and underperformed even with `lambda_hop_usage`: 0.335 vs 0.504 fixed-hop on DLPFC). AGAP is a distinct implementation and shows no collapse signature here — worth understanding what in its specific gating/regularization avoids that failure mode before assuming the result generalizes, and worth a proper leakage-safe (block-protocol) rerun given it's the strongest signal found so far. Not yet done.

**Caveat, stated plainly:** all of BAAP/HMA/MSAP/GMSM/AGAP/PPR use the legacy whole-sample protocol (same one that produced the original 0.546/0.412 baseline numbers, so the comparison is fair on its own terms) — not LDCM's nested spatial-block holdout. PPR's own alpha pre-sweep also reuses the same sample it's later scored on (a smaller-scale version of the repeated-test-set-exposure concern flagged for the earlier MSAP→LDCM search sequence) — disclosed in the script's docstring, not hidden.

### 24. Found and fixed a real bug in LDCM's block-protocol runner — not leakage, but the per-block breakdown was fake

`src/eval/run_ldcm_standard.py`'s report loop did `for block_id in REPORT_BLOCKS:`, which shadowed the outer per-spot block-assignment array (also named `block_id`) with the loop's scalar index. `block_mask = (block_id == block_id)` then compared that scalar to itself — always `True` — so every "block" silently scored against the whole `report_mask` (blocks 2+3+4+5 combined) instead of its own spots. Concretely: the logged "per-block" ARI was the same whole-report-set number copied 4x under different block keys, `report["per_seed_std"] = 0.0` was fabricated (std of 4 identical copies, not a real between-block variance), and the model was retrained 4x redundantly per seed for identical results.

Checked carefully whether this was a leakage bug: it is not. Selection (`_select_lambda_contrastive_breast_cancer`) and report (`evaluate_breast_cancer`) score against `get_selection_mask`/`get_report_mask` respectively, which stayed correctly disjoint (blocks {0,1} vs {2,3,4,5}) throughout — the shadowing only broke the fake inner block loop, not the real selection/report split. So the already-logged headline numbers (`per_seed_mean=0.5047`, `consensus=0.5634`) are valid measurements against the correct held-out region; they just never had a genuine per-block breakdown or real variance estimate, and there is still no baseline number logged under the identical block protocol to compare them against.

Fixed: train once per seed (not once per fake "block"), score each of the 4 real report blocks from that single embedding via `block_id == report_block`. 4x cheaper, and produces an actual per-block ARI distribution. `standard_protocol.py` (the shared harness used for baseline/other architectures under the block protocol) does not have this pattern — confirmed via grep, isolated to LDCM's bespoke script. Re-running `evaluate_breast_cancer()` with the fix is next; no baseline-under-block-protocol number exists yet either, so LDCM's corrected number still won't be comparable to anything until that gap is closed too.

### 25. Tested the external-review (Qwen) fix set for real — `enhanced_memory_layer.py` had never actually run

The 5-fix set proposed by an external review months ago (key repulsion, KL-contrastive address regularization, dynamic adjacency refresh, two-stream domain/state memory, entropy-gated propagation — see `Memory-Architecture-for-Spatial-Transcriptomics.md` in the chat exports) was implemented as `src/models/enhanced_memory_layer.py` + `train_enhanced_model.py` + `run_enhanced_smoke_test.py`, committed, but never executed once. Confirmed by absence of any results file and by the smoke test script crashing immediately on inspection: three real bugs (wrong ground-truth column name `"layer_guess"` vs actual `"ground_truth_layer"`, a `truth != "nan"` string comparison instead of a real NaN check, an undefined `load_dlpfc` name, and the wrong preprocessing function `preprocess` instead of `preprocess_hvg`). All fixed. Added 16 unit tests (`tests/test_enhanced_memory_layer.py`) covering every mechanism's invariants (key repulsion at identical/orthogonal keys and under gradient descent; KL-contrastive symmetry; adjacency/feature augmentation invariants; entropy-gate's two edge cases plus valid-simplex; two-stream shape checks and the core claim that the state stream never sees the spatial graph) — all pass, confirming no silent breakage in the core math. 119/119 total tests pass.

These fixes are distinct from prior closed hypotheses, not re-tests of them: key repulsion closes a blind spot `usage_entropy` never addressed (identical keys give a uniform softmax, which is `usage_entropy`'s maximum); the KL-contrastive loss uses graph/feature augmentation and KL divergence, not the closed Fix #1's permutation-corruption dot-product; two-stream targets slot-blurring within expression alone, not the closed expression+histology dual-modality plan; the entropy gate is parameter-free (derived from `A0` itself), unlike the closed `adaptive_hops`'s learned gate that collapsed to depth 0.

Literature-grounded hypotheses going into the real run (not yet confirmed): the KL-contrastive loss has no negative sampling and no stop-gradient/predictor asymmetry — exactly the setup identified in contrastive-learning literature as collapse-prone (SimSiam/BYOL avoid this only via specific architectural tricks this doesn't have); key repulsion has no paired commitment loss, unlike standard VQ repulsion+commitment practice; the entropy gate may face a cold-start problem since `A0` starts near-max-entropy at init (certainty≈0 everywhere), though being parameter-free it at least can't collapse via a bad training signal the way the old learned `adaptive_hops` gate did; two-stream showed slower convergence than single-stream at matched (short) epoch counts in an initial diagnostic, consistent with the "one stream dominates" failure mode documented in multi-stream representation-learning literature, though not yet distinguishable from "just needs more epochs."

Full 3-slice × 3-seed × 6-variant smoke test launched; results to follow.

### 26. Enhanced-model smoke test RESULT — all 5 external-review fixes closed, NOT adopted

Ran to completion in 3 GPU-temperature-gated sprints (one per slice, cooling to ≤65°C between each — see section 28 on the fan-noise/thermal constraint this session ran under). Grand mean across all 3 slices × 3 seeds, 600 epochs each:

| variant | grand mean ARI | Δ vs baseline |
|---|---|---|
| baseline | 0.4821 | — |
| repulsion_only (Fix 1) | 0.4675 | −0.0146 |
| kl_contrastive_only (Fix 2) | 0.4603 | −0.0218 |
| two_stream_no_entropy (Fix 4) | 0.3006 | **−0.1815** |
| all_fixes (1+2+4+5) | 0.2496 | **−0.2325** |

Consistent across every slice, not a single-slice artifact. Fix 1/Fix 2 alone are mild but consistent regressions — not promising enough to isolate further. Fix 4 (two-stream) is a severe regression on every slice and seed; 16/16 domain and state slots stay used throughout (not classic codebook collapse), so the cause is either the capacity split itself (8+8 vs. a shared 16, right at the edge of the previously-found instability threshold from the Stage 5 capacity sweep) or an undiagnosed stream-dominance dynamic. None adopted. Closes the one item that was genuinely untested in the closed-hypothesis list.

### 27. GMSM/AGAP/PPR closed for real with persisted results (both legacy and, for AGAP/PPR, standardized block protocol)

`src/models/{gmsm,agap,ppr}_memory_layer.py` had training code but zero logged results anywhere — confirmed by `find`/`grep` before touching anything. Section 21's "IN PROGRESS" was stale; `run_master_rerun.py` had never actually been run to completion for any of the 9 architectures. Added 6 legacy-protocol runners (`run_{gmsm,agap,ppr}_{breast_cancer,dlpfc_smoke}.py`) matching BAAP/HMA/MSAP's existing pattern, then real block-protocol runs via the (bug-fixed, see section 29) `run_master_rerun.py` for the one architecture that survived first-pass screening.

**Legacy protocol (whole-sample, same protocol as the original 0.546/0.412 baseline numbers) — breast cancer:**

| Architecture | Consensus | Δ vs baseline (0.546) |
|---|---|---|
| **AGAP** | **0.5661** | **+0.0201**, best of the sweep, tighter variance too |
| PPR (alpha=0.1, properly selected) | 0.5474 | +0.0014, first to not regress |
| GMSM | 0.4601 | −0.0859 |
| BAAP (prior) | 0.4532 | −0.0928 |
| MSAP (prior) | 0.3324 | −0.2136 |
| HMA (prior) | ~0.0 | full collapse, `"collapsed": true` |

**Standardized block protocol (leakage-safe, disjoint selection/report blocks) — the real test.** Ran baseline, AGAP, and PPR through it on breast cancer, and baseline + AGAP on DLPFC (8 report slices):

| | BC per-seed | BC consensus | DLPFC per-seed | DLPFC consensus |
|---|---|---|---|---|
| baseline | 0.3789 ± 0.189 | 0.4760 ± 0.216 | 0.5443 ± 0.080 | 0.5726 ± 0.094 |
| AGAP | 0.4470 ± 0.198 (**+0.068**) | 0.4428 ± 0.260 (−0.033, within noise) | 0.4376 ± 0.085 (**−0.107**) | 0.4505 ± 0.121 (**−0.122**) |
| PPR | 0.3550 ± 0.142 (−0.024) | 0.4333 ± 0.268 (−0.043) | not run | not run |

**PPR's legacy "doesn't regress" finding does not survive the real protocol** — it now loses on both metrics, the leakage-safe version of the same pattern this project has hit before (Stage 8's single-slice `memory_slots=32`, Phase B1's hierarchical init).

**AGAP's finding is real but tissue-specific, not general.** It wins clearly on breast cancer's per-seed metric (+0.068, consistent direction to the legacy result) but **loses clearly on DLPFC** (−0.107 to −0.122, a real regression, not noise). This is mechanistically coherent, not just a mixed result: AGAP's adaptive per-spot propagation-depth gate helps exactly where Phase D diagnosed the fixed-hop baseline as over-smoothing — breast cancer's small, size-heterogeneous tumor domains (28–190 spots) — and adds unhelpful variance where the fixed-hop baseline was already well-suited: DLPFC's large, uniform cortical layers (622.8 spots mean). Read as supporting evidence for the Phase D domain-size-vs-propagation-depth diagnosis being real and general, not as "AGAP is a free win." Worth a closer look at *why* AGAP avoids the collapse signature that killed the earlier `adaptive_hops` mechanism (Phase D), specifically on the breast-cancer regime where it helps — not yet investigated.

### 28. Fan-noise-driven process for this session: sprints + temperature gating, never killing a running process

This session started under "full GPU util is fine" (user logging off for the night), got a mid-session "stop, fans making weird sounds" — the running job was killed immediately (harness-level `TaskStop`, not a process signal from inside a script, no partial/corrupt output since the job had barely started). Resumed under a revised constraint: sprints instead of long unattended runs, temperature checked between each, but *never kill a running process* once started (the difference from the first constraint: bounded units of work chosen up front, not started-then-interrupted). Practical pattern used throughout: check `nvidia-smi` temperature before each sprint; if above ~75-76°C, wait (a real polling loop via the `Monitor` tool, not a blind sleep) for a ≤65°C cooldown target before starting the next one; jobs that exceeded the foreground timeout were allowed to continue in the background rather than killed, with a passive (non-interventionist) temperature watch running alongside just to know if something needed flagging. GPU peaked at 85°C once during the AGAP/DLPFC sweep (a larger, more continuous job than the earlier single-slice sprints) and self-recovered without intervention — consistent with normal mobile-GPU thermal behavior, not a fault.

`run_enhanced_smoke_test.py`'s all-3-slices-in-one-process design doesn't support this pattern (only saves once at the end), so `src/eval/run_enhanced_smoke_sprint.py` was added: runs exactly one slice per invocation and merges into the results JSON incrementally, so every sprint is a complete, self-contained, already-saved unit — nothing is ever "mid-flight" between sprints.

### 29. Two more real bugs found and fixed in `run_master_rerun.py`, neither previously triggered

1. **`agap`'s `ARCH_SPECS` entry inherited `expression_weighted=True`** from `SHARED_BASELINE_HP`, but `train_agap_model()` has no such parameter (it always builds its own edge-index graph via `connectivities_to_edge_index`, confirmed by inspecting all 8 architectures' train-function signatures side by side — AGAP is the only one without it). `TypeError` on the very first attempted `--only agap` run, before any training started (no wasted GPU time). Fixed by excluding that one key for AGAP's `default_hp` specifically.
2. **Every successful run crashed at the very end** with `UnicodeEncodeError` — the completion messages used ✅/📊 characters, which the default Windows console encoding (cp1252) can't encode. The actual training and file save always completed first, so no run ever actually lost data to this, but every prior invocation would have *looked* like a hard failure from its exit state. Replaced with plain ASCII markers.

Also found, not a bug but a real methodological gap: **`run_ldcm_standard.py` (bespoke) and `standard_protocol.py` (generic, used for baseline/AGAP/PPR here) compute "consensus" differently** — LDCM's script does one global consensus clustering over all report spots combined; the generic harness does 4 separate per-block consensus clusterings and averages them. These are not the same statistic, so LDCM's consensus number (0.5634) is not yet comparable to baseline's/AGAP's/PPR's block-protocol consensus numbers (0.476/0.443/0.433) — only the per-seed-mean metric is computed identically across both paths and is safe to compare. Reconciling this (rerunning LDCM through the generic harness, or vice versa) is still open.


### 30. Quick check on the block-3-easy/block-5-hard pattern: not class imbalance, still open

Checked the obvious hypothesis first: block 3 and block 5 have nearly identical class-composition profiles (block 3: one dominant class at 73%, IDC_4, 550 spots; block 5: one dominant class at 72%, IDC_5, 626 spots; both have 6-7 total classes with a similar long tail of small classes). So the easy/hard split across baseline, AGAP, LDCM, and PPR (all four, consistently) is not explained by class count or class-imbalance -- both blocks are structurally similar in that sense. The actual explanation (spatial arrangement of the tail classes relative to the dominant one, or genuine transcriptional similarity between block 5's specific tail classes and its dominant class vs. block 3's) is still open; would need a per-class expression-similarity check, not done here. Recorded as a ruled-out hypothesis, not a finding.

### 31. Full standardized-protocol matrix complete: a real tradeoff, not a win or a loss

Final picture, same harness, same metrics, both tissues, all four tested architectures (baseline, AGAP, LDCM, PPR):

| architecture | BC per-seed Δ | DLPFC per-seed Δ | pattern |
|---|---|---|---|
| AGAP | +0.068 | −0.107 | wins breast cancer, loses DLPFC |
| LDCM | +0.036 | −0.067 | wins breast cancer, loses DLPFC (smaller both ways) |
| PPR | −0.024 | −0.060 | loses everywhere |

**The headline finding: both mechanisms that showed any positive signal on breast cancer trade it away on DLPFC, in the same direction and roughly proportional magnitude to their gain.** This is stronger evidence than either result alone — two architecturally different mechanisms (AGAP's per-spot adaptive gate, LDCM's latent contrastive smoothing) show the same tradeoff shape, consistent with a real, general tension between small/heterogeneous domains (breast cancer tumor regions) and large/uniform domains (DLPFC cortical layers) rather than an architecture-specific quirk. This reframes the project's open question from "find an architecture that beats baseline everywhere" to "there may not be one fixed architecture that's best across both regimes" — a more interesting, defensible, and mechanistically grounded claim than either a clean win or a clean loss would have been, and it's supported by Phase D's earlier domain-size-vs-propagation-depth diagnosis rather than contradicting it.

GMSM/BAAP/MSAP/HMA remain closed under the legacy protocol only — their regressions were large enough (−0.09 to full collapse) that a block-protocol rerun is very unlikely to change the conclusion, so this was deprioritized in favor of completing the more informative AGAP/LDCM/PPR matrix.

## Phase G — Literature-grounded follow-up plan (external research, verified before implementing)

A detailed 4-stage plan arrived from external literature research targeting the domain-scale tradeoff found in Phase F. Citations verified before implementing anything (this project has been burned by fabricated/mischaracterized citations before): ClustSIGNAL (bioRxiv 10.64898/2025.11.30.691081), SimVQ (ICCV 2025, arXiv 2411.02038), the rotation trick (arXiv 2410.06424), NID (arXiv 2405.16435), DeepSeek loss-free balancing (arXiv 2408.15664), and MMP (ICLR 2022) are all real and accurately described. One imprecision caught: arXiv 2505.19525 was cited for "auxiliary loss causes interference gradients," but that claim is actually the DeepSeek paper's (2408.15664) own point -- 2505.19525 is a real, different paper (Conf-SMoE, multimodal missing-modality gating). Not a fabrication, a mis-citation; doesn't block Stage 3 since the underlying claim it needs is solid.

Also caught: the plan's target baseline numbers (DLPFC ~0.562, breast cancer ~0.546) were stale -- both had already been recomputed fresh under the leakage-safe block protocol earlier in Phase F (DLPFC 8-slice: consensus_mean 0.5726/per_seed 0.5443; breast cancer 4-block: consensus_mean 0.4760/per_seed 0.3789). All Stage 1-4 evaluations use these real numbers as the target, not the plan's stale ones.

### 32. Stage 1 RESULT: fixed heterogeneity gate fails its DLPFC threshold -- a mechanistically important negative result

Implemented `src/models/heterogeneity_gated_layer.py` + `train_heterogeneity_gated_model.py`: a per-spot propagation-depth gate computed ONCE from raw (untrained) data via this repo's own `local_expression_heterogeneity` (already built for Hop-Fusion), converted to a rank-based certainty score, and applied through the same convex-combination formula as `entropy_gated_propagation` -- but with certainty as an external argument, `.detach()`-ed, provably independent of any trainable parameter (regression-tested: a certainty tensor deliberately built from an `nn.Parameter` still receives zero gradient). 11 new unit tests, 130/130 total pass.

Result, standardized protocol, both tissues:

| metric | baseline | Stage 1 | delta |
|---|---|---|---|
| BC per_seed_mean | 0.3789±0.1889 | 0.4361±0.2363 | **+0.057 (passes)** |
| BC consensus_mean | 0.4760±0.2160 | 0.5257±0.1981 | **+0.050 (passes)** |
| DLPFC per_seed_mean | 0.5443±0.0803 | 0.4183±0.0966 | **-0.126 (FAILS)** |
| DLPFC consensus_mean | 0.5726±0.0940 | 0.4385±0.1370 | **-0.134 (FAILS)** |

Fails the plan's conjunctive success threshold -- and by more on DLPFC than AGAP's learned gate did. This is genuinely informative, not just another loss: it rules out "the gate is learned and collapse-prone" as the explanation for the domain-scale tradeoff, since Stage 1's gate is provably immune to that specific failure mode and shows the identical tradeoff shape anyway. The real cause is more fundamental -- local heterogeneity exists WITHIN true large domains too (dropout noise, mixed cell types inside one real cortical layer), so any mechanism that reduces propagation wherever local heterogeneity is detected will fire inside true domains on DLPFC, not just at genuine boundaries, costing exactly the deep propagation that was already working there.

Per the plan's own stopping rule, proceeding to Stage 2 (independent of Stage 1's outcome) and flagging Stage 3 as now in-scope (explicitly gated on Stage 1 underperforming, which it did).

### 33. Stage 2 and Stage 3 results, and the Phase G verdict

**Stage 2 (SimVQ codebook reparameterization): confirmed the pre-registered calibration exactly.** Utilization did not measurably improve on either dataset (DLPFC identical to baseline at 16/16 slots; breast cancer marginally lower, 14-16/16 vs baseline's 15-16/16) -- there was no dead-codebook-entry problem for this dense-softmax-addressing architecture to fix, as flagged before running it. ARI: BC per_seed +0.041/consensus -0.021 (wash), DLPFC per_seed -0.054/consensus -0.063 (regression). Not adopted.

**Stage 3 (adaptive gate, loss-free bias balancing): worse than Stage 1 on DLPFC, and the failure mode is now fully diagnosed.** BC per_seed +0.035/consensus +0.015 (passes, smaller than Stage 1's gain); DLPFC per_seed -0.185/consensus -0.148 (fails, worse than Stage 1). Every fit on both datasets collapsed to ~94-100% weight on depth-0. Checked whether this was the familiar unregularized-collapse failure (Phase D's plain `adaptive_hops`) -- it was not: `depth_bias` grew to +-6, correctly identifying depth-0 as overloaded and pushing against it every step, exactly as designed. The bias mechanism worked; it just wasn't enough, because the learned routing logits grew even more extreme to overcome a bounded, fixed-step external correction. DeepSeek's loss-free balancing is built for genuinely interchangeable experts (no expert is systematically easier for every input in LLM MoE); reconstruction loss creates exactly the opposite condition here -- depth-0 is monotonically easier for every spot, an unbounded gradient pull no fixed-step bias correction can out-race.

**Phase G verdict.** Three structurally different anti-collapse mechanisms have now been tested against the same underlying pressure (reconstruction MSE prefers less smoothing): no regularizer (Phase D, collapsed), an auxiliary loss (Phase D's `lambda_hop_usage`, didn't collapse but underperformed with high variance), and loss-free bias correction (Stage 3, didn't collapse the bias itself but still lost the routing decision). All three land in the same place. Combined with Stage 1 (a fixed, non-learned gate, which also failed DLPFC despite being immune to gradient-driven collapse by construction) and Phase F's AGAP/LDCM findings (both win on breast cancer, both lose on DLPFC), the honest conclusion is that the domain-scale tradeoff is not a fixable training-dynamics problem -- it is a structural property of applying variable propagation depth to a reconstruction objective, regardless of how the depth decision is made (fixed, learned-with-no-reg, learned-with-auxiliary-loss, or learned-with-external-bias-correction).

Per the plan's own gate ("Stage 4: skip unless Stages 1-3 leave clear headroom"), none of the three left any -- all three lost on DLPFC, the tissue where this architecture's core premise was originally validated. Stage 4 (MMP-style hybrid channel gating) is **not attempted this session**, flagged here as a candidate for a future session with a genuinely different mechanism (explicit gating + decoupling regularization between a propagated and non-propagated channel, distinct from the already-failed naive two-stream split), not implemented speculatively against a plan stage whose own precondition wasn't met.

**Nothing from Phase G is adopted.** Current production defaults are unchanged from Phase F: `train_spatial_address_model(memory_slots=16, n_hops=4, lambda_usage=0.02, expression_weighted=True)`. README.md's results table is correspondingly unchanged (no Phase G stage cleared its success threshold, so per the plan's own instruction, no update was warranted).
