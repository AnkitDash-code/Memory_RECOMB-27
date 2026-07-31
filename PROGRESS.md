# Progress & Pending Work

Continuing work, in progress. Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand — THE REAL RESULT: 12-slice, 5-seed held-out evaluation

**Everything tuned on 151673 alone turned out to be measuring an optimistic
special case — and that overfitting was itself found and progressively fixed.**
Four full 12-slice evaluations have now run:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours, `memory_slots=32` (single-slice-tuned on 151673) | 0.4391 ± 0.0883 | 0.4501 ± 0.0921 |
| Ours, `memory_slots=16`, `lambda_usage=0.1` (cross-validated capacity only), per-seed | 0.4815 ± 0.0979 | 0.4818 ± 0.0937 |
| Ours, `memory_slots=16`, `lambda_usage=0.1`, consensus | 0.4993 ± 0.1403 | 0.5033 ± 0.1349 |
| Ours, `memory_slots=16`, `lambda_usage=0.02` (fully cross-validated), per-seed | 0.5200 ± 0.0785 | 0.5230 ± 0.0758 |
| **Ours, `memory_slots=16`, `lambda_usage=0.02`, consensus** | **0.5486 ± 0.0948** | 0.5494 ± 0.0908 |
| GraphST (matched protocol), per-seed mean | 0.5685 ± 0.0825 | 0.5707 ± 0.0793 |
| GraphST, consensus across seeds | 0.5724 ± 0.0861 | 0.5693 ± 0.0830 |
| Gap, `lambda_usage=0.1`, consensus | 0.0731 | 0.0660 |
| **Gap, `lambda_usage=0.02`, consensus (current, fair to both methods)** | **0.0238** | 0.0199 |

The original tuning-slice gap was 0.026 — **~5× smaller than the first
(single-slice-tuned) held-out measurement.** Diagnosing why (Stage 7: our
within-subject variance nearly equals our across-slice variance — real model
fragility, not dataset difficulty) led to three concrete fixes:

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

The remaining gap (0.024 held-out consensus) is now smaller than GraphST's own
across-slice standard deviation (0.086) — close to parity on average, though
still uneven per-slice: 3 of 11 held-out slices now clearly beat GraphST
(151508, 151509, 151670) plus a near-exact tie on 151672 (+0.0003), while
GraphST leads on the remaining 7 — most starkly the three subject-3 slices
(151674–676), which improved substantially (gap 0.21–0.23 → 0.11–0.13) but are
still the worst, and one previously-strong slice (151671) regressed under
consensus at the new `lambda_usage` (0.717 → 0.594). Full detail in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md)
(Stages 6–11) and [`outputs/logs/results_table.md`](outputs/logs/results_table.md).

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
| Ours — Stage 11 (memory_slots=16, `lambda_usage=0.02`, fully cross-validated) | 0.556 ± 0.046 (5 seeds) |
| GraphST, our harness, 5 seeds (identical protocol) | 0.5972 ± 0.0120 |

**Honest bottom line:** the architecture learns real structure and beats a
from-scratch baseline, and the address-propagation mechanism is validated (ARI
rises monotonically with hop count, and pure addressing beat both tested hybrid
variants). It does **not clearly beat GraphST overall, but the two are now
close to statistical parity on average** (held-out consensus gap 0.024,
smaller than GraphST's own 0.086 across-slice std) after three genuine,
evidence-based fixes to how hyperparameters were selected and how seeds are
aggregated (0.129 → 0.087 → 0.073 → 0.024). It is still real and still uneven:
one subject's three slices remain the clear weak point (gap 0.11–0.13, down
from 0.21–0.23), and one slice regressed. This should be reported as "closed
most of the gap through rigorous cross-validation, not yet closed entirely,"
not as "beats state of the art."

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

### 9. Still open

- The subject-3 slices (151674-676) still show the largest gaps in the whole
  set (0.11-0.13) even after all three fixes (capacity, consensus, `lambda_usage`)
  -- worth investigating directly; no data-level explanation (sparsity, layer
  proportions, spot count) has been found so far.
- 151671 regressed under consensus clustering at the new `lambda_usage`
  (0.717 -> 0.594) -- worth understanding whether this is a real interaction
  or noise, since it's currently the single largest per-slice regression from
  an otherwise-broad improvement.
- Slide-seqV2 / Colab scale-up (`notebooks/04_colab_scaleup.ipynb`) untouched this
  session; STAGATE and Garfield remain blocked on Windows as documented.

## Honest framing for any write-up

The current result is real progress with a small remaining gap (0.024 ARI,
held-out, consensus-clustered, both methods compared identically -- smaller
than GraphST's own across-slice standard deviation). It should be described as
"closes most of the gap to a real spatial-transcriptomics SOTA method through
rigorous cross-validation," not as "beats state of the art" -- GraphST still
leads on 7 of 11 held-out slices, most clearly on one subject's three slices.
Several hypotheses were tested and rejected on evidence (per-row entropy as
the anti-collapse term; NB/ZINB likelihood; hybrid feature message passing in
two placements; naive embedding averaging across seeds; k-means codebook
initialization), and the tuning slice must stay out of any headline average.
