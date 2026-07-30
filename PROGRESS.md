# Progress & Pending Work

Status as of the last working session. Paused mid-effort at the user's request.
Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand — THE REAL RESULT: 12-slice, 5-seed held-out evaluation

**Everything tuned on 151673 alone turned out to be measuring an optimistic
special case — and that overfitting was itself found and partially fixed.**
Two full 12-slice evaluations have now run:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours, `memory_slots=32` (single-slice-tuned on 151673) | 0.4391 ± 0.0883 | 0.4501 ± 0.0921 |
| **Ours, `memory_slots=16` (cross-validated on 3 other slices)** | **0.4815 ± 0.0979** | **0.4818 ± 0.0937** |
| GraphST (matched protocol) | 0.5685 ± 0.0825 | 0.5707 ± 0.0793 |
| **Gap (current, cross-validated config)** | **0.0870** | 0.0889 |

The original tuning-slice gap was 0.026 — **~5× smaller than the first
(single-slice-tuned) held-out measurement.** Diagnosing why (Stage 7: our
within-subject variance nearly equals our across-slice variance — real model
fragility, not dataset difficulty) led to a concrete fix: `memory_slots` had
itself been chosen on 151673 alone. Cross-validating it across 3 different
slices (Stage 8) picked `memory_slots=16` instead of 32, and this **closed
about a third of the held-out gap** (0.129 → 0.087) — a real improvement from
fixing the tuning methodology, not a new architecture. It is uneven, though: 3
of 11 held-out slices now match or slightly beat GraphST, while three slices
from subject 3 (151674–676) still show a 0.21–0.23 gap. Full detail in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md) (Stages
6–8) and [`outputs/logs/results_table.md`](outputs/logs/results_table.md).

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
| Ours — Stage 8 (memory_slots=16, cross-validated) | 0.485 ± 0.108 (5 seeds) — *lower here, but generalizes better* |
| GraphST, our harness, 5 seeds (identical protocol) | 0.5972 ± 0.0120 |

**Honest bottom line:** the architecture learns real structure and beats a
from-scratch baseline, and the address-propagation mechanism is validated (ARI
rises monotonically with hop count, and pure addressing beat both tested hybrid
variants). It does **not** beat GraphST overall. The properly-measured,
held-out, multi-slice gap is real — smaller than first measured (0.087, not
0.129) once a genuine overfitting bug in hyperparameter selection was fixed,
but still real, still larger than any single-slice number in this project ever
suggested, and uneven: some slices are now competitive, three from one subject
are not.

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

41/41 tests pass across these stages, including a `scipy.stats.nbinom`
reference test that caught a real sign error in the NB likelihood, and two tests
pinning the hybrid's `feature_hops=0`/`latent_hops=0` semantics as true no-ops.

## Current tuned configuration (defaults updated in code)

`train_spatial_address_model(memory_slots=16, memory_dim=128, n_hops=4,
lambda_usage=0.1, feature_hops=0, latent_hops=0, epochs=600)` on
`preprocess_hvg()` output, clustered with `cluster_embedding(..., refine=True)`.

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
(151674–676) still show a 0.21–0.23 gap. `n_hops` and `lambda_usage` were **not**
re-validated this way and remain at their single-slice-tuned values (4, 0.1) —
the same overfitting risk could apply to them too, untested.

### 5. Figures — DONE

`src/viz/dlpfc_plots.py` has been run with the corrected (`memory_slots=16`)
config; `outputs/figures/dlpfc_ground_truth_vs_methods.png` reflects the current
model. `src/viz/spatial_plots.py` (old Phase 0 model, Visium crop) is unrelated
and still stale/unused for this story.

### 6. Still open

- Cross-validate `n_hops` and `lambda_usage` the same way `memory_slots` was —
  not done; both could be subject to the same single-slice overfitting.
- The subject-3 slices (151674–676) still show the largest gaps in the whole
  set (0.21–0.23) even after the capacity fix — worth investigating directly
  rather than assuming further capacity tuning alone will address it.
- Slide-seqV2 / Colab scale-up (`notebooks/04_colab_scaleup.ipynb`) untouched this
  session; STAGATE and Garfield remain blocked on Windows as documented.

## Honest framing for any write-up

The current result is real progress with a real remaining gap. It should not be
described as beating state of the art. Two hypotheses were tested and rejected on
evidence (per-row entropy as the anti-collapse term; NB/ZINB likelihood), and the
tuning slice must stay out of any headline average.
