# Progress & Pending Work

Status as of the last working session. Paused mid-effort at the user's request.
Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand — THE REAL RESULT: 12-slice, 5-seed held-out evaluation

**Everything tuned on 151673 alone turned out to be measuring an optimistic
special case.** The 12-slice evaluation (`src/eval/run_dlpfc_multislice.py`) has
now run to completion. Headline, 11 slices held out of the tuning slice:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours (`SpatialAddressMemoryAutoencoder`) | **0.4391 ± 0.0883** | 0.4501 ± 0.0921 |
| GraphST (matched protocol) | **0.5685 ± 0.0825** | 0.5707 ± 0.0793 |
| **Gap** | **0.1294** | 0.1206 |

**This gap is ~5× larger than the 0.026 measured on the tuning slice.** 151673 is
quantifiably the most favorable slice in the whole set for our method — highest
single-slice ARI (0.571) *and* by far the smallest gap to GraphST (next-closest
gap is 0.043, on 151671). Full per-slice table and discussion in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md) (Stage 6) and
[`outputs/logs/results_table.md`](outputs/logs/results_table.md).

Single-slice history, for context on how the architecture was built (all measured
on 151673 only, now known to be optimistic):

| Method | ARI on 151673 (tuning slice) |
|---|---|
| Ours — Phase 0 (PCA input, no propagation, Leiden clustering) | 0.303 |
| Ours — Stage 2 (HVG + address propagation, memory_slots=64) | 0.551 ± 0.018 (5 seeds) |
| Ours — Stage 3 (NB/ZINB + contrastive) | 0.183–0.346 — *worse, rejected* |
| Ours — Stage 4 (hybrid feature message passing, both placements) | 0.16–0.54 — *worse than pure, rejected* |
| Ours — Stage 5 (capacity-tuned, memory_slots=32) | 0.5713 ± 0.0057 (5 seeds) |
| GraphST, our harness, 5 seeds (identical protocol) | 0.5972 ± 0.0120 |

**Honest bottom line:** the architecture learns real structure and beats a
from-scratch baseline, and the address-propagation mechanism is validated (ARI
rises monotonically with hop count, and pure addressing beat both tested hybrid
variants). It does **not** beat GraphST, and the properly-measured gap
(~0.13 ARI, held-out, multi-slice) is real and larger than this project's own
single-slice tuning suggested at any earlier point.

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
10. **Stage 6 — the full 12-slice, 5-seed evaluation, run to completion.** See
    above. This is the real result; everything before it was tuning-slice-only.

41/41 tests pass across these stages, including a `scipy.stats.nbinom`
reference test that caught a real sign error in the NB likelihood, and two tests
pinning the hybrid's `feature_hops=0`/`latent_hops=0` semantics as true no-ops.

## Current tuned configuration (defaults updated in code)

`train_spatial_address_model(memory_slots=32, memory_dim=128, n_hops=4,
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

### 4. Still open

- **Figures**: `src/viz/dlpfc_plots.py` is written (ground truth vs. baseline vs.
  GraphST vs. ours, all under the shared clustering protocol, on DLPFC 151673)
  but has not been run yet. `src/viz/spatial_plots.py` (old Phase 0 model, Visium
  crop) is unrelated and still stale/unused for this story.
- Slide-seqV2 / Colab scale-up (`notebooks/04_colab_scaleup.ipynb`) untouched this
  session; STAGATE and Garfield remain blocked on Windows as documented.
- Hyperparameters were tuned on a single slice, which the 12-slice run showed
  does not generalize well. A proper next step (not started) would be
  cross-validating capacity/hop/usage-weight choices across multiple slices
  rather than one, before trying to close the ~0.13 ARI gap further.

## Honest framing for any write-up

The current result is real progress with a real remaining gap. It should not be
described as beating state of the art. Two hypotheses were tested and rejected on
evidence (per-row entropy as the anti-collapse term; NB/ZINB likelihood), and the
tuning slice must stay out of any headline average.
