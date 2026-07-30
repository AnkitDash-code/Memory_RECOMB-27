# Progress & Pending Work

Status as of the last working session. Paused mid-effort at the user's request.
Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand (DLPFC 151673, the tuning slice)

| Method | ARI vs. ground truth |
|---|---|
| Ours — Phase 0 (PCA input, no propagation, Leiden clustering) | 0.303 |
| Ours — Stage 2 (HVG + address propagation, memory_slots=64) | 0.551 ± 0.018 (5 seeds) |
| Ours — Stage 3 (NB/ZINB + contrastive) | 0.183–0.346 — *worse, rejected* |
| Ours — Stage 4 (hybrid feature message passing, both placements) | 0.16–0.54 — *worse than pure, rejected* |
| **Ours — Stage 5 (capacity-tuned, memory_slots=32)** | **0.5713 ± 0.0057** (5 seeds) — current best |
| GraphST, our harness, 5 seeds (identical protocol) | **0.5972 ± 0.0120** |
| GraphST, our harness, single default seed | 0.590 |
| GraphST, literature (Kang et al. 2025) | 0.633 |

**Net: +0.268 ARI over the starting point. We do not yet beat GraphST** — the
honestly-characterized gap (both sides now measured across 5 seeds, not our
5 vs. their 1) is **+0.026 in GraphST's favor**. Smaller than earlier framing
suggested, and GraphST's own seed variance (±0.012) is over 2× ours (±0.006), but
still a consistent gap: GraphST's worst of 5 seeds (0.585) beats our best (0.582).

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

39/41 → 41/41 tests pass across these stages, including a `scipy.stats.nbinom`
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
stays as the model. Not open anymore, no decision needed here.

### 2. Run the final 12-slice evaluation (harness written, never executed)

`src/eval/run_dlpfc_multislice.py` is complete and ready. It already implements
the methodology safeguards:
- headline = mean over the **11 held-out slices**, with 151673 excluded because it
  was the tuning slice (prevents leakage); all-12 mean reported separately
- identical clustering protocol and true K for every method
- all seeds reported, mean ± std, never best-of-N

```bash
uv run python -m src.eval.run_dlpfc_multislice --model mse --seeds 5
```
Estimated 2–3 hours on the RTX 4050; pace with GPU cooldown checks. Downloads the
12 slices (~1.3GB, CC BY 4.0) on first run into `data/` (gitignored).

### 3. Documentation not yet updated

`README.md` and `outputs/logs/results_table.md` still show the **old Phase 0
numbers** (ours 0.303, GraphST 0.491). They must be refreshed once the multi-slice
run lands — do not publish the current README as-is, it understates both our
method and GraphST.

### 4. Also still open (lower priority)

- Figures (`src/viz/spatial_plots.py`) still visualize the old Phase 0 model.
- Slide-seqV2 / Colab scale-up (`notebooks/04_colab_scaleup.ipynb`) untouched this
  session; STAGATE and Garfield remain blocked on Windows as documented.

## Honest framing for any write-up

The current result is real progress with a real remaining gap. It should not be
described as beating state of the art. Two hypotheses were tested and rejected on
evidence (per-row entropy as the anti-collapse term; NB/ZINB likelihood), and the
tuning slice must stay out of any headline average.
