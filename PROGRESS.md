# Progress & Pending Work

Status as of the last working session. Paused mid-effort at the user's request.
Detailed experimental numbers live in
[`outputs/logs/stage2_progress.md`](outputs/logs/stage2_progress.md); this file is
the handoff summary — where things stand, and exactly what is left.

## Where the numbers stand (DLPFC 151673, the tuning slice)

| Method | ARI vs. ground truth |
|---|---|
| Ours — Phase 0 (PCA input, no propagation, Leiden clustering) | 0.303 |
| **Ours — Stage 2 (HVG + address propagation, matched protocol)** | **0.551 ± 0.018** (5 seeds) |
| Ours — Stage 3 (NB/ZINB + contrastive) | 0.183–0.346 — *worse, rejected* |
| GraphST, our harness (identical protocol) | 0.590 |
| GraphST, literature (Kang et al. 2025) | 0.633 |

**Net: +0.248 ARI over the starting point. We do not yet beat GraphST** — 0.551 vs
0.590 is ~2σ against our own seed spread, a real deficit rather than noise.

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

39/39 tests pass, including a `scipy.stats.nbinom` reference test that caught a
real sign error in the NB likelihood.

## PENDING — what is left to do

### 1. Decide the architecture direction (blocked on a call that was never made)

The evidence points at one remaining structural difference vs. GraphST: it
aggregates neighbour *features*; we deliberately only propagate *addresses*.

- **Option A — hybrid feature message passing** (was pre-approved as a fallback,
  not yet implemented). Aggregate features as well as addresses:
  ```python
  x_prime = adjacency @ x        # <-- the missing piece
  q = encoder(x_prime)
  A = softmax(q @ keys.T)
  A = adjacency @ A              # existing address propagation
  ```
  Most likely to close the 0.04 gap. **Cost:** the paper's contribution must be
  re-framed as a hybrid, not a replacement for message passing.
- **Option B — keep the premise pure** and push unswept axes within address
  propagation: memory slot count, temperature, embedding capacity, epochs,
  learning-rate schedule. Prior sweeps suggest diminishing returns.

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
