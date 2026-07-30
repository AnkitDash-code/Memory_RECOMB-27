# Closing the SOTA gap — Embedded Memory Architecture for Spatial Transcriptomics

## Context

Phase 0 (already complete, in `recomb2027/`) built the scaffold, a trainable
`EmbeddedMemoryAutoencoder`, a Scanpy baseline, a GraphST comparison, and a real
ARI-vs-ground-truth benchmark on DLPFC 151673. The honest result: **ARI 0.303** for our
method vs. **0.491** for our local GraphST run and **0.633** for GraphST in the
literature. We beat our own baseline (0.253) but not the field.

This plan closes that gap. Four evidence-backed root causes were identified by reading
GraphST's installed source and the benchmarking literature — not guessed:

1. **No spatial aggregation in the forward pass.** GraphST's encoder does
   `z = adj @ (feat @ W1)` then `h = adj @ (z @ W2)` — explicit neighbor aggregation,
   twice. Our model has spatial information *only* as a soft penalty in the loss, so
   spot *i*'s embedding never actually sees neighbor *j*. This is the biggest gap.
2. **Wrong input/target representation.** GraphST consumes 3000 HVGs
   (`highly_variable_genes(flavor="seurat_v3", n_top_genes=3000)` → `normalize_total` →
   `log1p` → `scale(zero_center=False, max_value=10)`) and reconstructs that real
   expression matrix. We feed and reconstruct PCA-50, which discards gene-level
   biological signal and makes the objective doubly lossy.
3. **Clustering protocol mismatch.** GraphST's literature 0.633 uses **mclust**
   (`modelNames='EEE'`, i.e. a tied-covariance Gaussian mixture) on PCA-20 of the
   embedding, plus optional spatial label refinement. We ran everything through Leiden.
   Benchmarking literature reports ARI swings of ~11% from this choice alone — so our
   own GraphST number (0.491) is likely understated by our harness.
4. **Evaluation not credible yet.** One slice, one seed. The field reports mean over 12
   DLPFC slices; published work notes ARI varies ~±0.05 across seeds.

**User decisions:** preserve the "memory-addressing replaces message passing" premise via
address-space propagation, falling back to feature message passing only if that can't
close the gap; iterate on slice 151673 and do the full 12-slice run at the end; **check in
at each plateau** rather than iterating silently or stopping early.

## Approach

Staged, each stage gated on a real measured number before moving on.

### Stage 0 — Validate the harness first (cheap, highest trust value)

Before trusting any of our own numbers, confirm our benchmark can reproduce a known
result. Add to a new `src/eval/clustering.py`:

- `mclust_equivalent(embedding, n_clusters)` — `sklearn.mixture.GaussianMixture(
  covariance_type="tied")` on PCA-20 of the embedding. **`tied` is the correct mapping
  for mclust's `EEE`** (equal volume/shape/orientation = one shared full covariance),
  verified against GraphST's `utils.py::mclust_R`.
- `refine_labels_spatial(labels, coords, n_neighbors=50)` — majority vote over the *k*
  nearest spatial neighbours, mirroring GraphST's `utils.py::refine_label(radius=50)`.

Re-run GraphST on 151673 through this protocol.

**Gate:** GraphST should land near the literature's ~0.60–0.63 (up from our 0.491). If it
does, the harness is trustworthy. If it doesn't, stop and diagnose — every downstream
number depends on this. From here on, **every method uses the identical clustering
protocol** so comparisons stay fair.

### Stage 1 — Biologically-correct preprocessing

Add `preprocess_hvg()` to `src/data/preprocess.py` matching the field standard: HVG
`seurat_v3` (3000) on **raw counts**, then `normalize_total(1e4)` → `log1p` →
`scale(zero_center=False, max_value=10)`. Keep the existing `preprocess()` untouched so
the Phase 0 label-free Visium results stay reproducible.

The model then consumes and reconstructs the **3000-dim HVG matrix**, not PCA-50.

### Stage 2 — Architecture: spatial propagation in *address* space

New `SpatialAddressMemoryAutoencoder` in `src/models/memory_layer.py`, keeping the
existing `EmbeddedMemoryLayer`/`EmbeddedMemoryAutoencoder` intact as ablation baselines:

```
q       = encoder_mlp(x)              # per-spot only; no neighbour info
A       = softmax(q @ memory_keys.T)  # address distribution over slots
A       = rownorm(Â @ A)  × k hops    # <-- propagate ADDRESSES (Â = D⁻¹(adj+I))
z       = A @ memory_values           # embedding
x_hat   = decoder(z)                  # reconstruct 3000 HVGs
```

Spot **features never mix across spots** — only "which memory slot am I" mixes. This
preserves the novelty claim honestly while directly encoding the laminar prior (cortical
layers are spatially contiguous, so neighbouring spots should share slot identity).
Multi-hop (k = 1/2/3) follows MAEST's finding that combining one-hop and multi-hop
representations captures local *and* global structure. Reuse
`connectivities_to_edge_index` for graph construction; keep `attention_entropy` as the
slot-collapse check.

### Stage 3 — Loss functions matched to the biology

- **Zero-inflation-aware reconstruction.** Measured sparsity is 70–97% (`data_stats.txt`).
  Try an NB/ZINB head on raw counts as an ablation against MSE-on-scaled (stGRL uses ZINB
  for exactly this reason).
- **Anti-collapse regularization.** Feature-permutation contrastive loss (as GraphST) or
  masking (as MAEST); both papers cite preventing feature collapse as essential.

### Stage 4 — Biological validation (makes results trustable, not just high-scoring)

New `src/eval/biological_validation.py`: for each predicted domain, test enrichment of
canonical DLPFC layer markers **verified from Maynard et al. 2021** (the paper that
produced these annotations): `AQP4`→L1, `HPCAL1`→L2, `FREM3`→L3, `RORB`→L4 (widely used;
flag as convention rather than quoted from that paper), `PCP4`/`TRABD2A`→L5, `KRT17`→L6,
`MOBP`→WM. Also check predicted domains form ordered, contiguous laminar bands.

A high ARI with markers landing in the wrong domains would mean something is wrong — this
is the check that the clusters are biologically real, not just statistically separable.

### Stage 5 — Full, credible evaluation

- Extend `src/data/load_dlpfc.py` with a downloader for **all 12 DLPFC slices**
  (figshare DOI `10.6084/m9.figshare.22004273`, CC BY 4.0, 12 × ~110MB `.h5ad`, cached
  under `data/`, gitignored). Verified via the figshare API; these are the spatialLIBD
  slices used for CellCharter's benchmarking.
- New `src/eval/run_dlpfc_multislice.py`: 12 slices × 5 seeds for our method, GraphST and
  the baseline through the same protocol, reporting **mean ± std**.
- **Avoid test-set leakage:** 151673 is the tuning slice, so report the headline number as
  the mean over the **other 11 slices**, with the all-12 mean also given for comparability
  with published tables. State this split explicitly.
- Update `outputs/logs/results_table.md` and `README.md` with real numbers.

### Escalation and honesty constraints

If Stages 1–3 still trail GraphST, evaluate the user-approved fallback: a hybrid that also
aggregates features (standard message passing), with the paper's contribution re-framed
accordingly. **Check in at each plateau** before continuing.

Throughout: report every seed (no best-of-N cherry-picking), never tune against the
reported slices' labels, and if the result plateaus below SOTA, say so plainly rather than
manufacturing a win.

## Critical files

- `src/eval/clustering.py` *(new)* — mclust-equivalent + spatial refinement; the Stage 0 gate
- `src/models/memory_layer.py` — add `SpatialAddressMemoryAutoencoder`; keep existing classes as ablations
- `src/models/train_memory_layer.py` — train on HVGs, new loss terms, seed control
- `src/data/preprocess.py` — add `preprocess_hvg()` alongside existing `preprocess()`
- `src/data/load_dlpfc.py` — generalize from 151673 to all 12 slices + figshare downloader
- `src/eval/biological_validation.py` *(new)* — marker-gene enrichment
- `src/eval/run_dlpfc_benchmark.py` — single-slice iteration loop, switched to the new protocol
- `src/eval/run_dlpfc_multislice.py` *(new)* — final 12-slice × 5-seed evaluation
- `outputs/logs/results_table.md`, `README.md` — updated with real results

## Verification

- `uv run pytest` — existing 15 tests must stay green; add tests for `mclust_equivalent`
  (recovers known Gaussian blobs), `refine_labels_spatial` (cleans salt-and-pepper labels),
  address propagation (rows stay a valid simplex; k=0 reduces to the un-propagated case),
  and `preprocess_hvg` (exactly 3000 HVGs, no NaNs).
- **Stage 0 gate:** GraphST on 151673 reaches ≈0.60–0.63 under the new protocol.
- **Stage 2+ gate:** our ARI on 151673 vs. the current 0.303, same protocol, same K.
- **Stage 4 gate:** canonical markers enrich in the correct predicted domains.
- **Final:** `uv run python -m src.eval.run_dlpfc_multislice` → mean ± std ARI over 11
  held-out slices (and all 12), ours vs. GraphST vs. baseline, versus the literature table.
- GPU paced with cooldown checks between heavy runs (6GB card, ~5GB ceiling).
