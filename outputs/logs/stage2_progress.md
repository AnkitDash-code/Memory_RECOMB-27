# Stage 0-2 progress: closing the SOTA gap (DLPFC 151673, tuning slice)

All numbers measured locally on RTX 4050. Tuning slice only — final claims require
the multi-slice run (Stage 5). No number here is copied or estimated.

## Stage 0 — harness validation (PASSED)

Our Phase 0 GraphST number was understated by our own clustering protocol, not by
GraphST. Same GraphST embedding, three clustering protocols:

| Protocol | ARI |
|---|---|
| Leiden (Phase 0 protocol) | 0.4905 |
| mclust-equivalent (GMM, `covariance_type="tied"` = mclust `EEE`) | 0.5713 |
| mclust-equivalent + spatial refinement | **0.5902** |
| Literature (Kang et al. 2025) | 0.6327 |

0.590 vs. 0.633 is within the ±0.05 seed variance published work reports, so the
harness is trustworthy. **Every method is now scored with this identical protocol.**

## Data biological validation (PASSED, model-independent)

Canonical DLPFC layer markers from Maynard et al. 2021 vs. the annotated layers.
This never looks at a model, so it cannot be gamed — it validates the data itself.

| Layer | Marker | Provenance | Mean in layer | Mean outside | log2 FC | Enriched |
|---|---|---|---|---|---|---|
| Layer1 | AQP4 | verified | 0.935 | 0.649 | +0.53 | yes |
| Layer2 | HPCAL1 | verified | 1.682 | 0.722 | +1.22 | yes |
| Layer3 | FREM3 | verified | 0.107 | 0.030 | +1.82 | yes |
| Layer4 | RORB | convention | 0.520 | 0.214 | +1.28 | yes |
| Layer5 | TRABD2A | verified | 0.188 | 0.023 | +3.04 | yes |
| Layer5 | PCP4 | verified | 1.468 | 0.562 | +1.38 | yes |
| Layer6 | KRT17 | verified | 0.607 | 0.177 | +1.78 | yes |
| WM | MOBP | verified | 2.647 | 0.515 | +2.36 | yes |

**8/8 markers enriched in their annotated layer.** Independently, the figshare
`.h5ad` copy of this slice matches our Zenodo-derived copy exactly (3639 spots;
identical per-layer counts), and X is confirmed raw integer counts as required by
`seurat_v3` HVG selection.

## Stage 2 — architecture: spatial propagation in address space

**Slot collapse found and fixed.** The first run of the new architecture gave
ARI = 0.0000 with `slots_used=1` and identical `recon=0.8469` at every hop count —
an optimization failure, not an architecture failure: with an MSE objective and a
softmax bottleneck, routing every spot to one slot decoding the dataset mean is a
strong early optimum.

The fix is a **marginal usage-entropy** term, which is a different quantity from the
per-row entropy originally stubbed in:

- per-row entropy high → each spot smeared across slots (mushy, undesirable)
- **usage entropy high → all slots used across the dataset**, while each spot may
  still commit confidently

Hop-count sweep (`lambda_usage=0.1`), showing address propagation genuinely helps:

| hops | 1 | 2 | 4 |
|---|---|---|---|
| ARI | 0.4889 | 0.5375 | **0.5487** |

Usage-weight sweep: 0.1 best; 1.0 → ~0.34; 5.0 → ~0.22 (over-regularizing toward
uniform usage destroys structure). Per-spot sharpening (`lambda_sharpen`) was
harmful — at low `lambda_usage` it re-triggered full collapse.

## Where this stands

| Method | ARI (DLPFC 151673) |
|---|---|
| Ours, Phase 0 (PCA input, no propagation, Leiden) | 0.303 |
| **Ours, Stage 2 (HVG + address propagation, matched protocol)** | **0.5510 ± 0.0178** (5 seeds) |
| GraphST, our harness (same protocol) | 0.5902 |
| GraphST, literature | 0.6327 |

Real improvement of +0.248 ARI, and the mechanism is validated (more hops → higher
ARI monotonically). **We do not yet beat GraphST**: 0.551 vs 0.590 is roughly a 2σ
gap against our own ±0.018 seed spread, so it is a real deficit, not noise.

## Stage 3 — NB/ZINB likelihood + contrastive regularization (NEGATIVE RESULT)

Hypothesis: with 68–97% zeros, MSE on scaled expression is the wrong likelihood;
an NB/ZINB head on raw counts should help (it is stGRL's core contribution).
Implemented `src/models/count_losses.py` (NB + ZINB, verified against
`scipy.stats.nbinom` — the reference test caught a real sign error) and
`src/models/train_count_model.py`, then measured it. **The hypothesis was wrong
for this model:**

| Variant | ARI (3 seeds) |
|---|---|
| **MSE (Stage 2)** | **0.551 ± 0.018** |
| NB, no contrastive | 0.346 ± 0.095 |
| ZINB, no contrastive | 0.331 ± 0.072 |
| NB + contrastive | 0.251 ± 0.028 |
| ZINB + contrastive | 0.183 ± 0.141 |

Every count-likelihood variant is substantially worse, and seed variance grows
4–8×. Likely cause: the NB/ZINB decoder must fit per-gene dispersion across 3000
genes, a much harder optimization target than MSE, so the *encoder* receives a
noisier gradient and the embedding we actually cluster on degrades — a more
principled likelihood does not automatically yield a more clusterable
representation. Contrastive regularization made things worse in every pairing.

This code is kept (tested, working) as a documented negative result and ablation,
not deleted — but MSE remains the configuration to use.

## Current best configuration

`train_spatial_address_model(n_hops=4, lambda_usage=0.1, lambda_sharpen=0.0,
memory_slots=64, memory_dim=128, hidden_dim=256, epochs=600)` on
`preprocess_hvg()` output, clustered with
`cluster_embedding(..., refine=True)`.
