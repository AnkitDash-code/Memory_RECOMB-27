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

## Stage 4 — hybrid feature message passing (NEGATIVE RESULT, with a caveat)

The pre-approved fallback: aggregate neighbour *features* in addition to
propagating addresses, i.e. give up the "replaces message passing" premise if
that is what it takes to close the gap. Implemented as an ablatable
`feature_hops` parameter (0 = pure) rather than a fork.

Smoothing **raw features before the encoder**, 3 seeds each:

| feature_hops | addr_hops=2 | addr_hops=4 |
|---|---|---|
| **0 (pure)** | 0.5108 ± 0.037 | **0.5418 ± 0.015** |
| 1 | 0.2165 ± 0.008 | 0.2232 ± 0.013 |
| 2 | 0.1901 ± 0.011 | 0.2012 ± 0.006 |
| 3 | 0.1588 ± 0.114 | 0.1623 ± 0.116 |

Adding feature message passing roughly *halves* ARI, monotonically worse with
more hops. Plausible mechanism: pre-smoothing 3000-dim expression destroys the
per-spot signal the encoder needs to discriminate between memory slots — the
representation is over-smoothed before it ever reaches the bottleneck.

**Important caveat on this result.** GraphST does **not** aggregate raw features;
it aggregates after a learned projection (`z = adj @ (feat @ W1)`). The sweep
above therefore tests "smooth-then-encode", which is *not* the placement the
reference method uses, so on its own it does not fairly refute the hybrid idea.
A second variant (`latent_hops`, smoothing the encoded representation, matching
GraphST's placement) was added specifically to test this properly rather than
claim a conclusion the experiment did not support.

Smoothing the **encoded representation** (GraphST's actual placement), 3 seeds:

| latent_hops | addr_hops=2 | addr_hops=4 |
|---|---|---|
| **0 (pure)** | 0.5108 ± 0.037 | **0.5418 ± 0.015** |
| 1 | 0.4248 ± 0.113 | 0.3369 ± 0.056 |
| 2 | 0.3503 ± 0.033 | 0.4422 ± 0.123 |
| 4 | 0.3023 ± 0.020 | 0.4849 ± 0.043 |

Placement does matter — the GraphST-style hybrid recovers to 0.485 where the
raw-feature version bottomed out at 0.22, vindicating the caveat above. But it
**still loses to pure address propagation** (0.542), and it is markedly less
stable across seeds.

**Conclusion, now tested two ways:** feature message passing does not help this
architecture. That is a *positive* finding for the project's central claim —
"memory-addressing replaces message passing" is not merely a constraint being
paid for here, it is the better-performing design in this model. It does not,
however, close the gap to GraphST, which reaches 0.590 by a different route.

## Stage 5 — capacity sweep (found a real further improvement)

With only ~7 true domains, `memory_slots=512` (the Phase 0 default) is heavily
over-parameterized. Swept codebook size, embedding width, softmax temperature,
training length, holding the Stage 2 config fixed otherwise:

| memory_slots | ARI (3 seeds) |
|---|---|
| 8 | 0.496 ± 0.143 (unstable) |
| 12 | 0.530 ± 0.001 |
| 16 | 0.498 ± 0.133 (unstable) |
| 24 | 0.556 ± 0.009 |
| **32** | **0.569 ± 0.001** |
| 64 (Stage 2 default) | 0.542 ± 0.015 |
| 128 | 0.515 ± 0.074 |
| 256 | 0.408 ± 0.083 |

Clear inverted-U with a sharp optimum at 32: smaller than 32 becomes unstable
across seeds (a discrete near-tie between assignments, plausibly), larger
monotonically dilutes each slot's signal. `memory_dim`, `temperature`, and
longer training (1200 epochs) were all neutral-to-worse and did not change this
conclusion. Confirmed at `memory_slots=32` over 5 seeds: **0.5713 ± 0.0057** —
tighter variance than the Stage 2 config, not just a higher mean.

## GraphST's own seed variance (fairness check)

Every earlier comparison used GraphST's single default-seed result (0.5902).
Comparing a 5-seed mean against someone else's 1-seed number is not a fair
comparison, so GraphST was run across 5 seeds under the identical clustering
protocol:

| seed | 41 (GraphST's own default) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| ARI | 0.5902 | 0.6153 | 0.6077 | 0.5854 | 0.5876 |

**GraphST: 0.5972 ± 0.0120 (5 seeds).** Its own default seed was not even its
best. With variance now honestly characterized on both sides:

| | ARI (5 seeds) |
|---|---|
| GraphST | 0.5972 ± 0.0120 |
| **Ours (slots=32)** | **0.5713 ± 0.0057** |
| Difference | +0.0259 in GraphST's favor |

The gap is smaller than earlier framing suggested (0.026, not the 0.04–0.10
implied by single-seed comparisons at various points in this project), and
GraphST's variance (±0.012) is more than double ours (±0.006) — but it is still
a real, consistent gap: GraphST's worst seed (0.585) still beats our best (0.582).

## Current best configuration (defaults updated in code)

`train_spatial_address_model(n_hops=4, lambda_usage=0.1, memory_slots=32,
memory_dim=128, hidden_dim=256, epochs=600)` on `preprocess_hvg()` output,
clustered with `cluster_embedding(..., refine=True)`. **0.5713 ± 0.0057** on the
tuning slice (151673), vs. GraphST's **0.5972 ± 0.0120** under the identical
protocol. Not yet evaluated on the other 11 slices.
