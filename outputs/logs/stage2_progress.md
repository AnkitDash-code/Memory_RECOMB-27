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

## Stage 6 — full 12-slice, 5-seed evaluation (the real generalization number)

Everything above was measured on **151673 alone**, which was also the slice used
to choose every hyperparameter (memory_slots, n_hops, lambda_usage). That is a
tuning slice, not a test set. `src/eval/run_dlpfc_multislice.py` runs all 12
DLPFC slices × 5 seeds, holding 151673 out of the headline mean to avoid
reporting a leaked number as if it were held-out performance.

| Slice | Ours (mean ± std, 5 seeds) | GraphST (mean ± std, 5 seeds) | Gap |
|---|---|---|---|
| 151507 | 0.417 ± 0.048 | 0.514 ± 0.063 | 0.097 |
| 151508 | 0.291 ± 0.085 | 0.488 ± 0.015 | 0.197 |
| 151509 | 0.379 ± 0.034 | 0.441 ± 0.043 | 0.063 |
| 151510 | 0.479 ± 0.049 | 0.539 ± 0.030 | 0.061 |
| 151669 | 0.466 ± 0.031 | 0.592 ± 0.009 | 0.126 |
| 151670 | 0.487 ± 0.112 | 0.534 ± 0.029 | 0.047 |
| 151671 | 0.582 ± 0.110 | 0.625 ± 0.010 | 0.043 |
| 151672 | 0.589 ± 0.077 | 0.769 ± 0.006 | 0.180 |
| 151674 | 0.404 ± 0.102 | 0.618 ± 0.003 | 0.214 |
| 151675 | 0.388 ± 0.054 | 0.548 ± 0.058 | 0.161 |
| 151676 | 0.349 ± 0.054 | 0.584 ± 0.015 | 0.236 |
| **151673 (tuning slice)** | **0.571 ± 0.006** | 0.595 ± 0.014 | **0.026** |

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours | **0.4391 ± 0.0883** | 0.4501 ± 0.0921 |
| GraphST | **0.5685 ± 0.0825** | 0.5707 ± 0.0793 |
| **Gap** | **0.1294** | 0.1206 |

**This is the number that matters, and it is materially worse than the tuning-slice
result.** 151673 is not merely "a slice we happened to tune on" — it is
quantifiably the *most favorable* slice in the entire set for our method: it has
both the highest "ours" ARI (0.571, tied for best alongside 151672's 0.589) and by
a wide margin the smallest gap to GraphST (0.026, next-closest is 151671 at 0.043).
Every other slice shows a gap of 0.04–0.24. The earlier "closing the gap" framing,
built entirely on 151673, was not wrong about what was measured, but it was
measuring something that does not generalize — this is exactly the failure mode
holding out a tuning slice is meant to catch, and here it caught something real.

Also notable: our per-seed variance is high and inconsistent across slices
(0.006 on 151673 vs. 0.11 on 151670/151671) — the capacity tuning that produced a
strikingly *low* variance on 151673 (std 0.006) did not transfer that stability
either. Across the 11 held-out slices our std exceeds GraphST's in 9 of 11 (e.g.
151674: ours ±0.102 vs. GraphST ±0.003) -- so the gap is not just in central
tendency, our method is also generally less seed-stable on data it wasn't tuned
on. 151673's low variance was itself a symptom of overfitting to that slice, not
a property of the method.

**Honest conclusion:** the architecture works (it learns real structure, beats a
from-scratch baseline, and the address-propagation mechanism is validated), but it
does not beat GraphST, and the true gap (~0.13 ARI on held-out slices) is larger
than this project's local tuning suggested. Any further work should tune
hyperparameters via cross-validation across multiple slices, not a single one, to
avoid repeating this exact failure mode.

## Stage 7 — why the gap widened: within-subject variance analysis

DLPFC's 12 slices come from 3 subjects × 4 sections each (2 pairs of spatially
adjacent replicates per subject). Breaking the 12-slice result down by subject
(`src/eval/analyze_multislice_variance.py`) separates two competing
explanations for the widened gap: some biological samples are genuinely harder
(between-subject effect), vs. our model being fragile to section-to-section
variation even within one tissue block (within-subject effect — real model
instability, not dataset difficulty).

| | avg within-subject std | between-subject std |
|---|---|---|
| Ours | **0.0694** | 0.0592 |
| GraphST | 0.0494 | 0.0560 |

Our within-subject variance is **larger than our between-subject variance**, and
larger than GraphST's within-subject variance for 2 of 3 subjects (subject1:
0.068 vs. 0.036; subject3: 0.085 vs. 0.025). Concretely: 151673 (the tuning
slice, subject3) scores 0.571, but its same-subject, same-layer-count siblings
151674/151675/151676 score 0.404/0.388/0.349 — close to the worst slices in the
whole set. **151673 was not "an easy biological sample"; it was an outlier
within its own subject.** This points at real model fragility to section-level
nuisance variation (exact HVG set selected, exact spot layout, exact spatial
graph) as the dominant unaddressed problem, not something the architecture or
loss function choices investigated so far were built to handle.

Also checked and ruled out: slice size (`corr(n_spots, ARI) = -0.13`, weak).
Confirmed real: slices with only 5 annotated layers (subject2) score higher on
average (0.531) than 7-layer slices (0.410) — an easier task with fewer true
domains — but this does not explain the within-subject-3 spread since all four
of its slices have 7 layers.

## Stage 8 — cross-validated retuning fixes real overfitting (memory_slots: 32 → 16)

Stage 7 found the tuning-slice gap (0.026) was not representative because our
model is fragile to section-level variation, not because DLPFC is uniformly
easy. One concrete, fixable instance: the Stage 5 capacity sweep picked
`memory_slots=32` using **151673 alone** — a single-slice-tuned hyperparameter,
exactly the methodology that produced the misleading tuning-slice result in the
first place.

`src/eval/cross_validate_capacity.py` fixes this with a proper train/validation/
test split:

- **CV validation set** (used only to pick `memory_slots`, never 151673): one
  slice per DLPFC subject — 151508, 151670, 151674.
- **True held-out test set** (used for neither original tuning nor this CV):
  the other 8 slices.

Cross-validating `memory_slots ∈ {16, 24, 32, 48, 64}` on the validation set:

| memory_slots | CV mean (3 slices, 3 seeds) |
|---|---|
| **16** | **0.4672** |
| 24 | 0.4007 |
| 32 | 0.3839 |
| 48 | 0.4077 |
| 64 | 0.3968 |

**`memory_slots=32` — the single-slice-tuned choice — is the worst of the five
candidates under cross-validation.** `memory_slots=16` wins clearly.

Checked on the 8 true held-out slices (never used to pick anything):

| memory_slots | True held-out mean (8 slices, 3 seeds) |
|---|---|
| 32 (old, single-slice-tuned) | 0.4601 ± 0.0919 |
| **16 (new, cross-validated)** | **0.5025 ± 0.0892** |
| GraphST, same 8 slices | 0.5766 ± 0.0896 |

A real +0.042 ARI improvement from fixing the tuning *methodology*, not from a
new architecture or loss. The gap to GraphST on this fair comparison narrows
from **0.117 → 0.074** (roughly a third smaller), though it does not close.
`memory_slots=16` is now the default in `train_spatial_address_model`.

`n_hops` and `lambda_usage` were not re-validated by this cross-validation and
remain at their single-slice-tuned values (4 and 0.1) — a reasonable next step
if pursuing this further, now that the methodology for doing so correctly is
established.

## Stage 9 — consensus clustering across seeds (real, modest, applied fairly)

Motivated directly by Stage 7's finding: our per-seed ARI variance is often
higher than GraphST's on the same slice. Two aggregation strategies were
tested, both combining the SAME 5 independently-trained seeds already used
for the per-seed mean:

- **Naive embedding averaging** (average the 5 raw embeddings, cluster once):
  tested first, gave a mixed, unreliable result (2/4 test slices better, 2/4
  worse). Expected in hindsight -- each run randomly initializes its own
  `memory_keys`/`memory_values`, so different seeds' embeddings live in
  unrelated coordinate systems; averaging them blurs structure rather than
  reinforcing it.
- **Consensus clustering** (`src/eval/clustering.py::consensus_cluster`):
  cluster each seed independently as before, then combine the *label*
  assignments via a co-association matrix (fraction of seeds that agree spot
  i and j are in the same cluster) and re-cluster on that, via
  `AgglomerativeClustering(metric="precomputed")`. Coordinate-system-
  independent, so it doesn't have the alignment problem above.

**A real crash was found and fixed along the way.** The first full-run attempt
raised `ValueError: Linkage 'Z' contains excessive observations in a cluster`
from `scipy.cluster.hierarchy.fcluster(criterion="maxclust")` -- triggered by
GraphST's label sets on 151673, where its very low seed variance (std ~0.014)
produces near-identical labels across seeds and a co-association matrix full
of exact ties, a degenerate input scipy's maxclust cut is fragile to. Fixed by
switching to `sklearn.cluster.AgglomerativeClustering(metric="precomputed")`,
the standard tool for fixed-K clustering from a precomputed distance matrix;
verified directly against the real failing case (not just a synthetic repro,
which didn't reproduce scipy's specific internal fragility).

**Fairness check**: consensus was applied identically to GraphST, not just to
our method -- if it's a real improvement it must be offered to the baseline
too, or the comparison silently favors us. It is *not* uniformly better for
either method (e.g. GraphST: 0.595 → 0.536 on 151673, worse; ours: 0.694 →
0.705 on 151672, roughly neutral).

Full 12-slice, 5-seed result with consensus computed for both methods:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours, per-seed mean | 0.4815 ± 0.0979 | 0.4818 ± 0.0937 |
| **Ours, consensus** | **0.4993 ± 0.1403** | 0.5033 ± 0.1349 |
| GraphST, per-seed mean | 0.5685 ± 0.0825 | 0.5707 ± 0.0793 |
| GraphST, consensus | 0.5724 ± 0.0861 | 0.5693 ± 0.0830 |
| **Gap, per-seed mean** | 0.0870 | 0.0889 |
| **Gap, consensus (fair, both methods)** | **0.0731** | 0.0660 |

Consensus narrows the held-out gap further: 0.087 → 0.073 (about 0.014 more,
on top of the 0.042 already gained from cross-validating `memory_slots` in
Stage 8). **Honest caveat, not smoothed over**: consensus clustering
*increases* our across-slice variance (0.098 → 0.140) even as it improves the
mean -- some slices jump substantially (151671: 0.598 → 0.717; 151670: 0.506 →
0.613) while one drops hard (151676: 0.356 → 0.246). It is a real average
improvement, not a uniformly safer one.

k-means codebook initialization (`SpatialAddressMemoryLayer.
initialize_keys_kmeans`, VQ-style init from the data manifold instead of small
random noise) was implemented and unit-tested as a further candidate fix for
seed variance, but not yet evaluated at the full 12-slice scale -- a natural
next step.

## Stage 10 -- k-means codebook init at scale: REJECTED

Evaluated the k-means init from Stage 9 at the full 12-slice x 5-seed scale
(`uv run python -m src.eval.run_dlpfc_multislice --kmeans-init --skip-graphst`,
saved to `outputs/logs/dlpfc_multislice_results_kmeans_init.json`; GraphST
skipped since its numbers don't change and re-running it wastes GPU time).
The hypothesis was that seeding `memory_keys` from k-means centers of the
initial (mostly untrained) per-spot queries would reduce seed-to-seed
variance, complementing consensus clustering.

It did not. It made both the per-seed mean and the consensus result *worse*,
consistently, not just noisier:

| | Held-out per-seed | Held-out consensus | All-12 per-seed | All-12 consensus |
|---|---|---|---|---|
| Random init (current default) | 0.4815 ± 0.0979 | 0.4993 ± 0.1403 | 0.4818 ± 0.0937 | 0.5033 ± 0.1349 |
| k-means init | 0.4413 ± 0.0842 | 0.4625 ± 0.1291 | 0.4406 ± 0.0807 | 0.4619 ± 0.1236 |

A ~0.04 ARI drop on every one of the four numbers -- this is a directionally
consistent regression, not sampling noise. Per-slice variance (std) is
actually *slightly lower* with k-means init on some slices, so the original
premise (less seed-to-seed spread) has some truth to it, but the achieved
representations are worse on average. Plausible explanation: the k-means
centers are computed from the *initial* per-spot queries, before any training
signal has shaped the encoder -- these queries mostly reflect noise/PCA-like
structure rather than the layers we actually want, so k-means locks every
seed's codebook into similar, premature clusters. This removes exactly the
seed-diversity that consensus clustering depends on to average out individual
runs' idiosyncratic errors, without buying a better single run in exchange.

**Decision:** `kmeans_init` stays implemented and unit-tested
(`initialize_keys_kmeans`, `train_spatial_address_model(..., kmeans_init=True)`,
`--kmeans-init` on the harness) for reproducibility, but the default remains
`False`, and no headline number uses it.

## Current best configuration (defaults updated in code)

`train_spatial_address_model(n_hops=4, lambda_usage=0.1, memory_slots=16,
memory_dim=128, hidden_dim=256, epochs=600)` on `preprocess_hvg()` output,
clustered with `cluster_embedding(..., refine=True)`. `memory_slots=16` is
cross-validated (Stage 8), not single-slice-tuned. On the 8 truly-unseen slices:
**0.5025 ± 0.0892** vs. GraphST's **0.5766 ± 0.0896** (gap 0.074). On the tuning
slice alone (151673, not representative — see Stage 7/8):
`memory_slots=32` reached 0.5713 there specifically, but generalizes worse
(0.4601 on the true held-out set) than the CV-selected `memory_slots=16`.
