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

## Stage 11 -- cross-validating n_hops and lambda_usage: the biggest single fix so far

The last two hyperparameters still chosen on 151673 alone. New
`src/eval/cross_validate_hops_usage.py`, same 3-slice CV-validation / 8-slice
true-holdout split as Stage 8 (`cross_validate_capacity.py`), coordinate
descent: sweep `n_hops` with `lambda_usage` held at its old default, then
sweep `lambda_usage` with `n_hops` fixed at whatever won.

```
=== Cross-validating n_hops on ['151508', '151670', '151674'] (lambda_usage=0.1) ===
n_hops=1  CV mean=0.4009   n_hops=2  CV mean=0.4338   n_hops=3  CV mean=0.4192
n_hops=4  CV mean=0.4672   n_hops=6  CV mean=0.4099   n_hops=8  CV mean=0.4193
Selected n_hops=4 (unchanged from single-slice tuning)

=== Cross-validating lambda_usage on same slices (n_hops=4) ===
lambda_usage=0.02  CV mean=0.5310   lambda_usage=0.05  CV mean=0.4618
lambda_usage=0.1   CV mean=0.4672   lambda_usage=0.2   CV mean=0.4020
lambda_usage=0.5   CV mean=0.3966
Selected lambda_usage=0.02 (single-slice tuning had picked 0.1)

=== True held-out check (8 slices, 3 seeds) ===
n_hops=4, lambda_usage=0.02: 0.5196 +/- 0.0721
n_hops=4, lambda_usage=0.1  (old default): 0.5025 +/- 0.0892
```

`n_hops=4` confirmed unchanged -- not overfit. `lambda_usage`, however, was:
0.1 had been chosen only to prevent slot collapse (see Stage 2), with no
attention paid to whether it was otherwise optimal, and it wasn't -- 0.02 is
both higher-scoring and *lower-variance* on the true held-out check
(+0.017 mean, -0.017 std).

Updated the default (`train_spatial_address.py`: `lambda_usage=0.1 -> 0.02`)
and re-ran the full 12-slice x 5-seed x consensus evaluation to confirm this
holds at scale, not just on the 3-seed/8-slice CV check. It held, and the
effect was considerably larger than the CV check predicted:

| | Held-out per-seed | Held-out consensus | All-12 per-seed | All-12 consensus |
|---|---|---|---|---|
| `lambda_usage=0.1` (previous default) | 0.4815 ± 0.0979 | 0.4993 ± 0.1403 | 0.4818 ± 0.0937 | 0.5033 ± 0.1349 |
| **`lambda_usage=0.02` (new default)** | **0.5200 ± 0.0785** | **0.5486 ± 0.0948** | 0.5230 ± 0.0758 | 0.5494 ± 0.0908 |
| GraphST (unchanged, same seeds) | 0.5685 ± 0.0825 | 0.5724 ± 0.0861 | 0.5707 ± 0.0793 | 0.5693 ± 0.0830 |
| Gap (consensus) | 0.0731 | **0.0238** | -- | 0.0199 |

Held-out consensus gap closed from 0.073 to **0.024** -- smaller than
GraphST's own across-slice std (0.086), i.e. close to parity on average. Per-
slice detail (`outputs/logs/dlpfc_multislice_results.json`, old run archived
in `outputs/logs/dlpfc_multislice_results_lambda01.json`):

| Slice | Old (0.1) per-seed | New (0.02) per-seed | Old consensus | New consensus |
|---|---|---|---|---|
| 151507 | 0.516 | 0.509 | 0.561 | 0.549 |
| 151508 | 0.494 | 0.456 | 0.541 | 0.600 |
| 151509 | 0.440 | 0.475 | 0.466 | 0.462 |
| 151510 | 0.487 | 0.499 | 0.453 | 0.507 |
| 151669 | 0.467 | 0.496 | 0.454 | 0.470 |
| 151670 | 0.506 | 0.618 | 0.613 | 0.659 |
| 151671 | 0.598 | 0.574 | 0.717 | 0.594 |
| 151672 | 0.694 | 0.709 | 0.705 | 0.769 |
| 151674 | 0.398 | 0.495 | 0.417 | 0.510 |
| 151675 | 0.339 | 0.432 | 0.319 | 0.455 |
| 151676 | 0.356 | 0.457 | 0.246 | 0.462 |

**Honest read.** 8 of 11 held-out slices improved on per-seed mean (151507,
151508, 151671 went down); all three subject-3 slices (151674-676), the
persistent weak point since Stage 7, improved by +0.09 to +0.10 ARI each --
though they remain the worst slices in the set (gap now 0.11-0.13, down from
0.21-0.23). 151671 is a genuine regression under consensus (0.717 -> 0.594),
the one place this fix made things clearly worse. Plausible explanation for
the broad improvement: `lambda_usage=0.1` was pushing the marginal
slot-usage distribution to be flatter than the true ~7-domain structure
warranted, smearing signal across slots the harder slices especially needed
to keep distinct; weakening it to 0.02 (barely above the near-collapse regime)
let the model use its capacity more precisely without reintroducing collapse
(`slots_used=16` held throughout, confirmed in the figure-regen run log).

**Decision:** `lambda_usage=0.02` is now the default, alongside `n_hops=4`
(confirmed) and `memory_slots=16` (Stage 8) -- all three of the model's
previously-untested hyperparameters are now cross-validated, not
single-slice-tuned.

## Stage 12 -- paired significance test on the held-out gap (correction to prior framing)

External review flagged something real: this project's own docs had been
saying "the gap (0.024) is smaller than GraphST's own across-slice std
(0.086)" as if that were evidence of parity. It is not a significance test --
it is an eyeballed comparison of one method's spread against a point
difference. With 11 held-out slices measured identically for both methods
(same slices, same seeds, same clustering protocol), the correct tool is a
**paired** test, not an independent-samples comparison.

Added `src/eval/significance_test.py`: Wilcoxon signed-rank (primary, no
normality assumption) and a paired t-test (secondary; Shapiro-Wilk confirms
the paired differences are plausibly normal here, p=0.16-0.48, so the t-test
is informative too, not just a fallback), run on both the per-seed mean and
the consensus metric, since a prior finding in this project (Stage 9) already
established that they can disagree:

```
=== Per-seed mean (5 seeds/slice, more stable statistic) ===
  n slices = 11, ours wins on 2/11
  mean diff (ours - graphst) = -0.0485, std = 0.0643
  Wilcoxon signed-rank: stat=10.0, p=0.0420   <- SIGNIFICANT
  Paired t-test:        stat=-2.3848, p=0.0383

=== Consensus (headline metric) ===
  n slices = 11, ours wins on 4/11
  mean diff (ours - graphst) = -0.0237, std = 0.0932
  Wilcoxon signed-rank: stat=24.0, p=0.4648   <- not significant
  Paired t-test:        stat=-0.8057, p=0.4392
```

**Honest read.** These two tests disagree, and both need reporting, not just
whichever is more flattering. On the per-seed mean -- arguably the more
trustworthy statistic, since each point is already an average of 5
independent seeds rather than a single ensemble output -- GraphST's advantage
is **statistically significant at n=11** (p=0.042). On the consensus metric,
the gap is smaller and not significant (p=0.465), but the paired-difference
variance is also higher there (0.093 vs. 0.064), which is exactly the
condition that reduces a paired test's power -- so "not significant" here is
plausibly "underpowered given this sample size and this noisier statistic,"
not "genuinely no difference." Consensus is a single clustering-ensemble
output per slice with no repeated-measure information of its own, while the
per-seed mean already averages over 5 independent training runs, which is
presumably why it produces a cleaner (lower-variance) paired signal.

**Correction:** prior versions of README.md/PROGRESS.md described this result
as "close to parity" or "close to statistical parity." That was an
overstatement not backed by an actual test. The corrected claim, now used
everywhere this result is stated (until Stage 13 changed it further): three
real, evidence-based fixes closed most of a real gap (0.129 -> 0.024
consensus), but GraphST's lead remains statistically significant on the more
stable metric -- "closed most of a real, significant gap," not "reached
parity."

## Stage 13 -- expression-weighted adjacency (Fix #4): real improvement, changes the significance verdict

Prompted by an external review of this project (a second AI, given this
repo's plateaued 0.024-gap result, proposed four candidate fixes and
prioritized them by risk/cost -- see the design-history note in README.md for
the full context). The lowest-risk, most targeted suggestion: reweight the
spatial propagation graph by expression similarity, not just spatial
adjacency, so a spot's address doesn't blur toward a spatially-adjacent but
transcriptionally-different neighbor -- exactly the failure mode plausible on
DLPFC's ambiguous, ~50-100um-wide layer boundaries, and a specific,
mechanistic hypothesis for why the subject-3 slices in particular
(persistently the worst since Stage 7) might be failing.

Implemented as `expression_weighted_adjacency()` in `memory_layer.py`:
`exp(-||x_i - x_j||^2 / 2*sigma^2)` per structural edge (median-heuristic
sigma, no dataset-specific constant), self-loops kept at full weight,
row-normalized identically to `normalized_adjacency`. Unit-tested (rows sum
to 1; identical features reduce exactly to `normalized_adjacency`; a
dissimilar neighbor is measurably downweighted relative to a similar one).
Still respects the paper's core premise: only the softmax ADDRESS
distribution is ever propagated across spots; expression similarity is used
purely to reweight *how strongly* an edge propagates that address mass,
never to mix raw features into the embedding.

**Tested in two stages, per the reviewer's own recommendation** (cheapest,
most targeted fix first; verify on the specific failure mode before paying
for the full protocol):

1. Subject-3 slices only (151674-676, 5 seeds, GraphST skipped since its
   numbers are unaffected):

   | Slice | Baseline (uniform) per-seed | Expr-weighted per-seed | Baseline consensus | Expr-weighted consensus |
   |---|---|---|---|---|
   | 151674 | 0.4952 | 0.4679 | 0.5098 | 0.5238 |
   | 151675 | 0.4321 | 0.4688 | 0.4546 | 0.5511 |
   | 151676 | 0.4569 | 0.4914 | 0.4615 | 0.4734 |

   All 3 improved on consensus (+0.014 to +0.096); 2/3 improved on per-seed
   mean too. Promising enough to justify the full-scale run.

2. Full 12-slice x 5-seed run (`uv run python -m src.eval.run_dlpfc_multislice
   --skip-graphst`, since GraphST's own numbers are seed-deterministic and
   don't depend on our adjacency choice -- reused from the prior run rather
   than re-computed):

   | | Held-out per-seed | Held-out consensus | All-12 per-seed | All-12 consensus |
   |---|---|---|---|---|
   | Uniform adjacency (previous default) | 0.5200 ± 0.0785 | 0.5486 ± 0.0948 | 0.5230 ± 0.0758 | 0.5494 ± 0.0908 |
   | **Expression-weighted (new default)** | **0.5342 ± 0.0764** | **0.5621 ± 0.0821** | 0.5337 ± 0.0732 | 0.5620 ± 0.0786 |
   | GraphST (unchanged) | 0.5685 ± 0.0825 | 0.5724 ± 0.0861 | 0.5707 ± 0.0793 | 0.5693 ± 0.0830 |
   | Gap | 0.0343 | **0.0103** | -- | 0.0073 |

Both mean AND variance improved together on both metrics -- not a
mean/variance trade-off like consensus clustering (Stage 9) or the
lambda_usage fix (Stage 11) each were in their own way. Per-slice, the
picture is mixed but net positive: 151509/151669/151671 improved a lot,
151507/151670/151674 got slightly worse, subject-3 (151674-676) improved on
2 of 3 slices' consensus and one slightly worsened, but the aggregate moved
clearly in the right direction.

**Re-ran the Stage 12 significance test with this config** -- this is the
result that actually matters, since a smaller point estimate alone doesn't
tell you whether the underlying verdict changed:

| Metric | Before Fix #4 | After Fix #4 |
|---|---|---|
| Per-seed mean, Wilcoxon p | 0.042 (significant) | **0.123 (not significant)** |
| Consensus, Wilcoxon p | 0.465 (not significant) | 0.465 (not significant, but mean gap 0.024 -> 0.010) |
| Ours wins (consensus) | 4/11 | 5/11 (151508, 151509, 151670, 151671, 151672) |

**This is a real, verified change in the statistical conclusion, not just a
smaller number.** Before Fix #4, the per-seed metric showed GraphST reliably
ahead (p=0.042). After it, neither metric detects a significant difference at
n=11. "No significant difference detected" is not the same claim as "proven
equivalent" -- a significance test can fail to reject the null hypothesis
either because the null is true or because the sample is too small to detect
a real but modest effect; n=11 held-out slices cannot distinguish these. But
it is a real, honestly-earned improvement in the evidence, achieved by a
change that keeps the paper's core premise intact.

**Methodological caveat, stated plainly:** unlike `memory_slots`/`n_hops`/
`lambda_usage` (Stages 8, 11), this was not validated with a strict
CV-validation-slice / disjoint-test-slice split before being adopted -- the
decision to keep it was made after seeing the full 12-slice number, following
a smaller, genuinely blind check on the 3 subject-3 slices alone. The pattern
held consistently across both checks (subject-3-only and full-scale), which
is reassuring, but a fully disjoint validation (as was done for the other
three hyperparameters) would strengthen this further and is a reasonable
next step if pursuing this fix further.

**Decision:** `expression_weighted=True` is now the default in
`train_spatial_address_model` (`--uniform-adjacency` on the harness opts back
into the old behavior for the ablation). Previous uniform-adjacency full run
archived in `outputs/logs/dlpfc_multislice_results_uniform_adjacency.json`.

## Stage 14 -- entmax15 / sparsemax address distribution (Fix #2): NOT ADOPTED

The external review's second-priority suggestion: replace the dense softmax
mapping raw address scores to a probability simplex with a sparse
alternative (`entmax15` or `sparsemax` from the `deep-spin/entmax` package,
installed via `uv add entmax`), so each spot's address commits to a small
subset of memory slots rather than always giving every slot some nonzero
weight. Implemented as `address_distribution(scores, attention_fn, dim)` in
`memory_layer.py` (dispatches to `F.softmax`/`entmax15`/`sparsemax`, all
verified to produce valid, autograd-compatible simplex rows), wired through
`SpatialAddressMemoryLayer`/`SpatialAddressMemoryAutoencoder`/
`train_spatial_address_model` as an `attention_fn` parameter, `--attention-fn`
on the harness. Unit-tested: sparse variants produce exact zeros (unlike
softmax), sparsemax sparsest of the three, unknown values raise.

**Single-seed smoke test** (151674, the current default config otherwise):
softmax 0.4852, entmax15 0.4325, sparsemax 0.4818 -- inconclusive on its own,
but suggestive that neither sparse variant obviously helps.

**Subject-3 check (5 seeds, GraphST skipped), same protocol used to validate
Fix #4 before committing to a full run:**

| Slice | Baseline (softmax) per-seed | entmax15 | sparsemax | Baseline consensus | entmax15 | sparsemax |
|---|---|---|---|---|---|---|
| 151674 | 0.468 | 0.437 (-0.031) | 0.465 (-0.004) | 0.524 | 0.418 (-0.106) | 0.522 (-0.003) |
| 151675 | 0.469 | 0.393 (-0.076) | 0.449 (-0.020) | 0.551 | 0.463 (-0.088) | 0.461 (-0.090) |
| 151676 | 0.491 | 0.407 (-0.084) | 0.446 (-0.045) | 0.473 | 0.406 (-0.067) | 0.561 (+0.088) |
| **Mean delta** | | **-0.064** | **-0.023** | | **-0.087** | **-0.002** |

`entmax15` is a clear, consistent regression on every slice and both metrics
-- rejected outright, no full run needed. `sparsemax` is a wash: per-seed
mean slightly down, consensus mean essentially flat (one slice up, two down)
-- nothing close to the clear, consistent improvement Fix #4 showed on this
exact same subset (+0.015 per-seed, +0.041 consensus). Per the reviewer's own
risk-prioritization (cheap check first, only escalate on signal), neither
result justified the ~2x compute of a full 12-slice x 5-seed run for two
variants.

Plausible reason entmax15 in particular hurts: forcing hard sparsity on the
address distribution *before* spatial propagation may prematurely commit
each spot to too few slots, propagating a peakier (less informative)
distribution across hops and losing the soft blending that lets ambiguous
boundary spots average two layers' worth of address mass -- the opposite of
what expression-weighted adjacency achieves by respecting boundaries in
*where* mass flows rather than restricting *how much* of the codebook a
single spot can use.

**Decision:** both kept as opt-in, unit-tested ablations
(`attention_fn="entmax15"/"sparsemax"`, `--attention-fn` on the harness) for
reproducibility; `"softmax"` remains the default.

## Stage 15 -- address-space contrastive loss (Fix #1): NOT ADOPTED, instrumented first

The external review's highest-risk suggestion, and the one it explicitly
flagged needed care: Stage 3 (train_count_model.py) had already tested a
contrastive term and rejected it, bundled together with an NB/ZINB
likelihood, without ever isolating which change caused that regression.
Gemini's specific hypothesis for *why* Stage 3 failed -- that the contrastive
loss was applied to continuous embeddings rather than discrete address
distributions -- turned out to be **factually wrong on inspection**:
`contrastive_address_loss` was already operating in address space (mean
dot-product similarity between a spot's real address and its address under a
feature-permutation corruption, minimized). The actual untested variable was
never the space it operated in, but the *combination* with NB/ZINB.

Per the reviewer's own recommendation, instrumentation was added *before*
running anything: moved `contrastive_address_loss` into `memory_layer.py`
(shared, not NB/ZINB-specific) and added `key_cosine_similarity()` -- mean
pairwise cosine similarity of `memory_keys` rows, a codebook-collapse
diagnostic distinct from `usage_entropy` (usage entropy can look healthy
while the key vectors themselves have collapsed to near-duplicates; this
catches that "quiet" failure mode directly). Both logged every `log_every`
epochs alongside the existing entropy/slot-usage diagnostics whenever
`lambda_contrastive` is nonzero.

**Smoke test with instrumentation active** (151674, seed 0, `lambda_contrastive`
0.0 vs 0.1): `key_cosine_similarity` stayed low and negative throughout both
runs (-0.02 to -0.04), `slots_used` stayed at 16 in both -- no collapse
signal in either case. A quick sweep on the same slice/seed (0.0/0.05/0.1/0.3)
showed 0.1 scoring notably higher (+0.053 ARI) and 0.3 destabilizing
(-0.104) -- promising enough to check on more than one seed before deciding.

**Subject-3 check (`lambda_contrastive=0.1`, 5 seeds, GraphST skipped):**

| Slice | Baseline per-seed | Contrastive per-seed | Baseline consensus | Contrastive consensus |
|---|---|---|---|---|
| 151674 | 0.468 | 0.474 (+0.006) | 0.524 | 0.541 (+0.017) |
| 151675 | 0.469 | 0.452 (-0.017) | 0.551 | 0.491 (-0.060) |
| 151676 | 0.491 | 0.365 (-0.126) | 0.473 | 0.425 (-0.048) |
| **Mean delta** | | **-0.046** | | **-0.031** |

The single-seed sweep's promising result on 151674 (+0.053) turned out to be
noise, not signal -- across 5 seeds, 151674 shows only a marginal +0.006, and
the aggregate across all three subject-3 slices is a net regression, driven
mainly by a sharp per-seed drop on 151676 (-0.126). This is exactly the kind
of single-seed-optimistic pattern this project has now seen multiple times
(the whole reason cross-validation and multi-seed checks exist), and a good
reminder not to commit to a full 12-slice run off one seed's number, however
promising.

**Honest read.** No collapse (unlike the qualitative failure mode Stage 3's
combined NB/ZINB+contrastive run produced) -- the instrumentation confirms
this is a genuine "doesn't help" result, not a repeat of a training pathology.
Not enough signal to justify a full 12-slice x 5-seed run. `lambda_contrastive`
kept as an opt-in, unit-tested parameter (`train_spatial_address_model(...,
lambda_contrastive=...)`, `--lambda-contrastive` on the harness) for
reproducibility and further tuning if pursued; `0.0` (off) remains the
default.

**This completes the external review's full four-fix plan**, tested in the
prioritized order it suggested: significance test (Stage 12) confirmed the
gap was real; Fix #4 expression-weighted adjacency (Stage 13) was a genuine
improvement and adopted; Fix #2 entmax15/sparsemax (Stage 14) and Fix #1
address-space contrastive loss (Stage 15) were both tested properly and not
adopted. Fix #3 (temperature annealing) was never tested standalone, per the
reviewer's own reasoning that it only matters paired with a sparse
projection -- since neither sparse variant showed promise, that pairing was
moot.

## Stage 16 -- dual-modality (expression + morphology) memory addressing: FALSIFIED at the diagnostic gate

A new architecture plan proposed bringing histology back into the model as a
second, morphology-addressed memory bank fused with the expression bank via a
learned per-spot gate -- explicitly NOT a feature-concatenation or
contrastive-fusion bolt-on, to keep the addressing-based novelty claim
distinct from SpaGIC/stMMC/SpatialDG/MultiST-style multimodal methods. The
plan itself specified a mandatory falsification test to run *before* writing
any of the dual-memory code: if subject-3 (the persistent weak point since
Stage 7, untouched by every fix tried since) doesn't show elevated
expression/morphology disagreement AND elevated model error relative to
subjects 1/2, stop and reconsider rather than spending a week building an
architecture the data doesn't support.

**Data pipeline built first, verified working:**
- `src/data/extract_patches.py`: extracts a 64x64 H&E patch per spot from
  `adata.uns['spatial'][lib]['images']['hires']`, correctly rescaling
  `adata.obsm['spatial']` (full-resolution pixel coordinates) by
  `tissue_hires_scalef` before indexing -- verified all 12 DLPFC slices and
  the squidpy mouse crop dataset carry real, non-empty H&E images with
  consistent scalefactors. Edge-padded (not clipped) so every patch has
  identical shape regardless of spot position. Spot-count assertion plus a
  visual spot-check (`outputs/figures/patch_alignment_spotcheck.png`) against
  the full tissue image confirmed correct alignment (a patch centered
  directly on the hippocampal band shows that dark band cutting through it;
  patches away from it show uniform lighter tissue) before trusting the cache.
- `src/models/image_encoder.py`: frozen (no gradient) ImageNet-pretrained
  ResNet18, `fc` replaced with `Identity()` -> 512-dim features. ResNet18 over
  DINO/DINOv2 to start, per the plan's own reasoning (fewer moving parts to
  debug before reaching for a fancier encoder). 9 new unit tests across both
  modules (shape, determinism, no-trainable-params, boundary padding, distinct
  patches for distinct spots).

**The Section 2 diagnostic (`src/eval/image_diagnostic.py`), run across all 12
slices before any dual-memory code was written:** for each spot, mean cosine
similarity to its spatial neighbors was computed separately in expression
space and in frozen-ResNet18 image-feature space, z-scored within-slice, and
compared (`disagreement = |z(img_sim) - z(expr_sim)|`). Model error was the
current best (Stage 13) model's cluster-purity mismatch rate per spot
(majority-vote cluster-to-label mapping, then flag disagreements).

| Subject | Mean disagreement | Mean error | Mean ARI |
|---|---|---|---|
| subject1 (151507-510) | 1.1186 | 0.2670 | 0.5372 |
| subject2 (151669-672) | 1.1400 | 0.1642 | 0.6430 |
| **subject3 (151673-676)** | **1.0840 (lowest)** | **0.3080 (highest)** | **0.4873 (lowest)** |

**This falsifies the hypothesis, not just fails to confirm it.** Subject 3
does show the highest error and lowest ARI, consistent with every prior
stage -- but it shows the *lowest* expression/morphology disagreement of the
three subjects, not the highest. Subject 2 shows the *highest* disagreement
alongside the *lowest* error -- the opposite of what the plan needed. Per-spot
correlations between disagreement and error within each subject are all near
zero with inconsistent signs across slices (range -0.144 to +0.072,
`outputs/logs/image_diagnostic_results.json`), showing no within-subject
signal either. The scatter plot
(`outputs/figures/image_diagnostic_scatter.png`) makes this visually obvious:
subject3's cluster sits at low-disagreement/high-error, subject2's at
high-disagreement/low-error, subject1's in between -- if anything, an inverse
relationship across subjects, not the predicted positive one.

**Decision, per the plan's own explicit stopping rule:** the
`DualModalityMemoryLayer` architecture (Section 3 of the plan) was **not
built**. Whatever is driving subject-3's persistent gap (unexplained across
Stages 7, 9, 11, 13, 14, and 15 too), this diagnostic gives no evidence it is
expression/morphology disagreement the frozen ImageNet features can resolve.
Possible reasons the diagnostic came back negative, worth separating for any
future attempt: (a) the underlying biological hypothesis (image can
disambiguate expression-ambiguous boundaries) is simply wrong for this
tissue/resolution; (b) it's right but frozen ImageNet features are the wrong
morphology representation for H&E specifically (the plan itself flagged this
domain-gap risk and named DINO/DINOv2 as a fallback, not yet tried); (c) the
neighbor-similarity-agreement operationalization used here doesn't capture
the real effect even if it exists. This result doesn't distinguish between
those -- it only says the cheapest, fastest version of the test didn't
support spending a week on the full architecture, which is exactly what the
diagnostic step was for.

## Stage 17 -- revised diagnostic (Moran's I "signal rescue" reframing + DINOv2): falsified again, more decisively

A follow-up review of Stage 16's result made a specific, credible case that the
original diagnostic was the wrong test, not that the hypothesis was wrong: (1)
frozen ImageNet ResNet18 is domain-mismatched for H&E histology (trained to
classify macro-objects, not read cellular texture); (2) the original metric
assumed histology helps by *resolving conflicts* with expression, but the
mechanism this architecture actually needs is *signal rescue* -- histology
staying spatially coherent where transcriptomic signal has degraded (dropout,
low depth), which a spot-by-spot disagreement score can't distinguish from
genuine conflict; (3) single-spot patch embeddings are noisy and should be
spatially smoothed before comparison; (4) expression and image features live
in unaligned manifolds, so a naive gate can't fairly weigh them.

Point (2)'s reframing led to a sharper, cheaper, and more directly testable
metric: **Global Moran's I** (spatial autocorrelation), computed separately
per modality, per slice -- no model training required at all, since it's a
pure statistic of the feature matrices and the spatial graph.
`src/eval/morans_i_diagnostic.py`, `morans_i()` vectorized across all feature
columns via one sparse-dense matmul, unit-tested against synthetic ring-graph
signals (smooth sinusoid -> I > 0.8, random noise -> |I| < 0.2, checkerboard
-> I < -0.8, confirming the standard Moran's I sign/magnitude conventions
before trusting it on real data).

**The hypothesis this needs to hold:** on subject-3 slices, expression
Moran's I should be LOW (spatially incoherent, consistent with degraded
signal) while image Moran's I stays HIGH (tissue anatomy intact regardless of
transcriptomic quality) -- a bigger img-minus-expr gap on subject 3
specifically than on subjects 1/2.

**Result, ResNet18 features (reusing the already-cached Stage 16 features, no
retraining):**

| Subject | Mean expr Moran's I | Mean img Moran's I | Gap |
|---|---|---|---|
| subject1 | 0.0188 | 0.5550 | 0.5362 |
| subject2 | 0.0184 | 0.5713 | 0.5530 |
| **subject3** | **0.0362 (highest)** | 0.5796 | **0.5434 (smallest of the three)** |

This is the opposite of the needed pattern on both counts: subject 3 has the
*highest* expression spatial coherence of the three subjects (not the
lowest), and the *smallest* image-rescue gap (not the largest). Point (1) of
the critique -- that ResNet18 might be too weak a backbone to see the real
effect -- doesn't apply to the `expr_moran` half of this finding, since that
computation never touches an image encoder at all; it's a property of the
expression data and the spatial graph alone.

**Confirmed with DINOv2 anyway, for completeness** (`get_dinov2_encoder()`
added to `image_encoder.py`, ViT-S/14 via `torch.hub`, patches bilinearly
resized 64x64 -> 224x224 for ViT input, both new functions unit-tested,
6/6 passing including a network-unavailable skip guard):

| Subject | Mean expr Moran's I | Mean img Moran's I (DINOv2) | Gap |
|---|---|---|---|
| subject1 | 0.0188 | 0.7481 | 0.7293 |
| subject2 | 0.0184 | 0.7664 | 0.7480 |
| **subject3** | **0.0362 (highest)** | 0.7580 | **0.7217 (smallest of the three)** |

DINOv2's image Moran's I is uniformly higher than ResNet18's (0.75-0.77 vs.
0.55-0.58, as expected from a stronger, texture-sensitive backbone) -- but
the *subject-level ranking is identical*: subject 2 still has the highest
image coherence, subject 3 still has the smallest gap, and `expr_moran` is
unchanged (backbone-independent, as it must be). Switching encoders changed
the absolute numbers, not the qualitative conclusion.

**This is a more decisive falsification than Stage 16's, not merely a repeat.**
It directly answers the two most substantive points in the critique (wrong
backbone, wrong metric) with a cheaper, more targeted test, and both come
back negative: there is no evidence that subject-3's transcriptomic signal is
uniquely degraded relative to subjects 1/2, and no evidence that image
information uniquely compensates for it there specifically -- if anything,
subject 3 is where image information helps *least* on a relative basis.
Points (3) (spatial smoothing of image features) and (4) (cross-modal
alignment via contrastive pretraining) are properties of the eventual model
architecture, not of a pre-architecture data diagnostic -- Moran's I already
measures spatial coherence directly, without needing a GNN smoothing step
first, so they don't change what this diagnostic can say. They remain the
right next things to get right *if* a future attempt is made, but nothing in
this repo's data has yet shown a signal that would justify building the
architecture to test them.

`DualModalityMemoryLayer` remains **not built**.

## Stage 18 -- Topologically-Ordered Memory (TOM): FAILED at the gate, all variants rejected

Premise of the plan: every mechanism tried so far treats memory slots as an
*unordered* bag of prototypes, while the ground truth (cortical layers) is
strictly ordered -- so give the memory bank a 1D geometry via the
differentiable SOM mechanism from SOM-VAE (Fortuin et al., ICLR 2019), plus an
ordinal-smoothness loss on the resulting per-spot position.

**Section 0 literature check (done first, as the plan required).** SOM-VAE is
real and citable. SOMDE (Bioinformatics 2021) applies SOMs to spatial
transcriptomics but for spatially-variable-*gene* identification, a different
task. Every DLPFC domain method surveyed (GraphST, STAGATE, DeepST,
SemanticST, SpaBatch, SEDR, SpaGCN, BayesSpace, stLearn) uses an unordered
cluster space, so the mechanism was not preempted. Cortical depth IS used as
an ordering axis in the field (Science 2023, Nature 2025) but for
annotation/cell assignment, not as an architectural prior -- so the biological
premise is well-supported and any novelty claim is about the mechanism only.

**A latent bug in the plan's code sketch, caught before it could waste a run.**
The sketch set `slot_pos = linspace(0, 1, memory_slots)` while sweeping
`som_sigma` over 0.5-2.5. With a maximum slot distance of 1.0, that makes
`exp(-d^2 / 2*sigma^2) >= 0.80` for EVERY slot pair -- an almost perfectly flat
neighborhood kernel that pulls every slot toward every input, silently
reducing the whole mechanism to a no-op while still appearing to run. Fixed by
using integer slot indices (the standard SOM convention, under which the
plan's sigma values are meaningful) and reporting position on a normalized
[0,1] scale so `lambda_ordinal` stays independent of `memory_slots`. Pinned by
a regression test (`test_som_kernel_is_not_degenerately_flat_at_default_sigma`).

**GATE RESULT: catastrophic SOM collapse, across the entire hyperparameter
range.** The instrumentation the plan mandated caught it in the first run:
`slots_used=1`, `key_cosine_similarity=0.995`, `expected_pos_std=0.0000`.
Sigma sweep + loss ablation on 151673 (seed 0):

| config | ARI | abs rho pos-vs-depth | pos_std | slots | key_cos |
|---|---|---|---|---|---|
| ordinal-only | 0.5696 | 0.151 | 0.104 | 16 | -0.03 |
| som-only | 0.0000 | 0.029 | 0.000 | 1 | 0.995 |
| both (plan default) | 0.0000 | 0.029 | 0.000 | 1 | 0.995 |
| both, sigma=0.5 | 0.0000 | 0.011 | 0.000 | 1 | 0.442 |
| both, sigma=1.0 | 0.0000 | 0.021 | 0.000 | 1 | 0.674 |
| both, sigma=2.5 | 0.0000 | 0.071 | 0.000 | 1 | 1.000 |
| both, lambda_som=0.002 | 0.0000 | 0.165 | 0.000 | 1 | 0.490 |
| both, lambda_som=2e-4 | 0.4865 | 0.162 | 0.006 | 11 | 0.035 |
| neither (= Stage 13) | 0.5150 | 0.175 | 0.118 | 16 | -0.03 |

Correctness check passed: "neither" reproduces 0.5150, exactly the Stage 13
model's known single-seed 151673 result -- TOM is a strictly additive change
and the comparison is apples-to-apples.

**Root cause identified, not merely observed.** This is not mistuning: collapse
persists across the full sigma range AND two orders of magnitude of
lambda_som, disappearing only at 2e-4 where the term has effectively been
switched off (and still underperforms baseline). The mechanism:

  * In SOM-VAE, reconstruction flows THROUGH the quantized codebook entry, so
    codebook entries must stay spread out to reconstruct well. That is what
    keeps the map from collapsing.
  * In this architecture, addressing (`memory_keys`) and reconstruction
    (`memory_values`) are deliberately separate -- which is what makes
    "addressing replaces message passing" coherent. But it means `memory_keys`
    receive NO spreading pressure from the reconstruction objective at all,
    so the SOM neighborhood term can freely collapse every key onto the query
    centroid, unopposed.
  * The existing anti-collapse guard is structurally blind to this failure:
    when all keys are identical the softmax over identical scores is UNIFORM,
    which is the MAXIMUM of `usage_entropy`. The guard reads as perfectly
    satisfied while the model is degenerate. `usage_entropy` prevents "every
    spot routes to one slot"; it cannot prevent "every spot routes uniformly
    to all slots", which is equally uninformative.

The plan transplanted SOM-VAE's loss without SOM-VAE's structural constraint.
Both designs are individually sound and mutually incompatible.

**The surviving piece also fails at scale.** `ordinal-only` (lambda_som=0) was
the one configuration that looked better than baseline (0.5696 vs 0.5150 on
151673 seed 0). Escalated per this project's own discipline -- 4 slices x 5
seeds gave +0.025 mean, better on 3/4, but with one -0.061 regression. Full
12-slice x 5-seed protocol settled it:

| | ordinal-only | current (Stage 13) | delta | Wilcoxon p |
|---|---|---|---|---|
| held-out per-seed | 0.5206 | 0.5342 | -0.0136 | 0.41 |
| held-out consensus | 0.5417 | 0.5621 | -0.0204 | 0.37 |

Slightly worse, wins only 4/11, not significantly different. And it actively
REGRESSES the headline claim: ordinal-only vs GraphST on the per-seed metric
is p=0.042 (significant, GraphST ahead), losing the "no significant
difference" standing the current model holds at p=0.123. Rejected.

Note also that with lambda_som=0 the slot ordering is arbitrary by
construction, so even a positive result here would NOT have been the TOM
hypothesis -- it would have been a generic smoothness regularizer overlapping
with the n_hops propagation already in the model.

**The premise itself is inverted where it was aimed.** The gate's control
condition asked whether the EXISTING model already encodes layer order
implicitly (abs Spearman of embedding PC1 vs. true cortical depth), measured
over 12 slices x 3 seeds (`src/eval/baseline_ordinal_axis.py`):

| subject | mean abs rho | note |
|---|---|---|
| subject1 | 0.2372 | |
| subject2 | 0.4204 | |
| **subject3** | **0.8240** | strongest AND most stable (per-slice std 0.016-0.039) |

Subject 3 -- the persistent weak point this entire plan was designed to fix --
is precisely where the current model ALREADY recovers cortical depth ordering
most strongly and most reliably. Its clustering is worst exactly where its
laminar ordering is best. So subject 3's failure is not "the model cannot tell
where a spot sits along the depth axis"; it can, better there than anywhere
else. An explicit ordinal prior targets a deficit subject 3 does not have.

(Across all 12 slices the trend between ordinal-axis strength and ARI is
negative -- Spearman -0.44 -- but p=0.15 at n=12, so that is reported as
suggestive only, not established. The per-subject statement above is the solid
one, backed by tight per-slice variance.)

**Decision:** `TopologicalMemoryLayer` / `TopologicalMemoryAutoencoder` and
`train_topological_model` are kept as opt-in, unit-tested code (13 tests) for
reproducibility, exactly as `kmeans_init`, `attention_fn`, and
`lambda_contrastive` were kept after their own negative results.
`--lambda-ordinal` is wired into the harness. No default changed; the Stage 13
configuration stands.

## Stage 19 -- per-subject QC: the data-quality-ceiling hypothesis is falsified too

With four architecture-level hypotheses now specifically falsified on subject 3
(image/expression disagreement, address-space contrastive, missing laminar
order, and now this), the natural remaining explanation was a data-quality
ceiling: subject 3 is simply worse tissue, and no architecture will lift it.

**This had never actually been measured, despite the docs implying otherwise.**
PROGRESS.md had been asserting "no data-level explanation (sparsity, layer
proportions, spot count) has been found so far" -- but `data_stats.py` only
ever covered the mouse Visium and Slide-seq datasets, never DLPFC and never
per-subject. No read-depth or dropout QC existed. Corrected, and the claim in
PROGRESS.md has been amended.

`src/eval/per_subject_qc.py`, raw counts, pre-normalization:

| subject | median library | median genes | dropout | spot density | n_layers | ARI |
|---|---|---|---|---|---|---|
| subject1 | 2304 | 1324 | 0.9597 | 55.2 | 7 | 0.536 |
| subject2 | 3452 | 1734 | 0.9471 | 49.4 | **5** | 0.623 |
| **subject3** | **4003** | **2058** | **0.9356** | 48.7 | 7 | **0.527** |

Subject 3 has the BEST sequencing depth, the BEST library complexity, and the
LOWEST dropout of the three subjects -- and the worst ARI. The
data-quality-ceiling hypothesis is falsified, not supported.

Two further points this table settles:

  * **Subject 2's advantage is largely a task-difficulty artifact**: it has 5
    annotated layers where subjects 1 and 3 have 7. The confound-free
    comparison is subject1 vs subject3, both 7-layer -- and there subject 3
    has strictly better data and worse performance.
  * **Subject 3 is not intrinsically hard; it is hard FOR US.** Per-subject,
    ours vs. GraphST (consensus, identical protocol):

    | subject | ours | GraphST | gap |
    |---|---|---|---|
    | subject1 | 0.536 | 0.499 | **-0.037 (we win)** |
    | subject2 | 0.623 | 0.625 | +0.002 (tie) |
    | subject3 | 0.527 | 0.584 | **+0.057 (we lose)** |

    GraphST handles subject 3 fine -- 0.584, its second-best subject. We are
    the only method that degrades there.

**This is the most actionable finding of the session.** "Subject 3 is hard"
becomes "our architecture specifically underperforms, on the cleanest data in
the set, a method that handles that data well." That is a targeted
architectural defect, not a ceiling -- and it rules out the paper framing of
"both approaches degrade together on a low-quality subject", which the data
does not support on either half.

A concrete, untested suspect worth the next cheap experiment: subject 3 has by
far the richest per-spot signal (2058 genes/spot vs. 1324 for subject 1), and
our `n_hops=4` address propagation may be over-smoothing genuinely separable
layers precisely where signal is strongest, while GraphST's contrastive term
actively resists that. `n_hops` was cross-validated globally (Stage 11), never
per-subject. Not yet run.

## Phase A -- per-subject n_hops sweep on subject 3: NOT SUPPORTED, not adopted

The one open, already-specified experiment from Stage 19: subject 3 has the
richest per-spot signal (2058 genes/spot vs 1324 for subject 1) yet the worst
ARI, so `n_hops=4` address propagation may be over-smoothing genuinely
separable layers there. `n_hops` was cross-validated globally (Stage 11), never
per-subject.

**Leakage-safe design.** Selection used 151673 ONLY -- already the project's
global tuning slice, therefore already burned, so reusing it contaminates
nothing new. Held-out evaluation used 151674/675/676, none of which
participated in selection. 6 hop counts x 4 slices x 5 seeds = 120 training
runs.

**The real test was the boundary-vs-interior breakdown, not aggregate ARI.**
New `src/eval/boundary_mask.py` (7 unit tests) marks a spot as
boundary-adjacent if any spot within 2 graph hops carries a different
ground-truth layer label -- graph hops, not pixels, because that is the unit
the propagation mechanism actually operates in. Unannotated neighbours are
excluded so tissue edges are not spuriously marked as boundaries. On 151673
this splits 3611 annotated spots into 1716 boundary / 1895 interior.

Selection on 151673 picked `n_hops=3` (0.5872 vs 0.5288 for the global
default). Held-out result:

| metric | global n_hops=4 | selected n_hops=3 | delta |
|---|---|---|---|
| overall | 0.4760 | 0.4797 | +0.0037 |
| boundary | 0.3458 | 0.3515 | +0.0056 |
| interior | 0.6665 | 0.6663 | -0.0002 |

**VERDICT: NOT SUPPORTED. Not adopted.** Three independent reasons:

  * The boundary delta (+0.0056) is an order of magnitude below the typical
    per-slice seed noise (0.0368).
  * The per-slice boundary deltas disagree in direction: +0.0204, +0.0006,
    -0.0042. The aggregate "improvement" is one slice averaged with two nulls.
  * The global default is ALREADY the best hop count on 2 of the 3 held-out
    slices. Per-slice optima are 3, 5, 4, 4 -- no consistent per-subject
    value exists to adopt.

Boundary ARI simply does not respond systematically to hop count on any slice
(e.g. 151674 across hops 1-6: 0.378, 0.386, 0.370, 0.349, 0.381, 0.332 -- no
trend). So the over-smoothing-at-boundaries mechanism is not what is
happening, and per the plan's own stopping rule a hop count that moves only
aggregate ARI by noise must not be adopted. `n_hops=4` stands.

**A bug in this experiment's own verdict logic, caught and fixed.** The first
automated verdict printed "SUPPORTED" -- because it tested only the SIGN of
the deltas (`b_delta > 0 and b_delta > i_delta`), never their magnitude
against seed variance. With per-slice std of 0.03-0.06, a +0.005 mean delta
trivially satisfies a sign test while being pure noise. The check now requires
the effect to exceed typical seed noise AND to be direction-consistent across
held-out slices, and it reports the per-slice best hop so a spurious
"improvement" over an already-optimal default is visible. The saved JSON
carries a `superseded_verdict_note` recording the correction rather than
silently overwriting it. Worth flagging plainly: an automated verdict is only
as good as its threshold, and a sign-only test is not a significance test --
the same class of error this project already corrected once at Stage 12.

## Phase B2 -- effect sizes and bootstrap CIs: the headline claim needed qualifying

The "no significant difference from GraphST" claim rested on two p-values at
n=11. Added to `src/eval/significance_test.py`: matched-pairs rank-biserial
correlation (Kerby 2014) as the Wilcoxon effect size, and a percentile
bootstrap CI on the mean paired difference (10,000 slice resamples).

| metric | mean gap | bootstrap 95% CI | rank-biserial | Wilcoxon p |
|---|---|---|---|---|
| consensus | -0.0103 | [-0.053, +0.036] | -0.273 (small-medium) | 0.465 |
| per-seed | -0.0343 | [-0.072, +0.005] | **-0.545 (large)** | 0.123 |

**This qualifies the headline claim materially.** On the per-seed metric --
the more stable of the two, since each point already averages 5 seeds -- the
effect size is *large* by conventional thresholds, ours wins only 3/11, and
the CI only barely includes zero: the data are consistent with GraphST being
up to 0.072 ahead but at most 0.005 behind.

So "no statistically significant difference at n=11" was accurate but was
carrying more weight than it should. The honest reading is **"GraphST is
probably still modestly ahead, and 11 slices is too few to establish it"** --
not that the methods are equivalent. At this sample size "not significant" and
"no effect" are different claims. README.md has been updated accordingly.

Note this also *strengthens* the case for cross-platform work: an underpowered
comparison is resolved by more datasets, not by more DLPFC tuning.

## Phase B1 -- GraphST reproduction gap: explained, and NOT a handicap

Our GraphST run scores 0.597 +/- 0.012 on 151673 against a 0.633 literature
reference. Left unexplained, that threatens every "no significant difference
from GraphST" claim here, because the comparison would be against a weakened
baseline -- and the same shortfall would silently reappear on every Phase C
dataset.

**Model config ruled out by inspection, before spending compute.**
`run_graphst.py` already passes GraphST's published defaults verbatim
(dim_input=3000, dim_output=64, epochs=600, random_seed=41, lr=0.001,
alpha=10/beta=1/theta=0.1/lamda1=10/lamda2=1) and calls GraphST's own
`preprocess` / `construct_interaction` / `add_contrastive_label` /
`get_feature`, so HVG count, normalization and graph construction are
GraphST's own, not ours. The gap is not a config mismatch.

**Clustering swept on a FIXED embedding** (`src/eval/graphst_reproduction.py`,
64 variants: PCA dim x GMM init x covariance x refinement), so any difference
is attributable to clustering alone rather than retraining variance:

| variant (PCA-20, refine) | ARI |
|---|---|
| kmeans init, tied -- current protocol | 0.5902 |
| hierarchical init, tied | 0.6022 |
| literature reference | 0.6330 |

Initialization does matter: R's mclust initializes from model-based
hierarchical agglomeration while sklearn's GaussianMixture defaults to
k-means, and a Ward-agglomerative init gained +0.012 here.

**Explicitly NOT adopting the grid maximum** (0.6232, random_from_data +
full covariance). That estimator ranges from 0.029 to 0.623 across the grid --
selecting its maximum is choosing on the evaluation metric, the same leakage
corrected at Stage 8. A single-slice grid search is not a protocol change.

**The decisive test: does the protocol bias the GAP or only the absolute
numbers?** Since the protocol is applied identically to every method, a
handicap should move both together. `src/eval/protocol_invariance.py` scored
both methods' embeddings under both inits on 6 held-out slices x 3 seeds
(one embedding per method/slice/seed, re-clustered two ways):

| init | ours | GraphST | gap |
|---|---|---|---|
| kmeans (current) | 0.5222 | **0.5969** | 0.0747 |
| hierarchical | 0.5093 | 0.5181 | 0.0088 |

**Verdict: NO HANDICAP.** The gap narrows by 0.066 under hierarchical init,
but not because our method improves -- because GraphST *degrades* under it
(-0.079 overall; on 151672 it collapses 0.770 -> 0.460). The current protocol
is the one that scores the baseline HIGHER, so the reproduction shortfall is
not biasing the comparison in our favour. If anything the current protocol is
generous to GraphST. The headline comparison stands and no protocol change is
warranted.

**A correction to this project's own Phase B1 conclusion.** The "hierarchical
init is better" finding was measured on ONE slice (151673, +0.012). Across 6
slices it is substantially WORSE for GraphST (-0.079). It does not replicate,
and an earlier verdict string in `protocol_invariance.py` that called
hierarchical "the better protocol" was wrong on this evidence; both the script
and the saved JSON now record the correction rather than overwriting it
silently. This is the third time in this project that a single-slice result has
failed to generalize (after `memory_slots=32` at Stage 8 and entmax/contrastive
at Stages 14/15) -- single-slice evidence should be treated as a hypothesis,
never a conclusion.

Residual: ~0.031 of the original 0.043 shortfall remains unexplained after
initialization. Most plausibly Python-GMM vs R-mclust implementation
differences plus the fact that the 0.633 reference is Kang et al.'s
recomputation from GraphST's released predictions -- a different pipeline
end to end. Documented as an explained-but-not-closed gap, which is the honest
state.

## Phase B3 -- comparators: installability verified, runners written, Colab notebook added

Checked availability BEFORE committing time, per the plan:

| package | status |
|---|---|
| STAGATE | `git+https://github.com/QIFEIDKN/STAGATE_pyG.git` -- resolves (author's own repo) |
| Garfield | PyPI `garfield==1.0.0`, confirmed the spatial-omics package (Weige Zhou, github.com/zhou-1314/Garfield), not a name collision |
| stGRL / MAEST / SpaBatch | not on PyPI -- deprioritized per the plan's own "don't block Phase C" instruction |

`src/models/run_stagate.py` follows STAGATE's own DLPFC tutorial (3000
seurat_v3 HVGs, rad_cutoff=150), on the same principle as `run_graphst.py`:
a comparator should be run the way its authors run it, or the comparison
measures our preprocessing rather than their method. ~~Marked NOT YET EXECUTED
-- STAGATE is blocked locally because `torch_sparse` has no wheel for torch
2.11.0+cu128 on Windows.~~ **Correction, tested directly rather than left as
an inherited assumption: this was stale.** PyG's wheel index now publishes
builds through torch 2.13.0, including `torch-sparse 0.6.18+pt211cu128`. It
installs and STAGATE trains locally without Colab:

```
uv pip install torch_sparse torch_scatter \
  --find-links "https://data.pyg.org/whl/torch-2.11.0+cu128.html"
uv pip install git+https://github.com/QIFEIDKN/STAGATE_pyG.git
```

Single-slice smoke test (151673, seed 0) landed right next to the literature
number (STAGATE ARI 0.5828 vs. its published 0.589), so the full protocol was
run locally: `src/eval/run_stagate_dlpfc.py`, 12 slices x 5 seeds, identical
mclust-equivalent + spatial-refinement clustering and consensus-across-seeds
as every other method here.

| | Ours | GraphST | STAGATE |
|---|---|---|---|
| Held-out per-seed | 0.5342 +/- 0.076 | 0.5685 +/- 0.083 | 0.5432 +/- 0.082 |
| Held-out consensus | 0.5621 +/- 0.082 | 0.5724 +/- 0.086 | 0.5500 +/- 0.087 |

STAGATE lands between us and GraphST on both metrics. `src/eval/significance_test_stagate.py`
extends the Phase B2 machinery (Wilcoxon + rank-biserial + bootstrap CI) to
all three pairwise comparisons rather than a 3-group omnibus test (with n=11,
an omnibus test would have even less power than the pairwise tests already in
use, and the question that matters is "how do we compare to EACH established
method", not "are the three different in general"):

| pair (per-seed) | mean diff | Wilcoxon p | rank-biserial |
|---|---|---|---|
| ours vs graphst | -0.0343 | 0.123 | -0.545 |
| ours vs stagate | -0.0090 | 0.700 | -0.152 |
| graphst vs stagate | +0.0253 | 0.123 | +0.545 |

**None of the three pairwise tests reach significance at n=11.** On consensus,
ours edges STAGATE (6/11 slices, rank-biserial +0.152); GraphST edges both of
us on both metrics, consistent with the Phase B2 effect-size finding, but that
edge is itself not statistically established at this sample size either. A
second real comparator did not change the picture: broadly competitive with
the field on DLPFC, not proven ahead or behind, n=11 the binding constraint.

Garfield is genuinely blocked on Windows -- verified, not stale: it depends on
`pybedtools` -> `pysam` -> `htslib`, which has no Windows wheel, confirmed by
attempting the install directly (`pysam` fails to build: "Cython ... using
cythonize if necessary ... FileNotFoundError: [WinError 2]"). Unlike STAGATE's
"blocked" claim, this one held up -- but it works inside WSL2 Ubuntu (verified,
GPU passthrough confirmed via `nvidia-smi`/`torch.cuda.is_available()`), so it
was run there rather than abandoned.

**Getting a real embedding out required reverse-engineering the package from
tracebacks -- Garfield v1.0.0 (Jan 2025, "paper coming soon") ships no
tutorial, no quickstart, and no worked example anywhere in its repo, and both
its documented default entry point (`DataProcess`) and its own `GarfieldTrainer`
class turned out to be broken for single-sample spatial data.** Three
independent bugs, each confirmed by direct source inspection rather than
guessed:

1. `preprocessing_rna(...)` requires `adata.obs['batch']` to exist even for one
   unbatched sample (`sc.pp.highly_variable_genes(..., batch_key='batch')`
   KeyErrors otherwise).
2. `Garfield.model.Garfield.__init__` unconditionally re-runs `DataProcess` on
   `gf_params['adata_list']`, which is broken for both of its documented input
   shapes (a list of paths crashes at `adata.obsm`; a list of real AnnData
   objects crashes inside `concat_data`'s single-element branch, which assumes
   a path string). Worked around via an undocumented early-return: if an
   element of `adata_list` already carries a non-empty
   `obsm['garfield_latent']`, `DataProcess` returns the input unchanged --
   so a placeholder is stamped in purely to trip that path.
3. `GarfieldTrainer.train()` calls `self.model(data_batch=..., decoder_type=...,
   augment_type=...)`, but `Garfield` never implements a matching `forward()`
   (falls through to `nn.Module._forward_unimplemented`) -- it always raises
   `TypeError` on the first real batch. Separately, `Garfield` overrides
   `nn.Module.train()` with its own zero-argument method that runs a complete,
   self-contained training pipeline (data loaders, epoch loop, early stopping,
   held-out eval) -- but this breaks the standard `train(mode: bool)` contract,
   so `model.eval()` also raises `TypeError`. The real (if undocumented)
   training entry point turned out to be calling `model.train()` directly,
   bypassing `GarfieldTrainer` entirely.

Full details and the working wrapper: `src/models/run_garfield.py`.

**Result, DLPFC 151673, 3 seeds (default hyperparameters, same
`cluster_embedding(..., refine=True)` protocol as every other method):**

| seed | ARI |
|---|---|
| 0 | 0.2431 |
| 1 | 0.2322 |
| 2 | 0.2714 |
| **mean ± std** | **0.249 ± 0.017** |

Garfield lands well below our method (0.303) and far below GraphST/STAGATE
(~0.52-0.63) on this slice, with tight seed variance (std 0.017, tighter than
the ~0.05-0.08 typical of the other three methods). **Deliberately stopped at
n=3 (one slice) rather than running the full 12-slice x 5-seed protocol
(~5h GPU time):** the low variance already makes the gap look real rather than
noise, and three separate check-ins with the low-variance signal in hand
confirmed the full run was unlikely to change that conclusion. Recorded here
as a real, reproducible negative result for Garfield v1.0.0 with default
settings, not a full evaluation -- see `outputs/logs/garfield_dlpfc_151673_results.json`.

`notebooks/05_comparators_and_generalization.ipynb` covers B3 and scaffolds C.
Two deliberate design choices:

  * **It clones the repo rather than inlining model code.** The older
    `04_colab_scaleup.ipynb` pasted a copy of the Phase 0 `EmbeddedMemoryLayer`
    inline; that copy is now eight stages stale and would silently benchmark a
    model nobody uses.
  * **Garfield's API is discovered at runtime, not guessed.** Its README
    documents parameters but ships no quickstart, and writing confident-looking
    calls against an unverified API is worse than writing none.

## Phase C -- a literature check that REVERSES the plan's dataset priority

The forward plan proposed leading with mouse olfactory bulb (Stereo-seq) on
the rationale that it is "used directly in GraphST's own paper, so you have a
literature number to compare against on the same data GraphST was evaluated on
originally." Checked against the paper (Nat Commun 2023, PMC9977836) -- that
rationale does not hold:

| dataset | ground truth | GraphST ARI reported? |
|---|---|---|
| Mouse olfactory bulb (Stereo-seq) | authors' own DAPI-based laminar annotation | **No** -- qualitative, marker-gene overlap only |
| Mouse hippocampus (Slide-seqV2) | Allen Brain Atlas reference | **No** -- visual comparison only |
| **Human breast cancer (10x Visium)** | **pathologist annotation, 20 regions** | **Yes -- ARI 0.54-0.57** |

So of the three proposed platforms, **human breast cancer is the only one with
a published, directly comparable GraphST ARI**, and it should lead Phase C --
the opposite of the plan's stated ordering. Stereo-seq OB and Slide-seqV2 have
no published ARI from GraphST at all; on those, any quantitative claim would
have to be constructed by us against annotations GraphST never scored against,
which is a much weaker comparison than the plan assumed.

Two further notes for Phase C:

  * Slide-seqV2 as shipped by squidpy carries **cell-type** annotations, not
    spatial-domain annotations. ARI against cell type measures a different task
    than ARI against cortical layers -- a spatial-domain method is not supposed
    to recover cell types. STAGATE reportedly distributes an annotated
    Slide-seqV2 mouse OB, which is a possible route to a quantitative
    evaluation, but it is STAGATE's annotation rather than a shared standard.
  * Where no domain annotation exists, report unsupervised metrics (silhouette,
    spatial coherence) and qualitative marker-gene agreement, exactly as the
    source papers do -- rather than manufacturing a supervised score against
    the wrong label set.

## Phase C, first result: human breast cancer -- a real, consistent gap

**Sourcing the dataset and annotation took real verification, not a
download-and-go.** The raw 10x counts (3798 spots, 36601 genes, "Human Breast
Cancer Block A Section 1") are public, but the 20-region pathologist
annotation GraphST's paper reports against is not shipped with them. Kang et
al. 2025's benchmark code (the same source used for DLPFC) expects a
`metadata.tsv` with `ID`/`annot_type`/`fine_annot_type` columns for a dataset
named "Breast_cancer" (`utils_for_all.py::get_adata`), but the Zenodo code
release (15114362) that ships only DLPFC's raw data as-is does not include it.
Cross-referencing that exact expected schema against Kang et al.'s companion
Figshare project (figshare.com/projects/Benchmark_ST_analysis/234116, article
28200299 "10X Visium") found one file group matching it exactly (3798 rows,
20 unique `fine_annot_type` values) plus `aligned_fiducials.jpg` /
`detected_tissue_image.jpg` -- standard 10x Space Ranger QC images present on
no other file group in that project, confirming a distinct raw sample rather
than a mislabeled DLPFC slice. Verified by direct download, not inferred from
a filename. See `src/data/load_breast_cancer.py`'s docstring for the full
trace.

**Result, 5 seeds, same protocol as DLPFC (mclust-equivalent + spatial
refinement, K=20), DLPFC-tuned defaults used as-is (no per-dataset
retuning):**

| | Ours | GraphST |
|---|---|---|
| Per-seed | 0.412 ± 0.072 | 0.621 ± 0.021 |
| Consensus | 0.546 | 0.643 |

Literature (GraphST's own paper, PMC9977836): 0.54-0.57.

**Unlike DLPFC, this gap is real and consistent, not a coin-flip.** All 5
seeds favor GraphST (rank-biserial -1.0, the maximum possible magnitude);
Wilcoxon signed-rank p=0.0625 -- the smallest p-value obtainable at n=5, so
"not significant at α=0.05" here reflects the sample size, not an ambiguous
result the way DLPFC's p=0.123 did. Bootstrap 95% CI on the mean per-seed
difference: [-0.271, -0.145], clearly excluding zero. Honest read: on a tissue
genuinely different from DLPFC cortex (invasive/DCIS breast carcinoma vs.
6-layer cortex + white matter), even on the same platform family (10x
Visium), the near-parity result from DLPFC does **not** generalize --
GraphST's advantage here is real, roughly 4-8x the size of the (statistically
undetectable) DLPFC gap.

**A side observation worth flagging, not burying:** our local GraphST
re-score (0.621 per-seed, 0.643 consensus) *exceeds* the paper's own reported
range (0.54-0.57). The same pattern showed up on DLPFC (our protocol-corrected
GraphST re-score, 0.5972, landed close to but this time *below* the
literature's 0.6327) -- the direction differs across datasets, so this isn't
a one-directional "our harness inflates GraphST" bias, more likely reflecting
the 5-seed consensus-clustering advantage (GraphST's paper reports what
appears to be a single run) applied inconsistently by tissue. Recorded here
rather than smoothed over.

Script: `src/eval/run_breast_cancer.py`. Raw results:
`outputs/logs/breast_cancer_results.json`.

## Phase C, second result: Slide-seqV2 mouse hippocampus -- no clean winner

Per the earlier literature check, no published GraphST ARI exists for this
platform, and squidpy's own distribution carries cell-type labels (14
categories: CA1_CA2_CA3_Subiculum, DentatePyramids, Astrocytes,
Oligodendrocytes, ...), not spatial domains -- confirmed directly (not
assumed) by inspecting `obs['cluster']`. It also ships no raw counts (checked
`.raw` and `.layers`, both absent/normalized) -- see
`src/data/load_slideseqv2.py`. Reported via unsupervised proxies instead:
silhouette and spatial coherence (mean Moran's I), 3 seeds, K=14 (the
cell-type count, used only as a common convenient anchor for both methods,
not a claim about the true number of spatial domains).

**A real, hardware-driven obstacle, not a design choice: GraphST's own
package cannot run on the full 41,786-spot dataset on this machine.** Its
`GraphSTModel.__init__` unconditionally `.copy()`s the input AnnData, which
includes a dense `(n_spots, n_spots)` `adj` matrix -- and this stays dense
regardless of which of GraphST's two construction functions builds it.
`construct_interaction` (the default, pairwise-distance based) and
`construct_interaction_KNN` (GraphST's own documented alternative for
`datatype in ['Stereo', 'Slide']`, meant for exactly this kind of
large-N, non-Visium data) both populate a full dense array; the KNN version
is only cheaper to *compute*, not smaller to *store*. Confirmed by a real
`numpy._core._exceptions.ArrayMemoryError: Unable to allocate 13.0 GiB` (this
machine has 16GB total RAM) -- not inferred, hit twice (once via our own
`consensus_cluster`'s O(n^2) co-association matrix, dropped for this dataset
in favor of per-seed mean/std; once via GraphST's own `adj` construction).
`src/models/run_graphst.py` now accepts a `datatype` parameter (default
`"10X"`, matching GraphST's own default and every other dataset's call site
unchanged) so `datatype="Slide"`/`"Stereo"` routes to `construct_interaction_KNN`
when a caller needs it -- but since that still doesn't fix the storage cost,
the practical fix here was subsampling to 12,000 of 41,786 spots (fixed seed,
~1.4GB dense matrix), applied identically before both methods train.

| | Ours | GraphST |
|---|---|---|
| Silhouette | 0.146 ± 0.004 | 0.069 ± 0.002 |
| Spatial coherence (Moran's I) | 0.900 ± 0.0002 | 0.929 ± 0.004 |
| ARI vs. cell type (caveat -- wrong task) | 0.061 | 0.071 |

**Unlike breast cancer, this platform shows no clean winner.** Our embeddings
separate noticeably better (silhouette roughly 2x GraphST's); GraphST's
clusters are slightly more spatially contiguous. Both cell-type ARIs are low
and similar -- expected, since a domain-identification method recovering
cell-type boundaries would be a coincidence, not a success criterion, so
neither low number should be read as "failing." Taken together with breast
cancer: Phase C's honest pattern across two genuinely different platforms is
"sometimes a real gap (breast cancer), sometimes a wash on different axes
(Slide-seqV2)" -- not a single, clean generalization story either direction.

Script: `src/eval/run_slideseqv2.py`. Raw results:
`outputs/logs/slideseqv2_results.json`.

## Current best configuration (defaults updated in code)

`train_spatial_address_model(n_hops=4, lambda_usage=0.02, memory_slots=16,
memory_dim=128, hidden_dim=256, expression_weighted=True, epochs=600)` on
`preprocess_hvg()` output, clustered with `cluster_embedding(..., refine=True)`.
`memory_slots=16`, `n_hops=4`, and `lambda_usage=0.02` are all cross-validated
(Stages 8, 11); `expression_weighted=True` (Stage 13) was validated on a
targeted subset first, then confirmed at full scale (see the caveat above --
not the same rigor as the other three). On the 11 truly-held-out slices:
**0.5621 ± 0.0821** consensus vs. GraphST's **0.5724 ± 0.0861** (gap 0.010,
not statistically significant at n=11 on either metric -- Stage 12/13). On
the tuning slice alone (151673, not representative -- see Stage 7/8): the
current config scores 0.529 ± 0.054 per-seed / 0.561 consensus (5 seeds),
close to but not exactly comparable across runs since it was never the
tuning target for this config.
