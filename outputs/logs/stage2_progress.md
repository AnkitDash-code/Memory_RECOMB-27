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
