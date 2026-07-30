# Results Table

All numbers below are pulled directly from `outputs/logs/benchmark_results.json`
and `outputs/logs/dlpfc_ari_results.json` (local run, RTX 4050 Laptop / 6GB VRAM,
Python 3.11 / torch 2.11.0+cu128). Nothing here is fabricated or assumed —
anything not actually run is marked `TODO: not yet benchmarked` with the real
reason it wasn't run.

## Headline result: 12-slice, 5-seed held-out evaluation (cross-validated config)

**This section supersedes the single-slice numbers below as the number that
matters.** This is the *second* version of this evaluation. The first (see
`outputs/logs/dlpfc_multislice_results_memslots32.json`) used `memory_slots=32`,
chosen by tuning on 151673 alone; that slice turned out to be the most
favorable slice in the whole set for that config, and the resulting held-out
gap was 0.129. Cross-validating `memory_slots` across 3 *different* slices
(`src/eval/cross_validate_capacity.py`, none of them 151673) selected
`memory_slots=16` instead — a real, measured improvement, not a new
architecture:

| | Held-out (11 slices) | All 12 slices |
|---|---|---|
| Ours, `memory_slots=32` (single-slice-tuned) | 0.4391 ± 0.0883 | 0.4501 ± 0.0921 |
| **Ours, `memory_slots=16` (cross-validated)** | **0.4815 ± 0.0979** | **0.4818 ± 0.0937** |
| GraphST (matched protocol) | 0.5685 ± 0.0825 | 0.5707 ± 0.0793 |
| **Gap (cross-validated config)** | **0.0870** | 0.0889 |

Fixing the tuning methodology alone closed about a third of the gap (0.129 →
0.087). It does not close the remaining gap. Full per-slice breakdown, produced
by `src/eval/run_dlpfc_multislice.py` and logged in
`outputs/logs/dlpfc_multislice_results.json`:

| Slice | Ours (mean ± std, 5 seeds) | GraphST (mean ± std, 5 seeds) | Gap |
|---|---|---|---|
| 151507 | 0.516 ± 0.027 | 0.514 ± 0.063 | **−0.002** (ours ahead) |
| 151508 | 0.494 ± 0.043 | 0.488 ± 0.015 | **−0.006** (ours ahead) |
| 151509 | 0.440 ± 0.053 | 0.441 ± 0.043 | 0.001 (tied) |
| 151510 | 0.487 ± 0.032 | 0.539 ± 0.030 | 0.052 |
| 151669 | 0.467 ± 0.031 | 0.592 ± 0.009 | 0.125 |
| 151670 | 0.506 ± 0.122 | 0.534 ± 0.029 | 0.028 |
| 151671 | 0.598 ± 0.131 | 0.625 ± 0.010 | 0.027 |
| 151672 | 0.694 ± 0.074 | 0.769 ± 0.006 | 0.075 |
| 151674 | 0.398 ± 0.067 | 0.618 ± 0.003 | 0.221 |
| 151675 | 0.339 ± 0.057 | 0.548 ± 0.058 | 0.209 |
| 151676 | 0.356 ± 0.095 | 0.584 ± 0.015 | 0.228 |
| *151673 (tuning slice, excluded from headline)* | *0.485 ± 0.108* | *0.595 ± 0.014* | *0.110* |

**Read this honestly, both ways.** On 3 of 11 held-out slices (151507, 151508,
151509) we now match or slightly beat GraphST — genuinely encouraging. But on
three others (151674, 151675, 151676 — one full DLPFC subject, coincidentally
the same subject 151673 belongs to) the gap is 0.21–0.23, as large as before
cross-validation. The improvement is real and averaged out positively, but it
is uneven across slices, not a uniform win. Our per-seed variance still exceeds
GraphST's on most held-out slices (e.g. 151671: ±0.131 vs. ±0.010). See Stage
7/8 in `outputs/logs/stage2_progress.md` for the within-subject variance
analysis this result should be read alongside.

**Conclusion.** The architecture works — it learns real structure and beats a
from-scratch baseline — and the address-propagation mechanism is validated
(monotonic ARI gain with hop count, and it beats both tested hybrid variants that
add feature message passing). Fixing a real overfitting bug (cross-validating
hyperparameters instead of single-slice tuning) closed roughly a third of the
held-out gap (0.129 → 0.087), with the improvement concentrated on some slices
and absent on others. It does **not** beat GraphST overall. Any further tuning
should continue to cross-validate across multiple slices, not one — the
subject-3 slices (151674–676) that still show a large gap are the next
concrete thing to investigate.

## Single-slice detail (151673, the tuning slice — see above for the real result)

The numbers below were measured entirely on the tuning slice, before the 12-slice
evaluation existed, and are kept for the detailed ablation history (protocol
validation, hybrid rejection, capacity sweep). They should not be read as a
generalization claim on their own.

### Real ARI vs. ground truth (DLPFC 151673, spatialLIBD manual layer annotations)

The Visium crop/full/Slide-seqV2 datasets used elsewhere in this project have
no ground-truth region labels, so the section below (silhouette/spatial
coherence) explicitly could **not** validate biological correctness — only
internal cluster quality. This section closes that gap using a real
ground-truth dataset: **DLPFC sample 151673** (Maynard/spatialLIBD, 6-layer
human cortex + white matter, 7 classes), with raw counts + manually-annotated
layers recovered from Kang et al. 2025 (*Nucleic Acids Research*, "Benchmarking
computational methods for detecting spatial domains...")'s own released data
(Zenodo record 15114362) — see `src/data/load_dlpfc.py`.

The "literature" column's numbers were **computed directly by us** from that
paper's own released per-spot predictions for 14 methods against the same
ground-truth layers (`src/eval/run_dlpfc_benchmark.py`), not copied from the
paper's text — two independent attempts to read the paper's own summarized
numbers gave inconsistent figures (0.498 vs. 0.515 for STAGATE in two separate
fetches), so we went to the source data instead rather than trust either.

**All methods now share one clustering protocol** (`src/eval/clustering.py`):
mclust-equivalent (`GaussianMixture(covariance_type="tied")`, the correct mapping
for mclust's `EEE`) on PCA-20 of the embedding, then spatial label refinement —
which is what GraphST and the benchmarking studies actually use. K is set to the
true number of layers for every method, the standard convention in this benchmark.

This protocol change mattered a great deal and is worth stating plainly: an
earlier version of this table scored our methods with Leiden and reported GraphST
at 0.4911 (single seed). Re-scoring the *identical* GraphST embedding under the
correct protocol gives 0.5713, and 0.5902 with refinement (still single-seed) —
close to the literature's 0.6327. **Our harness had been understating GraphST,
not GraphST underperforming.**

Both methods are now also compared across **5 seeds each**, not one seed vs.
five — comparing a multi-seed mean against someone else's single run is not a
fair comparison. GraphST's own default seed (41 → 0.590) was not even its best
across the 5 tried (0.585–0.615). GraphST: 0.5972 ± 0.0120. Ours: 0.5713 ± 0.0057
after a capacity sweep found `memory_slots=32` clearly beats the Stage 2 default
of 64 (inverted-U: 8/16 unstable, 32 optimal, 256 → 0.408). **Remaining gap:
+0.026 in GraphST's favor** — smaller than earlier single-seed comparisons
suggested, and GraphST's variance (±0.012) is over 2× ours (±0.006), but the gap
is still real and consistent: GraphST's worst seed still beats our best.

| Method | ARI vs. ground truth | Source |
|---|---|---|
| GraphST | 0.6327 | Literature (Kang et al. 2025, computed by us from their released predictions) |
| **GraphST (our local run, matched protocol, 5 seeds)** | **0.5972 ± 0.0120** | Run locally, this repo |
| STAGATE | 0.5892 | Literature |
| Spatial_MGCN | 0.5561 | Literature |
| **SpatialAddressMemoryAutoencoder (ours, tuned, 5 seeds)** | **0.5713 ± 0.0057** | Run locally, this repo |
| BayesSpace | 0.5499 | Literature |
| DeepST | 0.5384 | Literature |
| conST | 0.5277 | Literature |
| STMGCN | 0.5107 | Literature |
| SEDR | 0.4723 | Literature |
| SpaGCN | 0.4652 | Literature |
| Seurat | 0.4295 | Literature |
| stLearn | 0.3681 | Literature |
| CCST | 0.3563 | Literature |
| SpaceFlow | 0.3510 | Literature |
| SCGDL | 0.3216 | Literature |
| *Ours, Phase 0 (PCA input, no propagation, Leiden)* | *0.3032* | Superseded |
| *Scanpy PCA+Leiden baseline (ours)* | *0.2532* | Superseded protocol |

### Ablations behind the 0.303 → 0.551 improvement

| Change | ARI | Note |
|---|---|---|
| Phase 0 starting point | 0.3032 | PCA-50 input, spatial info only as a loss penalty |
| + HVG input, address propagation, **no** usage regularizer | **0.0000** | **Total slot collapse** — `slots_used=1`, identical loss at every hop count |
| + marginal usage-entropy regularizer (1 address hop) | 0.4889 | The anti-collapse fix |
| + 2 address hops | 0.5375 | |
| + 4 address hops | **0.5487** | Monotonic in hops — the mechanism works |
| + 5-seed average at best config | **0.5510 ± 0.0178** | |

Two hypotheses were tested and **rejected on evidence**, recorded here rather than
quietly dropped:

| Rejected idea | ARI | Why it was plausible |
|---|---|---|
| Per-row attention entropy as the anti-collapse term | 0.0000 | Intuitive but the wrong quantity — the fix is *marginal* slot usage, not per-spot spread |
| NB likelihood on raw counts | 0.346 ± 0.095 | 68–97% zeros genuinely violate MSE's Gaussian assumption |
| ZINB likelihood | 0.331 ± 0.072 | Models dropout explicitly; stGRL's core contribution |
| NB/ZINB + contrastive regularization | 0.183–0.251 | GraphST/MAEST both cite contrastive terms as essential |

The count-likelihood result is a genuine negative: a more principled likelihood did
not produce a more clusterable embedding, plausibly because fitting per-gene
dispersion over 3000 genes gives the encoder a noisier gradient. Seed variance also
grew 4–8×. The code is retained and tested (`src/models/count_losses.py`, verified
against `scipy.stats.nbinom`) as a documented ablation.

### Biological validation of the data (model-independent)

Canonical layer markers from Maynard et al. 2021 vs. the annotated layers. This
never looks at a model, so no model can game it (`src/eval/biological_validation.py`).

| Layer | Marker | Provenance | log2 FC in-layer vs. rest | Enriched |
|---|---|---|---|---|
| Layer1 | AQP4 | verified | +0.53 | yes |
| Layer2 | HPCAL1 | verified | +1.22 | yes |
| Layer3 | FREM3 | verified | +1.82 | yes |
| Layer4 | RORB | convention | +1.28 | yes |
| Layer5 | TRABD2A | verified | +3.04 | yes |
| Layer5 | PCP4 | verified | +1.38 | yes |
| Layer6 | KRT17 | verified | +1.78 | yes |
| WM | MOBP | verified | +2.36 | yes |

**8/8 enriched.** Separately, the figshare `.h5ad` and Zenodo copies of this slice
agree exactly (3639 spots, identical per-layer counts), and the matrix is confirmed
raw integer counts as `seurat_v3` HVG selection requires.

**Honest reading of this table**: on real ground-truth ARI, our trained
memory layer beats our own Scanpy baseline (0.303 vs. 0.253), but it does
**not** beat GraphST — neither our own local GraphST run (0.491) nor any of
the 14 methods' literature-reported scores except SCGDL/SpaceFlow/CCST, which
it also trails once GraphST/STAGATE/BayesSpace etc. are accounted for. Our
local GraphST run (0.491) also underperforms the literature's own GraphST
number (0.633) — a real, honest gap likely from differences in random seed,
exact preprocessing, or number of training runs (the paper may report a
best-of-N), not evidence of a bug we've found yet. **This is not currently a
"beats SOTA" result** — it is real progress (a working, trained model that
beats a from-scratch baseline) with a clear, quantified gap left to close.

**Visium crop/full and Slide-seqV2 have no ground-truth cluster labels**, so
the section below reports internal-quality proxies only for those three. A
separate section further down uses a fourth dataset (DLPFC 151673) that *does*
have real ground-truth region labels, with real ARI-vs-truth numbers. For the
label-free datasets, what *is* reported:

- **Silhouette**: cluster separation in each method's own embedding space.
- **Spatial coherence (mean Moran's I)**: whether a method's clusters form
  spatially contiguous regions rather than scattered speckle (see
  `src/eval/metrics.py::spatial_coherence`).
- **ARI(baseline, X)**: agreement between two methods' cluster assignments.
  This is **not** an accuracy score — neither side is ground truth, it's a
  measure of how similarly two methods carve up the same tissue.

## Domain-identification comparison (real, run locally)

> **Which model this section refers to.** The rows below measure the **Phase 0**
> `EmbeddedMemoryLayer` (PCA-50 input, spatial information only as a loss penalty,
> Leiden clustering) — *not* the current `SpatialAddressMemoryAutoencoder`. They are
> kept because they are real measurements of that model on label-free data, but the
> ground-truth ARI section above supersedes them as the headline result. Note also
> the caveat below: scoring well on these unsupervised proxies did **not** translate
> into competitive ARI once ground truth was available, which is precisely why the
> proxies are not treated as evidence of correctness.

| Method | Dataset | n_clusters | Silhouette | Spatial coherence (Moran's I) | Wall time (s) | ARI vs. baseline |
|---|---|---|---|---|---|---|
| Scanpy PCA+Leiden (baseline) | crop | 10 | 0.1443 | 0.5925 | 19.2 | -- |
| **EmbeddedMemoryLayer (trained)** | crop | 14 | **0.2887** | **0.7790** | 5.0 | 0.314 |
| GraphST (Long et al. 2023, verified current SOTA) | crop | 10 | 0.1454 | 0.6381 | 18.9 | 0.502 |
| Scanpy PCA+Leiden (baseline) | full | 18 | 0.1293 | 0.7188 | 2.2 | -- |
| **EmbeddedMemoryLayer (trained)** | full | 25 | **0.3002** | **0.8093** | 1.5 | 0.400 |
| GraphST (Long et al. 2023, verified current SOTA) | full | 18 | 0.1470 | 0.5802 | 67.1 | 0.496 |

On both datasets tested, the trained EmbeddedMemoryLayer scores higher than
both the Scanpy baseline and GraphST on silhouette and spatial coherence.
**Read this carefully, not as a headline claim**: with no ground-truth
annotations, this shows the memory layer's clusters are more internally
separated and more spatially contiguous than the other two methods' clusters
*by these two specific unsupervised proxies* -- it does not establish that
they are more biologically correct. A method can score well on these proxies
without matching true anatomical/cell-type boundaries. Real ARI-vs-truth
validation (e.g. against a manually-annotated dataset) is the necessary next
step before this becomes a paper-ready superiority claim.

Training details: `lambda_spatial=10.0`, `epochs=300` (chosen via a sweep on
crop over `lambda_spatial in [0.1, 20]` and `epochs in [300, 1000]`; see
`src/models/train_memory_layer.py` docstring). Final median attention entropy
was 3.01 (crop) / 2.11 (full) out of a max of 6.24 (512 memory slots) --
healthy (entropy dipped as low as ~0.11 mid-training in an earlier tuning run
before settling higher, so slot collapse is a real failure mode that was
checked for, not assumed away).

## Not yet benchmarked (real reasons, not placeholders)

| Method | Status | Reason |
|---|---|---|
| SpaCeNet | Not ARI-comparable | It infers a gene-gene conditional-independence graphical model, not a spot clustering -- there is no cluster-label output to compute ARI/silhouette/coherence against. Keeping it out of this table is correct, not an omission. |
| STAGATE (STAGATE_pyG) | `TODO: infeasible locally` | Hard-imports `torch_sparse.SparseTensor`. PyG's official wheel index (data.pyg.org) has no prebuilt binary past torch 2.9.1; this project's torch is 2.11.0+cu128. `uv add torch-sparse` was attempted and fails during source build (no prebuilt wheel, build isolation can't see torch). Candidate for Colab, where an older PyG-compatible torch/CUDA combo can be selected. |
| Garfield | `TODO: infeasible on native Windows` | Depends on `pysam`, which has never shipped a Windows wheel (only manylinux/macOS on PyPI) -- confirmed by checking PyPI's file list directly. This is a permanent platform limitation, not a version mismatch. Needs Colab (Linux) or WSL. |
| SpatialDG | `TODO: not yet benchmarked` | No confirmed public pip-installable implementation as of this writing. |
| EmbeddedMemoryLayer / GraphST | `TODO: not yet benchmarked` | Slide-seqV2 (41,786 spots) -- deferred to `notebooks/04_colab_scaleup.ipynb` per the original Phase 0 scope split (local = fast iteration on Visium, Colab = larger-scale + additional baselines). |
