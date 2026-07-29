# Embedded Memory Architecture for Spatial Transcriptomics

A research prototype exploring **memory-addressing as an alternative to explicit graph
message passing** for spatial-domain identification in spatial transcriptomics (ST) data.
Instead of a GNN layer that aggregates each spot's neighbors directly, spots attend over a
bank of learned "memory slots," with spatial structure encouraged through a
spatial-smoothness loss on the resulting embeddings rather than architectural aggregation.

This is a **Phase 0 prototype with honestly-reported, in-progress results** — see
[Results](#results) and [Limitations](#limitations--honest-status) below before drawing
conclusions from it. It does not currently beat the field's state of the art.

## Results

All numbers below are real, reproducible, and logged in `outputs/logs/`. Nothing is
fabricated; anything not actually run is marked as such with the reason why.

### Real ARI vs. ground truth (DLPFC 151673, spatialLIBD manual cortical-layer annotations)

This is the benchmark that actually matters: **DLPFC sample 151673** (Maynard et al./
spatialLIBD, human dorsolateral prefrontal cortex, 6 cortical layers + white matter, 7
classes), with real, manually-annotated ground-truth labels — not a proxy metric.

| Method | ARI vs. ground truth | Source |
|---|---|---|
| GraphST | 0.633 | Literature ([Kang et al. 2025](https://academic.oup.com/nar/article/53/7/gkaf303/8114322), recomputed by us from their released per-spot predictions, not copied from the paper's text) |
| STAGATE | 0.589 | Literature |
| Spatial_MGCN | 0.556 | Literature |
| BayesSpace | 0.550 | Literature |
| DeepST | 0.538 | Literature |
| conST | 0.528 | Literature |
| STMGCN | 0.511 | Literature |
| **GraphST (our local run)** | **0.491** | Run locally, this repo |
| SEDR | 0.472 | Literature |
| SpaGCN | 0.465 | Literature |
| Seurat | 0.430 | Literature |
| stLearn | 0.368 | Literature |
| CCST | 0.356 | Literature |
| SpaceFlow | 0.351 | Literature |
| SCGDL | 0.322 | Literature |
| **EmbeddedMemoryLayer (trained, ours)** | **0.303** | Run locally, this repo |
| **Scanpy PCA+Leiden (baseline, ours)** | **0.253** | Run locally, this repo |

**Honest read**: the trained memory layer beats our own from-scratch baseline (0.303 vs.
0.253), which shows the architecture is learning real structure — but it does **not**
currently beat GraphST or most of the field's published methods. Full writeup, including
why the gap exists and what closing it would take, is in
[`outputs/logs/results_table.md`](outputs/logs/results_table.md).

### Domain identification on Visium mouse brain (unsupervised proxy metrics only)

`crop`/`full` (squidpy's Visium mouse-brain datasets) and Slide-seqV2 have **no
ground-truth region labels**, so only internal-quality proxies are reported here —
silhouette (embedding separation) and spatial coherence (mean Moran's I of cluster
labels on the spatial graph). These should **not** be read as accuracy claims.

| Method | Dataset | n_clusters | Silhouette | Spatial coherence (Moran's I) |
|---|---|---|---|---|
| Scanpy PCA+Leiden (baseline) | crop | 10 | 0.144 | 0.593 |
| EmbeddedMemoryLayer (trained) | crop | 14 | 0.289 | 0.779 |
| GraphST | crop | 10 | 0.145 | 0.638 |
| Scanpy PCA+Leiden (baseline) | full | 18 | 0.129 | 0.719 |
| EmbeddedMemoryLayer (trained) | full | 25 | 0.300 | 0.809 |
| GraphST | full | 18 | 0.147 | 0.580 |

Full table with wall-clock times and ARI-between-methods (agreement, not accuracy):
[`outputs/logs/results_table.md`](outputs/logs/results_table.md).

### Figures

**Trained memory-layer clusters on real tissue** (Visium mouse brain, crop) — spatially
coherent domains, not the salt-and-pepper noise an untrained/random-init model produces:

![Spatial memory clusters](outputs/figures/spatial_memory_clusters.png)

**UMAP of the trained memory embedding**, colored by the same clusters:

![UMAP of memory embedding](outputs/figures/umap_memory_embedding.png)

**Baseline vs. GraphST vs. trained memory layer**, side by side on the same tissue image:

![Baseline vs GraphST vs memory layer](outputs/figures/baseline_vs_memory_layer.png)

## Limitations & honest status

- **Not currently state of the art.** See the ARI table above. The gap is real and is
  discussed candidly in `outputs/logs/results_table.md`.
- **Hyperparameters were tuned on mouse Visium, not DLPFC.** Applied out of the box to
  DLPFC, the model over-segmented badly (34 clusters vs. 7 true layers, ARI 0.17) until
  a resolution-matching fix (`src/eval/metrics.py::search_leiden_resolution`, the same
  convention GraphST itself uses) was added and applied fairly to all methods.
- **Single run, no seed averaging.** Our local GraphST run (0.491) already undershoots
  its own literature-reported number (0.633) — some of the gap for our method is likely
  ordinary run-to-run variance on top of the real architectural gap, not yet disentangled.
  Even accounting for that, the current results are not competitive with the field,
  including several methods (e.g. stGRL, MAEST) published more recently than GraphST/STAGATE
  that were not attempted here.
- **STAGATE and Garfield could not be run locally** (Windows). STAGATE hard-depends on
  `torch_sparse`, which has no prebuilt wheel for this project's torch build
  (2.11.0+cu128; PyG's wheel index tops out at 2.9.1) and fails to build from source.
  Garfield depends on `pysam`, which has never shipped a Windows wheel at all. Both are
  wired up in `notebooks/04_colab_scaleup.ipynb` to run on Colab's Linux runtime instead.
- **SpaCeNet is intentionally excluded** from the ARI table — it infers a gene-gene
  conditional-independence graphical model, not spot clusters, so there is no
  cluster-label output to score.
- **SpatialDG has no confirmed public implementation** as of this writing.

## Repository structure

```
recomb2027/
  src/
    data/
      load_visium.py       # squidpy Visium crop/full loaders
      load_geo.py           # Slide-seqV2 loader (see note below)
      load_dlpfc.py         # DLPFC 151673 + real ground-truth layers
      preprocess.py         # filter/normalize/log1p/spatial graph + PCA helper
      data_stats.py         # real sparsity/shape stats -> outputs/logs/data_stats.txt
    models/
      memory_layer.py        # EmbeddedMemoryLayer + trainable autoencoder wrapper
      train_memory_layer.py  # training loop, tuned hyperparameters, clustering
      baseline_pca.py         # Scanpy PCA+Leiden baseline
      run_graphst.py          # GraphST wrapper (real package, method="leiden")
    eval/
      metrics.py               # silhouette, spatial coherence, ARI, resolution search
      run_benchmark.py         # baseline vs. memory layer vs. GraphST, crop/full
      run_dlpfc_benchmark.py   # real ARI-vs-ground-truth benchmark on DLPFC
      compute_literature_ari.py  # reproduces the literature ARI numbers from source data
      vram_profile.py           # VRAM/latency profiling utility
    viz/
      spatial_plots.py    # generates the figures above
  tests/                  # pytest suite, 15 tests
  notebooks/
    04_colab_scaleup.ipynb  # Slide-seqV2 scale-up + STAGATE/Garfield on Colab
  outputs/
    figures/    # PNGs referenced above
    logs/       # real logged results (JSON + results_table.md)
    checkpoints/  # trained model weights (.pt); large .h5ad snapshots are gitignored
```

`load_geo.py`'s name is a holdover from the original plan, which targeted GEO accession
GSE129788 as a "messier, larger" dataset — that accession turned out to be dissociated
scRNA-seq with no spatial coordinates, so it was replaced with `squidpy.datasets.slideseqv2()`.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11. Torch is pinned to the
`cu128` wheel index in `pyproject.toml` (see `[tool.uv.sources]` / `[[tool.uv.index]]`) —
`uv sync` will pull a CUDA build automatically if a compatible GPU is present, and fall
back to CPU otherwise.

```bash
uv sync
uv run pytest        # 15 tests
```

## Reproducing the results

```bash
# Data stats (real sparsity/shape numbers)
uv run python -m src.data.data_stats

# Baseline vs. trained memory layer vs. GraphST, on Visium crop + full
uv run python -m src.eval.run_benchmark

# Real ARI vs. ground truth, on DLPFC 151673 (requires data/dlpfc_151673/, not
# included in this repo -- see src/data/load_dlpfc.py for the expected layout)
uv run python -m src.eval.run_dlpfc_benchmark

# Regenerate the figures
uv run python -m src.viz.spatial_plots
```

The DLPFC 151673 raw data + ground truth used above were recovered from
[Kang et al. 2025](https://academic.oup.com/nar/article/53/7/gkaf303/8114322)'s own
released benchmark data ([Zenodo record 15114362](https://zenodo.org/records/15114362)),
which sidesteps needing spatialLIBD's R/Bioconductor distribution while still using the
field's standard ground-truth annotations for this slide. This data is not committed to
the repo (see `.gitignore`); re-download it from the Zenodo record to reproduce
`run_dlpfc_benchmark.py`.

## Related work

**Baselines compared against:**
[GraphST](https://github.com/JinmiaoChenLab/GraphST) (Long et al., *Nat Commun* 2023) —
run locally. [STAGATE](https://github.com/QIFEIDKN/STAGATE_pyG) (Dong & Zhang, *Nat
Commun* 2022), [Garfield](https://github.com/zhou-1314/Garfield) — blocked on Windows,
see [Limitations](#limitations--honest-status). [SpaCeNet](https://github.com/sschrod/SpaCeNet) —
a different task (gene-network inference), kept out of the clustering comparison.
SpatialDG (*Briefings in Bioinformatics* 2026) — no confirmed public implementation yet.

**Field benchmark used for ground truth comparison:** Kang, Zhang, Qian, Liang, Wu,
"Benchmarking computational methods for detecting spatial domains and domain-specific
spatially variable genes from spatial transcriptomics data," *Nucleic Acids Research*
53(7), 2025.

**Adjacent prior art on memory + graphs** (differentiated from, not built on):
Memory-Based Graph Networks (arXiv:2002.09518); Memory-Augmented GNNs: A Brain-Inspired
Review (arXiv:2209.10818).
