import numpy as np
import scanpy as sc
import squidpy as sq


def preprocess(
    adata,
    min_counts=500,
    min_cells=10,
    target_sum=1e4,
    n_neighs=6,
    coord_type="grid",
):
    """Filter, normalize, log-transform, and build a spatial neighbor graph.

    coord_type="grid" fits Visium's hex/square lattice. Non-lattice platforms
    (e.g. Slide-seqV2) should pass coord_type="generic", which builds the
    neighbor graph from raw spatial coordinates instead of grid adjacency.
    """
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if coord_type == "grid":
        sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=n_neighs)
    else:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs)
    return adata


def preprocess_hvg(
    adata,
    n_top_genes=3000,
    target_sum=1e4,
    n_neighs=6,
    coord_type="grid",
):
    """Field-standard ST preprocessing: HVG selection on raw counts, then
    normalize/log1p/scale, plus a spatial neighbor graph.

    This mirrors GraphST's own `preprocess()` (and STAGATE's, and what the
    benchmarking studies use), which matters for a like-for-like comparison:

      1. `highly_variable_genes(flavor="seurat_v3", n_top_genes=3000)` --
         seurat_v3 expects RAW COUNTS and must run before normalization.
      2. normalize_total -> log1p
      3. scale(zero_center=False, max_value=10) -- non-centered so the matrix
         stays non-negative, which suits count-derived data and keeps a
         downstream NB/ZINB head meaningful.

    Unlike `preprocess()` above, this does NOT aggressively filter spots. On
    annotated benchmark data (e.g. DLPFC) every spot carries a ground-truth
    label, so dropping spots for low counts silently shrinks the evaluation
    set. Genes are still filtered, since all-zero genes carry no signal.

    Raw counts are stashed in `adata.layers['counts']` before normalization so
    a count-based (NB/ZINB) reconstruction loss can still use them.
    """
    sc.pp.filter_genes(adata, min_cells=1)
    adata.layers["counts"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    if coord_type == "grid":
        sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=n_neighs)
    else:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs)
    return adata


def get_hvg_features(adata):
    """Dense matrix of the highly-variable genes only, for model input/target.

    Feeding real (HVG) expression rather than PCA scores means the model
    reconstructs biologically interpretable gene-level signal instead of
    reconstructing an already-lossy linear compression.
    """
    if "highly_variable" not in adata.var:
        raise ValueError("Run preprocess_hvg() first -- no 'highly_variable' column found.")
    subset = adata[:, adata.var["highly_variable"]]
    x = subset.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.ascontiguousarray(x, dtype=np.float32)


def get_pca_features(adata, n_comps=50):
    """Return adata.obsm['X_pca'], computing it if not already present.

    Feeding raw gene expression (~15-18k dims after filtering) directly into
    a feature_dim x feature_dim layer (e.g. EmbeddedMemoryLayer.query_proj)
    makes that layer hundreds of millions of parameters -- a real VRAM/compute
    blowup, not just an inefficiency. PCA to n_comps=50 matches what the
    Scanpy baseline already uses, keeping the comparison on equal footing.
    """
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_comps)
    # scanpy's PCA output can be a non-contiguous / negative-stride view,
    # which torch.tensor() refuses to wrap directly.
    return np.ascontiguousarray(adata.obsm["X_pca"])
