"""STAGATE (Dong & Zhang, Nat Commun 2022) wrapper, for use as a real comparator.

STATUS: **NOT YET EXECUTED.** STAGATE is blocked on this project's Windows
machine -- `torch_sparse` (a hard PyG dependency) has no prebuilt wheel for
torch 2.11.0+cu128 and fails to build from source there. This module is written
against STAGATE_pyG's published tutorial API and is validated in
`notebooks/05_comparators_and_generalization.ipynb` on Colab (Linux), where
the wheel problem does not exist. Until that notebook has been run, treat this
as unverified code, and do not put STAGATE numbers in any results table
sourced from anything but an actual run.

Installation (verified resolvable, not yet executed):
    pip install git+https://github.com/QIFEIDKN/STAGATE_pyG.git

Preprocessing mirrors STAGATE's own DLPFC tutorial (3000 seurat_v3 HVGs,
normalize_total 1e4, log1p) rather than this repo's `preprocess_hvg`, for the
same reason `run_graphst.py` defers to GraphST's own preprocessing: a
comparator should be run the way its authors run it, or the comparison
measures our preprocessing choices rather than their method.

The returned embedding is scored downstream by `src/eval/clustering.py`, the
protocol applied identically to every method here.
"""

import scanpy as sc

DEFAULT_RAD_CUTOFF = 150  # STAGATE's own DLPFC/Visium tutorial value


def run_stagate(adata, device, rad_cutoff=DEFAULT_RAD_CUTOFF, n_top_genes=3000,
                random_seed=0):
    """Train STAGATE on a raw-counts AnnData; returns adata with obsm['STAGATE'].

    Expects RAW counts (seurat_v3 HVG selection requires them), so pass a
    freshly-loaded slice, not one already run through this repo's preprocess().
    """
    import STAGATE_pyG as STAGATE

    adata = adata.copy()
    adata.var_names_make_unique()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    adata = STAGATE.train_STAGATE(adata, device=device, random_seed=random_seed)
    return adata
