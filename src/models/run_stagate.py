"""STAGATE (Dong & Zhang, Nat Commun 2022) wrapper, used as a real comparator.

STATUS: **EXECUTED, locally, no Colab needed.** This project had recorded
STAGATE as blocked on Windows because `torch_sparse` (a hard `GATConv`
dependency) supposedly had no prebuilt wheel for torch 2.11.0+cu128. That
claim was stale: PyG's wheel index now publishes builds through torch 2.13.0,
including `torch-sparse 0.6.18+pt211cu128`. Full 12-slice x 5-seed results are
in `outputs/logs/stagate_dlpfc_results.json` via `src/eval/run_stagate_dlpfc.py`;
see `outputs/logs/stage2_progress.md` (Phase B3) for the numbers and the
three-way significance test against GraphST.

Installation (verified working, not merely resolvable):
    uv pip install torch_sparse torch_scatter \
      --find-links "https://data.pyg.org/whl/torch-2.11.0+cu128.html"
    uv pip install git+https://github.com/QIFEIDKN/STAGATE_pyG.git

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
