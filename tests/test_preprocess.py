import numpy as np

from src.data.load_visium import load_visium_crop
from src.data.preprocess import get_hvg_features, preprocess, preprocess_hvg


def test_preprocess_crop():
    adata_raw = load_visium_crop()
    n_cells_before, n_genes_before = adata_raw.shape

    adata = preprocess(adata_raw)

    assert not np.isnan(adata.X.data if hasattr(adata.X, "data") else adata.X).any()

    assert "spatial_connectivities" in adata.obsp
    conn = adata.obsp["spatial_connectivities"]
    assert (conn != conn.T).nnz == 0

    n_cells_after, n_genes_after = adata.shape
    assert n_cells_after <= n_cells_before
    assert n_genes_after <= n_genes_before


def test_preprocess_hvg_selects_requested_genes_and_keeps_all_spots():
    adata_raw = load_visium_crop()
    n_cells_before = adata_raw.n_obs
    n_top_genes = 3000

    adata = preprocess_hvg(adata_raw, n_top_genes=n_top_genes)

    # No spot filtering: on annotated benchmark data every spot carries a label.
    assert adata.n_obs == n_cells_before
    assert int(adata.var["highly_variable"].sum()) == n_top_genes
    assert "counts" in adata.layers  # raw counts retained for NB/ZINB losses
    assert "spatial_connectivities" in adata.obsp

    features = get_hvg_features(adata)
    assert features.shape == (n_cells_before, n_top_genes)
    assert not np.isnan(features).any()
    # scale(zero_center=False, max_value=10) -> non-negative, capped
    assert features.min() >= 0
    assert features.max() <= 10 + 1e-5
