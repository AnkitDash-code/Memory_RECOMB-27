import numpy as np

from src.data.load_visium import load_visium_crop
from src.data.preprocess import preprocess


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
