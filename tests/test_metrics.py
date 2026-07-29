import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from src.eval.metrics import (
    cluster_agreement,
    embedding_silhouette,
    search_leiden_resolution,
    spatial_coherence,
)


def test_embedding_silhouette_basic():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(20, 4))
    cluster_b = rng.normal(loc=10.0, scale=0.1, size=(20, 4))
    embedding = np.vstack([cluster_a, cluster_b])
    labels = np.array([0] * 20 + [1] * 20)

    score = embedding_silhouette(embedding, labels)

    assert score is not None
    assert score > 0.9


def test_embedding_silhouette_degenerate_returns_none():
    embedding = np.random.default_rng(0).normal(size=(10, 3))
    labels = np.zeros(10, dtype=int)

    assert embedding_silhouette(embedding, labels) is None


def test_spatial_coherence_perfectly_clumped_beats_random():
    n = 40
    # a simple path graph: spot i connected to i-1 and i+1
    rows, cols = [], []
    for i in range(n - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
    connectivities = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    adata = ad.AnnData(X=np.zeros((n, 1)))
    adata.obsp["spatial_connectivities"] = connectivities

    clumped_labels = np.array([0] * (n // 2) + [1] * (n // 2))
    adata.obs["clumped"] = clumped_labels.astype(str)

    rng = np.random.default_rng(1)
    shuffled_labels = clumped_labels.copy()
    rng.shuffle(shuffled_labels)
    adata.obs["scattered"] = shuffled_labels.astype(str)

    clumped_score = spatial_coherence(adata, "clumped")["mean"]
    scattered_score = spatial_coherence(adata, "scattered")["mean"]

    assert clumped_score > scattered_score


def test_cluster_agreement_identical_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert cluster_agreement(labels, labels) == 1.0


def test_search_leiden_resolution_hits_target_cluster_count():
    rng = np.random.default_rng(0)
    n_per_cluster, n_clusters_target = 30, 4
    blobs = [
        rng.normal(loc=i * 20.0, scale=0.5, size=(n_per_cluster, 8))
        for i in range(n_clusters_target)
    ]
    x = np.vstack(blobs).astype(np.float32)

    adata = ad.AnnData(X=x)
    sc.pp.neighbors(adata, use_rep="X", key_added="test_neighbors")

    resolution = search_leiden_resolution(adata, "test_neighbors", n_clusters_target)
    sc.tl.leiden(adata, neighbors_key="test_neighbors", resolution=resolution, key_added="found")

    assert adata.obs["found"].nunique() == n_clusters_target
