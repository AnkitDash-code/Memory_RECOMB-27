import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score


def embedding_silhouette(embedding, labels):
    """Silhouette score of an embedding given cluster labels.

    Requires at least 2 clusters and fewer clusters than samples; returns
    None otherwise rather than raising, since a degenerate clustering
    (e.g. everything in one cluster) is a real outcome worth reporting as
    "undefined", not a crash.
    """
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return None
    return float(silhouette_score(embedding, labels))


def spatial_coherence(adata, cluster_key, connectivity_key="spatial_connectivities"):
    """Mean per-cluster Moran's I of one-hot cluster indicators.

    Moran's I is defined for continuous variables; applying it to a
    categorical cluster label directly does not have a meaningful
    interpretation, so each cluster is one-hot encoded and treated as its
    own binary spatial variable (a value near 1 means that cluster's spots
    are spatially clumped rather than scattered).
    """
    labels = adata.obs[cluster_key].to_numpy()
    categories = np.unique(labels)
    one_hot = np.stack([(labels == cat).astype(float) for cat in categories])
    connectivity = adata.obsp[connectivity_key]
    per_cluster_i = sc.metrics.morans_i(connectivity, one_hot)
    return {
        "mean": float(np.mean(per_cluster_i)),
        "per_cluster": dict(zip((str(c) for c in categories), (float(v) for v in per_cluster_i))),
    }


def cluster_agreement(labels_a, labels_b):
    """ARI between two methods' cluster assignments (agreement, not accuracy --
    neither side is ground truth)."""
    return float(adjusted_rand_score(labels_a, labels_b))


def search_leiden_resolution(adata, neighbors_key, n_clusters_target, start=0.1, end=3.0, increment=0.02):
    """Search for a Leiden resolution giving approximately n_clusters_target
    clusters (Leiden takes a resolution, not a cluster count directly).

    Without this, comparing methods at whatever cluster count their default
    resolution happens to produce is not a fair comparison -- e.g. leaving
    one method free to produce 34 clusters against a 7-cluster ground truth
    tanks its ARI for reasons unrelated to embedding quality. GraphST's own
    clustering() does the same kind of search; this makes our own methods
    follow the same convention instead of being compared unfairly.
    """
    best_resolution, best_diff = 1.0, float("inf")
    resolution = start
    while resolution <= end:
        sc.tl.leiden(adata, neighbors_key=neighbors_key, resolution=resolution, key_added="_res_search_tmp")
        n_clusters = adata.obs["_res_search_tmp"].nunique()
        diff = abs(n_clusters - n_clusters_target)
        if diff < best_diff:
            best_diff, best_resolution = diff, resolution
        if n_clusters == n_clusters_target:
            break
        resolution += increment
    del adata.obs["_res_search_tmp"]
    return best_resolution
