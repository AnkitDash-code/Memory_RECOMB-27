"""Clustering protocol matching what the spatial-transcriptomics field actually uses.

GraphST, STAGATE, DeepST and BayesSpace report their published numbers using
R's `mclust`, not Leiden, plus (commonly) a spatial label-refinement step.
Comparing our methods against those numbers while running Leiden ourselves is
not an apples-to-apples comparison -- benchmarking literature reports ARI
swings on the order of 10% from this choice alone.

These helpers reproduce that protocol in pure Python so every method in this
repo can be scored the same way.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def mclust_equivalent(embedding, n_clusters, n_pcs=20, random_seed=2020):
    """Python stand-in for GraphST's `mclust_R(..., modelNames='EEE')`.

    mclust's `EEE` means equal volume, equal shape, equal orientation -- i.e.
    every mixture component shares one common full covariance matrix. In
    scikit-learn that is exactly `covariance_type="tied"` (NOT the default
    "full", which gives each component its own covariance).

    The PCA-to-20-components step mirrors GraphST's `utils.py::clustering`,
    which PCAs the embedding before handing it to mclust.
    """
    embedding = np.asarray(embedding)
    n_pcs = min(n_pcs, embedding.shape[1], embedding.shape[0])
    reduced = PCA(n_components=n_pcs, random_state=random_seed).fit_transform(embedding)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied",
        random_state=random_seed,
        n_init=10,
    )
    return gmm.fit_predict(reduced).astype(str)


def refine_labels_spatial(labels, coords, n_neighbors=50):
    """Majority-vote label smoothing over each spot's nearest spatial neighbours.

    Mirrors GraphST's `utils.py::refine_label(radius=50)`, which is applied as a
    post-processing step by several methods and by the benchmarking studies. It
    cleans up isolated salt-and-pepper misassignments inside otherwise coherent
    domains -- appropriate here because real cortical layers are spatially
    contiguous bands, so a lone spot of a different label inside a band is far
    more likely noise than signal.
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)

    # +1 because the query point is its own nearest neighbour; drop it below.
    k = min(n_neighbors + 1, len(labels))
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nn.kneighbors(coords)

    refined = []
    for row in indices:
        neighbor_labels = labels[row[1:]]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        refined.append(values[np.argmax(counts)])
    return np.asarray(refined, dtype=str)


def cluster_embedding(embedding, n_clusters, coords=None, refine=True, random_seed=2020):
    """Full protocol: mclust-equivalent clustering, then optional spatial refinement.

    Use this for every method being compared so the comparison stays fair.
    """
    labels = mclust_equivalent(embedding, n_clusters, random_seed=random_seed)
    if refine and coords is not None:
        labels = refine_labels_spatial(labels, coords)
    return labels


def consensus_cluster(label_sets, n_clusters):
    """Combine several independent label assignments of the SAME spots into
    one consensus labeling, via a co-association matrix.

    Motivated by documented high per-seed ARI variance in the address-
    propagation model: each training run randomly initializes its own
    memory_keys/memory_values, so different seeds' embeddings live in
    unrelated coordinate systems -- averaging the raw embeddings across seeds
    was tested and gave a mixed, unreliable result (helped on some slices,
    hurt on others). Combining at the LABEL level instead is coordinate-
    system-independent: co_association[i, j] is simply the fraction of runs
    that placed spots i and j in the same cluster, regardless of what each
    run privately called that cluster or how its embedding was oriented.

    label_sets: list of label arrays, one per run, all for the same n spots.
    """
    n_spots = len(label_sets[0])
    co_association = np.zeros((n_spots, n_spots))
    for labels in label_sets:
        labels = np.asarray(labels)
        co_association += (labels[:, None] == labels[None, :]).astype(float)
    co_association /= len(label_sets)

    distance = 1 - co_association
    np.fill_diagonal(distance, 0)
    # AgglomerativeClustering on a precomputed distance matrix, not scipy's
    # linkage()+fcluster(criterion="maxclust"): the latter raised
    # "Linkage 'Z' contains excessive observations in a cluster" on a real
    # slice where one method's near-identical labels across seeds (GraphST is
    # very low-variance) produced a co-association matrix with many exact
    # ties, which is exactly the kind of degenerate input scipy's
    # maxclust cut is fragile to. AgglomerativeClustering is the standard,
    # more robust tool for "fixed K from a precomputed distance matrix".
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    return model.fit_predict(distance).astype(str)
