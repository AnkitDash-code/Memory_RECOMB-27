import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.eval.clustering import cluster_embedding, mclust_equivalent, refine_labels_spatial


def test_mclust_equivalent_recovers_known_blobs():
    rng = np.random.default_rng(0)
    n_per, n_clusters = 60, 3
    blobs = [rng.normal(loc=i * 15.0, scale=0.6, size=(n_per, 6)) for i in range(n_clusters)]
    embedding = np.vstack(blobs)
    truth = np.repeat(np.arange(n_clusters), n_per)

    labels = mclust_equivalent(embedding, n_clusters)

    assert len(np.unique(labels)) == n_clusters
    assert adjusted_rand_score(truth, labels) > 0.95


def test_refine_labels_spatial_cleans_salt_and_pepper():
    # Two spatially separated bands, with a few spots mislabeled inside each.
    coords = np.array([[x, 0.0] for x in range(40)] + [[x, 100.0] for x in range(40)])
    labels = np.array(["0"] * 40 + ["1"] * 40)

    noisy = labels.copy()
    noisy[5], noisy[17] = "1", "1"   # wrong label inside band 0
    noisy[50], noisy[63] = "0", "0"  # wrong label inside band 1

    refined = refine_labels_spatial(noisy, coords, n_neighbors=10)

    assert adjusted_rand_score(labels, refined) > adjusted_rand_score(labels, noisy)


def test_refine_labels_spatial_preserves_clean_labels():
    coords = np.array([[x, 0.0] for x in range(30)] + [[x, 500.0] for x in range(30)])
    labels = np.array(["0"] * 30 + ["1"] * 30)

    refined = refine_labels_spatial(labels, coords, n_neighbors=5)

    assert adjusted_rand_score(labels, refined) == 1.0


def test_cluster_embedding_end_to_end_shape():
    rng = np.random.default_rng(1)
    embedding = np.vstack([rng.normal(loc=i * 12.0, scale=0.5, size=(25, 5)) for i in range(4)])
    coords = np.array([[i, j] for i in range(10) for j in range(10)])

    labels = cluster_embedding(embedding, n_clusters=4, coords=coords, refine=True)

    assert len(labels) == embedding.shape[0]
    assert 1 <= len(np.unique(labels)) <= 4
